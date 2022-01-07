#  ------------------------------------------------------------------------------------------
#  Copyright (c) Microsoft Corporation. All rights reserved.
#  Licensed under the MIT License (MIT). See LICENSE in the repo root for license information.
#  ------------------------------------------------------------------------------------------

import logging
from pathlib import Path
import pandas as pd
import numpy as np
from typing import Any, Callable, Dict, Optional, Tuple, List
import torch

from pytorch_lightning import LightningModule
from torch import Tensor, argmax, mode, nn, no_grad, optim, round
from torchmetrics import AUROC, F1, Accuracy, Precision, Recall

from InnerEye.Common import fixed_paths
from InnerEye.ML.Histopathology.datasets.base_dataset import TilesDataset
from InnerEye.ML.Histopathology.models.encoders import TileEncoder
from InnerEye.ML.Histopathology.utils.metrics_utils import select_k_tiles, plot_slide_noxy, plot_scores_hist, plot_heatmap_slide, plot_slide
from InnerEye.ML.Histopathology.utils.naming import ResultsKey

from InnerEye.ML.Histopathology.datasets.panda_dataset import PandaDataset
from InnerEye.ML.Histopathology.datasets.panda_tiles_dataset import PandaTilesDataset
from monai.data.dataset import Dataset
from InnerEye.ML.Histopathology.utils.viz_utils import load_image_dict
from InnerEye.ML.Histopathology.utils.naming import SlideKey


RESULTS_COLS = [ResultsKey.SLIDE_ID, ResultsKey.TILE_ID, ResultsKey.IMAGE_PATH, ResultsKey.PROB,
                ResultsKey.PRED_LABEL, ResultsKey.TRUE_LABEL, ResultsKey.BAG_ATTN]


def _format_cuda_memory_stats() -> str:
    return (f"GPU {torch.cuda.current_device()} memory: "
            f"{torch.cuda.memory_allocated() / 1024 ** 3:.2f} GB allocated, "
            f"{torch.cuda.memory_reserved() / 1024 ** 3:.2f} GB reserved")


class DeepMILModule(LightningModule):
    """Base class for deep multiple-instance learning"""

    def __init__(self,
                 label_column: str,
                 n_classes: int,
                 encoder: TileEncoder,
                 pooling_layer: Callable[[int, int, int], nn.Module],
                 pool_hidden_dim: int = 128,
                 pool_out_dim: int = 1,
                 class_weights: Optional[Tensor] = None,
                 l_rate: float = 5e-4,
                 weight_decay: float = 1e-4,
                 adam_betas: Tuple[float, float] = (0.9, 0.99),
                 verbose: bool = False,
                 ) -> None:
        """
        :param label_column: Label key for input batch dictionary.
        :param n_classes: Number of output classes for MIL prediction.
        :param encoder: The tile encoder to use for feature extraction. If no encoding is needed,
        you should use `IdentityEncoder`.
        :param pooling_layer: Type of pooling to use in multi-instance aggregation. Should be a
        `torch.nn.Module` constructor accepting input, hidden, and output pooling `int` dimensions.
        :param pool_hidden_dim: Hidden dimension of pooling layer (default=128).
        :param pool_out_dim: Output dimension of pooling layer (default=1).
        :param class_weights: Tensor containing class weights (default=None).
        :param l_rate: Optimiser learning rate.
        :param weight_decay: Weight decay parameter for L2 regularisation.
        :param adam_betas: Beta parameters for Adam optimiser.
        :param verbose: if True statements about memory usage are output at each step
        """
        super().__init__()

        # Dataset specific attributes
        self.label_column = label_column
        self.n_classes = n_classes
        self.pool_hidden_dim = pool_hidden_dim
        self.pool_out_dim = pool_out_dim
        self.pooling_layer = pooling_layer
        self.class_weights = class_weights
        self.encoder = encoder
        self.num_encoding = self.encoder.num_encoding

        # Optimiser hyperparameters
        self.l_rate = l_rate
        self.weight_decay = weight_decay
        self.adam_betas = adam_betas

        self.save_hyperparameters()
        self.verbose = verbose

        self.aggregation_fn, self.num_pooling = self.get_pooling()
        self.classifier_fn = self.get_classifier()
        self.loss_fn = self.get_loss()
        self.activation_fn = self.get_activation()

        # Metrics Objects
        self.train_metrics = self.get_metrics()
        self.val_metrics = self.get_metrics()
        self.test_metrics = self.get_metrics()

    def get_pooling(self) -> Tuple[Callable, int]:
        pooling_layer = self.pooling_layer(self.num_encoding,
                                           self.pool_hidden_dim,
                                           self.pool_out_dim)
        num_features = self.num_encoding*self.pool_out_dim
        return pooling_layer, num_features

    def get_classifier(self) -> Callable:
        return nn.Linear(in_features=self.num_pooling,
                         out_features=self.n_classes)

    def get_loss(self) -> Callable:
        if self.n_classes > 1:
            return nn.CrossEntropyLoss(weight=self.class_weights.float())    # type: ignore
        else:
            pos_weight = None
            if self.class_weights is not None:
                pos_weight = Tensor([self.class_weights[1]/(self.class_weights[0]+1e-5)])
            return nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    def get_activation(self) -> Callable:
        if self.n_classes > 1:
            return nn.Softmax()
        else:
            return nn.Sigmoid()

    @staticmethod
    def get_bag_label(labels: Tensor) -> Tensor:
        # Get bag (batch) labels as majority vote
        bag_label = mode(labels).values
        return bag_label.view(1)

    def get_metrics(self) -> nn.ModuleDict:
        if self.n_classes > 1:
            return nn.ModuleDict({'accuracy': Accuracy(num_classes=self.n_classes, average='micro'),
                                  'macro_accuracy': Accuracy(num_classes=self.n_classes, average='macro'),
                                  'weighted_accuracy': Accuracy(num_classes=self.n_classes, average='weighted')})
        else:
            return nn.ModuleDict({'accuracy': Accuracy(),
                                   'auroc': AUROC(num_classes=self.n_classes),
                                   'precision': Precision(),
                                   'recall': Recall(),
                                   'f1score': F1()})

    def log_metrics(self,
                    stage: str) -> None:
        valid_stages = ['train', 'test', 'val']
        if stage not in valid_stages:
            raise Exception(f"Invalid stage. Chose one of {valid_stages}")
        for metric_name, metric_object in self.get_metrics_dict(stage).items():
            self.log(f'{stage}/{metric_name}', metric_object, on_epoch=True, on_step=False, logger=True, sync_dist=True)

    def forward(self, images: Tensor) -> Tuple[Tensor, Tensor]:  # type: ignore
        with no_grad():
            H = self.encoder(images)                        # N X L x 1 x 1
        A, M = self.aggregation_fn(H)                       # A: K x N | M: K x L
        M = M.view(-1, self.num_encoding * self.pool_out_dim)
        Y_prob = self.classifier_fn(M)
        return Y_prob, A

    def configure_optimizers(self) -> optim.Optimizer:
        return optim.Adam(self.parameters(), lr=self.l_rate, weight_decay=self.weight_decay,
                          betas=self.adam_betas)

    def get_metrics_dict(self, stage: str) -> nn.ModuleDict:
        return getattr(self, f'{stage}_metrics')

    def _shared_step(self, batch: Dict, batch_idx: int, stage: str) -> Dict[ResultsKey, Tensor]:
        # The batch dict contains lists of tensors of different sizes, for all bags in the batch.
        # This means we can't stack them along a new axis without padding to the same length.
        # We could alternatively concatenate them, but this would require other changes (e.g. in
        # the attention layers) to correctly split the tensors by bag/slide ID.
        bag_labels_list = []
        bag_logits_list = []
        bag_attn_list = []
        for bag_idx in range(len(batch[self.label_column])):
            images = batch[TilesDataset.IMAGE_COLUMN][bag_idx]
            labels = batch[self.label_column][bag_idx]
            bag_labels_list.append(self.get_bag_label(labels))
            logit, attn = self(images)
            bag_logits_list.append(logit.view(-1))
            bag_attn_list.append(attn)
        bag_logits = torch.stack(bag_logits_list)
        bag_labels = torch.stack(bag_labels_list).view(-1)

        if self.n_classes > 1:
            loss = self.loss_fn(bag_logits, bag_labels.long())
        else:
            loss = self.loss_fn(bag_logits.squeeze(1), bag_labels.float())

        probs = self.activation_fn(bag_logits)
        if self.n_classes > 1:
            preds = argmax(probs, dim=1)
        else:
            preds = round(probs)

        loss = loss.view(-1, 1)
        preds = preds.view(-1, 1)
        probs = probs.view(-1, 1)
        bag_labels = bag_labels.view(-1, 1)

        results = dict()
        for metric_object in self.get_metrics_dict(stage).values():
            metric_object.update(preds, bag_labels)
        results.update({ResultsKey.SLIDE_ID: batch[TilesDataset.SLIDE_ID_COLUMN],
                        ResultsKey.TILE_ID: batch[TilesDataset.TILE_ID_COLUMN],
                        ResultsKey.IMAGE_PATH: batch[TilesDataset.PATH_COLUMN], ResultsKey.LOSS: loss,
                        ResultsKey.PROB: probs, ResultsKey.PRED_LABEL: preds,
                        ResultsKey.TRUE_LABEL: bag_labels, ResultsKey.BAG_ATTN: bag_attn_list,
                        ResultsKey.IMAGE: batch[TilesDataset.IMAGE_COLUMN]})

        if (TilesDataset.TILE_X_COLUMN in batch.keys()) and (TilesDataset.TILE_Y_COLUMN in batch.keys()):
            results.update({ResultsKey.TILE_X: batch[TilesDataset.TILE_X_COLUMN],
                           ResultsKey.TILE_Y: batch[TilesDataset.TILE_Y_COLUMN]}
                           )
        else:
            logging.warning("Coordinates not found in batch. If this is not expected check your input tiles dataset.")

        return results

    def training_step(self, batch: Dict, batch_idx: int) -> Tensor:  # type: ignore
        train_result = self._shared_step(batch, batch_idx, 'train')
        self.log('train/loss', train_result[ResultsKey.LOSS], on_epoch=True, on_step=True, logger=True,
                 sync_dist=True)
        if self.verbose:
            print(f"After loading images batch {batch_idx} -", _format_cuda_memory_stats())
        self.log_metrics('train')
        return train_result[ResultsKey.LOSS]

    def validation_step(self, batch: Dict, batch_idx: int) -> Tensor:  # type: ignore
        val_result = self._shared_step(batch, batch_idx, 'val')
        self.log('val/loss', val_result[ResultsKey.LOSS], on_epoch=True, on_step=True, logger=True,
                 sync_dist=True)
        self.log_metrics('val')
        return val_result[ResultsKey.LOSS]

    def test_step(self, batch: Dict, batch_idx: int) -> Dict[ResultsKey, Any]:   # type: ignore
        test_result = self._shared_step(batch, batch_idx, 'test')
        self.log('test/loss', test_result[ResultsKey.LOSS], on_epoch=True, on_step=True, logger=True,
                 sync_dist=True)
        self.log_metrics('test')
        return test_result

    def test_epoch_end(self, outputs: List[Dict[str, Any]]) -> None:  # type: ignore
        # outputs object consists of a list of dictionaries (of metadata and results, including encoded features)
        # It can be indexed as outputs[batch_idx][batch_key][bag_idx][tile_idx]
        # example of batch_key ResultsKey.SLIDE_ID_COL
        # for batch keys that contains multiple values for slides e.g. ResultsKey.BAG_ATTN_COL
        # outputs[batch_idx][batch_key][bag_idx][tile_idx]
        # contains the tile value

        # collate the batches
        results: Dict[str, List[Any]] = {}
        [results.update({col: []}) for col in outputs[0].keys()]
        for key in results.keys():
            for batch_id in range(len(outputs)):
                results[key] += outputs[batch_id][key]

        print("Saving outputs ...")
        # collate at slide level
        list_slide_dicts = []
        list_encoded_features = []
        # any column can be used here, the assumption is that the first dimension is the N of slides
        for slide_idx in range(len(results[ResultsKey.SLIDE_ID])):
            slide_dict = dict()
            for key in results.keys():
                if key not in [ResultsKey.IMAGE, ResultsKey.LOSS]:
                    slide_dict[key] = results[key][slide_idx]
            list_slide_dicts.append(slide_dict)
            list_encoded_features.append(results[ResultsKey.IMAGE][slide_idx])

        print(f"Metrics results will be output to {fixed_paths.repository_root_directory()}/outputs")
        csv_filename = fixed_paths.repository_root_directory() / Path('outputs/test_output.csv')
        encoded_features_filename = fixed_paths.repository_root_directory() / Path('outputs/test_encoded_features.pickle')

        # Collect the list of dictionaries in a list of pandas dataframe and save
        df_list = []
        for slide_dict in list_slide_dicts:
            slide_dict = self.normalize_dict_for_df(slide_dict, use_gpu=False)
            df_list.append(pd.DataFrame.from_dict(slide_dict))
        df = pd.concat(df_list, ignore_index=True)
        df.to_csv(csv_filename, mode='w', header=True)

        # Collect all features in a list and save
        features_list = self.move_list_to_device(list_encoded_features, use_gpu=False)
        torch.save(features_list, encoded_features_filename)

        panda_dir = "/tmp/datasets/PANDA"
        panda_tiles_dir = "/tmp/datasets/PANDA_tiles"
        panda_dataset = Dataset(PandaDataset(root=panda_dir))
        panda_tiles_dataset = PandaTilesDataset(root=panda_tiles_dir)

        print("Selecting tiles ...")
        fn_top_tiles = select_k_tiles(results, n_slides=10, label=1, n_tiles=10, select=('lowest_pred', 'highest_att'))
        fn_bottom_tiles = select_k_tiles(results, n_slides=10, label=1, n_tiles=10, select=('lowest_pred', 'lowest_att'))
        tp_top_tiles = select_k_tiles(results, n_slides=10, label=1, n_tiles=10, select=('highest_pred', 'highest_att'))
        tp_bottom_tiles = select_k_tiles(results, n_slides=10, label=1, n_tiles=10, select=('highest_pred', 'lowest_att'))
        report_cases = {'TP': [tp_top_tiles, tp_bottom_tiles], 'FN': [fn_top_tiles, fn_bottom_tiles]}

        for key in report_cases.keys():
            print(f"Plotting {key} (tiles, thumbnails, attention heatmaps)...")
            output_path = Path(fixed_paths.repository_root_directory(), f'outputs/fig/{key}/')
            Path(output_path).mkdir(parents=True, exist_ok=True)
            nslides = len(report_cases[key][0])
            for i in range(nslides):
                slide, score, paths, top_attn = report_cases[key][0][i]
                fig = plot_slide_noxy(slide, score, paths, top_attn, key + '_top', ncols=4)
                figpath = Path(output_path, f'{slide}_top.png')
                fig.savefig(figpath, bbox_inches='tight')

                slide, score, paths, bottom_attn = report_cases[key][1][i]
                fig = plot_slide_noxy(slide, score, paths, bottom_attn, key + '_bottom', ncols=4)
                figpath = Path(output_path, f'{slide}_bottom.png')
                fig.savefig(figpath, bbox_inches='tight')

                slide_dict = list(filter(lambda entry: entry[SlideKey.SLIDE_ID] == slide, panda_dataset))[0]  # type: ignore
                load_image_dict(slide_dict, level=slide_dict['level'], margin=0)
                slide_image = slide_dict[SlideKey.IMAGE]

                fig = plot_slide(slide_image=slide_image, scale=1.0)
                figpath = Path(output_path, f'{slide}_thumbnail.png')
                fig.savefig(figpath, bbox_inches='tight')

                fig = plot_heatmap_slide(slide=slide, slide_image=slide_image, results=results)
                figpath = Path(output_path, f'{slide}_heatmap.png')
                fig.savefig(figpath, bbox_inches='tight')

        print("Plotting histogram ...")
        fig = plot_scores_hist(results)
        output_path = Path(fixed_paths.repository_root_directory(), 'outputs/fig/hist_scores.png')
        fig.savefig(output_path, bbox_inches='tight')

    @staticmethod
    def normalize_dict_for_df(dict_old: Dict[str, Any], use_gpu: bool) -> Dict:
        # slide-level dictionaries are processed by making value dimensions uniform and converting to numpy arrays.
        # these steps are required to convert the dictionary to pandas dataframe.
        device = 'cuda' if use_gpu else 'cpu'
        dict_new = dict()
        for key, value in dict_old.items():
            if isinstance(value, Tensor):
                value = value.squeeze(0).to(device).numpy()
                if value.ndim == 0:
                    bag_size = len(dict_old[ResultsKey.SLIDE_ID])
                    value = np.full(bag_size, fill_value=value)
            dict_new[key] = value
        return dict_new

    @staticmethod
    def move_list_to_device(list_encoded_features: List, use_gpu: bool) -> List:
        # a list of features on cpu obtained from original list on gpu
        features_list = []
        device = 'cuda' if use_gpu else 'cpu'
        for feature in list_encoded_features:
            feature = feature.squeeze(0).to(device)
            features_list.append(feature)
        return features_list


class DeepMILModule_Panda(DeepMILModule):
    """
    Child class of `DeepMILModule` for deep multiple-instance learning on PANDA dataset
    """
    def __init__(self,
                panda_dir: str,
                tile_size: int = 224,
                **kwargs: Any) -> None:     
        super().__init__(**kwargs)  
        self.panda_dir = panda_dir
        print("in child class")
        self.tile_size = tile_size

    def test_epoch_end(self, outputs: List[Dict[str, Any]]) -> None:  # type: ignore
        # outputs object consists of a list of dictionaries (of metadata and results, including encoded features)
        # It can be indexed as outputs[batch_idx][batch_key][bag_idx][tile_idx]
        # example of batch_key ResultsKey.SLIDE_ID_COL
        # for batch keys that contains multiple values for slides e.g. ResultsKey.BAG_ATTN_COL
        # outputs[batch_idx][batch_key][bag_idx][tile_idx]
        # contains the tile value

        # collate the batches
        results: Dict[str, List[Any]] = {}
        [results.update({col: []}) for col in outputs[0].keys()]
        for key in results.keys():
            for batch_id in range(len(outputs)):
                results[key] += outputs[batch_id][key]

        print("Saving outputs ...")
        # collate at slide level
        list_slide_dicts = []
        list_encoded_features = []
        # any column can be used here, the assumption is that the first dimension is the N of slides
        for slide_idx in range(len(results[ResultsKey.SLIDE_ID])):
            slide_dict = dict()
            for key in results.keys():
                if key not in [ResultsKey.IMAGE, ResultsKey.LOSS]:
                    slide_dict[key] = results[key][slide_idx]
            list_slide_dicts.append(slide_dict)
            list_encoded_features.append(results[ResultsKey.IMAGE][slide_idx])

        print(f"Metrics results will be output to {fixed_paths.repository_root_directory()}/outputs")
        csv_filename = fixed_paths.repository_root_directory() / Path('outputs/test_output.csv')
        encoded_features_filename = fixed_paths.repository_root_directory() / Path('outputs/test_encoded_features.pickle')

        # Collect the list of dictionaries in a list of pandas dataframe and save
        df_list = []
        for slide_dict in list_slide_dicts:
            slide_dict = self.normalize_dict_for_df(slide_dict, use_gpu=False)
            df_list.append(pd.DataFrame.from_dict(slide_dict))
        df = pd.concat(df_list, ignore_index=True)
        df.to_csv(csv_filename, mode='w', header=True)

        # Collect all features in a list and save
        features_list = self.move_list_to_device(list_encoded_features, use_gpu=False)
        torch.save(features_list, encoded_features_filename)

        panda_dataset = Dataset(PandaDataset(root=self.panda_dir))

        print("Selecting tiles ...")
        fn_top_tiles = select_k_tiles(results, n_slides=10, label=1, n_tiles=10, select=('lowest_pred', 'highest_att'))
        fn_bottom_tiles = select_k_tiles(results, n_slides=10, label=1, n_tiles=10, select=('lowest_pred', 'lowest_att'))
        tp_top_tiles = select_k_tiles(results, n_slides=10, label=1, n_tiles=10, select=('highest_pred', 'highest_att'))
        tp_bottom_tiles = select_k_tiles(results, n_slides=10, label=1, n_tiles=10, select=('highest_pred', 'lowest_att'))
        report_cases = {'TP': [tp_top_tiles, tp_bottom_tiles], 'FN': [fn_top_tiles, fn_bottom_tiles]}

        for key in report_cases.keys():
            print(f"Plotting {key} (tiles, thumbnails, attention heatmaps)...")
            output_path = Path(fixed_paths.repository_root_directory(), f'outputs/fig/{key}/')
            Path(output_path).mkdir(parents=True, exist_ok=True)
            nslides = len(report_cases[key][0])
            for i in range(nslides):
                slide, score, paths, top_attn = report_cases[key][0][i]
                fig = plot_slide_noxy(slide, score, paths, top_attn, key + '_top', ncols=4)
                figpath = Path(output_path, f'{slide}_top.png')
                fig.savefig(figpath, bbox_inches='tight')

                slide, score, paths, bottom_attn = report_cases[key][1][i]
                fig = plot_slide_noxy(slide, score, paths, bottom_attn, key + '_bottom', ncols=4)
                figpath = Path(output_path, f'{slide}_bottom.png')
                fig.savefig(figpath, bbox_inches='tight')

                slide_dict = list(filter(lambda entry: entry[SlideKey.SLIDE_ID] == slide, panda_dataset))[0]  # type: ignore
                load_image_dict(slide_dict, level=slide_dict['level'], margin=0)
                slide_image = slide_dict[SlideKey.IMAGE]

                fig = plot_slide(slide_image=slide_image, scale=1.0)
                figpath = Path(output_path, f'{slide}_thumbnail.png')
                fig.savefig(figpath, bbox_inches='tight')

                fig = plot_heatmap_slide(slide=slide, slide_image=slide_image, results=results)
                figpath = Path(output_path, f'{slide}_heatmap.png')
                fig.savefig(figpath, bbox_inches='tight')

        print("Plotting histogram ...")
        fig = plot_scores_hist(results)
        output_path = Path(fixed_paths.repository_root_directory(), 'outputs/fig/hist_scores.png')
        fig.savefig(output_path, bbox_inches='tight')
