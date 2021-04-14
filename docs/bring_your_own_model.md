# Bring Your Own PyTorch Lightning Model

The InnerEye toolbox is capable of training any PyTorch Lighting model inside of AzureML, making
use of all the usual InnerEye toolbox features:
- Working with different model in the same codebase, and selecting one by name
- Distributed training in AzureML
- Logging via AzureML's native capabilities
- Training on a local GPU machine or inside of AzureML without code changes
- Supply commandline overrides for model configuration elements, to quickly queue many jobs

## Setup

In order to use these capabilities, you need to implement a class deriving from `LightningContainer`. This class
encapsulates everything that is needed for training with PyTorch Lightning:
- The `create_model` method needs to return a subclass of `LightningModule`, that has
all the usual PyTorch Lightning methods required for training, like the `training_step` and `forward` methods. This
object needs to adhere to additional constraints, see below.
- The `get_data_module` method needs to return a `LightningDataModule` that has the data loaders for
training and validation data.
- The optional `get_inference_data_module` returns a `LightningDataModule` that is used to read the data for inference
(that is, evaluating the trained model). By default, this returns the same data as `get_training_data_module`, but you
can override this for special models like segmentation models that are trained on equal sized image patches, but 
evaluated on full images of varying size.

Your class needs to be defined in a Python file in the `InnerEye/ML/configs` folder, otherwise it won't be picked up
correctly. If you'd like to have your model defined in a different folder, please specify the Python namespace via
the `--model_configs_namespace` argument. For example, use `--model_configs_namespace=My.Own.configs` if your
model configuration classes reside in folder `My/Own/configs`.
 
There are further requirements for the object returned by `create_model`, as described below.

### Outputting files during training

The Lightning model returned by `create_model` can inherit the fields defined in `OutputParams`.
All files that are written during training should go into the folder `self.outputs_folder` (of type `pathlib.Path`).
Files written into this folder will get uploaded to blob storage at the end of the AzureML job at the very latest.

Additional log files need to go into the folder `self.logs_folder` (of type `pathlib.Path`). These files will be
streamed to blob storage during training in AzureML.

If respecting those guidelines is not feasible, we advise that all output files should go to the current working 
directory, rather than an absolute path.

### Trainer arguments
The Lightning model returned by `create_model` can optionally inherit the fields defined in `TrainerParams`.
This defines all arguments that control the PyTorch Lightning `Trainer` object that will be created by the InnerEye
toolbox. Most importantly, the `num_epochs` controls the `max_epochs` argument of the `Trainer`.
For further details how the `TrainerParams` are used, refer to the `create_lightning_trainer` method in 
`InnerEye/ML/model_training.py`

### Optimizer and LR scheduler arguments
Optionally, the Lightning model returned by `create_model` can inherit from the class 
`OptimizerParams`. `OptimizerParams` defines fields that control the PyTorch optimizer and learning rate scheduler. 
If you do not wish to make use of these, your Lightning model needs to also implement the `configure_optimizers` 
method. This method has the same signature as defined in `LightningModule.configure_optimizers`

### Evaluating the trained model
The InnerEye toolbox provides two possible routes of implementing that:

You can either use PyTorch Lightning's built-in capabilities, via the `test_step` method. If the model that is
returned by `create_model` implements the `test_step` method, the InnerEye toolbox will use the `trainer.test` method
(see [docs](https://pytorch-lightning.readthedocs.io/en/latest/common/trainer.html?highlight=trainer.test#test)).
In this case, the best checkpoint during training will be used. The test data is read via the data loader created
by the `test_dataloader` of the `LightningDataModule` that is used for training/validation.

Alternatively, the model can implement the methods defined in `LightningInference`. In this case, the methods will be
call in this order:
```
model.inference_start()
for dataset_split in [Train, Val, Test]
    model.on_inference_epoch_start(dataset_split, is_ensemble_model=False)
    for batch_idx, item in enumerate(dataloader[dataset_split])):
        model_outputs = model.forward(item)
        model.inference_step(item, batch_idx, model_outputs)
    model.on_inference_epoch_end()
model.on_inference_end()
```

## Overriding properties on the commandline

You can define hyperparameters that affect data and/or model, as in the following code snippet: 
```python
import param
from InnerEye.ML.lightning_container import LightningContainer, LightningWithInference
class DummyContainerWithParameters(LightningContainer):
    container_param = param.String(default="foo")
    def __init__(self):
        super().__init__()

    def create_lightning_module(self) -> LightningWithInference:
        return MyLightningModel(self.container_param)
```
All parameters so added will be automatically accessible from the commandline: When starting
training, you can add a flag like `--container_param=bar`.


## Examples


