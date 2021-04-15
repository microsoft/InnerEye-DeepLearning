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
- The `get_data_module` method of the container needs to return a `LightningDataModule` that has the data loaders for
training and validation data.
- The optional `get_inference_data_module` returns a `LightningDataModule` that is used to read the data for inference
(that is, evaluating the trained model). By default, this returns the same data as `get_training_data_module`, but you
can override this for special models like segmentation models that are trained on equal sized image patches, but 
evaluated on full images of varying size.

Your class needs to be defined in a Python file in the `InnerEye/ML/configs` folder, otherwise it won't be picked up
correctly. If you'd like to have your model defined in a different folder, please specify the Python namespace via
the `--model_configs_namespace` argument. For example, use `--model_configs_namespace=My.Own.configs` if your
model configuration classes reside in folder `My/Own/configs` from the repository root.
 
There are further requirements for the object returned by `create_model`, as described below.

### Outputting files during training

The Lightning model returned by `create_model` needs to write its output files to the current working directory.
When running the InnerEye toolbox outside of AzureML, the toolbox will change the current working directory to a 
newly created output folder, with a name that contains the time stamp and and the model name.
When running the InnerEye toolbox in AzureML, the folder structure will be set up such that all files written
to the current working directory are later uploaded to Azure blob storage at the end of the AzureML job. The files
will also be later available via the AzureML UI.

### Trainer arguments
All arguments that control the PyTorch Lightning `Trainer` object are defined in the class `TrainerParams`. A
`LightningContainer` object inherits from this class. The most essential one is the `num_epochs` field, which controls
the `max_epochs` argument of the `Trainer`.

Usage example:
```python
from pytorch_lightning import LightningModule
from InnerEye.ML.lightning_container import LightningContainer
class MyContainer(LightningContainer):
    def __init__(self):
        super().__init__()
        self.num_epochs = 42

    def create_lightning_module(self) -> LightningModule:
        return MyLightningModel()
```

For further details how the `TrainerParams` are used, refer to the `create_lightning_trainer` method in 
`InnerEye/ML/model_training.py`

### Optimizer and LR scheduler arguments
There are two possible ways of choosing the optimizer and LR scheduler:
- The Lightning model returned by `create_model` can define its own `configure_optimizers` method, with the same
signature as `LightningModule.configure_optimizers`. This is the typical way of configuring it for Lightning models.
- Alternatively, the model can inherit from `LightningModuleWithOptimizer`. This class implements a 
`configure_optimizers` method that uses settings defined in the `OptimizerParams` class. These settings are all
available from the command line, and you can, for example, start a new run with a different learning rate by
supplying the additional commandline flag `--l_rate=1e-2`. 

### Evaluating the trained model
The InnerEye toolbox provides two possible routes of implementing that:

You can either use PyTorch Lightning's built-in capabilities, via the `test_step` method. If the model that is
returned by `create_model` implements the `test_step` method, the InnerEye toolbox will use the `trainer.test` method
(see [docs](https://pytorch-lightning.readthedocs.io/en/latest/common/trainer.html?highlight=trainer.test#test)).
In this case, the best checkpoint during training will be used. The test data is read via the data loader created
by the `test_dataloader` of the `LightningDataModule` that is used for training/validation.

Alternatively, the model can implement the methods defined in `InnerEyeInference`. In this case, the methods will be
call in this order:
```
model.on_inference_start()
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
from pytorch_lightning import LightningModule
from InnerEye.ML.lightning_container import LightningContainer
class DummyContainerWithParameters(LightningContainer):
    container_param = param.String(default="foo")
    def __init__(self):
        super().__init__()

    def create_lightning_module(self) -> LightningModule:
        return MyLightningModel(self.container_param)
```
All parameters added in this form will be automatically accessible from the commandline: When starting
training, you can add a flag like `--container_param=bar`.


## Examples


