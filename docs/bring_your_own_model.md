# Bring Your Own (PyTorch Lightning) Model

The InnerEye toolbox is capable of training any PyTorch Lighting model inside of AzureML, making
use of all the usual InnerEye toolbox features:
- Working with different model in the same codebase, and selecting one by name
- Training on a local GPU machine or inside of AzureML
- Distributed training in AzureML
- Logging via AzureML's native capabilities

## Setup

In order to use these capabilities, you need to implement a class deriving from `LightningContainer`. This class
encapsulates everything that is needed for training with PyTorch Lightning:
- The `create_lightning_module` method needs to return a subclass of `LightningModule`, that has
all the usual PyTorch Lightning methods required for training, like the `training_step` and `forward` methods. This
object needs to adhere to additional constraints, see below.
- The `get_training_data_module` method needs to return a `LightningDataModule` that has the data loaders for
training and validation data.
- The optional `get_inference_data_module` returns a `LightningDataModule` that is used to read the data for inference
(that is, evaluating the trained model). By default, this returns the same data as `get_training_data_module`, but you
can override this for special models like segmentation models that are trained on equal sized image patches, but 
evaluated on full images of varying size.

There are further requirements for the object returned by `create_lightning_module`.

### Outputting files during training

The Lightning model returned by `create_lightning_module` needs to inherit the fields defined in `OutputParams`.
All files that are written during training need to go into the folder `self.outputs_folder` (of type `pathlib.Path`).
Files written into this folder will get uploaded to blob storage at the end of the AzureML job at the very latest.

Additional log files need to go into the folder `self.logs_folder` (of type `pathlib.Path`). These files will be
streamed to blob storage during training in AzureML.

### Trainer arguments
The Lightning model returned by `create_lightning_module` needs to inherit the fields defined in `TrainerParams`.
This defines all arguments that control the PyTorch Lightning `Trainer` object that will be created by the InnerEye
toolbox. Most importantly, the `num_epochs` controls the `max_epochs` argument of the `Trainer`.
For further details how the `TrainerParams` are used, refer to the `create_lightning_trainer` method in 
`InnerEye/ML/model_training.py`


### Optimizer arguments
Optionally, the Lightning model returned by `create_lightning_module` can inherit the fields defined in 
`OptimizerParams`. These fields control the PyTorch optimizer and learning rate scheduler. If you do not wish
to make use of these, your Lightning model needs to also implement the `configure_optimizers` method. This
method has the same signature as defined in `LightningModule.configure_optimizers`


## Overriding properties on the commandline

You can define hyperparameters that affect both data and model, as in the following code snippet: 
```
class DummyContainerWithParameters(LightningContainer):
    container_param = param.String(default="foo")
    def __init__(self):
        super().__init__()

    def create_lightning_module(self) -> LightningWithInference:
        return InferenceWithParameters(self.container_param)
```
All parameters so added will be automatically accessible from the commandline, that is, when starting
training, you can add a flag like `--container_param=bar`.
