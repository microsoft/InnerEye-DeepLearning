# Training a Hello World segmentation model

In the configs folder, you will find a config file called [HelloWorld.py](../InnerEye/ML/configs/segmentation/HelloWorld.py) 
We have created this file to demonstrate how to:

1. Subclass SegmentationModelBase which is the base config for all segmentation model configs
1. Configure the UNet3D implemented in this package
1. Configure Azure HyperDrive based parameter search

- This model can be trained from the commandline: ../InnerEye/runner.py --model=HelloWorld
- If you have set up AzureML then parameter search can be performed for this model by running:
../InnerEye/runner.py --model=HelloWorld --hyperdrive=True
