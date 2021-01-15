# Training a Hello World segmentation model

In the configs folder, you will find a config file called [HelloWorld.py](../InnerEye/ML/configs/segmentation/HelloWorld.py) 
We have created this file to demonstrate how to:

1. Subclass SegmentationModelBase which is the base config for all segmentation model configs
1. Configure the UNet3D implemented in this package
1. Configure Azure HyperDrive based parameter search

* This model can be trained from the commandline, from the root of the repo: `python InnerEye/runner.py --model=HelloWorld`
* If you want to test your AzureML workspace with the HelloWorld model:
    * Upload to datasets storage account for your AzureML workspace: `Test/ML/test_data/dataset.csv` and
    `Test/ML/test_data/train_and_test_data` and name the folder "hello_world"   
    * If you have set up AzureML then parameter search can be performed for this model by running:
    `python InnerEye/ML/ runner.py --model=HelloWorld --azureml=True --hyperdrive=True`
