from numpy import mean, max, min, std, median
from typing import List
from matplotlib import pyplot as plt
from pytorch_lightning import LightningDataModule

from InnerEye.ML.configs.histo_configs.classification.DeepSMILECrck import TcgaCrckImageNetMIL


def get_number_samples_per_bag(data_module: LightningDataModule, stage: str) -> List[int]:
    dist = []
    dataloader_fn = stage + '_dataloader'
    data_loader = getattr(data_module, dataloader_fn)()
    dataset = data_loader.dataset
    sampler = dataset.bag_sampler
    n_bags = len(sampler.unique_bag_ids)
    bag_sequence = range(n_bags)
    for id in bag_sequence:
        bag = sampler.get_bag(id)
        dist.append(len(bag))
    return dist


container = TcgaCrckImageNetMIL()
data_module = container.get_data_module()
train_samples_per_bag = get_number_samples_per_bag(data_module, 'train')
val_samples_per_bag = get_number_samples_per_bag(data_module, 'val')
test_samples_per_bag = get_number_samples_per_bag(data_module, 'test')

all_samples_per_bag = train_samples_per_bag + val_samples_per_bag + test_samples_per_bag

plt.figure()
plt.hist(all_samples_per_bag, bins=50)
plt.gca().set(title='Distribution of the number of samples per bag', xlabel='Number_of_samples_in_bag', ylabel='Frequency')
plt.savefig('./outputs/TcgaCrck_samplesperbag.png')

print("Mean number of samples in the bag: ", mean(all_samples_per_bag))
print("Standard deviation of number of samples in the bag: ", std(all_samples_per_bag))
print("Minimum number of number of samples in the bag: ", min(all_samples_per_bag))
print("Maximum number of number of samples in the bag: ", max(all_samples_per_bag))
print("Median number of samples in the bag: ", median(all_samples_per_bag))
