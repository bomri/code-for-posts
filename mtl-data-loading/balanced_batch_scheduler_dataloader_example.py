import torch
from torch.utils.data.dataset import ConcatDataset
from balanced_sampler import BalancedBatchSchedulerSampler
from basic_dataset_example import MyFirstDataset, MySecondDataset

first_dataset = MyFirstDataset()
second_dataset = MySecondDataset()
concat_dataset = ConcatDataset([first_dataset, second_dataset])

batch_size = 8

# dataloader with BalancedBatchSchedulerSampler
dataloader = torch.utils.data.DataLoader(dataset=concat_dataset,
                                         sampler=BalancedBatchSchedulerSampler(dataset=concat_dataset,
                                                                               batch_size=batch_size),
                                         batch_size=batch_size,
                                         shuffle=False)

for inputs in dataloader:
    print(inputs)
