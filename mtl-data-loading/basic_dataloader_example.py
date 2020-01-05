import torch
from torch.utils.data.dataset import ConcatDataset
from basic_dataset_example import MyFirstDataset, MySecondDataset


first_dataset = MyFirstDataset()
second_dataset = MySecondDataset()
concat_dataset = ConcatDataset([first_dataset, second_dataset])

batch_size = 8

# basic dataloader
dataloader = torch.utils.data.DataLoader(dataset=concat_dataset,
                                         batch_size=batch_size,
                                         shuffle=True)

for inputs in dataloader:
    print(inputs)
