import torch
from torch.utils.data.dataset import ConcatDataset


class MyFirstDataset(torch.utils.data.Dataset):
    def __init__(self):
        # dummy dataset
        self.samples = torch.cat((-torch.ones(5), torch.ones(5)))

    def __getitem__(self, index):
        # change this to your samples fetching logic
        return self.samples[index]

    def __len__(self):
        # change this to return number of samples in your dataset
        return self.samples.shape[0]


class MySecondDataset(torch.utils.data.Dataset):
    def __init__(self):
        # dummy dataset
        self.samples = torch.cat((torch.ones(50) * 5, torch.ones(5) * -5))

    def __getitem__(self, index):
        # change this to your samples fetching logic
        return self.samples[index]

    def __len__(self):
        # change this to return number of samples in your dataset
        return self.samples.shape[0]


first_dataset = MyFirstDataset()
second_dataset = MySecondDataset()
concat_dataset = ConcatDataset([first_dataset, second_dataset])
