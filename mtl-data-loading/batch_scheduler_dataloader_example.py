import torch
from torch.utils.data.dataset import ConcatDataset
from basic_dataset_example import MyFirstDataset, MySecondDataset
from multi_task_batch_scheduler import BatchSchedulerSampler

first_dataset = MyFirstDataset()
second_dataset = MySecondDataset()
concat_dataset = ConcatDataset([first_dataset, second_dataset])

batch_size = 8

# dataloader with BatchSchedulerSampler
dataloader = torch.utils.data.DataLoader(dataset=concat_dataset,
                                         sampler=BatchSchedulerSampler(dataset=concat_dataset,
                                                                       batch_size=batch_size),
                                         batch_size=batch_size,
                                         shuffle=False)

for inputs in dataloader:
    print(inputs)
