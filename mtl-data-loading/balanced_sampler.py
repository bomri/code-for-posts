import torch
from torch.utils.data import RandomSampler
from sampler import ImbalancedDatasetSampler


class ExampleImbalancedDatasetSampler(ImbalancedDatasetSampler):
    """
    ImbalancedDatasetSampler is taken from https://github.com/ufoym/imbalanced-dataset-sampler/blob/master/sampler.py
    In order to be able to show the usage of ImbalancedDatasetSampler in this example I am editing the _get_label
    to fit my datasets
    """
    def _get_label(self, dataset, idx):
        return dataset.samples[idx].item()


class BalancedBatchSchedulerSampler(torch.utils.data.sampler.Sampler):
    """
    iterate over tasks and provide a balanced batch per task in each mini-batch
    """
    def __init__(self, dataset, batch_size):
        self.dataset = dataset
        self.batch_size = batch_size
        self.number_of_datasets = len(dataset.datasets)

    def __len__(self):
        return len(self.dataset) * self.number_of_datasets

    def __iter__(self):
        samplers_list = []
        sampler_iterators = []
        datasets_length = []
        for dataset_idx in range(self.number_of_datasets):
            cur_dataset = self.dataset.datasets[dataset_idx]
            if dataset_idx == 0:
                # the first dataset is kept at RandomSampler
                sampler = RandomSampler(cur_dataset)
            else:
                # the second unbalanced dataset is changed
                sampler = ExampleImbalancedDatasetSampler(cur_dataset)
            samplers_list.append(sampler)
            cur_sampler_iterator = sampler.__iter__()
            sampler_iterators.append(cur_sampler_iterator)
            datasets_length.append(len(cur_dataset))

        push_index_val = [0] + self.dataset.cumulative_sizes[:-1]
        step = self.batch_size * self.number_of_datasets
        samples_to_grab = self.batch_size
        largest_dataset_index = torch.argmax(torch.as_tensor(datasets_length)).item()
        # for this case we want to get all samples in dataset, this force us to resample from the smaller datasets
        epoch_samples = datasets_length[largest_dataset_index] * self.number_of_datasets

        final_samples_list = []  # this is a list of indexes from the combined dataset
        for _ in range(0, epoch_samples, step):
            for i in range(self.number_of_datasets):
                cur_batch_sampler = sampler_iterators[i]
                cur_samples = []
                for _ in range(samples_to_grab):
                    try:
                        cur_sample_org = cur_batch_sampler.__next__()
                        cur_sample = cur_sample_org + push_index_val[i]
                        cur_samples.append(cur_sample)
                    except StopIteration:
                        if i == largest_dataset_index:
                            # largest dataset iterator is done we can break
                            samples_to_grab = len(cur_samples)  # adjusting the samples_to_grab
                            break  # got to the end of iterator - extend final list and continue to next task
                        else:
                            # restart the iterator - we want more samples until finishing with the largest dataset
                            sampler_iterators[i] = samplers_list[i].__iter__()
                            cur_batch_sampler = sampler_iterators[i]
                            cur_sample_org = cur_batch_sampler.__next__()
                            cur_sample = cur_sample_org + push_index_val[i]
                            cur_samples.append(cur_sample)
                final_samples_list.extend(cur_samples)

        return iter(final_samples_list)