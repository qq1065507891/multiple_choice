import torch

from torch.utils.data import Dataset

from src.utils.config import config


class CustomDataset(Dataset):
    """
    数据集的类
    """
    def __init__(self, data):
        self.input_ids, self.token_type_ids, self.attention_masks, self.labels = data

    def __getitem__(self, item):

        return self.input_ids[item], self.token_type_ids[item], self.attention_masks[item], self.labels[item]

    def __len__(self):
        return len(self.input_ids)


def collate_fn(batch):
    if config.train:

        input_ids = [x[0] for x in batch]
        token_type_ids = [x[1] for x in batch]
        attention_masks = [x[2] for x in batch]
        labels = [x[3] for x in batch]

        return torch.tensor(input_ids), torch.tensor(token_type_ids), \
               torch.tensor(attention_masks), torch.tensor(labels, dtype=torch.int64)
    else:
        input_ids = [x[0] for x in batch]
        token_type_ids = [x[1] for x in batch]
        attention_masks = [x[2] for x in batch]

        return torch.tensor(input_ids), torch.tensor(token_type_ids), torch.tensor(attention_masks)



