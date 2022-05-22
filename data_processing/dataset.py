from typing import List, Dict

import torch
from torch.utils.data import Dataset

from data_processing.frame_entry import FrameEntry


class SequenceDataSet(Dataset):
    SequenceMap = Dict[str, List[FrameEntry]]

    def __init__(self, sequences: SequenceMap, letter_idx):
        self.letter_idx = letter_idx
        self.sequences = [
            {
                # "name": name,
                "sequence": torch.stack([e.picture for e in entry_list]),
                "class": self.class_to_tensor(entry_list[0].data_class),
            }
            for name, entry_list in sequences.items()
        ]
        self.sequence_tuples = [tuple(seq.values()) for seq in self.sequences]

    def __len__(self):
        return len(self.sequence_tuples)

    def __getitem__(self, idx):
        return self.sequence_tuples[idx]

    @staticmethod
    def collate_fn(data):
        return [el[0] for el in data], torch.stack([el[1] for el in data])

    def class_to_tensor(self, class_name):
        if class_name[self.letter_idx] == "O":
            return torch.Tensor([1, 0])
        else:
            return torch.Tensor([0, 1])