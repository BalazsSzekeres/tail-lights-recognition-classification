from typing import List, Dict

import numpy as np
import torch
from torch.utils.data import Dataset
from tqdm import tqdm

from data_processing.frame_entry import FrameEntry
from feature_extraction.FeatureExtractionROI import FeatureExtractionROI


class SequenceDataSet:
    SequenceMap = Dict[str, List[FrameEntry]]

    def __init__(self, sequences: SequenceMap, letter_idx):
        self.letter_idx = letter_idx
        self.sequences = []
        # self.sequences = []
        # all_names = sequences.keys()
        # for name in all_names:
        #     entry_list = sequences[name]
        # self.sequences.append({
        #     # "name": name,
        #     "sequence": torch.stack([e.picture for e in entry_list]),
        #     "class": self.class_to_tensor(entry_list[0].data_class),
        # })
        # del sequences[name]
        # for name, entry_list in sequences.items():
        #     self.sequences.append({
        #         # "name": name,
        #         "sequence": torch.stack([e.picture for e in entry_list]),
        #         "class": self.class_to_tensor(entry_list[0].data_class),
        #     })

        self.sequences = [
            self.make_sequence([e.picture for e in entry_list], entry_list[0].data_class)
            for name, entry_list in tqdm(sequences.items())
        ]
        self.sequence_tuples = [tuple(seq.values()) for seq in self.sequences]

    def __len__(self):
        return len(self.sequence_tuples)

    def __getitem__(self, idx):
        return self.sequence_tuples[idx]

    def make_sequence(self, picture_list, class_name):
        if self.letter_idx in [1, 2]:
            # Apply feature extraction
            # Opencv2 : shape is H x W x C
            # Pytorch : shape is C x H X W
            picture_list = [(picture.numpy().transpose(1, 2, 0) * 255).astype('uint8') for picture in picture_list]
            feature_extraction_roi = FeatureExtractionROI(self.letter_idx)
            picture_list = feature_extraction_roi.extract_roi_sequence(picture_list)
            picture_list = [(picture.transpose(2, 0, 1) / 255).astype(np.float32) for picture in picture_list]
            picture_list = [torch.from_numpy(picture) for picture in picture_list]
        return {
            # "name": name,
            "sequence": torch.stack(picture_list).to('cuda'),
            "class": self.class_to_tensor(class_name),
        }

    @staticmethod
    def collate_fn(data):
        return [el[0] for el in data], torch.stack([el[1] for el in data])

    def class_to_tensor(self, class_name):
        if class_name[self.letter_idx] == "O":
            return torch.Tensor([1, 0])
            # return torch.Tensor([1])
        else:
            return torch.Tensor([0, 1])
            # return torch.Tensor([0])


class CustomDataLoader:
    def __init__(self, dataset: SequenceDataSet, batch_size, shuffle=True):
        self.dataset = dataset
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.idx = []
        self.iter_i = 0

    def __iter__(self):
        if self.shuffle:
            self.idx = np.random.permutation(len(self.dataset)).astype(int)
        else:
            self.idx = np.arange(len(self.dataset), dtype=int)
        self.iter_i = 0
        return self

    def __len__(self):
        return len(self.dataset) // self.batch_size + 1

    def __next__(self):
        if self.iter_i == len(self.dataset):
            raise StopIteration
        if self.iter_i + self.batch_size > len(self.dataset):
            indices = self.idx[self.iter_i:]
            self.iter_i = len(self.dataset)
        else:
            indices = self.idx[self.iter_i:self.iter_i + self.batch_size]
            self.iter_i += self.batch_size
        return self.dataset.collate_fn([self.dataset[i] for i in indices])
