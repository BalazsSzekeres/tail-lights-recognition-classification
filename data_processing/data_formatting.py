import random
from functools import partial
from multiprocessing import Pool, cpu_count, Manager
from typing import List

import ahocorasick
import torch as th
from torch.utils.data import Dataset
from tqdm import tqdm

from data_processing.dataset import SequenceDataSet, CustomDataLoader
from data_processing.frame_entry import FrameEntry

classes = ['OOO', 'BOO', 'OLO', 'BLO', 'OOR', 'BOR', 'OLR', 'BLR']


class DataProcessor:
    def __init__(self, raw_data: List):
        self.raw_data = raw_data

    def get_frame_list(self):
        automaton = ahocorasick.Automaton()
        for name in classes:
            automaton.add_word(name, name)
        automaton.make_automaton()

        data_dict_list = []

        for element in self.raw_data:
            append_entry(data_dict_list, automaton, element)

        return data_dict_list

    def get_frame_list_async(self):
        automaton = ahocorasick.Automaton()
        for name in classes:
            automaton.add_word(name, name)
        automaton.make_automaton()

        manager = Manager()
        data_dict_list = manager.list()

        with Pool(processes=(cpu_count())) as pool:
            max_ = len(self.raw_data)
            with tqdm(total=max_) as pbar:
                for i, _ in enumerate(pool.imap_unordered(partial(append_entry,
                                                                  data_dict_list,
                                                                  automaton), self.raw_data)):
                    pbar.update()

        return data_dict_list

    @staticmethod
    def convert_to_train_test(data_dict_list, n_train_sequences=None, n_test_sequences=None, val_ratio=None, shuffle=True, max_sequence_size=None):
        """
        Convert the data_dict_list to sequence maps for training and for testing
        Also return a list of all images to allow easy loading
        """
        sequence_names = list(set(entry.name for entry in data_dict_list))

        all_sequences = {}
        for name in sequence_names:
            all_sequences[name] = sorted([e for e in data_dict_list if name == e.name], key=lambda e: e.name)

            if max_sequence_size:
                all_sequences[name] = all_sequences[name][:max_sequence_size]

        train_sequence_names = [name for name in sequence_names if "test" not in name]
        test_sequence_names = [name for name in sequence_names if "test" in name]

        if shuffle:
            random.shuffle(test_sequence_names)
            random.shuffle(train_sequence_names)
        if n_train_sequences is not None:
            train_sequence_names = train_sequence_names[:n_train_sequences]
        if n_test_sequences is not None:
            test_sequence_names = test_sequence_names[:n_test_sequences]

        # Split train set into train - val
        train_val_split = int(len(train_sequence_names) * val_ratio)
        val_sequence_names = train_sequence_names[:train_val_split]
        train_sequence_names = train_sequence_names[train_val_split:]

        train_sequences = {k: v for k, v in all_sequences.items() if k in train_sequence_names}
        val_sequences = {k: v for k, v in all_sequences.items() if k in val_sequence_names}
        test_sequences = {k: v for k, v in all_sequences.items() if k in test_sequence_names}

        if n_test_sequences is not None or n_train_sequences is not None or max_sequence_size is not None:
            # Update the list of all frame entries
            data_dict_list = []
            for name, frame_list in train_sequences.items():
                data_dict_list += frame_list
            for name, frame_list in val_sequences.items():
                data_dict_list += frame_list
            for name, frame_list in test_sequences.items():
                data_dict_list += frame_list
        return train_sequences, val_sequences, test_sequences, data_dict_list

    def sequence_map_to_torch_loader(self, sequences, batch_size) -> th.utils.data.DataLoader:
        # 0 because we focus on brakes for now
        data_set = SequenceDataSet(sequences, 0)
        # data_loader = CustomDataLoader(data_set, batch_size=batch_size)
        data_loader = th.utils.data.DataLoader(data_set, batch_size=batch_size, collate_fn=data_set.collate_fn,
                                               shuffle=True, pin_memory=True)
        return data_loader


def append_entry(list_to_append, automaton_instance, raw_data_point):
    new_entry = FrameEntry(name=raw_data_point[1],
                           location=raw_data_point[0],
                           data_class=findit_with_ahocorasick_name(automaton_instance, raw_data_point[0]),
                           # picture=read_image_to_tensor(raw_data_point[0])
                           )

    list_to_append.append(new_entry)


def findit_with_ahocorasick_name(automaton, element):
    try:
        class_id = next(automaton.iter(element))[1]
        return class_id
    except StopIteration:
        return None
