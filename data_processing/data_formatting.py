from dataclasses import dataclass
from multiprocessing import Pool, cpu_count, Manager
from typing import List
import re

import torch
from tqdm import tqdm
from functools import partial
import ahocorasick

from data_processing import DataLoader, read_image_to_tensor

classes = ['OOO', 'BOO', 'OLO', 'BLO', 'OOR', 'BOR', 'OLR', 'BLR']


@dataclass
class FrameEntry:
    name: str
    location: str
    data_class: str
    picture: torch.Tensor = None


class DataProcessor:
    def __init__(self, raw_data: List):
        self.raw_data = raw_data

    def get_frame_list(self):
        automaton = ahocorasick.Automaton()
        for name in classes:
            automaton.add_word(name, name)
        automaton.make_automaton()

        data_dict_list = []

        for element in tqdm(self.raw_data):
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


def append_entry(list_to_append, automaton_instance, raw_data_point):
    new_entry = FrameEntry(name=raw_data_point[1],
                           location=raw_data_point[0],
                           data_class=findit_with_ahocorasick_name(automaton_instance, raw_data_point[0]),
                           #picture=read_image_to_tensor(raw_data_point[0])
                           )

    list_to_append.append(new_entry)


def findit_with_ahocorasick_name(automaton, element):
    try:
        class_id = next(automaton.iter(element))[1]
        return class_id
    except StopIteration:
        return None
