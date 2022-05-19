from concurrent.futures import ProcessPoolExecutor
from multiprocessing import Pool, cpu_count, Manager
from typing import List
import re
from tqdm import tqdm
from functools import partial
import ahocorasick

from data_processing import DataLoader, read_image_to_tensor

classes = ['OOO', 'BOO', 'OLO', 'BLO', 'OOR', 'BOR', 'OLR', 'BLR']


class DataFormatting:
    def __init__(self, raw_data: List):
        self.raw_data = raw_data

    def get_list_dict(self):
        automaton = ahocorasick.Automaton()
        for name in classes:
            automaton.add_word(name, name)
        automaton.make_automaton()

        data_dict_list = []

        for element in tqdm(self.raw_data):
            append_entry(data_dict_list, automaton, element)

        return data_dict_list

    def get_list_dict_async(self):
        automaton = ahocorasick.Automaton()
        for name in classes:
            automaton.add_word(name, name)
        automaton.make_automaton()

        manager = Manager()
        data_dict_list = manager.list(self.raw_data)

        # data_dict_list = []
        # data_dict_list = (self.raw_data).copy()

        with Pool(processes=(cpu_count())) as pool:
            max_ = len(self.raw_data)
            with tqdm(total=max_) as pbar:
                for i, _ in enumerate(pool.imap_unordered(partial(get_entry,
                                                                  data_dict_list,
                                                                  automaton), self.raw_data)):
                    pbar.update()
        # max_ = len(self.raw_data)
        # with tqdm(total=max_) as pbar:
        #     with ProcessPoolExecutor(max_workers=cpu_count()) as executor:
        #         for i, r in enumerate(executor.map(partial(get_entry, data_dict_list, automaton), self.raw_data)):
        #             data_dict_list[i] = r
        #             pbar.update()

        return data_dict_list


def append_entry(list_to_append, automaton_instance, raw_data_point):
    new_entry = {
        'name': get_label(raw_data_point),
        'location': raw_data_point,
        'class': findit_with_ahocorasick_name(automaton_instance, raw_data_point),
        'picture': read_image_to_tensor(raw_data_point)}
    # print(new_entry)
    list_to_append.append(new_entry)


def get_entry(list_to_append, automaton_instance, raw_data_point):
    new_entry = {
        'name': get_label(raw_data_point),
        'location': raw_data_point,
        'class': findit_with_ahocorasick_name(automaton_instance, raw_data_point),
        'picture': read_image_to_tensor(raw_data_point)}
    # new_entry = read_image_to_tensor(raw_data_point)
    return new_entry


def get_label(raw_data_point):
    label = re.search('/rear_signal_dataset/(.*?)/light_mask', raw_data_point).group(1)
    return label


def findit_with_ahocorasick_name(automaton, element):
    try:
        class_id = next(automaton.iter(element))[1]
        return class_id
    except StopIteration:
        return None
