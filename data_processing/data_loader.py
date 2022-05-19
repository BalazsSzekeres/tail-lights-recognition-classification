import os.path
import typing
from typing import List, Tuple
from .data_processing_helpers import get_frame_list, get_list_of_pictures, findit_with_ahocorasick
import ahocorasick


class DataLoader:
    def __init__(self, root_directory: str, difficulty: str = 'Easy'):

        self.root_directory = root_directory

        self.difficulty = difficulty

        self._easy_files_list: typing.Optional[list] = None
        self._moderate_files_list: typing.Optional[list] = None
        self._hard_files_list: typing.Optional[list] = None

        self._list_of_all_pictures: typing.Optional[list] = None

        self._filtered_list: typing.Optional[list] = None

    def __call__(self, *args, **kwargs):
        return self.filtered_list

    @property
    def easy_files_list(self):
        if self._easy_files_list is not None:
            return self._easy_files_list
        else:
            self._easy_files_list = get_frame_list(os.path.join(self.root_directory, 'Easy.txt'))
            return self._easy_files_list

    @property
    def moderate_files_list(self):
        if self._moderate_files_list is not None:
            return self._moderate_files_list
        else:
            self._moderate_files_list = get_frame_list(os.path.join(self.root_directory, 'Moderate.txt'))
            return self._moderate_files_list

    @property
    def hard_files_list(self):
        if self._hard_files_list is not None:
            return self._hard_files_list
        else:
            self._hard_files_list = get_frame_list(os.path.join(self.root_directory, 'Hard.txt'))
            return self._hard_files_list

    @property
    def list_of_all_pictures(self):
        if self._list_of_all_pictures is not None:
            return self._list_of_all_pictures
        else:
            self._list_of_all_pictures = get_list_of_pictures(self.root_directory)
            return self._list_of_all_pictures

    @property
    def filtered_list(self):
        if self._filtered_list is not None:
            return self._filtered_list
        else:
            self._filtered_list = self.filter_list()
            return self._filtered_list

    def filter_list(self):

        match self.difficulty:
            case 'Easy':
                files = self.easy_files_list
            case 'Moderate':
                files = self.moderate_files_list
            case 'Hard':
                files = self.hard_files_list
            case _:
                raise ValueError("Difficulty must be Easy, Moderate or Hard!")

        automaton = ahocorasick.Automaton()

        for name in files:
            automaton.add_word(name, name)

        automaton.make_automaton()

        output = [findit_with_ahocorasick(automaton, element)
                  for element in self.list_of_all_pictures]

        output = list(filter(None, output))

        return output

