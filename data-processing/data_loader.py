import typing
from typing import List, Tuple


class DataLoader:
    def __init__(self, root_directory: str, difficulty: str = 'Easy', type_filter: dict = None):
        raise NotImplementedError

    def read_file_list(self) -> List[str]:
        raise NotImplementedError

    def load_files(self) -> List[typing.Dict[str, str, str]]:
        raise NotImplementedError
