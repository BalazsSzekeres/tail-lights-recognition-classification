import os
from typing import List, Optional
# import torch
# from PIL import Image
# import torchvision.transforms as transforms


def get_frame_list(path: str) -> List:
    """
This function reads a textfile that contains the instance numbers.
    :param path: The path to the textfile that contains the instance list.
    :return: A list with the instance numbers.
    """
    with open(path, 'r') as file:
        lines = file.readlines()
        lines = [text_line.rstrip() for text_line in lines]

    return lines


def get_list_of_pictures(directory: str) -> List:
    picture_list = []
    for root, dirs, files in os.walk(directory, topdown=False):
        for name in files:
            if name.endswith('.png'):
                picture_list.append(os.path.join(root, name))
    return picture_list


def findit_with_ahocorasick(automaton, element, list_of_pictures):
    try:
        index = next(automaton.iter(element))[0]
        return list_of_pictures[index]
    except StopIteration:
        return None

def read_image_to_tensor(file_path: str):
    image = Image.open(file_path)
    transform = transforms.ToTensor()
    tensor = transform(image)
    return tensor
