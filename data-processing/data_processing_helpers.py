

def get_frame_list(path: str) -> list:
    """
This function reads a textfile that contains the instance numbers.
    :param path: The path to the textfile that contains the instance list.
    :return: A list with the instance numbers.
    """
    with open(path, 'r') as file:
        lines = file.readlines()
        lines = [text_line.rstrip() for text_line in lines]

    return lines
