from concurrent.futures import ProcessPoolExecutor
from multiprocessing import cpu_count, Pool

from tqdm import tqdm

from data_processing import DataLoader, DataProcessor, read_image_to_tensor
import time


def new_read_img(data_list_element):
    location = data_list_element.location
    return read_image_to_tensor(location)


if __name__ == "__main__":
    data_loader = DataLoader(root_directory="./rear_signal_dataset/", difficulty="Easy")

    data_processor = DataProcessor(data_loader.filtered_list)
    data_list = data_processor.get_frame_list()

    # for i, element in enumerate(tqdm(data_list)):
    #     data_list[i].picture = read_image_to_tensor(file_path=element.location)

    with Pool(processes=(cpu_count())) as pool:
        max_ = len(data_list)
        with tqdm(total=max_) as pbar:
            for i, _ in enumerate(pool.imap_unordered(new_read_img, data_list)):
                pbar.update()

    # max_ = len(data_list)
    # with tqdm(total=max_) as pbar:
    #     with ProcessPoolExecutor(max_workers=cpu_count()) as executor:
    #         for i, r in enumerate(executor.map(new_read_img, data_list)):
    #             data_list[i].picture = r
    #             pbar.update()
