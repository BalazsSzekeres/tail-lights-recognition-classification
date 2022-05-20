from tqdm import tqdm

from data_processing import DataLoader, DataProcessor, read_image_to_tensor
import time

if __name__ == "__main__":
    data_loader = DataLoader(root_directory="../rear_signal_dataset/", difficulty="Easy")

    data_processor = DataProcessor(data_loader.filtered_list)
    data_list = data_processor.get_frame_list()

    for i, element in enumerate(tqdm(data_list)):
        data_list[i].picture = read_image_to_tensor(file_path=element.location)
