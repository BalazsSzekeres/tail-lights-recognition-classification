from data_processing import DataLoader, DataFormatting
import time
if __name__ == "__main__":
    data_loader = DataLoader(root_directory="./rear_signal_dataset/", difficulty="Easy")
    data_formatter = DataFormatting(data_loader.filtered_list)

    start_time = time.time()
    data_list = data_formatter.get_list_dict_async()
    print(data_list)
    print(f"Async takes a total of {time.time() - start_time} seconds")

    start_time = time.time()
    data_list = data_formatter.get_list_dict()
    print(f"Sync takes a total of {time.time() - start_time} seconds")


