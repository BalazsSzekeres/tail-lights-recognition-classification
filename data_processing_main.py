import ahocorasick
from data_processing import DataLoader, get_list_of_pictures, findit_with_ahocorasick


if __name__ == "__main__":

    data_loader = DataLoader(root_directory="./rear_signal_dataset/", difficulty="Easy")
    print(data_loader())




