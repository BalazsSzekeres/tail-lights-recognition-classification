import argparse
import os
import random
import shutil

import torch
import torch as th
# import torch.nn as nn
# import torch.nn.functional as F
# from torch.utils.data import TensorDataset, DataLoader

import yaml
from tqdm import tqdm

from data_processing import DataLoader, DataProcessor, read_image_to_tensor
import wandb
from torch.utils.data import Dataset

from runner import Runner

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def read_img(data_list_element):
    data_list_element.picture = read_image_to_tensor(data_list_element.location)
    return data_list_element

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("data_root")
    parser.add_argument("-c", "--config", default=os.path.join("config", "run", "default.yaml"))
    return parser.parse_args()


def main():
    args = parse_args()
    config = yaml.safe_load(open(args.config).read())
    torch.cuda.empty_cache()
    random.seed(config["random_seed"])
    torch.manual_seed(config["random_seed"])
    #print(f"{config=}")
    if config["wandb"]:
        # Weights and biases
        wandb.init(project="tail-lights-recognition-classification", entity="bramvanriessen")

        # Weights and biases config
        wandb.config = config
    if config["save_weights"]:
        # Create save location
        save_weights = 'weights'  # Model weights will be saved here.
        shutil.rmtree(save_weights, ignore_errors=True)
        os.makedirs(save_weights)

    # Load dataset
    data_loader = DataLoader(root_directory=args.data_root, difficulty=config["difficulty"])
    data_processor = DataProcessor(data_loader.filtered_list)
    data_list = data_processor.get_frame_list()

    key_val_list = ['OOO', 'BOO']
    data_list = [el for el in data_list if el.data_class in key_val_list]

    train_sequences, val_sequences, test_sequences, data_list = data_processor.convert_to_train_test(
        data_list,
        config.get("n_train"),
        config.get("n_test"),
        config.get("val_ratio"),
        max_sequence_size=config.get("max_sequence_size")
    )

    data_list[:] = map(read_img, tqdm(data_list))
    train_loader = data_processor.sequence_map_to_torch_loader(train_sequences, config["batch_size"])
    val_loader = data_processor.sequence_map_to_torch_loader(val_sequences, config["batch_size"])
    test_loader = data_processor.sequence_map_to_torch_loader(test_sequences, config["batch_size"])

    runner = Runner(train_loader, val_loader, test_loader, config, device)
    runner.run()
    # run('lstm', train_loader, test_loader, save_weights)


if __name__ == "__main__":
    main()
