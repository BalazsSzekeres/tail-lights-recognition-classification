from .data_loader import DataLoader

class DataVisualization:
    def __init__(self, data: DataLoader):
        raise NotImplementedError

    def plot_frame(self, frame_number):
        raise NotImplementedError

    def plot_random(self, number_of_frames: int = 5):
        raise NotImplementedError
