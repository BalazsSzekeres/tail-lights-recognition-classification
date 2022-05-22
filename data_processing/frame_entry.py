from dataclasses import dataclass
import torch

@dataclass
class FrameEntry:
    name: str
    location: str
    data_class: str
    picture: torch.Tensor = None