import glob

import torch
import yaml
from net.Net import Net
from PIL import Image
from torchsummary import summary
from torchvision import transforms

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

path = "/home/bas/Documents/Master/Y1/Q4/Seminar Computer Vision by Deep " \
       "Learning/project/rear_signal_dataset_copy/20160805_g1k17-08-05-2016_15-57-59_idx99/20160805_g1k17-08-05" \
       "-2016_15-57-59_idx99_BOO/20160805_g1k17-08-05-2016_15-57-59_idx99_BOO_00002671/light_mask"
transform = transforms.Compose([transforms.Resize(size=(227, 227)),
                                transforms.ToTensor()])
images = []
for i, f in enumerate(glob.iglob(f"{path}/*.png")):
    if i == 10:
        break
    images.append(transform(Image.open(f)))
input = torch.stack(images).to(device)

config_file = "config/net.yaml"
config = yaml.load(open(config_file), Loader=yaml.FullLoader)

net = Net(config).to(device)
summary(net.model['cnn'], (config["input"]["dim"], config["input"]["size"], config["input"]["size"]))

with torch.no_grad():
    y = net(input)
    print(y)