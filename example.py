import glob

import torch
import yaml
from net.Net import Net
from PIL import Image
from torchsummary import summary
from torchvision import transforms

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

path = "/home/bramvr/datasets/rear_signal_dataset_sampleset/lighting"

transform = transforms.Compose([transforms.Resize(size=(227, 227)),
                                transforms.ToTensor()])
images = []
for i, f in enumerate(glob.iglob(f"{path}/*.png")):
    if i == 1:
        break
    images.append(transform(Image.open(f)))
input = torch.stack(images).to(device)

config_file = "config/net/net.yaml"
config = yaml.load(open(config_file), Loader=yaml.FullLoader)

net = Net(config).to(device)
summary(net.model['cnn'], (config["input"]["dim"], config["input"]["size"], config["input"]["size"]))

checkpoint = torch.load('weights/weights_29.pt', map_location=device)
net.load_state_dict(checkpoint['model'])

# Example usage
y_list = net.forward_sequence_list([input, input])
y1 = net(input)
print(y1)
print(y_list)