import cv2
from matplotlib import pyplot as plt

from feature_extraction.FeatureExtractionROI import FeatureExtractionROI

fer = FeatureExtractionROI()

p = "/home/bas/Documents/Master/Y1/Q4/Seminar Computer Vision by Deep Learning/project/rear_signal_dataset_copy/" \
    "20160805_g1k17-08-05-2016_15-57-59_idx99/20160805_g1k17-08-05-2016_15-57-59_idx99_OLO/" \
    "20160805_g1k17-08-05-2016_15-57-59_idx99_OLO_00003358/light_mask"
start = 58
imgs = [
    cv2.imread(f'{p}/frame00003367.png'),
    cv2.imread(f'{p}/frame00003368.png'),
    cv2.imread(f'{p}/frame00003369.png'),
]
imgs = [
    cv2.resize(img, (227, 227)) for img in imgs
]
fig = plt.figure(figsize=(15, 5))
fig.add_subplot(1, 2, 1)
plt.imshow(cv2.cvtColor(imgs[1], cv2.COLOR_BGR2RGB))
fig.add_subplot(1, 2, 2)
plt.imshow(cv2.cvtColor(imgs[2], cv2.COLOR_BGR2RGB))

result = fer.extract_roi_sequence(imgs)
fig = plt.figure(figsize=(15, 5))
fig.add_subplot(1, 2, 1)
plt.imshow(cv2.cvtColor(result[1][0], cv2.COLOR_BGR2RGB))
fig.add_subplot(1, 2, 2)
plt.imshow(cv2.cvtColor(result[1][1], cv2.COLOR_BGR2RGB))
plt.show()