import laspy
import os
from tqdm import tqdm
import numpy as np

dataset_dir = "/home/moussabendjilali/superpoint_transformer/data/fractal/raw"
sum = 0
class_counts = {}
for folder in os.listdir(dataset_dir):
    if folder == "train" or folder == "val":
        continue
    for file in tqdm(os.listdir(os.path.join(dataset_dir, folder))):
        if file.endswith(".laz"):
            las = laspy.read(os.path.join(dataset_dir, folder, file))
            sum += las.xyz.shape[0]
            unique, counts = np.unique(las.classification, return_counts=True)  
            for cls, count in zip(unique, counts):
                if cls not in class_counts:
                    class_counts[cls] = count
                else:
                    class_counts[cls] += count
print("Total number of points: ", sum)
print("Class counts: ", class_counts)
