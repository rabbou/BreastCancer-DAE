import torch
import torch.utils.data as tud
from sklearn.metrics import f1_score
import numpy as np
import matplotlib.pyplot as plt

class DataSet(object):
    def __init__(self, images, classes):

        assert images.shape[0] == classes.shape[0], ('images.shape: %s labels.shape: %s' % (images.shape,classes.shape))
        self._num_examples = images.shape[0]
        self._images = images
        self._classes = classes
        self._index_in_epoch = 0
    
    @property
    def images(self):
        return self._images
    
    @property
    def classes(self):
        return self._classes
    
class breakHis(tud.Dataset):
    def __init__(self, data, transform=None):
        self.X = data.images
        self.y = data.classes
        self.transform = transform

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        x = self.X[idx]
        y = self.y[idx]
        if self.transform:
            x = self.transform(x)
        return x, y
    
def add_noise(img, strength=25):
    row,col=img.shape[:2]
    mean=0
    var=10
    sigma=var**.5
    noise=np.random.normal(-strength,strength,img.shape)
    noise=noise.reshape(row,col,3)
    img=img+noise
    return img
    
class noisy_unlabelled_breakHis(tud.Dataset):
    def __init__(self, data, noise_strength, transform=None):
        self.X = data
        self.transform = transform
        self.noise_strength = noise_strength

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        x = self.X[idx]
        x_noise = add_noise(x, self.noise_strength)
        if self.transform:
            x = self.transform(x)
            x_noise = self.transform(x_noise)
        return x, x_noise
    
def plot_samples(data):
    fig, axs = plt.subplots(1, 10, constrained_layout = True, figsize=(10,2))
    for k in range(10):
        im = data[k]
        if im.shape[0] == 3:
            im = im.transpose(1, 2, 0)
        axs[k].imshow(im.astype(np.uint8))
        axs[k].axis("off")
    plt.show()