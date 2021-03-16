import torch
import torch.utils.data as tud
from sklearn.metrics import f1_score

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