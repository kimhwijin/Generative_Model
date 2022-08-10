from torch.utils.data import Dataset, DataLoader
from torchvision import datasets


class VAEFashionMNIST(Dataset):
    def __init__(self):
        datasets.FashionMNIST('datasets', 'train')