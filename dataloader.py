import torchvision.datasets as datasets
from torchvision.transforms import ToTensor
from torch.utils.data import DataLoader
from PIL import Image


def get_data(DOWNLOAD_DATASET: bool = False, BATCH_SIZE:int = 32):
    ROOT: str = "./data"

    train_dataset = datasets.MNIST(root=ROOT, train=True, download=DOWNLOAD_DATASET, transform=ToTensor(), target_transform=None)
    test_dataset = datasets.MNIST(root=ROOT, train=False, download=DOWNLOAD_DATASET, transform=ToTensor(), target_transform=None)
    trainLoader = DataLoader(dataset=train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    testLoader = DataLoader(dataset=test_dataset, batch_size=BATCH_SIZE, shuffle=False)
    
    class_names = train_dataset.class_to_idx

    return (trainLoader, testLoader, class_names)
    
