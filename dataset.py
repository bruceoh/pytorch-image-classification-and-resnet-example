from torchvision.io import read_image
from torch.utils.data import Dataset
from PIL import Image


class CustomImageDataset(Dataset):
    def __init__(
        self,
        path_list,
        transform,
    ):
        self.transform = transform
        self.path_list = path_list

    def __len__(self):
        return len(self.path_list)

    def __getitem__(self, idx):
        img_path = self.path_list[idx]
        image = Image.open(img_path)

        label = self.get_label(img_path)
        if self.transform:
            image = self.transform(image)

        return image, label

    def get_label(self, image_path: str):
        paths = image_path.split("/")
        classes = ('airplane', 'automobile', 'bird', 'cat',
           'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

        for index, x in enumerate(classes):
            if x == paths[-2]:
                return index
        
        return -1


    