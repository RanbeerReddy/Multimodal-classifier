import numpy as np
import torchvision.transforms as T
from PIL import ImageFilter
from configs.dataset.imagedata_config import CFG


class GaussianBlur:
    """Random Gaussian blur, following the SimCLR augmentation recipe."""

    def __init__(self, radius_min=0.1, radius_max=2.0):
        self.radius_min = radius_min
        self.radius_max = radius_max

    def __call__(self, img):
        radius = np.random.uniform(self.radius_min, self.radius_max)
        return img.filter(ImageFilter.GaussianBlur(radius))


simclr_transform = T.Compose([
    T.RandomResizedCrop(size=CFG["img_size"], scale=(0.3, 1.0)),
    T.RandomHorizontalFlip(p=0.5),
    T.RandomRotation(degrees=10),
    T.RandomAffine(degrees=0, translate=(0.05, 0.05), scale=(0.9, 1.1)),
    T.RandomApply([T.ColorJitter(brightness=0.4, contrast=0.4)], p=0.8),
    T.RandomApply([GaussianBlur()], p=0.5),
    T.Grayscale(num_output_channels=3),
    T.ToTensor(),
    T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])


eval_transform = T.Compose([
    T.Resize((CFG["img_size"], CFG["img_size"])),
    T.Grayscale(num_output_channels=3),
    T.ToTensor(),
    T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])


class SimCLRTransform:
    """Wraps a single transform to produce two augmented views per image (SimCLR)."""

    def __init__(self, transform=simclr_transform):
        self.transform = transform

    def __call__(self, image):
        view1 = self.transform(image)
        view2 = self.transform(image)
        return view1, view2
