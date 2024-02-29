import os
from PIL import Image
import torchvision.transforms as transforms
from torch.utils.data import Dataset


def convert_to_rgb(image):
    rgb_image = Image.new("RGB", image.size)
    rgb_image.paste(image)
    return rgb_image


class ImageDataset(Dataset):
    def __init__(self, root_dir, transforms_=None):
        self.root_dir = root_dir
        self.transform = transforms.Compose(transforms_) if transforms_ else None
        # Adjusted to include both '.jpg' and '.jpeg' files
        self.images = [os.path.join(root_dir, img) for img in os.listdir(root_dir) if
                       img.endswith('.png') or img.endswith('.jpg') or img.endswith('.jpeg')]

        # Attempt to determine the size of the first image
        if self.images:
            with Image.open(self.images[0]) as img:
                self.image_size = img.size  # This should be (width, height)
        else:
            self.image_size = (0, 0)  # Default/fallback size

    def __getitem__(self, index):
        # Your existing __getitem__ logic, adjusted for dynamic image size if necessary
        image_path = self.images[index]
        image = Image.open(image_path)

        # Convert grayscale images to rgb
        if image.mode != "RGB":
            image = convert_to_rgb(image)

        if self.transform:
            image = self.transform(image)

        return image

    def __len__(self):
        return len(self.images)
