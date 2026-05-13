import os.path
import cv2
from PIL import Image
from .base_dataset import BaseDataset
import torchvision.transforms as transforms
from .image_folder import make_dataset


class CrackDataset(BaseDataset):
    """A dataset class for crack dataset (Inference Mode Only)."""

    def __init__(self, args):
        """Initialize this dataset class."""
        BaseDataset.__init__(self, args)

    
        self.img_paths = make_dataset(os.path.join(args.dataset_path, '{}_img'.format(args.phase)))


        self.img_transforms = transforms.Compose([transforms.ToTensor(),
                                                  transforms.Normalize((0.5, 0.5, 0.5),
                                                                       (0.5, 0.5, 0.5))])
        self.phase = args.phase

    def __getitem__(self, index):
        """
        Return a data point and its metadata information.
        """
   
        img_path = self.img_paths[index]

   
        img = cv2.imread(img_path, cv2.IMREAD_UNCHANGED)


        if img is None:
            raise ValueError(f"error: {img_path}")

        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # Resize
        w, h = self.args.load_width, self.args.load_height
        if w > 0 or h > 0:
            img = cv2.resize(img, (w, h), interpolation=cv2.INTER_CUBIC)

   
        img = self.img_transforms(Image.fromarray(img.copy()))


        return {'image': img, 'A_paths': img_path}

    def __len__(self):
        """Return the total number of images in the dataset."""
        return len(self.img_paths)
