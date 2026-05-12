import os.path
import cv2
from PIL import Image
from .base_dataset import BaseDataset
import torchvision.transforms as transforms
from .image_folder import make_dataset


# 注意：删除了 MaskToTensor 的引用，因为不需要处理 Mask 了

class CrackDataset(BaseDataset):
    """A dataset class for crack dataset (Inference Mode Only)."""

    def __init__(self, args):
        """Initialize this dataset class."""
        BaseDataset.__init__(self, args)

        # 1. 设定图片文件夹路径
        # 代码会自动去 dataset_path 下寻找 '{phase}_img' 文件夹 (例如 test_img)
        self.img_paths = make_dataset(os.path.join(args.dataset_path, '{}_img'.format(args.phase)))

        # 2. 定义图片预处理 (转Tensor + 归一化)
        self.img_transforms = transforms.Compose([transforms.ToTensor(),
                                                  transforms.Normalize((0.5, 0.5, 0.5),
                                                                       (0.5, 0.5, 0.5))])
        self.phase = args.phase

    def __getitem__(self, index):
        """
        Return a data point and its metadata information.
        """
        # 1. 获取图片路径
        img_path = self.img_paths[index]

        # 2. 读取图片
        img = cv2.imread(img_path, cv2.IMREAD_UNCHANGED)

        # 防错：如果图片坏了或路径不对，抛出异常
        if img is None:
            raise ValueError(f"无法读取图片: {img_path}")

        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # 3. 调整尺寸 (Resize)
        w, h = self.args.load_width, self.args.load_height
        if w > 0 or h > 0:
            img = cv2.resize(img, (w, h), interpolation=cv2.INTER_CUBIC)

        # 4. 转为 Tensor
        img = self.img_transforms(Image.fromarray(img.copy()))

        # 5. 返回字典
        # 【重点】这里只返回 image 和路径，不再返回 label
        return {'image': img, 'A_paths': img_path}

    def __len__(self):
        """Return the total number of images in the dataset."""
        return len(self.img_paths)
