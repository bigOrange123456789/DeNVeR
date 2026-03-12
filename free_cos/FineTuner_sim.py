import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import numpy as np
script_path = os.path.abspath(__file__)
ROOT = os.path.dirname(script_path)
import torch.backends.cudnn as cudnn
from free_cos.ModelSegment import ModelSegment

# ========== 数据集定义 ==========
class SegmentationDataset(Dataset):
    """有标签数据集（保留以备兼容，但本简化版本中未使用）"""
    def __init__(self, img_dir, gt_dir, transform=None, target_transform=None):
        self.img_dir = img_dir
        self.gt_dir = gt_dir
        self.transform = transform
        self.target_transform = target_transform
        self.img_names = sorted([f for f in os.listdir(gt_dir) if f.endswith('.png')])
        if len(self.img_names) == 0:
            raise RuntimeError(f"No PNG images found in {img_dir}")

    def __len__(self):
        return len(self.img_names)

    def __getitem__(self, idx):
        img_name = self.img_names[idx]
        img_path = os.path.join(self.img_dir, img_name)
        gt_path = os.path.join(self.gt_dir, img_name)
        image = Image.open(img_path).convert('L')
        mask = Image.open(gt_path).convert('L')
        if self.transform:
            image = self.transform(image)
        else:
            image = transforms.ToTensor()(image)
        if self.target_transform:
            mask = self.target_transform(mask)
        else:
            mask = transforms.ToTensor()(mask)
            mask = (mask > 0.5).float()
        return image, mask

class SegmentationDatasetNoGT(Dataset):
    """无标签数据集，仅返回图像"""
    def __init__(self, img_dir, transform=None):
        self.img_dir = img_dir
        self.transform = transform
        self.img_names = sorted([f for f in os.listdir(img_dir) if f.endswith('.png')])
        if len(self.img_names) == 0:
            raise RuntimeError(f"No PNG images found in {img_dir}")

    def __len__(self):
        return len(self.img_names)

    def __getitem__(self, idx):
        img_name = self.img_names[idx]
        img_path = os.path.join(self.img_dir, img_name)
        image = Image.open(img_path).convert('L')
        if self.transform:
            image = self.transform(image)
        else:
            image = transforms.ToTensor()(image)
        return image

# ========== 微调类（仅保留推理保存功能） ==========
class FineTuner:
    def __init__(self, model):
        self.model = model
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    @torch.no_grad()
    def seg(self, test_loader, output_dir=None, apply_sigmoid=False, threshold=0.5):
        """
        推理并保存分割结果
        - test_loader: 数据加载器，要求dataset包含img_names属性
        - output_dir: 保存目录，若为None则不保存
        - apply_sigmoid: True保存概率图(0-255)，False保存二值图(0/255)
        - threshold: 二值化阈值
        """
        self.model.eval()
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
            print(f"Segmentation results will be saved to {output_dir}")

        dataset = test_loader.dataset
        if not hasattr(dataset, 'img_names'):
            raise AttributeError("Dataset must have 'img_names' attribute to save images.")

        sample_idx = 0
        for images in test_loader:          # 数据集返回单元素（图像）
            images = images.to(self.device)
            outputs = self.model(images, mask=None, trained=False, fake=False)
            pred_logits = outputs['pred']   # 形状 (B,1,H,W)
            pred_probs = pred_logits         # 根据模型输出，可能是logits或概率
            pred_binary = (pred_probs > threshold).float()

            if output_dir:
                if apply_sigmoid:
                    save_tensor = (pred_probs * 255).byte()
                else:
                    save_tensor = (pred_binary * 255).byte()

                for i in range(save_tensor.size(0)):
                    filename = dataset.img_names[sample_idx + i]
                    save_path = os.path.join(output_dir, filename)
                    img_array = save_tensor[i, 0].cpu().numpy()
                    Image.fromarray(img_array, mode='L').save(save_path)

            sample_idx += len(images)

# ========== 模型加载 ==========
def getModel(pathParam):
    os.environ['MASTER_PORT'] = '169711'
    os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
    cudnn.benchmark = True

    n_channels = 1
    num_classes = 1
    Segment_model = ModelSegment(n_channels, num_classes)
    if torch.cuda.is_available():
        Segment_model = Segment_model.cuda()
    checkpoint = torch.load(pathParam, map_location=torch.device('cpu'))
    Segment_model.load_state_dict(checkpoint['state_dict'])
    return Segment_model

# ========== 计算图像均值和标准差（用于归一化）==========
def calculate_mean_variance(image_folder):
    total_pixels = 0
    sum_pixels = 0.0
    sum_squared_pixels = 0.0
    image_files = [f for f in os.listdir(image_folder) if
                   f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff'))]
    if not image_files:
        print("文件夹中没有找到图片文件")
        return None, None
    for img_file in image_files:
        try:
            img_path = os.path.join(image_folder, img_file)
            img = Image.open(img_path).convert('L')
            img_array = np.array(img).astype(np.float32) / 255.0
            num_pixels = img_array.size
            total_pixels += num_pixels
            sum_pixels += np.sum(img_array)
            sum_squared_pixels += np.sum(img_array.astype(np.float64) ** 2)
        except Exception as e:
            print(f"处理图片 {img_file} 时出错: {e}")
            continue
    if total_pixels == 0:
        return None, None
    mean = sum_pixels / total_pixels
    variance = (sum_squared_pixels / total_pixels) - (mean ** 2)
    return mean, variance**0.5

# ========== 辅助函数：对单个视频进行推理保存 ==========
def process_video(img_dir, output_base_dir, segment_model, batch_size=4, mean=None, std=None):
    """处理单个视频文件夹，保存分割结果到 output_base_dir/视频名/"""
    if mean is None or std is None:
        mean, std = calculate_mean_variance(img_dir)
        print(f"  mean={mean:.4f}, std={std:.4f}")
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std),
    ])
    dataset = SegmentationDatasetNoGT(img_dir, transform=transform)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=2)
    video_name = os.path.basename(img_dir)
    output_dir = os.path.join(output_base_dir, video_name) if output_base_dir else None
    FineTuner(segment_model).seg(loader, output_dir=output_dir)
    print(f"  Saved to {output_dir}")

# ========== 主程序：批量处理数据集 ==========
if __name__ == "__main__":
    segment_model = getModel("../DeNVeR_in/models_config/freecos_Seg.pt")
    dataset_root = "../DeNVeR_in/xca_dataset"
    output_root = "temp/mask3"          # 结果保存根目录

    mean = None   # 若为None则自动计算每个视频的均值和标准差
    std = None

    for user_id in os.listdir(dataset_root):
        images_dir = os.path.join(dataset_root, user_id, "images")
        if not os.path.isdir(images_dir):
            continue
        for video_id in os.listdir(images_dir):
            video_img_dir = os.path.join(images_dir, video_id)
            print(f"Processing: {video_img_dir}")
            process_video(video_img_dir, output_root, segment_model,
                          batch_size=16, mean=mean, std=std)
    print("All videos processed.")