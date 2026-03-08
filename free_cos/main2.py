import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import numpy as np

# 假设您的模型定义在某个模块中，请根据实际情况导入
# from your_model_file import ModelSegment

class SegmentationDataset(Dataset):
    """自定义分割数据集，假设图像和标签均为灰度PNG文件，文件名一一对应"""
    def __init__(self, img_dir, gt_dir, transform=None, target_transform=None):
        self.img_dir = img_dir
        self.gt_dir = gt_dir
        self.transform = transform
        self.target_transform = target_transform
        self.img_names = sorted([f for f in os.listdir(img_dir) if f.endswith('.png')])
        # 确保标签文件存在（简单检查第一个文件）
        if len(self.img_names) == 0:
            raise RuntimeError(f"No PNG images found in {img_dir}")
        sample_gt = os.path.join(gt_dir, self.img_names[0])
        if not os.path.exists(sample_gt):
            raise FileNotFoundError(f"Ground truth file {sample_gt} not found. Ensure filenames match.")

    def __len__(self):
        return len(self.img_names)

    def __getitem__(self, idx):
        img_name = self.img_names[idx]
        img_path = os.path.join(self.img_dir, img_name)
        gt_path = os.path.join(self.gt_dir, img_name)

        image = Image.open(img_path).convert('L')   # 灰度图
        mask = Image.open(gt_path).convert('L')     # 标签图

        if self.transform:
            image = self.transform(image)
        else:
            # 默认转为Tensor，数值范围[0,1]
            image = transforms.ToTensor()(image)

        if self.target_transform:
            mask = self.target_transform(mask)
        else:
            # 标签也转为Tensor，并保持为0~1（假设标签值为0或255）
            mask = transforms.ToTensor()(mask)
            # 将255的像素映射为1（二分类）
            mask = (mask > 0.5).float()

        return image, mask


class FineTuneSegmenter:
    """
    图像分割模型微调类
    支持二值分割（num_classes=1），使用BCEWithLogitsLoss + DiceLoss组合损失
    """
    def __init__(self, model):
        """
        参数:
            model_class: 模型类（如 ModelSegment），用于实例化新模型
            pretrained_params_path: 预训练权重文件路径（.pth.tar 或 .pth）
            num_classes: 输出类别数（默认为1，即二分类）
            device: 训练设备，若为None则自动选择cuda或cpu
        """
        self.model = model
        self.device = next(model.parameters()).device
        # print("self.device", self.device)
        # exit(0)

        # 损失函数：组合 BCE + Dice
        self.bce_loss = nn.BCEWithLogitsLoss()
        self.dice_loss = self._dice_loss

        # 优化器、调度器等将在训练时设置
        self.optimizer = None
        self.scheduler = None

    def _dice_loss(self, pred, target, smooth=1e-6):
        """Dice损失，适用于二分类，pred为logits，target为0/1浮点数"""
        # pred = torch.sigmoid(pred)  # 转换为概率
        intersection = (pred * target).sum(dim=(2,3))
        union = pred.sum(dim=(2,3)) + target.sum(dim=(2,3))
        dice = (2.0 * intersection + smooth) / (union + smooth)
        return 1 - dice.mean()

    def _compute_loss(self, pred, target):
        """组合损失: BCE + Dice"""
        bce = self.bce_loss(pred, target)
        dice = self.dice_loss(pred, target)
        weight_dice = 1.0
        weight_ce = 3.0
        return (
            weight_ce * bce + weight_dice * dice
            ), bce, dice

    def train_one_epoch(self, dataloader, epoch, print_freq=10):
        """训练一个epoch"""
        self.model.train()
        total_loss = 0.0
        total_bce = 0.0
        total_dice = 0.0

        for i, (images, masks) in enumerate(dataloader):
            images = images.to(self.device)
            masks = masks.to(self.device)

            # 前向传播（注意模型返回格式与推理代码保持一致）
            # 假设模型返回字典，包含 'pred' 键
            outputs = self.model(images, mask=None, trained=True, fake=False)  # 训练模式
            pred = outputs['pred']  # 形状 (B,1,H,W) 的logits

            loss, bce, dice = self._compute_loss(pred, masks)

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            total_loss += loss.item()
            total_bce += bce.item()
            total_dice += dice.item()

            if (i+1) % print_freq == 0:
                print(f"Epoch {epoch} | Batch {i+1}/{len(dataloader)} | Loss: {loss.item():.4f} (BCE: {bce.item():.4f}, Dice: {dice.item():.4f})")

        avg_loss = total_loss / len(dataloader)
        avg_bce = total_bce / len(dataloader)
        avg_dice = total_dice / len(dataloader)
        return avg_loss, avg_bce, avg_dice

    @torch.no_grad()
    def evaluate(self, dataloader):
        """在验证/测试集上评估，返回平均损失和Dice系数"""
        self.model.eval()
        total_loss = 0.0
        total_dice = 0.0

        for images, masks in dataloader:
            images = images.to(self.device)
            masks = masks.to(self.device)

            outputs = self.model(images, mask=None, trained=False, fake=False)  # 推理模式
            pred = outputs['pred']
            loss, _, dice = self._compute_loss(pred, masks)

            total_loss += loss.item()
            total_dice += dice.item()

        avg_loss = total_loss / len(dataloader)
        avg_dice = total_dice / len(dataloader)
        return avg_loss, avg_dice

    def fit(self, train_loader, val_loader, epochs, lr=1e-4, weight_decay=1e-5, 
            save_path='best_model.pth', early_stopping_patience=None):
        """
        完整训练流程
        参数:
            train_loader: 训练数据加载器
            val_loader: 验证数据加载器
            epochs: 最大训练轮数
            lr: 学习率
            weight_decay: 权重衰减
            save_path: 最佳模型保存路径
            early_stopping_patience: 早停耐心值（可选）
        """
        self.optimizer = optim.Adam(self.model.parameters(), lr=lr, weight_decay=weight_decay)
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, mode='min', patience=5, factor=0.5)

        best_val_dice = 0.0
        no_improve_epochs = 0

        for epoch in range(1, epochs+1):
            print(f"\n========== Epoch {epoch}/{epochs} ==========")
            train_loss, train_bce, train_dice = self.train_one_epoch(train_loader, epoch)
            val_loss, val_dice = self.evaluate(val_loader)

            # 调整学习率
            self.scheduler.step(val_loss)

            print(f"Train Loss: {train_loss:.4f} (BCE: {train_bce:.4f}, Dice: {train_dice:.4f}) | Val Loss: {val_loss:.4f}, Val Dice: {val_dice:.4f}")

            # 保存最佳模型（根据验证集Dice）
            if val_dice > best_val_dice:
                best_val_dice = val_dice
                if not save_path is None:
                    torch.save({
                        'epoch': epoch,
                        'model_state_dict': self.model.state_dict(),
                        'optimizer_state_dict': self.optimizer.state_dict(),
                        'val_dice': val_dice,
                    }, save_path)
                    print(f"Best model saved to {save_path} (Val Dice: {val_dice:.4f})")
                no_improve_epochs = 0
            else:
                no_improve_epochs += 1

            # 早停
            if early_stopping_patience and no_improve_epochs >= early_stopping_patience:
                print(f"Early stopping triggered after {epoch} epochs.")
                break

        print("Training completed.")

    @torch.no_grad()
    def test(self, test_loader, model_path=None):
        """在测试集上评估，可加载指定模型"""
        if model_path:
            checkpoint = torch.load(model_path, map_location=self.device)
            self.model.load_state_dict(checkpoint['model_state_dict'])
            print(f"Loaded model from {model_path} for testing")

        test_loss, test_dice = self.evaluate(test_loader)
        print(f"Test Loss: {test_loss:.4f}, Test Dice: {test_dice:.4f}")
        return test_loss, test_dice

    @torch.no_grad()
    def test3(self, test_loader, model_path=None, output_dir=None, apply_sigmoid=False, threshold=0.5):
        """
        在测试集上评估，计算 Dice、Recall、Precision，并可选择保存分割结果。

        参数:
            test_loader: 测试数据加载器（要求 dataset 具有 img_names 属性，且 shuffle=False）
            model_path: 可选，加载指定模型权重
            output_dir: 可选，保存分割结果的文件夹路径，如果为 None 则不保存
            apply_sigmoid: 如果为 True，保存的图像为经过 sigmoid 的概率图（0-255 映射）；
                        否则保存为二值化后的图像（0 或 255）
            threshold: 二值化阈值，默认为 0.5
        """
        if model_path:
            checkpoint = torch.load(model_path, map_location=self.device)
            self.model.load_state_dict(checkpoint['model_state_dict'])
            print(f"Loaded model from {model_path} for testing")

        self.model.eval()

        # 用于累积指标
        total_tp = 0
        total_fp = 0
        total_fn = 0
        total_images = 0

        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
            print(f"Segmentation results will be saved to {output_dir}")

        dataset = test_loader.dataset
        if not hasattr(dataset, 'img_names'):
            raise AttributeError("Dataset must have 'img_names' attribute to save images.")

        sample_idx = 0
        for images, masks in test_loader:
            images = images.to(self.device)
            masks = masks.to(self.device)  # masks 是 0/1 的 float 张量

            outputs = self.model(images, mask=None, trained=False, fake=False)
            pred_logits = outputs['pred']  # 形状 (B,1,H,W)

            # 转换为概率并二值化
            pred_probs = pred_logits # torch.sigmoid(pred_logits)
            pred_binary = (pred_probs > threshold).float()

            # 将 masks 转为整数便于统计
            masks_int = masks.long()

            # 展平每个样本（保留 batch 维度）
            pred_binary_flat = pred_binary.view(pred_binary.size(0), -1).cpu()  # (B, H*W)
            masks_flat = masks_int.view(masks_int.size(0), -1).cpu()

            # 计算每个样本的 TP, FP, FN
            tp = (pred_binary_flat * masks_flat).sum(dim=1)  # (B,)
            fp = (pred_binary_flat * (1 - masks_flat)).sum(dim=1)
            fn = ((1 - pred_binary_flat) * masks_flat).sum(dim=1)

            total_tp += tp.sum().item()
            total_fp += fp.sum().item()
            total_fn += fn.sum().item()
            total_images += images.size(0)

            # 保存预测结果
            if output_dir:
                # 根据 apply_sigmoid 决定保存内容
                if apply_sigmoid:
                    # 保存概率图 (0-1) 映射到 0-255
                    save_tensor = (pred_probs * 255).byte()
                else:
                    # 保存二值图 (0/1) 映射到 0/255
                    save_tensor = (pred_binary * 255).byte()

                for i in range(save_tensor.size(0)):
                    filename = dataset.img_names[sample_idx + i]
                    save_path = os.path.join(output_dir, filename)
                    img_array = save_tensor[i, 0].cpu().numpy()
                    Image.fromarray(img_array, mode='L').save(save_path)

            sample_idx += len(images)

        # 计算总体指标
        if (2 * total_tp + total_fp + total_fn) > 0:
            dice = (2 * total_tp) / (2 * total_tp + total_fp + total_fn)
        else:
            dice = 0.0
        recall = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0.0
        precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0.0

        print(f"Test Results: Dice={dice:.4f}, Recall={recall:.4f}, Precision={precision:.4f}")
        return dice, recall, precision ##############################为啥推理结果不对


import torch.backends.cudnn as cudnn
from free_cos.ModelSegment import ModelSegment
def getModel(pathParam):
    os.environ['MASTER_PORT'] = '169711' #“master_port”的意思是主端口
    os.environ['CUDA_LAUNCH_BLOCKING'] = '1'

    cudnn.benchmark = True #benchmark的意思是基准


    n_channels = 1
    num_classes =  1
    Segment_model = ModelSegment(n_channels, num_classes)

    # 有1个cuda 。torch.cuda.device_count()=1
    if torch.cuda.is_available():
        # print("cuda_is available")
        Segment_model = Segment_model.cuda() # 分割模型


    ##############################   predictor.lastInference()   ##############################
    checkpoint = torch.load(pathParam, map_location=torch.device('cpu'))  # 如果模型是在GPU上训练的，这里指定为'cpu'以确保兼容性
    Segment_model.load_state_dict(checkpoint['state_dict'])  # 提取模型状态字典并赋值给模型

    return Segment_model

def calculate_mean_variance(image_folder):
    # 初始化变量
    total_pixels = 0
    sum_pixels = 0.0
    sum_squared_pixels = 0.0

    # 获取文件夹中所有图片文件
    image_files = [f for f in os.listdir(image_folder) if
                   f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff'))]

    if not image_files:
        print("文件夹中没有找到图片文件")
        return None, None

    # 遍历所有图片
    for img_file in image_files:
        try:
            img_path = os.path.join(image_folder, img_file)
            img = Image.open(img_path).convert('L')  # 确保是灰度图
            img_array = np.array(img).astype(np.float32)/255.0
            # print("img_array",np.max(img_array),np.min(img_array))
            # exit(0)

            # 更新统计量
            num_pixels = img_array.size
            total_pixels += num_pixels
            sum_pixels += np.sum(img_array)
            sum_squared_pixels += np.sum(img_array.astype(np.float64) ** 2)

        except Exception as e:
            print(f"处理图片 {img_file} 时出错: {e}")
            continue

    if total_pixels == 0:
        print("没有有效的像素数据")
        return None, None

    # 计算均值和方差
    mean = sum_pixels / total_pixels
    variance = (sum_squared_pixels / total_pixels) - (mean ** 2)

    return mean, variance**0.5
# ========== 使用示例 ==========
if __name__ == "__main__":
    # 路径配置
    train_img_dir= "../DeNVeR_in/dataset_test/img"
    train_gt_dir = "../DeNVeR_in/dataset_test/gt"
    test_img_dir = "../DeNVeR_in/dataset_test/img"
    test_gt_dir  = "../DeNVeR_in/dataset_test/gt"
    pretrained_path = "your_pretrained_model.pth"  # 请替换为实际路径
    pathParam = "../DeNVeR_in/models_config/freecos_Seg.pt"
    batch_size = 10#14#16

    # 定义数据预处理（与推理代码保持一致，计算均值和标准差）
    # 注意：您可以使用 calculate_mean_variance 函数计算训练集的均值和标准差
    # 此处简单使用默认的ToTensor()，即归一化到[0,1]
    mean, std = calculate_mean_variance(train_img_dir)
    print("{mean:",mean, "std:", std, "}")
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std),
        # 如果需要标准化，可以添加 transforms.Normalize(mean, std)
    ])
    target_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std),
        lambda x: (x > 0.5).float()  # 将标签转换为0/1
    ])

    # 创建数据集和数据加载器
    train_dataset = SegmentationDataset(train_img_dir, train_gt_dir, transform=transform, target_transform=target_transform)
    val_dataset = SegmentationDataset(test_img_dir, test_gt_dir, transform=transform, target_transform=target_transform)  # 这里用测试集作为验证集示例，实际应划分验证集
    test_dataset = SegmentationDataset(test_img_dir, test_gt_dir, transform=transform, target_transform=target_transform)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True,  num_workers=2)
    val_loader =  DataLoader(val_dataset,    batch_size=batch_size, shuffle=False, num_workers=2)
    test_loader = DataLoader(test_dataset,   batch_size=batch_size, shuffle=False, num_workers=2)

    # 初始化微调器
    # 假设 ModelSegment 已在某处定义，此处用占位
    # from your_model import ModelSegment
    # 如果您没有该类的定义，需要先导入或定义。这里假设存在。
    segment_model = getModel(pathParam)
    
    # 使用占位模型演示，实际应替换为您的 ModelSegment
    finetuner = FineTuneSegmenter(segment_model)

    # 测试
    finetuner.test3(test_loader, output_dir="inference_new2") #finetuner.test(test_loader)
    
    exit(0)
    # 开始训练
    finetuner.fit(
        train_loader, 
        val_loader, 
        epochs=5,#20, 
        lr=1e-4, save_path=None)

    # 测试
    finetuner.test3(test_loader, output_dir="inference2") #finetuner.test(test_loader)
