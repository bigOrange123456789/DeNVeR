import os
import torch
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import numpy as np
from concurrent.futures import ThreadPoolExecutor
import random

from free_cos.FineTuner import getAna
def testSave(tag,x):
    x = x.clone().detach().cpu()
    print("tag",x.shape,type(x))
    # os.makedirs("temp/"+tag, exist_ok=True)
    os.makedirs("temp_test", exist_ok=True)
    from PIL import Image
    import numpy as np

    # 假设你的张量变量名为 tensor，形状为 [4, 1, 512, 512]
    # tensor = your_tensor_here

    # 将张量转换为numpy并保存为4张图片
    for i in range(x.shape[0]):
        # 提取单张图片 [1, 512, 512] -> [512, 512]
        img_tensor = x[i, 0]  # 形状变为 [512, 512]
        
        # 归一化到 0-255 范围（假设张量值范围可能不是0-1）
        img_np = img_tensor.numpy()
        img_np = ((img_np - img_np.min()) / (img_np.max() - img_np.min()) * 255).astype(np.uint8)
        
        # 转换为PIL Image并保存（'L'模式表示灰度图）
        img = Image.fromarray(img_np, mode='L')
        img.save(f"temp_test/{tag}"+f'_{i+1}.png')
        # print(f"已保存: output_image_{i+1}.png")

    print("全部保存完成！")

# ==================== 辅助函数：计算视频均值和标准差 ====================
def compute_video_mean_std(video_img_dir):
    """计算单个视频文件夹内所有图像的均值和标准差（假设为灰度图，像素值范围 [0,1]）"""
    pixel_sum = 0.0
    pixel_sq_sum = 0.0
    total_pixels = 0
    for img_name in os.listdir(video_img_dir):
        img_path = os.path.join(video_img_dir, img_name)
        if not img_name.lower().endswith(('.png', '.jpg', '.jpeg')):
            continue
        img = Image.open(img_path).convert('L')
        img_np = np.array(img).astype(np.float32) / 255.0
        h, w = img_np.shape
        pixel_sum += np.sum(img_np)
        pixel_sq_sum += np.sum(img_np ** 2)
        total_pixels += h * w
    if total_pixels == 0:
        raise RuntimeError(f"No valid images found in {video_img_dir}")
    mean = pixel_sum / total_pixels
    variance = (pixel_sq_sum / total_pixels) - (mean ** 2)
    std = np.sqrt(max(variance, 1e-10))
    return mean.item(), std.item()

def compute_video_stats(video_dirs, max_workers=4):
    """并行计算多个视频目录的均值和标准差"""
    video_stats = {}
    # print("并行计算多个视频目录的均值和标准差")
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_dir = {executor.submit(compute_video_mean_std, d): d for d in video_dirs}
        for future in future_to_dir:
            video_dir = future_to_dir[future]
            try:
                mean, std = future.result()
                video_stats[video_dir] = (mean, std)
                print(f"\rComputed stats for {video_dir}: mean={mean:.4f}, std={std:.4f}",end="")
            except Exception as e:
                print(f"Failed to compute stats for {video_dir}: {e}")
    return video_stats

# ==================== 数据集类 ====================
class LabeledSegmentationDataset(Dataset):
    """有标注数据集：加载图像和对应的二值标签，并使用视频特定的归一化"""
    def __init__(self, label_root, image_root, video_to_user, video_stats,singleVideoId=None):
        """
        label_root: 标注根目录，如 './log_26/outputs/high_precision_refine'
        image_root: 图像根目录，如 '../DeNVeR_in/xca_dataset'
        video_to_user: 字典，videoId -> userId
        video_stats: 字典，视频目录 -> (mean, std)
        """
        self.samples = []  # 每个元素为 (img_path, label_path, video_dir)
        for videoId in os.listdir(label_root):
            if singleVideoId is None or singleVideoId==videoId:
                label_video_dir = os.path.join(label_root, videoId)
                if not os.path.isdir(label_video_dir):
                    continue
                userId = video_to_user.get(videoId)
                if userId is None:
                    print(f"Warning: videoId {videoId} not found in video_to_user mapping")
                    continue
                img_video_dir = os.path.join(image_root, userId, 'images', videoId)
                if not os.path.isdir(img_video_dir):
                    print(f"Warning: image video dir {img_video_dir} not found")
                    continue
                # 遍历标签文件
                for fname in os.listdir(label_video_dir):
                    if fname.endswith('.png'):
                        label_path = os.path.join(label_video_dir, fname)
                        img_path = os.path.join(img_video_dir, fname)
                        if os.path.exists(img_path):
                            self.samples.append((img_path, label_path, img_video_dir))
        self.video_stats = video_stats
        print(f"Labeled dataset: {len(self.samples)} samples")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        idx = np.random.randint(0, len(self.samples))
        img_path, label_path, video_dir = self.samples[idx]
        image = Image.open(img_path).convert('L')
        label = Image.open(label_path).convert('L')
        mean, std = self.video_stats[video_dir]  # 必须存在
        # 图像归一化
        image = transforms.ToTensor()(image)
        image = transforms.Normalize(mean=mean, std=std)(image)
        # 标签处理：假设白色(255)为前景，转为0/1
        label = transforms.ToTensor()(label)
        label = (label > 0.5).float()
        return image, label

import torchvision.transforms.functional as F #用于BackgroundDataset的数据增强
from torchvision.transforms import InterpolationMode #用于BackgroundDataset的数据增强
class BackgroundDataset(Dataset):
    """背景数据集：加载背景图像，要求模型输出全0。每个epoch随机采样，可重复使用"""
    def __init__(self, bg_paths, video_stats, vessel_folder_path=None, length=None, crop_size=(256,256)):
        """
        bg_paths: 列表，每个元素为 (img_path, video_dir)
        video_stats: 视频统计字典
        length: 数据集长度，若为None则等于背景样本数；若指定，则每次随机采样，用于实现重复使用
        """
        self.vessel_folder_path = vessel_folder_path
        self.bg_paths = bg_paths
        self.video_stats = video_stats
        self.num_real_samples = len(self.bg_paths) 
        self.length = length if length is not None else self.num_real_samples
        self.crop_size = crop_size

    def _perturbation(self,vessel0):
        random_float = random.uniform(1, 4) # 生成一个1到4之间的随机小数
        return  vessel0 ** random_float

    def _augmentation(self,img, gt, 
                    # crop_size=(256,256), 
                    p_flip=0.5, 
                    p_rotate=0.5, 
                    p_color_jitter=0.5,
                    brightness_range=(0.5, 1.5), 
                    contrast_range=(0.8, 2.1), 
                    saturation_range=(0.5, 1.5),
                    angle_range=(-180, 180)):
        crop_size = self.crop_size
        """
        对 img 和 gt 同时应用数据增强。

        Args:
            img (torch.Tensor): 图像张量，形状 (C, H, W)，值范围 [0, 255] 或归一化后的任意范围。
            gt (torch.Tensor): 标签张量，形状 (C, H, W)，通常为单通道（C=1）的掩码（值为0或1）。
            crop_size (int or tuple): 裁剪后的尺寸。若为 int，则生成正方形裁剪；若为 (h, w)，则按指定尺寸裁剪。
            p_flip (float): 应用随机翻转的概率（水平/垂直各独立概率，此处统一控制）。
            p_rotate (float): 应用随机旋转的概率。
            p_color_jitter (float): 应用颜色抖动的概率。
            brightness_range (tuple): 亮度调整因子的范围，如 (0.5, 1.5)。
            contrast_range (tuple): 对比度调整因子的范围。
            saturation_range (tuple): 饱和度调整因子的范围。
            angle_range (tuple): 旋转角度的范围，如 (-180, 180)。

        Returns:
            (torch.Tensor, torch.Tensor): 增强后的 img 和 gt。
        """
        # 确保输入为张量
        assert isinstance(img, torch.Tensor) and isinstance(gt, torch.Tensor)
        assert img.shape[1:] == gt.shape[1:], "img 和 gt 的空间尺寸必须相同"

        # 1. 随机水平翻转
        if random.random() < p_flip:
            img = F.hflip(img)
            gt = F.hflip(gt)

        # 2. 随机垂直翻转
        if random.random() < p_flip:
            img = F.vflip(img)
            gt = F.vflip(gt)

        # 3. 颜色抖动（仅对 img）
        if random.random() < p_color_jitter:
            # 随机生成调整因子
            brightness_factor = random.uniform(*brightness_range)
            contrast_factor = random.uniform(*contrast_range)
            saturation_factor = random.uniform(*saturation_range)
            # 依次应用亮度、对比度、饱和度调整（顺序通常固定）
            img = F.adjust_brightness(img, brightness_factor)
            img = F.adjust_contrast(img, contrast_factor)
            img = F.adjust_saturation(img, saturation_factor)

        # 4. 随机旋转
        if random.random() < p_rotate:
            angle = random.uniform(*angle_range)
            # 对 img 使用双线性插值，对 gt 使用最近邻插值（避免标签值被平滑）
            img = F.rotate(img, angle, interpolation=InterpolationMode.BILINEAR, expand=True)
            gt = F.rotate(gt, angle, interpolation=InterpolationMode.NEAREST, expand=True)

        # 5. 随机裁剪到指定尺寸
        if not crop_size is None: 
            # 获取当前图像尺寸（旋转后可能已改变）
            _, h, w = img.shape
            if isinstance(crop_size, int):
                crop_h, crop_w = crop_size, crop_size
            else:
                crop_h, crop_w = crop_size

            # 确保裁剪尺寸不大于当前尺寸（否则无法裁剪）
            assert crop_h <= h and crop_w <= w, f"裁剪尺寸 ({crop_h}, {crop_w}) 不能大于当前图像尺寸 ({h}, {w})"

            # 生成随机裁剪起始点
            top = random.randint(0, h - crop_h)
            left = random.randint(0, w - crop_w)

            img = F.crop(img, top, left, crop_h, crop_w)
            gt = F.crop(gt, top, left, crop_h, crop_w)

        return img, gt
    
    def _hasVessel(self, background, mean, std):
        if self.vessel_folder_path is None:
            return None, None, None
        else:
            l = len(os.listdir(self.vessel_folder_path))
            idx = np.random.randint(0, l)
            fileName =os.listdir(self.vessel_folder_path)[idx]
            image = Image.open(os.path.join(self.vessel_folder_path,fileName)).convert('L')
            vessel = transforms.ToTensor()(image)
            vessel = self._perturbation(vessel)
            gt = torch.zeros_like(vessel) # images_vessel.clone()[images_vessel]
            gt[vessel<1]=1
            synthesis = vessel*background
            synthesis, gt = self._augmentation(synthesis,gt)
            synthesis = transforms.Normalize(mean=mean, std=std)(synthesis)
            return synthesis, gt, vessel

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        idx = np.random.randint(0, self.num_real_samples) # 随机选择
        img_path, video_dir = self.bg_paths[idx]
        bg = Image.open(img_path).convert('L')
        mean, std = self.video_stats[video_dir]
        bg = transforms.ToTensor()(bg)
        synthesis, gt, vessel = self._hasVessel(bg, mean, std)
        bg = transforms.Normalize(mean=mean, std=std)(bg)
        return bg, synthesis, gt, vessel

# ==================== 扩展 FineTuner 类 ====================
from free_cos.FineTuner import FineTuner
class FineTuner2(FineTuner):
    # ...（此处保留原有 __init__、_dice_loss、_compute_loss 等方法，省略以节省空间）
    # 您需要将之前的 FineTuner 类代码复制到这里，下面只展示新增的 fit2 方法

    def fit2(self, labeled_loader, background_loader, #vessel_loader,
             epochs, lr=1e-4, weight_decay=1e-5,
             save_path=None, 
             bg_loss_weight=0.05, #1.0
             reweightBg=0.2,
             singleVideoId=None,
             moduleOpen_positive = True,
            #  moduleOpen_rigid = True,
             ):
        """
        使用有标注数据和背景数据联合训练
        labeled_loader: 有标注数据加载器
        background_loader: 背景数据加载器（要求其 __len__ 与 labeled_loader 的批次数大致相当）
        bg_loss_weight: 背景损失的权重系数
        """
        self.optimizer = optim.Adam(self.model.parameters(), lr=lr, weight_decay=weight_decay)
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, mode='min', patience=5, factor=0.5)

        best_val_dice = 0.0
        no_improve_epochs = 0

        for epoch in range(1, epochs + 1):
            start_time = time.time() # 记录开始时间
            self.model.train()
            total_loss = 0.0
            total_bce = 0.0
            total_dice = 0.0
            num_batches = 0

            labeled_iter = iter(labeled_loader)
            bg_iter = iter(background_loader)

            # 以 labeled_loader 为主循环，确保每个 epoch 都遍历所有标注数据
            num_batches_all = 0
            for labeled_batch in labeled_iter:
                num_batches_all = num_batches_all+1
            labeled_iter = iter(labeled_loader)

            for labeled_batch in labeled_iter:
                try:
                    bg_batch = next(bg_iter)
                except StopIteration:
                    bg_iter = iter(background_loader)
                    bg_batch = next(bg_iter)

                # 1.处理有标注数据（真实标注）
                if moduleOpen_positive:
                    images_l, masks_l = labeled_batch
                    images_l = images_l.to(self.device)
                    masks_l = masks_l.to(self.device)
                    outputs_l = self.model(images_l, mask=None, trained=True, fake=False)
                    pred_l = outputs_l['pred']
                    # print("masks_l",masks_l.max(),masks_l.min(),masks_l.mean())
                    # exit(0)
                    loss_sup, bce_sup, dice_sup = self._compute_loss(pred_l*masks_l, masks_l) #_compute_loss(pred_l, masks_l)
                else:
                    loss_sup=torch.tensor(0) 
                    bce_sup=torch.tensor(0) 
                    dice_sup=torch.tensor(0) 

                # 2.处理背景数据（无标注）
                images_bg, images_syn, images_gt, images_vessel = bg_batch
                images_bg = images_bg.to(self.device)
                images_syn = images_syn.to(self.device)
                images_gt = images_gt.to(self.device)
                images_vessel = images_vessel.to(self.device)
                # 生成全零掩膜，形状与输出一致 (B,1,H,W)
                zero_masks = torch.zeros_like(images_bg) #torch.zeros(images_bg.size(0), 1, images_bg.size(2), images_bg.size(3)).to(self.device)
                outputs_bg = self.model(images_bg, mask=None, trained=True, fake=False)
                pred_bg = outputs_bg['pred']
                loss_bg, bce_bg, dice_bg = self._compute_loss(pred_bg, zero_masks) #self._compute_loss(pred_bg, zero_masks)

                # 3.处理合成标签数据（合成标注）
                # testSave("images_syn",images_syn)
                # testSave("images_vessel",images_vessel)
                pred_syn = self.model(images_syn, mask=None, trained=True, fake=False)['pred']
                # testSave("pred_syn",pred_syn)
                # exit(0)
                reweight_mask = images_gt.clone()*(1-reweightBg)+reweightBg
                reweight_mask[pred_syn.clone().detach()>0.5] = 1
                loss_syn, bce_syn, dice_syn = self._compute_loss(reweight_mask*pred_syn, reweight_mask*images_gt)
                # loss_v
                # images_vessel, _ = vessel_batch
                # testSave("images_vessel",images_vessel)
                # def perturbation(vessel0):
                #     random_float = 4#random.uniform(1, 4) # 生成一个1到4之间的随机小数
                #     return  vessel0 ** random_float
                # images_vessel = images_vessel.to(self.device)
                # images_vessel_p = perturbation(images_vessel)
                # testSave("images_vesselP",images_vessel_p)
                # images_in = images_vessel_p * images_bg
                # testSave("images_in",images_in)
                # images_gt = torch.zeros_like(images_vessel)#images_vessel.clone()[images_vessel]
                # images_gt[images_vessel<1]=1
                # # print("images_gt",images_gt.shape,type(images_gt),images_bg.shape[0])
                # testSave("images_gt",images_gt)
                # testSave("images_bg",images_bg)
                # pred_v = self.model(images_in, mask=None, trained=True, fake=False)["pred"]
                # testSave("images_perd",pred_v)
                # exit(0)
                # loss_v, bce_v, dice_v = self._compute_loss(pred_v, images_gt)



                # 总损失
                loss = (2 * loss_syn + 
                        loss_sup + # 
                        bg_loss_weight * loss_bg) 
                # loss = loss_syn

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                total_loss += loss.item()
                total_bce += bce_sup.item() + bce_bg.item()
                total_dice += dice_sup.item() + dice_bg.item()
                num_batches += 1
                print("\r",f"Epoch {epoch}/{epochs} | batches {num_batches}/{int(num_batches_all)} Loss_sup {loss_sup.item():.4f} Loss_b{(bg_loss_weight * loss_bg).item():.4f} Loss_syn {loss_syn.item():.4f}", end="")
                if num_batches>100:
                    break

            avg_loss = total_loss / num_batches
            avg_bce = total_bce / num_batches
            avg_dice = total_dice / num_batches

            # 调整学习率（基于平均损失）
            self.scheduler.step(avg_loss)

            if False:
                print(f"Epoch {epoch}/{epochs} | Loss: {avg_loss:.4f} (BCE: {avg_bce:.4f}, Dice: {avg_dice:.4f})")
            else:
                end_time = time.time()# 记录结束时间
                # 计算运行时间
                elapsed_time = (end_time - start_time)/(60)
                print(f"程序运行时间: {elapsed_time:.4f} 分钟")
                print(f"Epoch {epoch}/{epochs} | Loss: {avg_loss:.4f} (BCE: {avg_bce:.4f}, Dice: {avg_dice:.4f})", getAna(segment_model=self.model, singleVideoId=singleVideoId))

            # 可选：验证（如果有验证集）
            # if val_loader is not None:
            #     val_loss, val_dice = self.evaluate(val_loader)
            #     print(f"Val Loss: {val_loss:.4f}, Val Dice: {val_dice:.4f}")
            #     if val_dice > best_val_dice:
            #         best_val_dice = val_dice
            #         if save_path:
            #             torch.save(self.model.state_dict(), save_path)
            #         no_improve_epochs = 0
            #     else:
            #         no_improve_epochs += 1
            #     if early_stopping_patience and no_improve_epochs >= early_stopping_patience:
            #         print(f"Early stopping triggered.")
            #         break

        print("Training completed.")

# ==================== 构建数据集所需映射 ====================
def build_video_to_user(image_root,singleVideoId=None):
    """遍历 image_root 下的 userId/images，构建 videoId -> userId 映射"""
    video_to_user = {}
    for userId in os.listdir(image_root):
        user_img_dir = os.path.join(image_root, userId, 'images')
        if not os.path.isdir(user_img_dir):
            continue
        for videoId in os.listdir(user_img_dir):
            if singleVideoId is None or singleVideoId==videoId:
                if os.path.isdir(os.path.join(user_img_dir, videoId)):
                    video_to_user[videoId] = userId
    return video_to_user

def collect_background_paths(image_root, decouple_root,singleVideoId,moduleOpen_rigid=True):
    """
    收集所有背景图像路径
    返回列表，每个元素为 (img_path, video_dir)
    """
    bg_paths = []
    # 扫描所有 userId
    for userId in os.listdir(image_root):
        # 来源1：第一帧
        user_img_dir = os.path.join(image_root, userId, 'images')
        if not os.path.isdir(user_img_dir):
            continue
        for videoId in os.listdir(user_img_dir):
            if singleVideoId is None or singleVideoId==videoId:
                img_dir = os.path.join(user_img_dir, videoId)
                first_frame = os.path.join(img_dir, '00000.png')
                if os.path.isfile(first_frame):
                    bg_paths.append((first_frame, img_dir))  # video_dir 为原始图像目录

        # 来源2：decouple 背景
        if moduleOpen_rigid:
            user_decouple_dir = os.path.join(decouple_root, userId, 'decouple')
            if not os.path.isdir(user_decouple_dir):
                continue
            for videoId in os.listdir(user_decouple_dir):
                if singleVideoId is None or singleVideoId==videoId:
                    bg_dir = os.path.join(user_decouple_dir, videoId, 'A26-03.rigid.non1')
                    bg_file = os.path.join(bg_dir, '00000.png')
                    if os.path.isfile(bg_file):
                        # 对应的原始视频目录
                        original_video_dir = os.path.join(image_root, userId, 'images', videoId)
                        if os.path.isdir(original_video_dir):
                            bg_paths.append((bg_file, original_video_dir))
    return bg_paths

# ==================== 主训练流程 ====================
def main(
    # 路径配置
    image_root = '../DeNVeR_in/xca_dataset',
    label_root = './log_26/outputs/high_precision_refine',
    decouple_root = './log_26/xca_dataset',
    vessel_folder_path='../DeNVeR_in/fake_grayvessel_bend',
    model_param_path = '../DeNVeR_in/models_config/freecos_Seg.pt',
    model = None,

    singleVideoId = "CVAI-1207LAO44_CRA29",#None 
    moduleOpen_positive = True,
    moduleOpen_rigid = True,
    output_dir=None
    
):
    

    # 超参数
    batch_size = 4 #5溢出 #6溢出 #7溢出 #8溢出 #4可行
    epochs = 50 #5
    lr = 1e-4
    bg_loss_weight = 2#2.5 #3过大 #2 略小 #1.2 # 0.8(太小) #0.4(太小) #0.1 #0.05 # 1.0  # 背景损失权重，可调整 #可减少面积
    
    reweightBg = 0.4 #0.2
    crop_size=(512,512)
    
    # 当bg_loss_weight = 2，reweightBg = 0.2的时候，指标几乎不改变 #{'dice': 0.7639873924309365, 'recall': 0.820330952737491, 'precision': 0.7148}
    print("bg_loss_weight:",bg_loss_weight,"reweightBg:",reweightBg,"crop_size:",crop_size)

    # 1. 构建 videoId -> userId 映射
    print("Building video->user mapping...")
    video_to_user = build_video_to_user(image_root,singleVideoId=singleVideoId)
    print(f"Found {len(video_to_user)} videos.")

    # 2. 收集所有背景路径
    print("Collecting background images...")
    bg_paths = collect_background_paths(image_root, decouple_root,singleVideoId=singleVideoId,moduleOpen_rigid=moduleOpen_rigid)
    print(f"Collected {len(bg_paths)} background images.")

    # 3. 收集所有需要计算统计量的视频目录
    #   来自标注数据集和背景数据集的视频目录
    video_dirs_set = set()
    # 从背景路径中收集 video_dir
    for _, video_dir in bg_paths:
        video_dirs_set.add(video_dir)
    # 从标注数据集中收集 video_dir（需要扫描 label_root）
    for videoId in os.listdir(label_root):
        userId = video_to_user.get(videoId)
        if userId:
            if singleVideoId is None or singleVideoId==videoId:
                video_dir = os.path.join(image_root, userId, 'images', videoId)
                if os.path.isdir(video_dir):
                    video_dirs_set.add(video_dir)
    video_dirs = list(video_dirs_set)
    print(f"Need to compute stats for {len(video_dirs)} video directories.")

    # 4. 计算视频统计量（如果已有缓存可跳过，这里简单计算）
    video_stats = compute_video_stats(video_dirs, max_workers=4)

    # 5. 构建数据集
    print("Building labeled dataset...")
    num_workers = 4 #2
    # 5.1 构建标注数据集
    labeled_dataset = LabeledSegmentationDataset(label_root, image_root, video_to_user, video_stats, singleVideoId=singleVideoId)
    labeled_loader = DataLoader(labeled_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    dataset_length = len(labeled_loader)  # 每个 epoch 背景 loader 会产生这么多批

    # 5.2 构建背景数据集
    #    设置背景数据集长度为 labeled_loader 的批次数，使得每个 epoch 中背景样本数与标注样本数相当
    background_dataset = BackgroundDataset(
        bg_paths, 
        video_stats, 
        vessel_folder_path=vessel_folder_path,
        length=dataset_length,
        crop_size=crop_size,#(512,512),#(256,256),#None,
        # moduleOpen_rigid=moduleOpen_rigid,
        )
    background_loader = DataLoader(background_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    # 注意：shuffle=False 因为我们已经在 __getitem__ 中随机采样


    # 6. 加载模型
    if model is None:
        from free_cos.ModelSegment import ModelSegment  # 根据实际路径导入
        n_channels = 1
        num_classes = 1
        model = ModelSegment(n_channels, num_classes)
        checkpoint = torch.load(model_param_path, map_location='cpu')
        model.load_state_dict(checkpoint['state_dict'])
    if torch.cuda.is_available():
        model = model.cuda()

    # 7.1. 可选：在测试集上评估
    if True:
        result1 = getAna(segment_model=model, singleVideoId=singleVideoId,output_dir=output_dir)
        print("微调前的效果:",result1)

    # 8. 初始化微调器
    finetuner = FineTuner2(model)

    # 9. 开始训练
    finetuner.fit2(labeled_loader, background_loader, 
                #    vessel_loader,
                   epochs=epochs, lr=lr,
                   bg_loss_weight=bg_loss_weight, reweightBg=reweightBg,
                   singleVideoId=singleVideoId,
                   moduleOpen_positive = moduleOpen_positive,
                #    moduleOpen_rigid = moduleOpen_rigid,
                   save_path=None)

    # 10. 可选：在测试集上评估
    if True:
        result2 = getAna(segment_model=model, singleVideoId=singleVideoId,output_dir=output_dir)
        print("微调前的效果:",result1)
        print("微调后的效果:",result2)

def start(
        moduleOpen_positive = True,
        moduleOpen_rigid = True,  
        image_root = '../DeNVeR_in/xca_dataset',
        output_dir=None,
):
    if not output_dir is None:
        os.makedirs(output_dir, exist_ok=True)
    datasetPath = image_root
    print("moduleOpen_positive",moduleOpen_positive)
    print("moduleOpen_rigid:",moduleOpen_rigid)
    for userId in os.listdir(datasetPath):
        for videoId in os.listdir(os.path.join(datasetPath,userId,"images")):
            main(
                singleVideoId = videoId, #"CVAI-1207LAO44_CRA29",#None 
                moduleOpen_positive = moduleOpen_positive,
                moduleOpen_rigid = moduleOpen_rigid,   
                image_root=image_root,
                output_dir=output_dir
            )

if __name__ == "__main__":
    import time
    start_time = time.time() # 记录开始时间
    # start(
    #     moduleOpen_positive = True,
    #     moduleOpen_rigid = True, 
    #     output_dir = os.path.join("temp","mask_YesYes")
    # )
    start(
        moduleOpen_positive = False,#True,
        moduleOpen_rigid = False,#True, 
        output_dir = os.path.join("temp","mask_NoNo")
    )
    start(
        moduleOpen_positive = False,#True,
        moduleOpen_rigid = True, 
        output_dir = os.path.join("temp","mask_NoPositive")
    )
    start(
        moduleOpen_positive = True,
        moduleOpen_rigid = False,#True, 
        output_dir = os.path.join("temp","mask_NoRigid")
    )
    #main()
    end_time = time.time()# 记录结束时间
    # 计算运行时间
    elapsed_time = (end_time - start_time)/(60*60)
    print(f"程序运行时间: {elapsed_time:.4f} 小时")