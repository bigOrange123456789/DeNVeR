import os
import torch
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import numpy as np
from concurrent.futures import ThreadPoolExecutor
import random
import time
import gc

from free_cos.FineTuner import getAna, FineTuner

os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'

def testSave(tag,x):
    x = x.clone().detach().cpu()
    print("tag",x.shape,type(x))
    os.makedirs("temp_test", exist_ok=True)
    from PIL import Image
    import numpy as np

    for i in range(x.shape[0]):
        img_tensor = x[i, 0]
        img_np = img_tensor.numpy()
        img_np = ((img_np - img_np.min()) / (img_np.max() - img_np.min()) * 255).astype(np.uint8)
        img = Image.fromarray(img_np, mode='L')
        img.save(f"temp_test/{tag}"+f'_{i+1}.png')
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
    def __init__(self, label_root, image_root, video_to_user, video_stats, singleVideoId=None):
        self.samples = []
        for videoId in os.listdir(label_root):
            if singleVideoId is None or singleVideoId == videoId:
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
        mean, std = self.video_stats[video_dir]
        image = transforms.ToTensor()(image)
        image = transforms.Normalize(mean=mean, std=std)(image)
        label = transforms.ToTensor()(label)
        label = (label > 0.5).float()
        return image, label

import torchvision.transforms.functional as F
from torchvision.transforms import InterpolationMode

class BackgroundDataset(Dataset):
    """背景数据集：加载背景图像，要求模型输出全0。每个epoch随机采样，可重复使用"""
    def __init__(self, bg_paths, video_stats, vessel_folder_path=None, length=None, crop_size=(256,256)):
        self.vessel_folder_path = vessel_folder_path
        self.bg_paths = bg_paths
        self.video_stats = video_stats
        self.num_real_samples = len(self.bg_paths)
        self.length = length if length is not None else self.num_real_samples
        self.crop_size = crop_size

    def _perturbation(self, vessel0):
        random_float = random.uniform(1, 4)
        return vessel0 ** random_float

    def _augmentation(self, img, gt, p_flip=0.5, p_rotate=0.5, p_color_jitter=0.5,
                      brightness_range=(0.5, 1.5), contrast_range=(0.8, 2.1),
                      saturation_range=(0.5, 1.5), angle_range=(-180, 180)):
        crop_size = self.crop_size
        assert isinstance(img, torch.Tensor) and isinstance(gt, torch.Tensor)
        assert img.shape[1:] == gt.shape[1:], "img 和 gt 的空间尺寸必须相同"

        if random.random() < p_flip:
            img = F.hflip(img)
            gt = F.hflip(gt)
        if random.random() < p_flip:
            img = F.vflip(img)
            gt = F.vflip(gt)
        if random.random() < p_color_jitter:
            brightness_factor = random.uniform(*brightness_range)
            contrast_factor = random.uniform(*contrast_range)
            saturation_factor = random.uniform(*saturation_range)
            img = F.adjust_brightness(img, brightness_factor)
            img = F.adjust_contrast(img, contrast_factor)
            img = F.adjust_saturation(img, saturation_factor)
        if random.random() < p_rotate:
            angle = random.uniform(*angle_range)
            img = F.rotate(img, angle, interpolation=InterpolationMode.BILINEAR, expand=True)
            gt = F.rotate(gt, angle, interpolation=InterpolationMode.NEAREST, expand=True)
        if crop_size is not None:
            _, h, w = img.shape
            if isinstance(crop_size, int):
                crop_h, crop_w = crop_size, crop_size
            else:
                crop_h, crop_w = crop_size
            assert crop_h <= h and crop_w <= w, f"裁剪尺寸 ({crop_h}, {crop_w}) 不能大于当前图像尺寸 ({h}, {w})"
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
            fileName = os.listdir(self.vessel_folder_path)[idx]
            image = Image.open(os.path.join(self.vessel_folder_path, fileName)).convert('L')
            vessel = transforms.ToTensor()(image)
            vessel = self._perturbation(vessel)
            gt = torch.zeros_like(vessel)
            gt[vessel < 1] = 1
            synthesis = vessel * background
            synthesis, gt = self._augmentation(synthesis, gt)
            synthesis = transforms.Normalize(mean=mean, std=std)(synthesis)
            return synthesis, gt, vessel

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        idx = np.random.randint(0, self.num_real_samples)
        img_path, video_dir = self.bg_paths[idx]
        bg = Image.open(img_path).convert('L')
        mean, std = self.video_stats[video_dir]
        bg = transforms.ToTensor()(bg)
        synthesis, gt, vessel = self._hasVessel(bg, mean, std)
        bg = transforms.Normalize(mean=mean, std=std)(bg)
        return bg, synthesis, gt, vessel

# ==================== 扩展 FineTuner 类 ====================
class FineTuner2(FineTuner):
    def __init__(self, model, use_proto=True):
        super().__init__(model)
        self.use_proto = use_proto          # 是否使用原型预测
        self.bg_proto = None
        self.fg_proto = None
        self.temperature = 1.0

    def compute_prototypes(self, bg_loader, labeled_loader, num_batches=10):
        """使用当前模型从数据加载器中采样，计算背景和前景的原型向量（包含进度输出）"""
        self.model.eval()
        bg_sum = None
        bg_count = 0
        fg_sum = None
        fg_count = 0

        print("开始计算背景原型...")
        with torch.no_grad():
            # 计算背景原型
            for i, batch in enumerate(bg_loader):
                if i >= num_batches:
                    break
                images_bg, _, _, _ = batch
                images_bg = images_bg.to(self.device)
                features = self.model(images_bg, mask=None, trained=True, fake=False)['feature']
                C = features.size(1)
                feat_flat = features.permute(0, 2, 3, 1).reshape(-1, C)
                if bg_sum is None:
                    bg_sum = feat_flat.sum(dim=0).cpu()
                else:
                    bg_sum += feat_flat.sum(dim=0).cpu()
                bg_count += feat_flat.size(0)
                # 每处理2个batch打印一次进度
                if (i + 1) % 2 == 0 or i == num_batches - 1:
                    print(f"\r  背景 batch {i+1}/{num_batches} 处理完成，已累积 {bg_count} 个像素", end="")

            # 计算前景原型
            print("开始计算前景原型...")
            for i, batch in enumerate(labeled_loader):
                if i >= num_batches:
                    break
                images_l, masks_l = batch
                images_l = images_l.to(self.device)
                masks_l = masks_l.to(self.device)
                features = self.model(images_l, mask=None, trained=True, fake=False)['feature']
                C = features.size(1)
                feat_flat = features.permute(0, 2, 3, 1).reshape(-1, C)  # [B*H*W, C]
                mask_flat = (masks_l > 0.5).view(-1)                     # [B*H*W]
                fg_features = feat_flat[mask_flat]                       # [N, C]
                if fg_features.size(0) > 0:
                    if fg_sum is None:
                        fg_sum = fg_features.sum(dim=0).cpu()
                    else:
                        fg_sum += fg_features.sum(dim=0).cpu()
                    fg_count += fg_features.size(0)
                # 每处理2个batch打印一次进度
                if (i + 1) % 2 == 0 or i == num_batches - 1:
                    print(f"\r  前景 batch {i+1}/{num_batches} 处理完成，已累积 {fg_count} 个前景像素", end="")

        # 计算均值并归一化
        if bg_sum is not None and bg_count > 0:
            bg_proto = bg_sum / bg_count
            self.bg_proto = bg_proto.to(self.device)
            self.bg_proto = self.bg_proto / (self.bg_proto.norm(p=2) + 1e-8)
            print(f"背景原型向量形状: {self.bg_proto.shape}")
        else:
            print("警告：未收集到背景特征，原型未初始化")
            self.bg_proto = None

        if fg_sum is not None and fg_count > 0:
            fg_proto = fg_sum / fg_count
            self.fg_proto = fg_proto.to(self.device)
            self.fg_proto = self.fg_proto / (self.fg_proto.norm(p=2) + 1e-8)
            print(f"前景原型向量形状: {self.fg_proto.shape}")
        else:
            print("警告：未收集到前景特征，原型未初始化")
            self.fg_proto = None

        self.model.train()
        print("原型计算完成。")

    def compute_proto_pred(self, feature):
        if self.bg_proto is None or self.fg_proto is None:
            raise RuntimeError("Prototypes not initialized. Call compute_prototypes first.")
        B, C, H, W = feature.shape
        if self.fg_proto.numel() != C:
            raise RuntimeError(f"Prototype dimension mismatch: fg_proto has {self.fg_proto.numel()} elements, but feature has {C} channels.")
        # 归一化特征向量
        feat_norm = feature / (feature.norm(p=2, dim=1, keepdim=True) + 1e-8)
        fg_proto = self.fg_proto.view(1, C, 1, 1).to(feature.device)
        bg_proto = self.bg_proto.view(1, C, 1, 1).to(feature.device)
        cos_fg = (feat_norm * fg_proto).sum(dim=1, keepdim=True)
        cos_bg = (feat_norm * bg_proto).sum(dim=1, keepdim=True)
        score = cos_fg - cos_bg
        pred = torch.sigmoid(score / self.temperature)
        return pred

    def fit2(self, labeled_loader, background_loader,
         epochs, lr=1e-4, weight_decay=1e-5,
         save_path=None,
         bg_loss_weight=0.05,
         reweightBg=0.2,
         singleVideoId=None,
         moduleOpen_positive=True,
         proto_update_freq=1,
         proto_num_batches=10):
        self.optimizer = optim.Adam(self.model.parameters(), lr=lr, weight_decay=weight_decay)
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, mode='min', patience=5, factor=0.5)

        best_val_dice = 0.0
        no_improve_epochs = 0

        if epochs==0:
            self.compute_prototypes(background_loader, labeled_loader, num_batches=proto_num_batches)
        for epoch in range(1, epochs + 1):
            epoch_start_time = time.time()
            self.model.train()
            total_loss = 0.0
            total_bce = 0.0
            total_dice = 0.0
            num_batches = 0

            # 仅在启用原型且需要更新时计算原型
            if self.use_proto and (epoch == 1 or (proto_update_freq > 0 and (epoch - 1) % proto_update_freq == 0)):
                print(f"\n[Epoch {epoch}] Updating prototypes...")
                self.compute_prototypes(background_loader, labeled_loader, num_batches=proto_num_batches)
                torch.cuda.empty_cache()
                print(f"[Epoch {epoch}] Prototypes updated.")

            labeled_iter = iter(labeled_loader)
            bg_iter = iter(background_loader)

            num_batches_all = len(labeled_loader)
            print(f"\n[Epoch {epoch}] Starting training with {num_batches_all} batches. Batch size: {labeled_loader.batch_size}")

            print_interval = 1  # 按需调整进度输出频率
            last_print_time = time.time()

            for batch_idx, labeled_batch in enumerate(labeled_iter, 1):
                try:
                    bg_batch = next(bg_iter)
                except StopIteration:
                    bg_iter = iter(background_loader)
                    bg_batch = next(bg_iter)

                # ---------- 有标注图像 ----------
                if moduleOpen_positive:
                    images_l, masks_l = labeled_batch
                    images_l = images_l.to(self.device)
                    masks_l = masks_l.to(self.device)
                    outputs_l = self.model(images_l, mask=None, trained=True, fake=False)
                    if self.use_proto:
                        pred_l = self.compute_proto_pred(outputs_l['feature'])
                    else:
                        pred_l = outputs_l['pred']
                    loss_sup, bce_sup, dice_sup = self._compute_loss(pred_l * masks_l, masks_l)
                else:
                    loss_sup = torch.tensor(0.0, device=self.device)
                    bce_sup = torch.tensor(0.0, device=self.device)
                    dice_sup = torch.tensor(0.0, device=self.device)

                # ---------- 背景与合成图像 ----------
                images_bg, images_syn, images_gt, images_vessel = bg_batch
                images_bg = images_bg.to(self.device)
                images_syn = images_syn.to(self.device)
                images_gt = images_gt.to(self.device)
                images_vessel = images_vessel.to(self.device)

                # 背景监督
                zero_masks = torch.zeros_like(images_bg)
                outputs_bg = self.model(images_bg, mask=None, trained=True, fake=False)
                if self.use_proto:
                    pred_bg = self.compute_proto_pred(outputs_bg['feature'])
                else:
                    pred_bg = outputs_bg['pred']
                loss_bg, bce_bg, dice_bg = self._compute_loss(pred_bg, zero_masks)

                # 合成血管监督
                outputs_syn = self.model(images_syn, mask=None, trained=True, fake=False)
                if self.use_proto:
                    pred_syn = self.compute_proto_pred(outputs_syn['feature'])
                else:
                    pred_syn = outputs_syn['pred']
                reweight_mask = images_gt.clone() * (1 - reweightBg) + reweightBg
                reweight_mask[pred_syn.clone().detach() > 0.5] = 1
                loss_syn, bce_syn, dice_syn = self._compute_loss(reweight_mask * pred_syn, reweight_mask * images_gt)

                loss = (2 * loss_syn + loss_sup + bg_loss_weight * loss_bg)

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                total_loss += loss.item()
                total_bce += bce_sup.item() + bce_bg.item()
                total_dice += dice_sup.item() + dice_bg.item()
                num_batches += 1

                # 进度输出
                if batch_idx % print_interval == 0 or batch_idx == num_batches_all:
                    elapsed = time.time() - last_print_time
                    avg_loss_sofar = total_loss / num_batches
                    print("\r", f"[Epoch {epoch}] Batch {batch_idx}/{num_batches_all} | "
                        f"Loss: {loss.item():.4f} (sup: {loss_sup.item():.4f}, bg: {loss_bg.item():.4f}, syn: {loss_syn.item():.4f}) | "
                        f"Avg loss so far: {avg_loss_sofar:.4f} | "
                        f"Time since last print: {elapsed:.2f}s", end="")
                    last_print_time = time.time()
                    torch.cuda.empty_cache() # 清理可能存在的临时变量

                # 可选的提前停止调试（例如处理 100 个 batch 后停止）
                if batch_idx >= 150:
                    break

            avg_loss = total_loss / num_batches
            avg_bce = total_bce / num_batches
            avg_dice = total_dice / num_batches

            self.scheduler.step(avg_loss)

            epoch_time = time.time() - epoch_start_time
            print(f"\n[Epoch {epoch}] Completed in {epoch_time:.2f}s | Avg Loss: {avg_loss:.4f} (BCE: {avg_bce:.4f}, Dice: {avg_dice:.4f})")

            # 可选评估
            if hasattr(self, 'evaluate') or 'getAna' in globals():
                result = getAna(segment_model=self.model, singleVideoId=singleVideoId)
                print(f"[Epoch {epoch}] Evaluation: {result}")

        print("Training completed.")

# ==================== 构建数据集所需映射 ====================
def build_video_to_user(image_root, singleVideoId=None):
    video_to_user = {}
    for userId in os.listdir(image_root):
        user_img_dir = os.path.join(image_root, userId, 'images')
        if not os.path.isdir(user_img_dir):
            continue
        for videoId in os.listdir(user_img_dir):
            if singleVideoId is None or singleVideoId == videoId:
                if os.path.isdir(os.path.join(user_img_dir, videoId)):
                    video_to_user[videoId] = userId
    return video_to_user

def collect_background_paths(image_root, decouple_root, singleVideoId, moduleOpen_rigid=True):
    bg_paths = []
    for userId in os.listdir(image_root):
        user_img_dir = os.path.join(image_root, userId, 'images')
        if not os.path.isdir(user_img_dir):
            continue
        for videoId in os.listdir(user_img_dir):
            if singleVideoId is None or singleVideoId == videoId:
                img_dir = os.path.join(user_img_dir, videoId)
                first_frame = os.path.join(img_dir, '00000.png')
                if os.path.isfile(first_frame):
                    bg_paths.append((first_frame, img_dir))

        if moduleOpen_rigid:
            user_decouple_dir = os.path.join(decouple_root, userId, 'decouple')
            if not os.path.isdir(user_decouple_dir):
                continue
            for videoId in os.listdir(user_decouple_dir):
                if singleVideoId is None or singleVideoId == videoId:
                    bg_dir = os.path.join(user_decouple_dir, videoId, 'A26-03.rigid.non1')
                    bg_file = os.path.join(bg_dir, '00000.png')
                    if os.path.isfile(bg_file):
                        original_video_dir = os.path.join(image_root, userId, 'images', videoId)
                        if os.path.isdir(original_video_dir):
                            bg_paths.append((bg_file, original_video_dir))
    return bg_paths

# ==================== 主训练流程 ====================
def main(
    image_root = '../DeNVeR_in/xca_dataset',
    label_root = './log_26/outputs/high_precision_refine',
    decouple_root = './log_26/xca_dataset',
    vessel_folder_path = '../DeNVeR_in/fake_grayvessel_bend',
    model_param_path = '../DeNVeR_in/models_config/freecos_Seg.pt',
    model = None,
    singleVideoId = "CVAI-1207LAO44_CRA29", #是否只对单个视频进行微调
    moduleOpen_positive = True, #是否对标定区域进行学习
    moduleOpen_rigid = True, #是否去除刚体
    output_dir = None,
    epochs = 50,
    crop_size = (512, 512),#(256,256),#
    batch_size = 4,
    rigidTag="A26-02.rigid.non1",
    use_proto = True,          # 新增：是否使用特征原型
):
    
    # 超参数
    lr = 1e-4
    bg_loss_weight = 0#0.1 #0.4#2  #背景图像的学习权重
    reweightBg = 0.1#0.4 #合成血管的背景区域的学习权重 #判断一下这个权重是否为0对结果是否有影响
    num_workers = 0  # 修改为 0，避免多进程导致的显存残留
    print("bg_loss_weight:", bg_loss_weight, "reweightBg:", reweightBg, "crop_size:", crop_size)

    # 1. 构建 videoId -> userId 映射
    print("Building video->user mapping...")
    video_to_user = build_video_to_user(image_root, singleVideoId=singleVideoId)
    print(f"Found {len(video_to_user)} videos.")

    # 2. 收集所有背景路径
    print("Collecting background images...")
    bg_paths = collect_background_paths(image_root, decouple_root, singleVideoId=singleVideoId, moduleOpen_rigid=moduleOpen_rigid)
    print(f"Collected {len(bg_paths)} background images.")

    # 3. 收集所有需要计算统计量的视频目录
    video_dirs_set = set()
    for _, video_dir in bg_paths:
        video_dirs_set.add(video_dir)
    for videoId in os.listdir(label_root):
        userId = video_to_user.get(videoId)
        if userId:
            if singleVideoId is None or singleVideoId == videoId:
                video_dir = os.path.join(image_root, userId, 'images', videoId)
                if os.path.isdir(video_dir):
                    video_dirs_set.add(video_dir)
    video_dirs = list(video_dirs_set)
    print(f"Need to compute stats for {len(video_dirs)} video directories.")

    # 4. 计算视频统计量
    video_stats = compute_video_stats(video_dirs, max_workers=4)

    # 5. 构建数据集
    print("Building labeled dataset...")
    labeled_dataset = LabeledSegmentationDataset(label_root, image_root, video_to_user, video_stats, singleVideoId=singleVideoId)
    labeled_loader = DataLoader(labeled_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    dataset_length = len(labeled_loader)

    background_dataset = BackgroundDataset(
        bg_paths,
        video_stats,
        vessel_folder_path=vessel_folder_path,
        length=dataset_length,
        crop_size=crop_size,
    )
    background_loader = DataLoader(background_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    # 6. 加载模型
    if model is None:
        from free_cos.ModelSegment import ModelSegment
        n_channels = 1
        num_classes = 1
        model = ModelSegment(n_channels, num_classes)
        checkpoint = torch.load(model_param_path, map_location='cpu')
        model.load_state_dict(checkpoint['state_dict'])
    if torch.cuda.is_available():
        model = model.cuda()

    # 7. 微调前评估
    if False:
        result1 = getAna(segment_model=model, singleVideoId=singleVideoId)
        print("微调前的效果:", result1)
        result1_nr = getAna(segment_model=model, 
                            singleVideoId=singleVideoId, 
                            output_dir=None if output_dir is None else (output_dir+"nr"),
                            rigidTag=rigidTag,
                            decouple_root=decouple_root)
        print("微调后的效果(去除刚体层):", result1_nr)
        # exit(0)

    # 8. 初始化微调器
    finetuner = FineTuner2(model,use_proto=use_proto)

    # 9. 开始训练
    finetuner.fit2(labeled_loader, background_loader,
                   epochs=epochs, lr=lr,
                   bg_loss_weight=bg_loss_weight, reweightBg=reweightBg,
                   singleVideoId=singleVideoId,
                   moduleOpen_positive=moduleOpen_positive,
                   save_path=None)

    # # 10. 微调后评估（可选）
    # if True:
    #     result2 = getAna(segment_model=model, singleVideoId=singleVideoId, output_dir=output_dir)
    #     print("微调后的效果:", result2)
    #     result2_nr = getAna(segment_model=model, 
    #                         singleVideoId=singleVideoId, 
    #                         output_dir=None if output_dir is None else (output_dir+"nr"),
    #                         rigidTag=rigidTag,
    #                         decouple_root=decouple_root)
    #     print("微调后的效果(去除刚体层):", result2_nr)
    # 10. 微调后评估（可选）
    if True:
        # 从 finetuner 获取原型参数
        use_proto_eval = finetuner.use_proto
        bg_proto_eval = finetuner.bg_proto.cpu() if finetuner.bg_proto is not None else None
        fg_proto_eval = finetuner.fg_proto.cpu() if finetuner.fg_proto is not None else None
        temperature_eval = finetuner.temperature

        print("use_proto_eval",use_proto_eval)
        print("bg_proto_eval",type(bg_proto_eval))
        print("fg_proto_eval",type(fg_proto_eval))
        result2 = getAna(
            segment_model=model,
            singleVideoId=singleVideoId,
            output_dir=output_dir,
            use_proto=use_proto_eval,
            bg_proto=bg_proto_eval,
            fg_proto=fg_proto_eval,
            temperature=temperature_eval
        )
        print("微调后的效果:", result2)

        result2_nr = getAna(
            segment_model=model,
            singleVideoId=singleVideoId,
            output_dir=None if output_dir is None else (output_dir + "nr"),
            rigidTag=rigidTag,
            decouple_root=decouple_root,
            use_proto=use_proto_eval,
            bg_proto=bg_proto_eval,
            fg_proto=fg_proto_eval,
            temperature=temperature_eval
        )
        print("微调后的效果(去除刚体层):", result2_nr)

    # ========== 关键：清理当前视频的显存占用 ==========
    print(f"Cleaning up for video {singleVideoId}...")
    del finetuner
    del model
    del labeled_loader, background_loader
    del labeled_dataset, background_dataset
    if 'optimizer' in locals():
        del optimizer
    gc.collect()
    torch.cuda.empty_cache()
    print("Cleanup done.")

def start(
    image_root = '../DeNVeR_in/xca_dataset',
    label_root = './log_26/outputs/high_precision_refine',
    decouple_root = './log_26/xca_dataset',
    vessel_folder_path = '../DeNVeR_in/fake_grayvessel_bend',
    model_param_path = '../DeNVeR_in/models_config/freecos_Seg.pt',

    moduleOpen_positive=True,
    moduleOpen_rigid=False, # True, # 不用刚体效果更好
    output_dir=None,
    epochs = 50,
    crop_size = (512, 512),
    use_proto = True,          # 新增：是否使用特征原型
):
    if output_dir is not None:
        os.makedirs(output_dir, exist_ok=True)
    print("output_dir:", output_dir)
    datasetPath = image_root
    print("moduleOpen_positive:", moduleOpen_positive)
    print("moduleOpen_rigid:", moduleOpen_rigid)

    for userId in os.listdir(datasetPath):
        user_images_dir = os.path.join(datasetPath, userId, "images")
        if not os.path.isdir(user_images_dir):
            continue
        for videoId in os.listdir(user_images_dir):
            print(f"\n=== Processing video {videoId} ===")
            main(
                image_root=image_root,
                label_root=label_root,
                decouple_root=decouple_root,
                vessel_folder_path=vessel_folder_path,
                model_param_path=model_param_path,
                ########################################
                singleVideoId=videoId,
                moduleOpen_positive=moduleOpen_positive,
                moduleOpen_rigid=moduleOpen_rigid,
                output_dir=output_dir,
                epochs = epochs,
                crop_size=crop_size,
                use_proto=use_proto,
            )
            # 再次清理（双重保障）
            gc.collect()
            torch.cuda.empty_cache()

if __name__ == "__main__":
    start_time = time.time()
    if True:
        import yaml
        script_path = os.path.abspath(__file__)
        ROOT1 = os.path.dirname(script_path)
        file_path = os.path.join(ROOT1, "../",'confs/newConfig.yaml')
        print("file_path:",file_path)
        with open(file_path, 'r', encoding='utf-8') as file:
            config0 = yaml.safe_load(file)
            image_root = config0["my"]["datasetPath"] # "../DeNVeR_in/xca_dataset"
            decouple_root = config0["my"]["datasetPath_rigid.out"] # "log_27/xca_dataset"
            label_root = os.path.join(
                config0["my"]["filePathRoot"], # "log_27"
                config0["my"]["subPath"]["outputs"], #"outputs"
                "high_precision_refine"
            )
            #A26-02.rigid.non1
    # 只用首帧学习或只用刚体学习、两者差异？

    # 运行三种配置
    # start(
    #     moduleOpen_positive=True,
    #     moduleOpen_rigid=True,
    #     output_dir=os.path.join("temp", "mask_YesYes"),
    #     epochs = 5,
    # )

    # 需要验证是整体微调效果好、还是逐视频微调效果好 #我猜测逐个视频微调效果更好
    # 一、逐帧微调
    if False:
        print("逐帧微调")
        start(#
            image_root = image_root,
            label_root = label_root,#'./log_26/outputs/high_precision_refine',
            decouple_root = decouple_root,#'./log_26/xca_dataset',

            moduleOpen_positive=False,#True,
            moduleOpen_rigid=False,
            output_dir=os.path.join("temp", "mask_test02"), #Autodl-H设备
            epochs = 10, #训练多少批次比较合适？       
            use_proto = True,     
        )
    if True: # 二、整体微调
        print("整体微调") # 指标下降了：0.76=>0.73
        batch_size = 4#5
        print("batch_size",batch_size)
        # main( # 微调前是76，微调后是75
        #     image_root = image_root,
        #     label_root = label_root,#'./log_26/outputs/high_precision_refine',
        #     decouple_root = decouple_root,#'./log_26/xca_dataset',
        #     moduleOpen_positive=True,
        #     moduleOpen_rigid=True,
        #     output_dir = os.path.join("temp", "mask_test_g02"),  #Autodl-G设备
        #     epochs = 5,
        #     singleVideoId = None,
        #     batch_size = batch_size,#15
        # )


        print("crop_size = (256,256)") #没有显著区别
        main(
            image_root = image_root,
            label_root = label_root,#'./log_26/outputs/high_precision_refine',
            decouple_root = decouple_root,#'./log_26/xca_dataset',
            moduleOpen_positive=True,
            moduleOpen_rigid=True,
            output_dir = os.path.join("temp", "mask_test_g03"),  #Autodl-J设备
            epochs = 5,
            singleVideoId = None,
            batch_size = batch_size,#15

            crop_size = (512,512),#(256,256), 
            use_proto = True,
        )
        # main(
        #     image_root = image_root,
        #     label_root = label_root,#'./log_26/outputs/high_precision_refine',
        #     decouple_root = decouple_root,#'./log_26/xca_dataset',
        #     moduleOpen_positive=True,
        #     moduleOpen_rigid=True,
        #     output_dir = os.path.join("temp", "mask_test_g03"),  #Autodl-J设备
        #     epochs = 5,
        #     singleVideoId = None,
        #     batch_size = batch_size,#15

        #     crop_size = (512,512),#(256,256), 
        #     use_proto = False,#True,
        # )

    # start(
    #     moduleOpen_positive=False,
    #     moduleOpen_rigid=False,
    #     output_dir=os.path.join("temp", "mask_NoNo")
    # )
    # start(
    #     moduleOpen_positive=False,
    #     moduleOpen_rigid=True,
    #     output_dir=os.path.join("temp", "mask_NoPositive")
    # )
    # start(
    #     moduleOpen_positive=True,
    #     moduleOpen_rigid=False,
    #     output_dir=os.path.join("temp", "mask_NoRigid")
    # )

    end_time = time.time()
    elapsed_time = (end_time - start_time) / (60 * 60)
    print(f"程序总运行时间: {elapsed_time:.4f} 小时")