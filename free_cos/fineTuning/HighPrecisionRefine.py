import os
import struct
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
import cv2

class HighPrecisionRefine:
    """
    极高查准率后处理：
        1. 计算光流幅值和前后向误差，取高幅值且高误差的区域作为高置信度运动区域。
        2. 原始预测与该区域取交集。
        3. 保留面积最大的连通组件（或面积大于阈值的组件），其余去除。
        4. 可选形态学平滑。
    """

    def __init__(self, fwd_flo_path, bwd_flo_path, pred_mask_path, output_path,
                 gt_path=None, black_thresh=50, white_thresh=200,
                 use_otsu=True, motion_combine='and',    # 固定为 'and' 以获得最严格区域
                 keep_largest_only=True,                  # 是否只保留最大连通组件
                 min_component_area=200,                   # 最小连通组件面积（若 keep_largest_only=False 则使用）
                 morph_kernel=3):
        """
        Args:
            fwd_flo_path: 前向光流 .flo 路径
            bwd_flo_path: 后向光流 .flo 路径
            pred_mask_path: 预测掩码 .png 路径（灰度，前景概率）
            output_path: 输出修正后掩码路径
            gt_path: 真实掩码 .png 路径（三值图），若提供则自动评估
            black_thresh: 黑色背景阈值（≤此值视为背景）
            white_thresh: 白色导管阈值（≥此值视为忽略区域）
            use_otsu: 是否使用 Otsu 自适应阈值
            motion_combine: 运动区域组合方式，建议 'and' 以获得最高精度
            keep_largest_only: 若 True，只保留面积最大的连通组件；若 False，保留面积 >= min_component_area 的组件
            min_component_area: 最小连通组件面积（当 keep_largest_only=False 时有效）
            morph_kernel: 形态学操作的核大小（可选，设为0则跳过）
        """
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        self.fwd_flo_path = fwd_flo_path
        self.bwd_flo_path = bwd_flo_path
        self.pred_mask_path = pred_mask_path
        self.output_path = output_path
        self.gt_path = gt_path if os.path.exists(gt_path) else None
        self.black_thresh = black_thresh
        self.white_thresh = white_thresh
        self.use_otsu = use_otsu
        self.motion_combine = motion_combine.lower()
        self.keep_largest_only = keep_largest_only
        self.min_component_area = min_component_area
        self.morph_kernel = morph_kernel

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self._load_data()

    # ---------- 读取工具 ----------
    def _read_flo(self, path):
        with open(path, 'rb') as f:
            magic = struct.unpack('i', f.read(4))[0]
            if magic == 1234567890:
                byte_order = 'i'
            elif magic == 1212500304:
                byte_order = 'I'
            else:
                raise ValueError(f"Invalid .flo magic: {magic}")
            w = struct.unpack(byte_order, f.read(4))[0]
            h = struct.unpack(byte_order, f.read(4))[0]
            data = struct.unpack(f'{h*w*2}f', f.read(h*w*2*4))
            return np.array(data, dtype=np.float32).reshape((h, w, 2))

    def _read_mask(self, path, normalize=True):
        img = Image.open(path).convert('L')
        mask = np.array(img, dtype=np.float32)
        if normalize:
            mask /= 255.0
        return mask

    def _load_data(self):
        flow_f = self._read_flo(self.fwd_flo_path)
        flow_b = self._read_flo(self.bwd_flo_path)
        pred = self._read_mask(self.pred_mask_path, normalize=True)

        assert flow_f.shape[:2] == flow_b.shape[:2] == pred.shape, "尺寸不一致！"
        self.H, self.W = flow_f.shape[:2]

        self.flow_f = torch.from_numpy(flow_f).float().to(self.device)
        self.flow_b = torch.from_numpy(flow_b).float().to(self.device)
        self.pred = torch.from_numpy(pred).float().to(self.device)

        if self.gt_path:
            gt_raw = self._read_mask(self.gt_path, normalize=False)
            gt_raw = torch.from_numpy(gt_raw).float().to(self.device)
            self.gt_white = gt_raw >= self.white_thresh
            self.gt_gray = (gt_raw > self.black_thresh) & (gt_raw < self.white_thresh)
            self.gt_black = gt_raw <= self.black_thresh
            self.valid = ~self.gt_white
        else:
            self.gt_white = self.gt_gray = self.gt_black = self.valid = None

    # ---------- 特征计算 ----------
    def _compute_features(self):
        # 幅值
        mag_fwd = torch.norm(self.flow_f, dim=-1)
        mag_bwd = torch.norm(self.flow_b, dim=-1)

        # 前后向一致性误差
        yy, xx = torch.meshgrid(torch.arange(self.H, device=self.device),
                                 torch.arange(self.W, device=self.device),
                                 indexing='ij')
        grid_pixel = torch.stack([xx, yy], dim=-1).float()
        target_f = grid_pixel + self.flow_f
        target_norm_f = target_f.clone()
        target_norm_f[..., 0] = 2.0 * target_f[..., 0] / (self.W - 1) - 1.0
        target_norm_f[..., 1] = 2.0 * target_f[..., 1] / (self.H - 1) - 1.0
        grid_f = target_norm_f.unsqueeze(0)

        flow_b_4d = self.flow_b.permute(2, 0, 1).unsqueeze(0)
        sampled_flow_b = F.grid_sample(flow_b_4d, grid_f, mode='bilinear',
                                       padding_mode='border', align_corners=False)
        sampled_flow_b = sampled_flow_b.squeeze(0).permute(1,2,0)

        err = torch.sum((self.flow_f + sampled_flow_b)**2, dim=-1)

        return mag_fwd, err

    # ---------- 自适应阈值 ----------
    def _otsu_threshold(self, tensor):
        arr = tensor.cpu().numpy().flatten()
        norm = (arr - arr.min()) / (arr.max() - arr.min() + 1e-8) * 255
        norm = norm.astype(np.uint8)
        th, _ = cv2.threshold(norm, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        orig_th = arr.min() + (th / 255.0) * (arr.max() - arr.min())
        mask = tensor > orig_th
        return mask, orig_th

    # ---------- 评估指标 ----------
    def _compute_metrics(self, pred_bin, gt_bin, valid_mask=None):
        if valid_mask is None:
            valid_mask = torch.ones_like(pred_bin, dtype=torch.bool)
        tp = (pred_bin & gt_bin & valid_mask).sum().item()
        fp = (pred_bin & ~gt_bin & valid_mask).sum().item()
        fn = (~pred_bin & gt_bin & valid_mask).sum().item()
        dice = 2 * tp / (2*tp + fp + fn + 1e-8)
        precision = tp / (tp + fp + 1e-8)
        recall = tp / (tp + fn + 1e-8)
        return {'TP': tp, 'FP': fp, 'FN': fn,
                'Dice': dice, 'Precision': precision, 'Recall': recall}

    # ---------- 形态学后处理 ----------
    def _morphology(self, mask_np):
        if self.morph_kernel <= 0:
            return mask_np
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (self.morph_kernel, self.morph_kernel))
        opened = cv2.morphologyEx(mask_np, cv2.MORPH_OPEN, kernel)
        closed = cv2.morphologyEx(opened, cv2.MORPH_CLOSE, kernel)
        return closed

    # ---------- 连通组件筛选 ----------
    def _select_components(self, mask_np):
        """根据设定保留连通组件"""
        num_labels, labels = cv2.connectedComponents(mask_np.astype(np.uint8))
        if num_labels <= 1:
            return mask_np

        areas = np.bincount(labels.ravel())
        # 忽略背景标签0
        if self.keep_largest_only:
            # 找到面积最大的组件（排除背景）
            largest_label = np.argmax(areas[1:]) + 1
            selected = (labels == largest_label)
        else:
            # 保留面积 >= min_component_area 的组件
            keep_labels = set(np.where(areas >= self.min_component_area)[0])
            selected = np.zeros_like(mask_np, dtype=bool)
            for label in keep_labels:
                if label == 0:
                    continue
                selected = selected | (labels == label)

        return selected.astype(np.uint8) * 255

    # ---------- 主处理 ----------
    def run(self):
        myprint=False
        if not self.output_path:
            raise ValueError("输出路径不能为空！")
        out_dir = os.path.dirname(self.output_path)
        if out_dir:
            os.makedirs(out_dir, exist_ok=True)
        if not os.path.splitext(self.output_path)[1]:
            self.output_path += '.png'

        if myprint:print("计算光流特征...")
        mag_fwd, err = self._compute_features()

        # 阈值
        if self.use_otsu:
            if myprint:print("应用 Otsu 自适应阈值...")
            high_mag, th_mag = self._otsu_threshold(mag_fwd)
            high_err, th_err = self._otsu_threshold(err)
            if myprint:print(f"  前向幅值阈值: {th_mag:.4f}, 误差阈值: {th_err:.4f}")
        else:
            # 固定阈值模式（可自定义）
            th_mag = 5.0
            th_err = 10.0
            high_mag = mag_fwd > th_mag
            high_err = err > th_err
            if myprint:print(f"  固定幅值阈值: {th_mag:.4f}, 误差阈值: {th_err:.4f}")

        # 高置信度运动区域（AND）
        if self.motion_combine == 'and':
            high_motion = high_mag & high_err
            if myprint:print("运动区域组合: 高幅值 AND 高误差")
        elif self.motion_combine == 'or':
            high_motion = high_mag | high_err
            if myprint:print("运动区域组合: 高幅值 OR 高误差")
        else:
            raise ValueError("motion_combine 必须是 'and' 或 'or'")

        # 原始预测二值化
        pred_binary = self.pred > 0.5

        # 取交集
        candidate = pred_binary & high_motion
        candidate_np = candidate.cpu().numpy().astype(np.uint8) * 255

        # 连通组件筛选
        if myprint:print("进行连通组件筛选...")
        selected = self._select_components(candidate_np)

        # 形态学后处理
        refined = self._morphology(selected)

        # 去除小面积（可选，已通过组件筛选处理）
        refined_final = torch.from_numpy(refined.astype(bool)).to(self.device)

        # 保存结果
        result_img = Image.fromarray(refined)
        result_img.save(self.output_path)
        if myprint:print(f"修正后掩码已保存至: {self.output_path}")

        # ---------- 评估 ----------
        if self.gt_path is not None:
            if myprint: #是否输出对这种图片的分析
                print("\n" + "="*50)
                print("评估结果（仅有效区域，排除白色导管）")

            # 原始预测
            metrics_orig = self._compute_metrics(pred_binary, self.gt_gray, self.valid)
            if myprint:
                print("\n【原始预测】")
                print(f"  TP: {metrics_orig['TP']}, FP: {metrics_orig['FP']}, FN: {metrics_orig['FN']}")
                print(f"  Dice: {metrics_orig['Dice']:.4f}, Precision: {metrics_orig['Precision']:.4f}, Recall: {metrics_orig['Recall']:.4f}")

            # 最终结果
            metrics_final = self._compute_metrics(refined_final, self.gt_gray, self.valid)
            if myprint:
                print("\n【修正后】")
                print(f"  TP: {metrics_final['TP']}, FP: {metrics_final['FP']}, FN: {metrics_final['FN']}")
                print(f"  Dice: {metrics_final['Dice']:.4f}, Precision: {metrics_final['Precision']:.4f}, Recall: {metrics_final['Recall']:.4f}")
            delta = metrics_final['Precision'] - metrics_orig['Precision']
            if myprint:
                print(f"  Precision 变化: {delta:+.4f}")
                print("="*50)

            # 保存中间结果用于调试
            if False:
                debug_dir = os.path.join(os.path.dirname(self.output_path), "debug")
                os.makedirs(debug_dir, exist_ok=True)
                base = os.path.splitext(os.path.basename(self.output_path))[0]
                Image.fromarray((high_mag.cpu().numpy() * 255).astype(np.uint8)).save(
                    os.path.join(debug_dir, f"{base}_high_mag.png"))
                Image.fromarray((high_err.cpu().numpy() * 255).astype(np.uint8)).save(
                    os.path.join(debug_dir, f"{base}_high_err.png"))
                Image.fromarray((high_motion.cpu().numpy() * 255).astype(np.uint8)).save(
                    os.path.join(debug_dir, f"{base}_high_motion.png"))
                Image.fromarray(candidate_np).save(
                    os.path.join(debug_dir, f"{base}_candidate.png"))

        # print("\n处理完成。")
        # 在 run() 方法的末尾（原评估打印之后）
        if self.gt_path is not None:
            # ... 原有打印代码保持不变 ...
            return metrics_orig, metrics_final   # 新增返回
        else:
            return None, None
import yaml
script_path = os.path.abspath(__file__)
ROOT1 = os.path.dirname(script_path)
file_path = os.path.join(ROOT1, "../..", 'confs/newConfig.yaml')
def getPrecisionMask(tag,needAna=False):
    with open(file_path, 'r', encoding='utf-8') as file:
        config0 = yaml.safe_load(file)
    datasetPath = config0["my"]["datasetPath_rigid.in"]
    customPath = config0["my"]["datasetPath_rigid.in_custom"]

    # 初始化累加器
    total_tp_orig = 0
    total_fp_orig = 0
    total_fn_orig = 0
    total_tp_refined = 0
    total_fp_refined = 0
    total_fn_refined = 0

    num_all = 0
    for userId in os.listdir(datasetPath):
        user_img_dir = os.path.join(datasetPath, userId, "images")
        user_gt_dir = os.path.join(datasetPath, userId, "ground_truth")
        if not os.path.isdir(user_img_dir) or not os.path.isdir(user_gt_dir):
            continue
        for videoId in os.listdir(user_img_dir):
            img_dir = os.path.join(user_img_dir, videoId)
            num_all = num_all+len(os.listdir(img_dir))-2
    
    num_all0 =0
    print("getPrecisionMask...")
    for userId in os.listdir(datasetPath):
        user_img_dir = os.path.join(datasetPath, userId, "images")
        user_gt_dir = os.path.join(datasetPath, userId, "ground_truth")
        if not os.path.isdir(user_img_dir) or not os.path.isdir(user_gt_dir):
            continue
        for videoId in os.listdir(user_img_dir):
            img_dir = os.path.join(user_img_dir, videoId)
            # gt_dir = os.path.join(user_gt_dir, videoId)
            outpath = os.path.join(
                config0["my"]["datasetPath_rigid.out"],
                userId, "decouple", videoId)
            maskPath = os.path.join(outpath, tag + ".orig_mask")
            for frameId_png in os.listdir(img_dir):
                frameId = frameId_png.split(".png")[0]
                if not int(frameId)==0 and not int(frameId)==len(os.listdir(img_dir))-1:
                    num_all0 = num_all0+1
                    print("\rprocess:",num_all0,"/",num_all,end="")
                    
                    fwd_flo = os.path.join(customPath, "raw_flows_gap1", videoId, frameId + ".flo")
                    bwd_flo = os.path.join(customPath, "raw_flows_gap-1", videoId, frameId + ".flo")
                    gt_mask = os.path.join(datasetPath, userId, "ground_truth", videoId + "CATH", frameId + ".png")
                    if not needAna: gt_mask = None
                    pred_mask = os.path.join(maskPath, frameId + ".png")
                    output_mask = os.path.join(
                        config0["my"]["filePathRoot"],
                        config0["my"]["subPath"]["outputs"],
                        "high_precision_refine",
                        videoId, frameId + ".png")
                    
                    refiner = HighPrecisionRefine(
                        fwd_flo, bwd_flo, pred_mask, output_mask,
                        gt_path=gt_mask,
                        black_thresh=50, white_thresh=200,
                        use_otsu=True,
                        motion_combine='and',
                        keep_largest_only=True,
                        min_component_area=200,
                        morph_kernel=3
                    )

                # 运行并获取指标
                metrics_orig, metrics_refined = refiner.run()

                if metrics_orig is not None and metrics_refined is not None:
                    total_tp_orig += metrics_orig['TP']
                    total_fp_orig += metrics_orig['FP']
                    total_fn_orig += metrics_orig['FN']
                    total_tp_refined += metrics_refined['TP']
                    total_fp_refined += metrics_refined['FP']
                    total_fn_refined += metrics_refined['FN']

    # 计算总体 Dice
    dice_orig = 2 * total_tp_orig / (2 * total_tp_orig + total_fp_orig + total_fn_orig + 1e-8)
    dice_refined = 2 * total_tp_refined / (2 * total_tp_refined + total_fp_refined + total_fn_refined + 1e-8)

    def pre(TP,FP,FN):
        return TP/(TP+FP)
        # Precision（精确率） = TP / (TP + FP) —— 预测为正例中实际为正例的比例
    def sn(TP,FP,FN):
        return TP/(TP+FN)
        # Recall（召回率） = TP / (TP + FN) —— 实际为正例中被正确预测的比例
    print("\n" + "=" * 50)
    print("整个数据集评估结果（所有帧聚合）")
    print(f"原始预测: TP={total_tp_orig}, FP={total_fp_orig}, FN={total_fn_orig}, Dice={dice_orig:.4f},pre={pre(total_tp_orig,total_fp_orig,total_fn_orig):.4f},sn={sn(total_tp_orig,total_fp_orig,total_fn_orig):.4f}")
    print(f"修正后:   TP={total_tp_refined}, FP={total_fp_refined}, FN={total_fn_refined}, Dice={dice_refined:.4f},pre={pre(total_tp_refined, total_fp_refined, total_fn_refined):.4f},sn={sn(total_tp_refined, total_fp_refined, total_fn_refined):.4f}")
    print("=" * 50)

# ---------- 使用示例 ----------
if __name__ == "__main__":
    # from nir.param import configs 
    # tag=configs[0]["decouple"]["tag"]
    tag = "A26-03"
    getPrecisionMask(tag, needAna=True)
