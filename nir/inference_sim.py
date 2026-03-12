import os
import yaml
import numpy as np
import torch
from PIL import Image
from tqdm import tqdm
from torchvision import transforms
from pathlib import Path
import shutil

from free_cos.ModelSegment import ModelSegment


class Config:
    """配置管理类"""
    def __init__(self):
        self.script_path = os.path.abspath(__file__)
        self.ROOT = os.path.dirname(self.script_path)
        self._load_config()
        
    def _load_config(self):
        """加载YAML配置文件"""
        config_path = os.path.join(self.ROOT, "../", 'confs/newConfig.yaml')
        with open(config_path, 'r', encoding='utf-8') as file:
            config = yaml.safe_load(file)
            self.root_path = config["my"]["filePathRoot"]
            self.dataset_path_gt = config["my"]["datasetPath_gt"]
            self.dataset_path = config["my"]["datasetPath_rigid.out"]
            
        print(f"真值标签的路径: {self.dataset_path_gt}")
        print(f"解耦数据的路径: {self.dataset_path}")


class ImageLoader:
    """图像加载器类"""
    
    IMAGE_PATHS = {
        "tDSA": "{dataset_path}/{patient_id}/decouple/{video_id}/0.TDSA/{frame_id}",
        "fluid": "{dataset_path}/{patient_id}/decouple/{video_id}/recon_non2/{frame_id}",
        "fluid2": "{dataset_path}/{patient_id}/decouple/{video_id}/C.recon_non2/{frame_id}",
        "fluid3": "{dataset_path}/{patient_id}/decouple/{video_id}/D.recon_non2/{frame_id}",
        "noRigid1": "{dataset_path}/{patient_id}/decouple/{video_id}/A.rigid.main_non1/{frame_id}",
        "noRigid2": "{dataset_path}/{patient_id}/decouple/{video_id}/A.rigid.main_non2/{frame_id}",
        "A-01-epoch2000.rigid.main_non1": "{dataset_path}/{patient_id}/decouple/{video_id}/A-01-epoch2000.rigid.main_non1/{frame_id}",
        "A-01-epoch1000.rigid.main_non1": "{dataset_path}/{patient_id}/decouple/{video_id}/A-01-epoch1000.rigid.main_non1/{frame_id}",
        "A-01-epoch500.rigid.main_non1": "{dataset_path}/{patient_id}/decouple/{video_id}/A-01-epoch500.rigid.main_non1/{frame_id}",
        "A-02-smooth.rigid.main_non1": "{dataset_path}/{patient_id}/decouple/{video_id}/A-02-smooth.rigid.main_non1/{frame_id}",
        "pred": "{dataset_path}/{patient_id}/decouple/{video_id}/A.mask.main_nr2.cf/filter/{frame_id}"
    }
    
    def __init__(self, dataset_path, transform):
        self.dataset_path = dataset_path
        self.transform = transform
    
    def load_image(self, tag, patient_id, video_id, frame_id):
        """根据标签加载单张图像"""
        if tag in self.IMAGE_PATHS:
            img_path = self.IMAGE_PATHS[tag].format(
                dataset_path=self.dataset_path,
                patient_id=patient_id,
                video_id=video_id,
                frame_id=frame_id
            )
            return self._load_single_image(img_path)
        elif tag.startswith("mix"):
            return self._load_mixed_image(tag, patient_id, video_id, frame_id)
        else:
            IMAGE_PATH = "{dataset_path}/{patient_id}/decouple/{video_id}/" + tag + "/{frame_id}"
            img_path = IMAGE_PATH.format(
                dataset_path=self.dataset_path,
                patient_id=patient_id,
                video_id=video_id,
                frame_id=frame_id
            )
            return self._load_single_image(img_path)
    
    def _load_single_image(self, img_path):
        """加载单张图像"""
        img = Image.open(img_path).convert('L')
        img_tensor = self.transform(img).unsqueeze(0).cuda()
        return img_tensor
    
    def _load_mixed_image(self, tag, patient_id, video_id, frame_id):
        """加载混合图像"""
        mix_configs = {
            "mix": [("recon_non2", "orig")],
            "mix2": [("recon_non2", "A.rigid.main_non1")],
            "mix4": [("C.recon_non2", "A.rigid.main_non1")],
            "mix5": [("C.recon_non2", "A.rigid.main_non1", "multiply")],
            "mix6": [("C.recon_non", "A.rigid.main_non1", "multiply")]
        }
        
        if tag not in mix_configs:
            raise ValueError(f"未知的混合类型: {tag}")
            
        config = mix_configs[tag]
        images = []
        
        for sub_path in config[:2]:
            img_path = os.path.join(
                self.dataset_path, patient_id, "decouple", video_id, sub_path, frame_id
            )
            images.append(self._load_single_image(img_path))
        
        if len(config) > 2 and config[2] == "multiply":
            return images[0] * images[1]
        else:
            return (images[0] + images[1]) / 2


class NormalizationCalculator:
    """归一化参数计算器"""
    
    def __init__(self, dataset_path, image_loader):
        self.dataset_path = dataset_path
        self.image_loader = image_loader
    
    def calculate_mean_variance(self, tag, transform, patient_id, video_id, use_quantile=False):
        """计算指定标签图像的均值和方差"""
        orig_path = os.path.join(self.dataset_path, patient_id, "decouple", video_id, "orig")
        frame_ids = os.listdir(orig_path)
        
        images = []
        for frame_id in frame_ids:
            img = self.image_loader.load_image(tag, patient_id, video_id, frame_id)
            images.append(img)
        
        all_images = torch.cat(images, dim=0)
        
        if use_quantile:
            all_images = self._filter_outliers_with_quantile(all_images)
        
        return all_images.mean(), all_images.std()
    
    def _filter_outliers_with_quantile(self, tensor, lower_percentile=5, upper_percentile=95):
        """使用分位数过滤离群值"""
        flat_array = tensor.detach().cpu().numpy().flatten()
        q_min = np.percentile(flat_array, lower_percentile)
        q_max = np.percentile(flat_array, upper_percentile)
        
        mask = (tensor >= q_min) & (tensor <= q_max)
        return tensor[mask]


class ModelManager:
    """模型管理器"""
    
    def __init__(self):
        self.transform = transforms.Compose([transforms.ToTensor()])
    
    def init_model(self, param_path):
        """初始化分割模型"""
        os.environ['MASTER_PORT'] = '169711'
        os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
        torch.backends.cudnn.benchmark = True
        
        model = ModelSegment(n_channels=1, num_classes=1)
        if torch.cuda.is_available():
            model = model.cuda()
            
        checkpoint = torch.load(param_path, map_location=torch.device('cpu'))
        model.load_state_dict(checkpoint['state_dict'])
        return model.eval()


class Main:
    """多组结果综合比较分析器（仅保留推理与保存功能）"""
    
    def __init__(self, config, model_manager, image_loader, norm_calculator, usedVideoId):
        self.config = config
        self.model_manager = model_manager
        self.image_loader = image_loader
        self.norm_calculator = norm_calculator
        self.usedVideoId = usedVideoId
    
    def _get_patient_names(self):
        """获取所有患者名称"""
        return [name for name in os.listdir(self.config.dataset_path) 
                if os.path.isdir(os.path.join(self.config.dataset_path, name))]
    
    def _get_normalization_params(self, config, patient_id, video_id):
        """获取归一化参数"""
        if config.get("precomputed", False):
            return 0, 1
        else:
            norm_method = config["norm_method"]
            return norm_method(config["input_mode"], self.model_manager.transform, patient_id, video_id)
    
    def _get_input_image(self, config, patient_id, video_id, frame_id):
        """根据配置获取输入图像并保存到inputs目录"""
        def save_image_tensor(image, subdir):
            from torchvision.utils import save_image
            path_save = os.path.join(self.config.root_path, subdir, config['name'], video_id)
            os.makedirs(path_save, exist_ok=True)
            save_image(image, os.path.join(path_save, frame_id))
        
        if config.get("precomputed", True):
            input_mode = config.get("input_mode_for_display", "orig")
        else:
            input_mode = config["input_mode"]
        
        img = self.image_loader.load_image(input_mode, patient_id, video_id, frame_id)
        save_image_tensor(img, "inputs")
        
        if "noise_label" in config:
            img_noise = self.image_loader.load_image(config["noise_label"], patient_id, video_id, frame_id)
            save_image_tensor(img_noise, "noiseLayer")
        
        return img
    
    def _normalize_image(self, config, img, mean, std):
        """根据配置对图像进行归一化"""
        if config.get("precomputed", False):
            return img
        else:
            return (img - mean) / std
    
    def _get_prediction2(self, config, model, img_norm, threshold, patient_id, video_id, frame_id):
        """获取预测结果并保存到outputs目录"""
        from torchvision.utils import save_image
        
        if config["mergeMask"]:
            subdir = config['name'] + "-temp"
        else:
            subdir = config['name']
        
        path_save = os.path.join(self.config.root_path, "outputs", subdir, video_id)
        os.makedirs(path_save, exist_ok=True)
        
        if config.get("precomputed", False):
            result_path_template = config["result_path_template"]
            pred_path = result_path_template.format(
                patientID=patient_id, 
                videoId=video_id, 
                frameId=frame_id
            )
            if os.path.exists(pred_path):
                pred = Image.open(pred_path).convert('L')
                pred_tensor = self.model_manager.transform(pred).unsqueeze(0).cuda()
                save_image(pred_tensor, os.path.join(path_save, frame_id))
            else:
                raise FileNotFoundError(f"预计算预测结果不存在: {pred_path}")
        else:
            pred = model(img_norm)["pred"]
            save_image(pred, os.path.join(path_save, frame_id))
    
    def inference0(self, configs, save_path="", block_cath=False, threshold=0.85, onlyInferGT=True):
        """
        执行推理并保存分割结果图
        """
        if len(configs) == 0:
            return
        
        need_model = any(not config.get("precomputed", False) for config in configs)
        if need_model:
            param_path = "../DeNVeR_in/models_config/freecos_Seg.pt"
            model = self.model_manager.init_model(param_path)
        else:
            model = None
        
        patient_names = self._get_patient_names()
        
        # 1. 计算总图像数量
        print("1/3 : 计算总图像数量")
        total_images = 0
        for patient_id in patient_names:
            if onlyInferGT:
                patient_gt_path = os.path.join(self.config.dataset_path_gt, patient_id, "ground_truth")
            else:
                patient_gt_path = os.path.join(self.config.dataset_path_gt, patient_id, "images")
            
            video_names = [name for name in os.listdir(patient_gt_path) 
                           if os.path.isdir(os.path.join(patient_gt_path, name))]
            
            for video_id in video_names:
                if self.usedVideoId is None or video_id in self.usedVideoId:
                    if "CATH" not in video_id:
                        video_path = os.path.join(patient_gt_path, video_id)
                        total_images += len(os.listdir(video_path))
        print(f"总图像数量: {total_images}")
        
        # 2. 推理分割每张图片
        print("2/3 : 推理分割每张图片")
        with tqdm(total=total_images, desc="收集预测结果") as progress_bar:
            for patient_id in patient_names:
                patient_gt_path = os.path.join(self.config.dataset_path, patient_id, "decouple")
                
                video_names = [name for name in os.listdir(patient_gt_path) 
                               if os.path.isdir(os.path.join(patient_gt_path, name))]
                
                for video_id in video_names:
                    if self.usedVideoId is None or video_id in self.usedVideoId:
                        if "CATH" not in video_id:
                            if onlyInferGT:
                                patient_gt_path = os.path.join(self.config.dataset_path_gt, patient_id, "ground_truth")
                            else:
                                patient_gt_path = os.path.join(self.config.dataset_path_gt, patient_id, "images")
                            
                            # 计算归一化参数（如果需要）
                            normalization_params = {}
                            for config in configs:
                                if not config.get("precomputed", False):
                                    mean, std = self._get_normalization_params(config, patient_id, video_id)
                                    normalization_params[config["name"]] = (mean, std)
                            
                            video_names = [name for name in os.listdir(patient_gt_path) 
                                           if os.path.isdir(os.path.join(patient_gt_path, name))]
                            for video_id in video_names:
                                if self.usedVideoId is None or video_id in self.usedVideoId:
                                    if "CATH" not in video_id:
                                        video_path = os.path.join(patient_gt_path, video_id)
                                        for frame_id in os.listdir(video_path):
                                            self._process_single_image_predictions(
                                                patient_id, video_id, frame_id, configs, model,
                                                threshold, block_cath, normalization_params
                                            )
                                            progress_bar.update(1)
    
    def _process_single_image_predictions(self, patient_id, video_id, frame_id, configs,
                                          model, threshold, block_cath, normalization_params):
        """处理单张图像的所有配置预测结果"""
        for config in configs:
            config_name = config["name"]
            img = self._get_input_image(config, patient_id, video_id, frame_id)
            
            if not config.get("precomputed", False) and config_name in normalization_params:
                mean, std = normalization_params[config_name]
                img_norm = self._normalize_image(config, img, mean, std)
            else:
                img_norm = img
            
            self._get_prediction2(config, model, img_norm, threshold, patient_id, video_id, frame_id)


def getUsedVideoId(folder_path="xca_dataset"):
    """获取有标注的视频ID列表"""
    usedVideoId = []
    for userId in os.listdir(folder_path):
        path_gt_user = os.path.join(folder_path, userId, "ground_truth")
        if not os.path.exists(path_gt_user):
            continue
        videoId_list = os.listdir(path_gt_user)
        for videoId in videoId_list:
            if "CATH" not in videoId:
                usedVideoId.append(videoId)
    return usedVideoId


if __name__ == "__main__":
    # 初始化配置和各个管理器
    config = Config()
    model_manager = ModelManager()
    image_loader = ImageLoader(config.dataset_path, model_manager.transform)
    norm_calculator = NormalizationCalculator(config.dataset_path, image_loader)
    
    threshold = 0.5
    block_cath = True
    
    configs = [
        {
            "name": "myFreeCOS_test4",
            "precomputed": False,
            "noise_label": "orig",
            "input_mode": "orig",
            "norm_method": norm_calculator.calculate_mean_variance,
            "binarize": True,
            "inferenceAll": False,
            "mergeMask": False,
            "fineTuning": True,
        },
    ]
    
    # 获取有标注的视频ID列表
    with open(os.path.join(os.path.dirname(__file__), "../", 'confs/newConfig.yaml'), 'r', encoding='utf-8') as file:
        config0 = yaml.safe_load(file)
    usedVideoId = getUsedVideoId(config0["my"]["datasetPath_rigid.in"])
    
    print("二、分割推理部分")
    main = Main(config, model_manager, image_loader, norm_calculator, usedVideoId)
    main.inference0(configs, config.root_path + "/", block_cath, threshold, onlyInferGT=True)

