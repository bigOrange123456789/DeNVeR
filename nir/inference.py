import os
import numpy as np
from PIL import Image
from tqdm import tqdm
import torch


from nir.new_batch_topK_lib import Config, ImageLoader, NormalizationCalculator, ModelManager, Evaluator, StatisticalAnalyzer, ResultVisualizer, ResultSaver, denoising

import cv2
import numpy as np
from pathlib import Path     
import os   
class PostProcessing:
    def __init__(self, inpath, outpath, progress_bar):
        self.inpath =inpath
        self.outpath= outpath
        self.progress_bar = progress_bar        
        self.update_masks()

    def update_masks(self):
        input_folder=self.inpath 
        output_folder=self.outpath
        """
        通过相邻帧更新所有mask图片
        
        参数:
            input_folder: 输入图片文件夹路径
            output_folder: 输出图片文件夹路径
        """
        # 创建输出文件夹
        Path(output_folder).mkdir(parents=True, exist_ok=True)
        
        # 获取所有png文件并按名称排序
        mask_files = sorted([f for f in os.listdir(input_folder) if f.endswith('.png')])
        
        if not mask_files:
            print(f"在文件夹 {input_folder} 中没有找到png文件")
            return
        
        # 一、处理第一帧：取前三帧的最大值
        if len(mask_files) >= 3:
            # 有至少3帧，第一帧取前3帧的最大值
            frame0 = cv2.imread(os.path.join(input_folder, mask_files[0]), cv2.IMREAD_GRAYSCALE)
            frame1 = cv2.imread(os.path.join(input_folder, mask_files[1]), cv2.IMREAD_GRAYSCALE)
            frame2 = cv2.imread(os.path.join(input_folder, mask_files[2]), cv2.IMREAD_GRAYSCALE)
            
            # 计算前三帧的最大值
            first_frame_max = np.maximum.reduce([frame0, frame1, frame2])
            cv2.imwrite(os.path.join(output_folder, mask_files[0]), first_frame_max)
        else:
            # 只有2帧，第一帧取前2帧的最大值
            frame0 = cv2.imread(os.path.join(input_folder, mask_files[0]), cv2.IMREAD_GRAYSCALE)
            frame1 = cv2.imread(os.path.join(input_folder, mask_files[1]), cv2.IMREAD_GRAYSCALE)
            
            if frame1.shape != frame0.shape:
                frame1 = cv2.resize(frame1, (frame0.shape[1], frame0.shape[0]))
            
            first_frame_max = np.maximum(frame0, frame1)
            cv2.imwrite(os.path.join(output_folder, mask_files[0]), first_frame_max)
        
        # 二、处理中间帧：M2(i) = max(M(i-1), M(i), M(i+1))
        for i in range(1, len(mask_files) - 1):
            # 读取前一张、当前和后一张mask
            prev_mask = cv2.imread(os.path.join(input_folder, mask_files[i-1]), cv2.IMREAD_GRAYSCALE)
            curr_mask = cv2.imread(os.path.join(input_folder, mask_files[i]), cv2.IMREAD_GRAYSCALE)
            next_mask = cv2.imread(os.path.join(input_folder, mask_files[i+1]), cv2.IMREAD_GRAYSCALE)
            
            # 计算三帧的最大值
            updated_mask = np.maximum.reduce([prev_mask, curr_mask, next_mask])
            
            # 保存更新后的mask
            output_path = os.path.join(output_folder, mask_files[i])
            cv2.imwrite(output_path, updated_mask)
            
            # if (i + 1) % 100 == 0:  # 每处理100帧打印一次进度
            #     print(f"已处理第 {i+1}/{len(mask_files)} 帧")
            self.progress_bar.set_postfix(p=f"已处理第 {i+1}/{len(mask_files)} 帧")
        
        # 三、处理最后一帧：取最后三帧的最大值
        if len(mask_files)>=3:
            # 有至少3帧，最后一帧取最后3帧的最大值
            frame_last3 = cv2.imread(os.path.join(input_folder, mask_files[-3]), cv2.IMREAD_GRAYSCALE)
            frame_last2 = cv2.imread(os.path.join(input_folder, mask_files[-2]), cv2.IMREAD_GRAYSCALE)
            frame_last1 = cv2.imread(os.path.join(input_folder, mask_files[-1]), cv2.IMREAD_GRAYSCALE)
            
            # 计算最后三帧的最大值
            last_frame_max = np.maximum.reduce([frame_last3, frame_last2, frame_last1])
            cv2.imwrite(os.path.join(output_folder, mask_files[-1]), last_frame_max)
        else:
            # 只有2帧，最后一帧取最后2帧的最大值（与第一帧相同）
            frame_last2 = cv2.imread(os.path.join(input_folder, mask_files[-2]), cv2.IMREAD_GRAYSCALE)
            frame_last1 = cv2.imread(os.path.join(input_folder, mask_files[-1]), cv2.IMREAD_GRAYSCALE)
            
            if frame_last2.shape != frame_last1.shape:
                frame_last2 = cv2.resize(frame_last2, (frame_last1.shape[1], frame_last1.shape[0]))
            
            last_frame_max = np.maximum(frame_last2, frame_last1)
            cv2.imwrite(os.path.join(output_folder, mask_files[-1]), last_frame_max)
        
        # print(f"所有帧处理完成！结果保存在: {output_folder}")

class Main:
    """多组结果综合比较分析器（整合了单对比较功能）"""
    
    def _get_patient_names(self):
        """获取所有患者名称"""
        return [name for name in os.listdir(self.config.dataset_path) 
                if os.path.isdir(os.path.join(self.config.dataset_path, name))]
    
    def _count_annotated_images(self, patient_names):
        """计算有标注的图像总数"""
        total_count = 0
        for patient_id in patient_names:
            # patient_gt_path = os.path.join(self.config.dataset_path_gt, patient_id, "ground_truth")
            patient_gt_path = os.path.join(self.config.dataset_path, patient_id, "decouple")
            if not os.path.exists(patient_gt_path):
                continue
                
            video_names = [name for name in os.listdir(patient_gt_path) 
                          if os.path.isdir(os.path.join(patient_gt_path, name))]
            
            for video_id in video_names:
                if "CATH" not in video_id:  # 只统计非导管视频
                    video_path = os.path.join(patient_gt_path, video_id,"A.rigid.main_non1")#每个视频的全部关键帧
                    total_count += len(os.listdir(video_path))
        
        return total_count
    
    def _get_normalization_params(self, config, patient_id, video_id):
        """获取归一化参数"""
        if config.get("precomputed", False):
            return 0, 1  # 预计算方法不需要归一化
        else:
            # 使用配置中指定的归一化方法
            norm_method = config["norm_method"]
            return norm_method(config["input_mode"], self.model_manager.transform, patient_id, video_id)
    
    def _get_input_image(self, config, patient_id, video_id, frame_id):
        """根据配置获取输入图像"""
        if config.get("precomputed", True):
            # 预计算方法使用显示模式
            input_mode = config.get("input_mode_for_display", "orig") #读取指定键的值，若键不存在则使用默认值
        else:
            # 没有进行预计算的时候，模型推理方法使用自己的输入模式
            input_mode = config["input_mode"]
        
        # print(
        #     "input_mode",input_mode, 
        #     "patient_id:",patient_id, 
        #     "video_id:",video_id, 
        #     "frame_id",frame_id
        # )
        return self.image_loader.load_image(input_mode, patient_id, video_id, frame_id)
    
    def _normalize_image(self, config, img, mean, std):
        """根据配置对图像进行归一化"""
        if config.get("precomputed", False):
            return img  # 预计算方法不需要归一化
        else:
            return (img - mean) / std  # 模型推理方法需要归一化
        
    def _get_prediction2(self, config, model, img_norm, threshold, patient_id, video_id, frame_id):
        """获取预测结果"""
        def s(pred):
            from torchvision.utils import save_image
            if config["mergeMask"]:
                path_save = os.path.join(self.config.root_path, "outputs", config['name']+"-temp", video_id)
            else:
                path_save = os.path.join(self.config.root_path, "outputs", config['name'], video_id)
            os.makedirs(path_save, exist_ok=True)
            save_image(pred, os.path.join(path_save,frame_id))
        
        if config.get("precomputed", False):
            # 读取预计算的分割结果
            result_path_template = config["result_path_template"]
            pred_path = result_path_template.format(
                patientID=patient_id, 
                videoId=video_id, 
                frameId=frame_id
            )
            
            if os.path.exists(pred_path):
                pred = Image.open(pred_path).convert('L')
                pred_tensor = self.model_manager.transform(pred).unsqueeze(0).cuda()
                s(pred_tensor)
            else:
                raise FileNotFoundError(f"预计算预测结果不存在: {pred_path}")
        else:
            # 使用模型进行推理
            pred = model(img_norm)["pred"]
            s(pred)
    
    def _get_image_path(self, config, patient_id, video_id, frame_id):
        """获取图像路径"""
        if config.get("precomputed", False):
            # 预计算方法的结果路径
            result_path_template = config["result_path_template"]
            return result_path_template.format(
                patientID=patient_id, 
                videoId=video_id, 
                frameId=frame_id
            )
        else:
            # 模型推理方法的输入图像路径
            input_mode = config["input_mode"]
            return self.image_loader.IMAGE_PATHS[input_mode].format(
                dataset_path=self.config.dataset_path,
                patient_id=patient_id,
                video_id=video_id,
                frame_id=frame_id
            )
    
    def _apply_catheter_mask(self, pred_tag1, pred_tag2, gt_tensor):
        """
        应用导管掩码到预测结果
            mask_cath = torch.zeros_like(gt_tensor)
            mask_vessel = torch.zeros_like(gt_tensor)
            mask_cath[gt_tensor>0.75]=1
            mask_vessel[(gt_tensor>0.25) & (gt_tensor<0.75)]=1 
            gt_tensor = mask_vessel
            pred_tag1 = pred_tag1 * ( 1 - mask_cath )
            pred_tag2 = pred_tag2 * ( 1 - mask_cath )
        """
        # 重新计算导管掩码
        mask_cath = torch.zeros_like(gt_tensor)
        mask_cath[gt_tensor>0.75]=1
        # mask_cath
        # 这里需要根据实际的ground truth值来确定导管区域
        # 假设在之前的处理中已经正确设置了gt_tensor
        
        # 屏蔽导管区域的预测
        pred_tag1 = pred_tag1 * (1 - mask_cath)
        pred_tag2 = pred_tag2 * (1 - mask_cath)
        
        return pred_tag1, pred_tag2
    
    def __init__(self, config, model_manager, image_loader, norm_calculator,usedVideoId):
        self.config = config
        self.model_manager = model_manager
        self.image_loader = image_loader
        self.norm_calculator = norm_calculator
        self.usedVideoId = usedVideoId
        self.statistical_analyzer = StatisticalAnalyzer()
        self.visualizer = ResultVisualizer()
        self.saver = ResultSaver()
    
    # ... 之前的其他方法保持不变 ...

    def inference0(self, configs, save_path="", block_cath=False, threshold=0.85,onlyInferGT=True):    
        """        
        参数:
            configs: 配置列表
            threshold: 分割阈值
            block_cath: 是否屏蔽导管区域
            
        返回:
            dict: 包含所有配置预测结果的字典
        """
        if len(configs)==0: return
        # 检查哪些配置需要模型
        need_model = any(not config.get("precomputed", False) for config in configs)
        
        # 初始化模型（如果需要）
        if need_model:
            param_path = "../DeNVeR_in/models_config/freecos_Seg.pt"
            model = self.model_manager.init_model(param_path)
        else:
            model = None
        
        # 获取所有患者
        patient_names = self._get_patient_names()
        
        # 1.计算总图像数量
        print("1/3 : 计算总图像数量")
        # total_images = self._count_annotated_images(patient_names)
        total_images = 0
        for patient_id in patient_names:
            if onlyInferGT:#***只收集有标注图片的预测结果***
                patient_gt_path = os.path.join(self.config.dataset_path_gt, patient_id, "ground_truth")
            else: #推理全部
                patient_gt_path = os.path.join(self.config.dataset_path_gt, patient_id, "images")
                
            video_names = [name for name in os.listdir(patient_gt_path) 
                          if os.path.isdir(os.path.join(patient_gt_path, name))]
            
            for video_id in video_names:
             if self.usedVideoId is None or video_id in self.usedVideoId:
                if "CATH" not in video_id:  # 只统计非导管视频   
                    # if onlyInferGT:#***只收集有标注图片的预测结果***
                    #     video_path = os.path.join(patient_gt_path, video_id,"A.rigid.main_non1")#每个视频的全部关键帧
                    # else: #推理全部
                    #     video_path = os.path.join(patient_gt_path, video_id,"A.rigid.main_non1")
                    video_path = os.path.join(patient_gt_path, video_id)
                    total_images += len(os.listdir(video_path))
                    # print("os.listdir(video_path)",os.listdir(video_path))
        print(f"总图像数量: {total_images}")
        
        # 2.推理分割每张图片
        print("2/3 : 推理分割每张图片")
        if True:
         with tqdm(total=total_images, desc="收集预测结果") as progress_bar:
            for patient_id in patient_names:
                """处理单个患者的所有配置预测结果"""
                # patient_gt_path = os.path.join(self.config.dataset_path_gt, patient_id, "ground_truth")
                patient_gt_path = os.path.join(self.config.dataset_path, patient_id, "decouple")
                
                video_names = [name for name in os.listdir(patient_gt_path) 
                            if os.path.isdir(os.path.join(patient_gt_path, name))]
                
                for video_id in video_names:
                 if self.usedVideoId is None or video_id in self.usedVideoId:
                    if "CATH" not in video_id:  # 只处理非导管视频
                        if onlyInferGT:#***只收集有标注图片的预测结果***
                            patient_gt_path = os.path.join(self.config.dataset_path_gt, patient_id, "ground_truth")
                        else: #推理全部
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
                            if "CATH" not in video_id:  # 只统计非导管视频  
                                video_path = os.path.join(patient_gt_path, video_id)
                                for frame_id in os.listdir(video_path):
                                    self._process_single_image_predictions(
                                        patient_id, video_id, frame_id, configs, model, threshold, 
                                        block_cath, normalization_params
                                    )#path_save = os.path.join(self.config.root_path, "outputs", config['name']+"-orig", video_id)
                                    progress_bar.update(1)
                        
        
        print("3/3 : 根据需要对分割结果进行后处理")
        num_video=0
        for patient_id in patient_names:
                patient_gt_path = os.path.join(self.config.dataset_path, patient_id, "decouple")
                video_names = [name for name in os.listdir(patient_gt_path) 
                            if os.path.isdir(os.path.join(patient_gt_path, name))]
                for video_id in video_names:
                    if "CATH" not in video_id:  # 只处理非导管视频
                        for config in configs:
                         if config["mergeMask"]:
                            num_video=num_video+1
        if configs and num_video>0:
         with tqdm(total=num_video, desc="收集预测结果") as progress_bar:
            for patient_id in patient_names:
                """处理单个患者的所有配置预测结果"""
                patient_gt_path = os.path.join(self.config.dataset_path, patient_id, "decouple")
                
                video_names = [name for name in os.listdir(patient_gt_path) 
                            if os.path.isdir(os.path.join(patient_gt_path, name))]
                
                for video_id in video_names:
                    if "CATH" not in video_id:  # 只处理非导管视频
                        for config in configs:
                         if config["mergeMask"]:
                            path_in = os.path.join(self.config.root_path, "outputs", config['name']+"-temp", video_id)
                            path_out = os.path.join(self.config.root_path, "outputs", config['name'], video_id)
                            PostProcessing(path_in, path_out,progress_bar)
                            progress_bar.update(1)

    def _process_single_image_predictions(self, patient_id, video_id, frame_id, configs, 
                                        model, threshold, block_cath, normalization_params):
        """处理单张图像的所有配置预测结果"""
        
        # 处理每个配置的预测
        for config in configs:
            config_name = config["name"]
            
            # 获取输入图像
            img = self._get_input_image(config, patient_id, video_id, frame_id)
            
            # 归一化处理（如果需要）
            if not config.get("precomputed", False) and config_name in normalization_params:
                mean, std = normalization_params[config_name]
                img_norm = self._normalize_image(config, img, mean, std)
            else:
                img_norm = img
            
            # 获取预测结果
            self._get_prediction2(config, model, img_norm, threshold, patient_id, video_id, frame_id)
            
def main(): 
    """主函数"""
    # 2025.11.26-15:59:做了一个诡异的梦
    # 被聚集到一个城市中的一个庭院中，天空中飞过三个外星飞船，还隐约能够看到巨大的鱿鱼
    # 
    # 初始化配置和各个管理器
    config = Config()
    model_manager = ModelManager()
    image_loader = ImageLoader(config.dataset_path, model_manager.transform)
    norm_calculator = NormalizationCalculator(config.dataset_path, image_loader)
    #os.makedirs(save_masks_dir, exist_ok=True)
    
    # 设置参数
    threshold = 0.5
    block_cath = True
    
    # 定义多个配置
    configs = [#在短视频数据上的测试结果
        ##########################  DeNVeR.010  ##########################  
        # {
        #     "name": "1.masks", 
        #     "precomputed": True,
        #     "result_path_template": os.path.join("../DeNVeR.006/log_6/outputs", "{videoId}", "1.masks", "{frameId}"),
        #     "input_mode_for_display": "orig",
        #     "binarize": True
        # },
        # {
        #     "name": "2.2.planar", 
        #     "precomputed": True,
        #     "result_path_template": os.path.join("../DeNVeR.006/log_6/outputs", "{videoId}", "2.2.planar", "{frameId}"),
        #     "input_mode_for_display": "orig",
        #     "binarize": True
        # },
        # {
        #     "name": "3.parallel", 
        #     "precomputed": True,
        #     "result_path_template": os.path.join("../DeNVeR.006/log_6/outputs", "{videoId}", "3.parallel", "{frameId}"),
        #     "input_mode_for_display": "orig",
        #     "binarize": True
        # },
        # {
        #     "name": "4.deform", 
        #     "precomputed": True,#复制结果
        #     "result_path_template": os.path.join("../DeNVeR.006/log_6/outputs", "{videoId}", "4.deform", "{frameId}"),
        #     "input_mode_for_display": "orig",
        #     "binarize": True
        # },
        # {
        #     "name": "5.refine", 
        #     "precomputed": True,
        #     "result_path_template": os.path.join("../DeNVeR.006/log_6/outputs", "{videoId}", "5.refine", "{frameId}"),
        #     "input_mode_for_display": "orig",
        #     "binarize": True
        # },
        # {
        #     "name": "orig",
        #     "precomputed": False,
        #     "input_mode": "orig",
        #     "norm_method": norm_calculator.calculate_mean_variance,
        #     "binarize": True
        # },
        ##########################  DeNVeR.011  ##########################  
        # {
        #     "name": "_011_continuity_01",
        #     "precomputed": False,
        #     "input_mode": "orig",
        #     "norm_method": norm_calculator.calculate_mean_variance,
        #     "binarize": True,
        #     "inferenceAll":True,
        #     "mergeMask":True,
        # },#"_011_continuity_01-temp" : orig
        # {
        #     "name": "_011_continuity_02",
        #     "precomputed": False,
        #     "input_mode": "noRigid1",
        #     "norm_method": norm_calculator.calculate_mean_variance,
        #     "binarize": True,
        #     "inferenceAll":True,
        #     "mergeMask":True,
        # }, #"_011_continuity_02-temp" : noRigid1
        ##########################  DeNVeR.012  ##########################  
        # {
        #     "name": "_012_continuity_01",
        #     "precomputed": False,
        #     "input_mode": "fluid2",
        #     "norm_method": norm_calculator.calculate_mean_variance,
        #     "binarize": True,
        #     "inferenceAll":False,
        # },
        # {
        #     "name": "_012_02_bigMaskFluid",
        #     "precomputed": False,
        #     "input_mode": "fluid3",
        #     "norm_method": norm_calculator.calculate_mean_variance,
        #     "binarize": True,
        #     "inferenceAll":False,
        # },
        ####################### 改用长视频子集合测试(没有相机移动) ####################### 
        ##########################  DeNVeR.013  ##########################  
        # {
        #     "name": "_013_long01_noRigid1",
        #     "precomputed": False,
        #     "input_mode": "noRigid1",
        #     "norm_method": norm_calculator.calculate_mean_variance,
        #     "binarize": True,
        #     "inferenceAll":True,
        #     "mergeMask":True,
        # },
        # {
        #     "name": "_013_long02_bigMaskFluid",
        #     "precomputed": False,
        #     "input_mode": "fluid3",#bigMask
        #     "norm_method": norm_calculator.calculate_mean_variance,
        #     "binarize": True,
        #     "inferenceAll":True,#False,
        #     "mergeMask":False,
        # },
        # {
        #     "name": "_013_long03_smallMaskFluid",
        #     "precomputed": False,
        #     "input_mode": "fluid2",#smallMask
        #     "norm_method": norm_calculator.calculate_mean_variance,
        #     "binarize": True,
        #     "inferenceAll":True,#False,
        #     "mergeMask":False,
        # },
        # {
        #     "name": "_013_04_traditionalDSA",
        #     "precomputed": False,
        #     "input_mode": "tDSA",#smallMask
        #     "norm_method": norm_calculator.calculate_mean_variance,
        #     "binarize": True,
        #     "inferenceAll":True,
        #     "mergeMask":False,
        # },
        # {
        #     "name": "_013_05_orig",
        #     "precomputed": False,
        #     "input_mode": "orig",
        #     "norm_method": norm_calculator.calculate_mean_variance,
        #     "binarize": True,
        #     "inferenceAll":True,
        #     "mergeMask":False,
        # },
        ##########################  测试整个数据集  ##########################  
        ##########################  DeNVeR.014  ##########################  
        # {
        #     "name": "noRigid1",
        #     "precomputed": False,
        #     "input_mode": "noRigid1",
        #     "norm_method": norm_calculator.calculate_mean_variance,
        #     "binarize": True,
        #     "inferenceAll": False,
        #     "mergeMask": False,
        # },
        # {
        #     "name": "fluid2",
        #     "precomputed": False,
        #     "input_mode": "fluid2",
        #     "norm_method": norm_calculator.calculate_mean_variance,
        #     "binarize": True,
        #     "inferenceAll": False,
        #     "mergeMask": False,
        # },
        # 使用单视频进行优化，在无显著下降的情况下提高运行速度
        ####################### 长视频子集合 xca_dataset_sub1 ####################### 
        ##########################  DeNVeR.015  ##########################  
        # {#刚体去噪,训练4000批次
        #     # "decouple":{#解耦
        #     #     ########################
        #     #     "de-rigid":"1",
        #     #     "epoch":4000,          #只兼容了startDecouple1
        #     #     "tag":"A",#只兼容了startDecouple1
        #     #     ########################
        #     #     "de-soft":None,
        #     # },
        #     "name": "_015_01_noRigid1",
        #     "precomputed": False,
        #     "input_mode": "noRigid1",
        #     "norm_method": norm_calculator.calculate_mean_variance,
        #     "binarize": True,
        #     "inferenceAll": True,#False,
        #     "mergeMask": False,
        # },
        # {#只训练2000批次
        #     # "decouple":{#解耦
        #     #     ########################
        #     #     "de-rigid":"1",
        #     #     "epoch":2000,          #只兼容了startDecouple1
        #     #     "tag":"A-01-epoch2000",#只兼容了startDecouple1
        #     #     ########################
        #     #     "de-soft":None,
        #     # },
        #     "name": "_015_02_noRigid1(b2000)",
        #     "precomputed": False,
        #     "input_mode": "A-01-epoch2000.rigid.main_non1",
        #     "norm_method": norm_calculator.calculate_mean_variance,
        #     "binarize": True,
        #     "inferenceAll": False,
        #     "mergeMask": False,
        # },
        # {#只训练1000批次
        #     # "decouple":{#解耦
        #     #     ########################
        #     #     "de-rigid":"1",
        #     #     "epoch":1000,          #只兼容了startDecouple1
        #     #     "tag":"A-01-epoch1000",#只兼容了startDecouple1
        #     #     ########################
        #     #     "de-soft":None,
        #     # },
        #     "name": "_015_03_noRigid1(b1000)",
        #     "precomputed": False,
        #     "input_mode": "A-01-epoch1000.rigid.main_non1",
        #     "norm_method": norm_calculator.calculate_mean_variance,
        #     "binarize": True,
        #     "inferenceAll": True,#False,
        #     "mergeMask": False,
        # },
        # {#只训练500批次
        #     # "decouple":{#解耦
        #     #     ########################
        #     #     "de-rigid":"1",
        #     #     "epoch":500,          #只兼容了startDecouple1
        #     #     "tag":"A-01-epoch500",#只兼容了startDecouple1
        #     #     ########################
        #     #     "de-soft":None,
        #     # },
        #     "name": "_015_04_noRigid1(b500)",
        #     "precomputed": False,
        #     "input_mode": "A-01-epoch500.rigid.main_non1",
        #     "norm_method": norm_calculator.calculate_mean_variance,
        #     "binarize": True,
        #     "inferenceAll": True,#False,
        #     "mergeMask": False,
        # },
        ##########################  DeNVeR.016  ##########################  
        # {#进行刚体平滑测试
        #     "decouple":{#解耦
        #         ########################
        #         "de-rigid":"1",
        #         "epoch":1000,          #只兼容了startDecouple1
        #         "tag":"A-02-smooth",#只兼容了startDecouple1
        #         "useSmooth":True,
        #         ########################
        #         "de-soft":None,
        #     },
        #     "name": "_016_01_noRigid1(b1000)",
        #     "precomputed": False,
        #     "input_mode": "A-02-smooth.rigid.main_non1",
        #     "norm_method": norm_calculator.calculate_mean_variance,
        #     "binarize": True,
        #     "inferenceAll": False,#True,#False,
        #     "mergeMask": False,
        # },
        # {#测试总刚体去噪效果是否更好: 基于主刚体层(old) vs 基于总刚体层(now)
        #     # "decouple":{#解耦
        #     #     ########################
        #     #     "de-rigid":"1",
        #     #     "epoch":1000,          #只兼容了startDecouple1
        #     #     "tag":"A-02-smooth",#只兼容了startDecouple1
        #     #     "useSmooth":True,
        #     #     ########################
        #     #     "de-soft":None,
        #     # },
        #     "name": "_016_01_noRigidAll1(b1000)",
        #     "precomputed": False,
        #     "input_mode": "A-02-smooth.rigid.non1",
        #     "norm_method": norm_calculator.calculate_mean_variance,
        #     "binarize": True,
        #     "inferenceAll": False,#True,#False,
        #     "mergeMask": False,
        # },
        # {#测试不同训练批次的影响 #对相机运动的视频应该会有影响
        #     "decouple":{#解耦
        #         ########################
        #         "de-rigid":"1",
        #         "epoch":4000,          #只兼容了startDecouple1
        #         "tag":"A-02-smooth-e4000",#只兼容了startDecouple1
        #         "useSmooth":True,
        #         ########################
        #         "de-soft":None,
        #     },
        #     "name": "_016_02_noRigid1(b4000)",
        #     "precomputed": False,
        #     "input_mode": "A-02-smooth-e4000.rigid.main_non1",
        #     "norm_method": norm_calculator.calculate_mean_variance,
        #     "binarize": True,
        #     "inferenceAll": False,#True,#False,
        #     "mergeMask": False,
        # },
        ##########################  DeNVeR.017(测试smooth损失对运动数据集的影响)  ##########################  
        {#在相机运动的场景下测试
            "decouple":{#解耦
                ########################
                "de-rigid":"1",
                "epoch":2000,          #只兼容了startDecouple1
                "tag":"A-02-smooth-e2000",#只兼容了startDecouple1
                "useSmooth":True,
                ########################
                "de-soft":None,
            },
            "name": "_017_01_nr(b2000)smooth",
            "precomputed": False,
            "input_mode": "A-02-smooth-e2000.rigid.main_non1",
            "norm_method": norm_calculator.calculate_mean_variance,
            "binarize": True,
            "inferenceAll": False,#True,#False,
            "mergeMask": False,
        },
        {#在相机运动的场景下测试
            "decouple":{#解耦
                ########################
                "de-rigid":"1",
                "epoch":2000,          #只兼容了startDecouple1
                "tag":"A-02-e2000",#只兼容了startDecouple1
                "useSmooth":False,
                ########################
                "de-soft":None,
            },
            "name": "_017_02_nr(b2000)",
            "precomputed": False,
            "input_mode": "A-02-e2000.rigid.main_non1",
            "norm_method": norm_calculator.calculate_mean_variance,
            "binarize": True,
            "inferenceAll": False,#True,#False,
            "mergeMask": False,
        },
    ]#不要重复做实验，要相信之前的结果
    [#没有进行的测试
        # {#测试不同的平滑权重的影响（因为最终收敛到0了(在2000epoch之内)，所以我感觉权重影响不大）
        #     "decouple":{#解耦
        #         ########################
        #         "de-rigid":"1",
        #         "epoch":1000,          #只兼容了startDecouple1
        #         "tag":"A-03-smooth-e1000",#只兼容了startDecouple1
        #         "useSmooth":True,
        #         ########################
        #         "de-soft":None,
        #     },
        #     "name": "_016_02_noRigid1(b4000)",
        #     "precomputed": False,
        #     "input_mode": "A-02-smooth-e4000.rigid.main_non1",
        #     "norm_method": norm_calculator.calculate_mean_variance,
        #     "binarize": True,
        #     "inferenceAll": False,#True,#False,
        #     "mergeMask": False,
        # },
    ]
    usedVideoId = None
    # usedVideoId = [
    #     'CVAI-1207LAO44_CRA29', 'CVAI-1207RAO2_CAU30', 
    #     # 'CVAI-1247RAO30_CAU24', 'CVAI-1250LAO31_CRA27', 'CVAI-1250LAO50_CAU1', 
    #     # 'CVAI-1251LAO30_CRA19', 'CVAI-1251LAO59_CAU24', 'CVAI-1253LAO0_CAU29', 'CVAI-1253RAO29_CAU19', 
    #     # 'CVAI-1255LAO52_CAU1', 'CVAI-1255LAO57_CRA18'
    #     ]
    ''' 将视频在这里进行解耦 '''
    
    print("代码分为三个阶段:视频解耦->分割推理->对比分析")
    print("一、视频解耦部分")
    for c in configs:
        if "decouple" in c:
            c["decouple"]["name"]=c["name"]#用于给处理进度文件命名
            denoising(c["decouple"],usedVideoId=usedVideoId)
    print("视频解耦完成!")
    # exit(0)

    print("二、分割推理部分")
    configs1=[]
    configs2=[]
    for c in configs:
        if (not "inferenceAll" in c) or (c["inferenceAll"]==False):
            c["inferenceAll"]=False
            configs1.append(c)
        else:
            configs2.append(c)

    main = Main(config, model_manager, image_loader, norm_calculator,usedVideoId)
    main.inference0(
        configs1, config.root_path + "/", block_cath, threshold,onlyInferGT=True
    )#推理全部图像
    main.inference0(
        configs2, config.root_path + "/", block_cath, threshold,onlyInferGT=False
    )#推理全部图像

if __name__ == "__main__":
    main()#测试在训练过程中f1、recall、precise的变化
