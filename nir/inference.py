import os
# import yaml
# import json
import datetime
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
# from scipy import stats
from PIL import Image
from tqdm import tqdm
import torch
# from torchvision import transforms
# import torch.backends.cudnn as cudnn

# Import custom modules
# from nir.new import startDecouple1, startDecouple3
from free_cos.newTrain import initCSV, save2CVS, getIndicators
# from nir.analysisOri import getModel
# from free_cos.ModelSegment import ModelSegment
# from free_cos.main import mainFreeCOS
# from preprocess.mySkeleton import getCatheter


from nir.new_batch_topK_lib import Config, ImageLoader, NormalizationCalculator, ModelManager, Evaluator, StatisticalAnalyzer, ResultVisualizer, ResultSaver, denoising

class ComprehensiveMulComparison:
    """多组结果综合比较分析器（整合了单对比较功能）"""
    
    # def analyze_pair(self, config1, config2, K=5, threshold=0.85, save_path="", block_cath=False):
    #     """
    #     对两种配置进行全面的比较分析（原ComprehensiveComparison的功能）
        
    #     参数:
    #         config1: 第一种配置的字典
    #         config2: 第二种配置的字典  
    #         K: 对比图中显示的图像数量
    #         threshold: 分割阈值
    #         save_path: 保存路径
    #         block_cath: 是否屏蔽导管区域
        
    #     返回:
    #         tuple: (数据DataFrame, t检验结果, 最佳结果列表, 最差结果列表)
    #     """
    #     tag1 = config1["name"]
    #     tag2 = config2["name"]
        
    #     print(f"开始单对比较分析: {tag1} vs {tag2}")
    #     print(f"模式: {'屏蔽标签中的导管' if block_cath else '不屏蔽标签中的导管'}")
    #     print("=" * 60)
        
    #     # 设置保存路径
    #     timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    #     mode_suffix = "_hasCath" if block_cath else "_nonCath"
        
    #     save_img_path_best = f"{save_path}{tag1}_vs_{tag2}_best{mode_suffix}_{timestamp}.jpg"
    #     save_img_path_worst = f"{save_path}{tag1}_vs_{tag2}_worst{mode_suffix}_{timestamp}.jpg"
    #     save_excel_path = f"{save_path}{tag1}_vs_{tag2}_statistical_analysis{mode_suffix}_{timestamp}.xlsx"
        
    #     # 步骤1: 收集所有图像的指标数据
    #     print("步骤1/4: 收集所有图像的指标数据...")
    #     df, best_results, worst_results = self._collect_all_metrics(
    #         config1, config2, K, threshold, block_cath
    #     )
        
    #     # 步骤2: 执行统计检验
    #     print("步骤2/4: 执行统计检验...")
    #     t_test_results = self.statistical_analyzer.perform_paired_t_test(df, tag1, tag2)
        
    #     # 步骤3: 保存对比图像
    #     print("步骤3/4: 保存对比图像...")
    #     self.visualizer.save_comparison_figure(best_results, K, tag1, tag2, save_img_path_best, "Best")
    #     self.visualizer.save_comparison_figure(worst_results, K, tag1, tag2, save_img_path_worst, "Worst")
        
    #     # 步骤4: 保存Excel统计结果
    #     print("步骤4/4: 保存Excel统计结果...")
    #     self.saver.save_results_to_excel(df, t_test_results, tag1, tag2, save_excel_path)
        
    #     print("\n" + "=" * 60)
    #     print("单对比较分析完成!")
    #     print(f"✓ 最佳对比图像已保存: {save_img_path_best}")
    #     print(f"✓ 最差对比图像已保存: {save_img_path_worst}") 
    #     print(f"✓ 统计结果已保存: {save_excel_path}")
        
    #     # 打印结果摘要
    #     valid_best = [r for r in best_results if r.get('has_ground_truth', True)]
    #     valid_worst = [r for r in worst_results if r.get('has_ground_truth', True)]
        
    #     print(f"\n{tag1} 优于 {tag2} 的前 {min(K, len(valid_best))} 张图像:")
    #     for i, result in enumerate(valid_best):
    #         print(f"{i+1}. {result['patientID']}/{result['videoId']}/{result['frameId']} - "
    #               f"F1差异: {result['f1_diff']:.4f}")
        
    #     print(f"\n{tag2} 优于 {tag1} 的前 {min(K, len(valid_worst))} 张图像:")
    #     for i, result in enumerate(valid_worst):
    #         print(f"{i+1}. {result['patientID']}/{result['videoId']}/{result['frameId']} - "
    #               f"F1差异: {result['f1_diff']:.4f}")
        
    #     return df, t_test_results, best_results, worst_results
    
    # def analyze_multiple_configs(self, configs, K=5, threshold=0.85, save_path="", block_cath=False):
    #     """
    #     对多个配置进行两两比较分析
        
    #     参数:
    #         configs: 配置字典列表
    #         K: 每对比较中显示的图像数量
    #         threshold: 分割阈值
    #         save_path: 保存路径
    #         block_cath: 是否屏蔽导管区域
            
    #     返回:
    #         dict: 包含所有比较结果的字典
    #     """
    #     print("=" * 80)
    #     print("开始多配置综合比较分析")
    #     print(f"配置数量: {len(configs)}")
    #     print(f"比较模式: {'屏蔽导管区域' if block_cath else '不屏蔽导管区域'}")
    #     print("=" * 80)
        
    #     n_configs = len(configs)
    #     timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    #     mode_suffix = "_hasCath" if block_cath else "_nonCath"
        
    #     # 存储所有比较结果
    #     all_results = {}
        
    #     # 步骤1: 执行所有两两比较
    #     print("步骤1: 执行所有两两比较...")
    #     comparison_matrix = self._perform_all_pairwise_comparisons(
    #         configs, K, threshold, block_cath, all_results
    #     )
        
    #     # 步骤2: 生成比较矩阵图
    #     print("步骤2: 生成比较矩阵图...")
    #     matrix_img_path = f"{save_path}comparison_matrix{mode_suffix}_{timestamp}.jpg"
    #     self._create_comparison_matrix_plot(comparison_matrix, configs, matrix_img_path)
        
    #     # 步骤3: 生成详细比较结果Excel
    #     print("步骤3: 生成详细比较结果Excel...")
    #     excel_path = f"{save_path}multi_comparison_results{mode_suffix}_{timestamp}.xlsx"
    #     self._save_multi_comparison_excel(all_results, configs, excel_path)
        
    #     # 步骤4: 生成显著性差异汇总
    #     print("步骤4: 生成显著性差异汇总...")
    #     significance_path = f"{save_path}significance_summary{mode_suffix}_{timestamp}.jpg"
    #     self._create_significance_summary(comparison_matrix, configs, significance_path)
        
    #     print("\n" + "=" * 80)
    #     print("多配置比较分析完成!")
    #     print(f"✓ 比较矩阵图已保存: {matrix_img_path}")
    #     print(f"✓ 详细结果Excel已保存: {excel_path}")
    #     print(f"✓ 显著性差异汇总已保存: {significance_path}")
        
    #     return all_results, comparison_matrix
    
    # def _collect_all_metrics(self, config1, config2, K=5, threshold=0.85, block_cath=False):
    #     """
    #     收集所有图像的指标数据
        
    #     参数:
    #         config1: 第一种配置
    #         config2: 第二种配置
    #         K: 返回的最佳/最差结果数量
    #         threshold: 分割阈值
    #         block_cath: 是否屏蔽导管区域
        
    #     返回:
    #         tuple: (所有结果的DataFrame, 最佳结果列表, 最差结果列表)
    #     """
    #     # 检查哪些配置需要模型
    #     need_model1 = not config1.get("precomputed", False)
    #     need_model2 = not config2.get("precomputed", False)
        
    #     # 初始化模型（如果需要）
    #     if need_model1 or need_model2:
    #         param_path = "../DeNVeR_in/models_config/freecos_Seg.pt"
    #         model = self.model_manager.init_model(param_path)
    #     else:
    #         model = None
        
    #     # 获取配置信息
    #     tag1 = config1["name"]
    #     tag2 = config2["name"]
        
    #     # 获取所有患者
    #     patient_names = self._get_patient_names()
        
    #     # 计算总图像数量
    #     total_images = self._count_annotated_images(patient_names)
    #     print(f"总图像数量: {total_images}")
        
    #     # 存储结果
    #     all_results = []
    #     comparison_results = []
        
    #     # 处理所有图像
    #     with tqdm(total=total_images, desc="处理图像") as progress_bar:
    #         for patient_id in patient_names:
    #             patient_results = self._process_patient_metrics(
    #                 patient_id, config1, config2, tag1, tag2, model, 
    #                 threshold, block_cath, progress_bar
    #             )
    #             all_results.extend(patient_results["all_results"])
    #             comparison_results.extend(patient_results["comparison_results"])
        
    #     # 转换为DataFrame
    #     df = pd.DataFrame(all_results)
        
    #     # 提取有效结果用于对比
    #     valid_results = [r for r in comparison_results if r.get('has_ground_truth', True)]
        
    #     # 找出最佳和最差结果
    #     best_results = sorted(valid_results, key=lambda x: x['f1_diff'], reverse=True)[:K]
    #     worst_results = sorted(valid_results, key=lambda x: x['f1_diff'])[:K]
        
    #     return df, best_results, worst_results
    
    def _get_patient_names(self):
        """获取所有患者名称"""
        return [name for name in os.listdir(self.config.dataset_path) 
                if os.path.isdir(os.path.join(self.config.dataset_path, name))]
    
    def _count_annotated_images(self, patient_names):
        """计算有标注的图像总数"""
        total_count = 0
        for patient_id in patient_names:
            patient_gt_path = os.path.join(self.config.dataset_path_gt, patient_id, "ground_truth")
            if not os.path.exists(patient_gt_path):
                continue
                
            video_names = [name for name in os.listdir(patient_gt_path) 
                          if os.path.isdir(os.path.join(patient_gt_path, name))]
            
            for video_id in video_names:
                if "CATH" not in video_id:  # 只统计非导管视频
                    video_path = os.path.join(patient_gt_path, video_id)
                    total_count += len(os.listdir(video_path))
        
        return total_count
    
    # def _process_patient_metrics(self, patient_id, config1, config2, tag1, tag2, 
    #                            model, threshold, block_cath, progress_bar):
    #     """处理单个患者的所有图像指标"""
    #     patient_gt_path = os.path.join(self.config.dataset_path_gt, patient_id, "ground_truth")
    #     if not os.path.exists(patient_gt_path):
    #         return {"all_results": [], "comparison_results": []}
        
    #     video_names = [name for name in os.listdir(patient_gt_path) 
    #                   if os.path.isdir(os.path.join(patient_gt_path, name))]
        
    #     all_results = []
    #     comparison_results = []
        
    #     for video_id in video_names:
    #         if "CATH" not in video_id:  # 只处理非导管视频
    #             video_results = self._process_video_metrics(
    #                 patient_id, video_id, config1, config2, tag1, tag2,
    #                 model, threshold, block_cath, progress_bar
    #             )
    #             all_results.extend(video_results["all_results"])
    #             comparison_results.extend(video_results["comparison_results"])
        
    #     return {"all_results": all_results, "comparison_results": comparison_results}
    
    # def _process_video_metrics(self, patient_id, video_id, config1, config2, tag1, tag2,
    #                          model, threshold, block_cath, progress_bar):
    #     """处理单个视频的所有帧指标"""
    #     video_gt_path = os.path.join(self.config.dataset_path_gt, patient_id, "ground_truth", video_id)
    #     if not os.path.exists(video_gt_path):
    #         return {"all_results": [], "comparison_results": []}
        
    #     # 计算归一化参数（如果需要）
    #     mean_tag1, std_tag1 = self._get_normalization_params(config1, patient_id, video_id)
    #     mean_tag2, std_tag2 = self._get_normalization_params(config2, patient_id, video_id)
        
    #     all_results = []
    #     comparison_results = []
    #     frame_ids = os.listdir(video_gt_path)
        
    #     progress_bar.set_postfix(patient=patient_id, video=video_id, mode="annotated")
        
    #     for frame_id in frame_ids:
    #         try:
    #             result = self._process_single_image(
    #                 patient_id, video_id, frame_id, config1, config2, tag1, tag2,
    #                 mean_tag1, std_tag1, mean_tag2, std_tag2, model, threshold, block_cath
    #             )
    #             all_results.append(result["all_result"])
    #             if result.get("comparison_result"):
    #                 comparison_results.append(result["comparison_result"])
                    
    #         except Exception as e:
    #             print(f"处理错误 {patient_id}/{video_id}/{frame_id}: {str(e)}")
            
    #         progress_bar.update(1)
        
    #     return {"all_results": all_results, "comparison_results": comparison_results}
    
    def _get_normalization_params(self, config, patient_id, video_id):
        """获取归一化参数"""
        if config.get("precomputed", False):
            return 0, 1  # 预计算方法不需要归一化
        else:
            # 使用配置中指定的归一化方法
            norm_method = config["norm_method"]
            return norm_method(config["input_mode"], self.model_manager.transform, patient_id, video_id)
    
    def _process_single_image(self, patient_id, video_id, frame_id, config1, config2, 
                            tag1, tag2, mean_tag1, std_tag1, mean_tag2, std_tag2,
                            model, threshold, block_cath):
        """处理单张图像的指标计算"""
        # 获取输入图像
        img_tag1 = self._get_input_image(config1, patient_id, video_id, frame_id)
        img_tag2 = self._get_input_image(config2, patient_id, video_id, frame_id)
        
        # 归一化处理
        img_tag1_norm = self._normalize_image(config1, img_tag1, mean_tag1, std_tag1)
        img_tag2_norm = self._normalize_image(config2, img_tag2, mean_tag2, std_tag2)
        
        # 获取预测结果
        pred_tag1 = self._get_prediction(config1, model, img_tag1_norm, threshold, patient_id, video_id, frame_id)
        pred_tag2 = self._get_prediction(config2, model, img_tag2_norm, threshold, patient_id, video_id, frame_id)
        
        # 获取图像路径
        tag1_img_path = self._get_image_path(config1, patient_id, video_id, frame_id)
        tag2_img_path = self._get_image_path(config2, patient_id, video_id, frame_id)
        
        # 基本结果
        all_result = {
            'patientID': patient_id,
            'videoId': video_id,
            'frameId': frame_id,
            'image_id': f"{patient_id}_{video_id}_{frame_id}",
            'tag1_image_path': tag1_img_path,
            'tag2_image_path': tag2_img_path,
            'has_ground_truth': True
        }
        
        # 获取ground truth并处理导管区域
        gt_tensor = self._load_and_process_ground_truth(patient_id, video_id, frame_id, block_cath)
        
        # 处理导管区域对预测的影响
        if block_cath:
            pred_tag1, pred_tag2 = self._apply_catheter_mask(pred_tag1, pred_tag2, gt_tensor)
        
        # 计算指标
        ind_tag1 = getIndicators(
            pred_tag1[0, 0].detach().cpu() * 255,
            gt_tensor[0, 0].detach().cpu() * 255
        )
        ind_tag2 = getIndicators(
            pred_tag2[0, 0].detach().cpu() * 255,
            gt_tensor[0, 0].detach().cpu() * 255
        )
        
        # 处理指标结果
        f1_tag1, f1_tag2 = self._process_metrics(ind_tag1["f1"], ind_tag2["f1"])
        f1_diff = float(f1_tag1 - f1_tag2)
        
        # 完善结果
        all_result.update(self._build_result_dict(tag1, tag2, ind_tag1, ind_tag2, f1_diff, gt_tensor))
        
        # 构建对比结果
        comparison_result = self._build_comparison_result(
            patient_id, video_id, frame_id, tag1, tag2, f1_tag1, f1_tag2, f1_diff,
            img_tag1, img_tag2, pred_tag1, pred_tag2, gt_tensor, tag1_img_path, tag2_img_path
        )
        
        return {
            "all_result": all_result,
            "comparison_result": comparison_result
        }
    
    def _get_input_image(self, config, patient_id, video_id, frame_id):
        """根据配置获取输入图像"""
        if config.get("precomputed", False):
            # 预计算方法使用显示模式
            input_mode = config.get("input_mode_for_display", "orig")
        else:
            # 模型推理方法使用自己的输入模式
            input_mode = config["input_mode"]
        
        return self.image_loader.load_image(input_mode, patient_id, video_id, frame_id)
    
    def _normalize_image(self, config, img, mean, std):
        """根据配置对图像进行归一化"""
        if config.get("precomputed", False):
            return img  # 预计算方法不需要归一化
        else:
            return (img - mean) / std  # 模型推理方法需要归一化
    
    # def _get_prediction(self, config, model, img_norm, threshold, patient_id, video_id, frame_id):
    #     """获取预测结果"""
    #     if config.get("precomputed", False):
    #         # 读取预计算的分割结果
    #         result_path_template = config["result_path_template"]
    #         pred_path = result_path_template.format(
    #             patientID=patient_id, 
    #             videoId=video_id, 
    #             frameId=frame_id
    #         )
            
    #         if os.path.exists(pred_path):
    #             pred = Image.open(pred_path).convert('L')
    #             pred_tensor = self.model_manager.transform(pred).unsqueeze(0).cuda()
    #             if config.get("binarize", True):
    #                 pred_tensor[pred_tensor > threshold] = 1
    #                 pred_tensor[pred_tensor <= threshold] = 0
    #             return pred_tensor
    #         else:
    #             raise FileNotFoundError(f"预计算预测结果不存在: {pred_path}")
    #     else:
    #         # 使用模型进行推理
    #         pred = model(img_norm)["pred"]
    #         if config.get("binarize", True):
    #             pred[pred > threshold] = 1
    #             pred[pred <= threshold] = 0
    #         return pred
        
    def _get_prediction2(self, config, model, img_norm, threshold, patient_id, video_id, frame_id):
        """获取预测结果"""
        def s(pred):
            from torchvision.utils import save_image
            if False: path_save = os.path.join(self.config.root_path, "outputs", video_id, config['name'])
            path_save = os.path.join(self.config.root_path, "outputs", config['name'], video_id)
            os.makedirs(path_save, exist_ok=True)
            # print("存储路径为：",os.path.join(path_save,frame_id))
            # print("pred",type(pred))
            save_image(pred, os.path.join(path_save,frame_id))

            # if config.get("binarize", True):
            #     pred[pred > threshold] = 1
            #     pred[pred <= threshold] = 0
            # # gt_tensor = self._load_and_process_ground_truth(patient_id, video_id, frame_id, block_cath)
            # # gt = gt_tensor[0, 0].detach().cpu().numpy()
            # # f1 = self._calculate_f1_score(pred, gt)
            # return pred
            # pred.save(os.path.join(path_save,frame_id))
        
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
                # if config.get("binarize", True):
                #     pred_tensor[pred_tensor > threshold] = 1
                #     pred_tensor[pred_tensor <= threshold] = 0
                # return pred_tensor
            else:
                raise FileNotFoundError(f"预计算预测结果不存在: {pred_path}")
        else:
            # 使用模型进行推理
            pred = model(img_norm)["pred"]
            s(pred)
            # if config.get("binarize", True):
            #     pred[pred > threshold] = 1
            #     pred[pred <= threshold] = 0
            # return pred
    
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
    
    def _load_and_process_ground_truth(self, patient_id, video_id, frame_id, block_cath):
        """加载并处理ground truth"""
        if block_cath:
            gt_path = os.path.join(self.config.dataset_path_gt, patient_id, "ground_truth", video_id + "CATH", frame_id)
        else:
            gt_path = os.path.join(self.config.dataset_path_gt, patient_id, "ground_truth", video_id, frame_id)
        
        gt = Image.open(gt_path).convert('L')
        gt_tensor = self.model_manager.transform(gt).unsqueeze(0).cuda()
        
        if block_cath:
            # 处理导管区域：分离血管和导管掩码
            mask_cath = torch.zeros_like(gt_tensor)
            mask_vessel = torch.zeros_like(gt_tensor)
            mask_cath[gt_tensor > 0.75] = 1  # 导管区域
            mask_vessel[(gt_tensor > 0.25) & (gt_tensor < 0.75)] = 1  # 血管区域
            gt_tensor = mask_vessel  # 只保留血管区域作为ground truth
        else:
            # 简单二值化
            gt_tensor[gt_tensor > 0.5] = 1
            gt_tensor[gt_tensor <= 0.5] = 0
        
        return gt_tensor
    
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
    
    def _process_metrics(self, metric1, metric2):
        """处理指标值，确保为标量"""
        if hasattr(metric1, 'dim') and not metric1.dim() == 0:
            metric1 = metric1[0]
        if hasattr(metric2, 'dim') and not metric2.dim() == 0:
            metric2 = metric2[0]
        
        return metric1, metric2
    
    def _build_result_dict(self, tag1, tag2, ind_tag1, ind_tag2, f1_diff, gt_tensor):
        """构建结果字典"""
        result = {
            'ground_truth_path': "processed",  # 实际路径在单独字段中
            'f1_diff': f1_diff
        }
        
        # 添加tag1的指标
        for metric_name, metric_value in ind_tag1.items():
            processed_value = metric_value
            if hasattr(metric_value, 'dim') and not metric_value.dim() == 0:
                processed_value = metric_value[0]
            result[f'{tag1}_{metric_name}'] = float(processed_value)
        
        # 添加tag2的指标
        for metric_name, metric_value in ind_tag2.items():
            processed_value = metric_value
            if hasattr(metric_value, 'dim') and not metric_value.dim() == 0:
                processed_value = metric_value[0]
            result[f'{tag2}_{metric_name}'] = float(processed_value)
        
        return result
    
    def _build_comparison_result(self, patient_id, video_id, frame_id, tag1, tag2, 
                               f1_tag1, f1_tag2, f1_diff, img_tag1, img_tag2, 
                               pred_tag1, pred_tag2, gt_tensor, tag1_img_path, tag2_img_path):
        """构建对比结果字典"""
        return {
            'patientID': patient_id,
            'videoId': video_id,
            'frameId': frame_id,
            'f1_tag1': float(f1_tag1),
            'f1_tag2': float(f1_tag2),
            'f1_diff': f1_diff,
            'img_tag1': img_tag1.detach().cpu().numpy()[0, 0],
            'img_tag2': img_tag2.detach().cpu().numpy()[0, 0],
            'pred_tag1': pred_tag1.detach().cpu().numpy()[0, 0],
            'pred_tag2': pred_tag2.detach().cpu().numpy()[0, 0],
            'gt': gt_tensor.detach().cpu().numpy()[0, 0],
            'tag1_image_path': tag1_img_path,
            'tag2_image_path': tag2_img_path,
            'has_ground_truth': True
        }
    
    def _perform_all_pairwise_comparisons(self, configs, K, threshold, block_cath, all_results):
        """
        执行所有两两比较
        
        返回:
            dict: 比较矩阵结果
        """
        n_configs = len(configs)
        comparison_matrix = {
            'f1_differences': np.zeros((n_configs, n_configs)),
            'p_values': np.zeros((n_configs, n_configs)),
            'significance': np.zeros((n_configs, n_configs)),
            'best_images': [[None for _ in range(n_configs)] for _ in range(n_configs)],
            'worst_images': [[None for _ in range(n_configs)] for _ in range(n_configs)]
        }
        
        # 创建进度条
        total_comparisons = n_configs * (n_configs - 1) // 2
        with tqdm(total=total_comparisons, desc="两两比较") as pbar:
            for i in range(n_configs):
                for j in range(i + 1, n_configs):
                    config1 = configs[i]
                    config2 = configs[j]
                    tag1 = config1["name"]
                    tag2 = config2["name"]
                    
                    pbar.set_postfix(comparison=f"{tag1} vs {tag2}")
                    
                    try:
                        # 使用analyze_pair方法执行单次比较
                        df, t_test_results, best_results, worst_results = self.analyze_pair(
                            config1, config2, K, threshold, "", block_cath
                        )
                        
                        # 存储详细结果
                        comparison_key = f"{tag1}_vs_{tag2}"
                        all_results[comparison_key] = {
                            'df': df,
                            't_test_results': t_test_results,
                            'best_results': best_results,
                            'worst_results': worst_results,
                            'config1': config1,
                            'config2': config2
                        }
                        
                        # 更新比较矩阵
                        self._update_comparison_matrix(
                            comparison_matrix, i, j, tag1, tag2, 
                            t_test_results, best_results, worst_results, df
                        )
                        
                    except Exception as e:
                        print(f"比较 {tag1} vs {tag2} 时出错: {str(e)}")
                        # 在矩阵中标记错误
                        comparison_matrix['f1_differences'][i, j] = np.nan
                        comparison_matrix['p_values'][i, j] = np.nan
                        comparison_matrix['significance'][i, j] = -1
                    
                    pbar.update(1)
        
        return comparison_matrix
    
    def _update_comparison_matrix(self, matrix, i, j, tag1, tag2, t_test_results, best_results, worst_results, df):
        """更新比较矩阵"""
        if 'f1' in t_test_results:
            f1_result = t_test_results['f1']
            
            # F1均值差异 (tag2 - tag1)
            matrix['f1_differences'][i, j] = f1_result['mean_difference']
            matrix['f1_differences'][j, i] = -f1_result['mean_difference']  # 对称位置取负值
            
            # p值
            matrix['p_values'][i, j] = f1_result['p_value']
            matrix['p_values'][j, i] = f1_result['p_value']
            
            # 显著性水平 (1: p<0.001, 2: p<0.01, 3: p<0.05, 0: 不显著)
            p_val = f1_result['p_value']
            if p_val < 0.001:
                significance_level = 3
            elif p_val < 0.01:
                significance_level = 2
            elif p_val < 0.05:
                significance_level = 1
            else:
                significance_level = 0
                
            matrix['significance'][i, j] = significance_level
            matrix['significance'][j, i] = significance_level
            
            # 存储最佳和最差图像信息
            if best_results:
                matrix['best_images'][i][j] = {
                    'images': best_results,
                    'max_f1_diff': max([r['f1_diff'] for r in best_results]) if best_results else 0
                }
            
            if worst_results:
                matrix['worst_images'][i][j] = {
                    'images': worst_results,
                    'min_f1_diff': min([r['f1_diff'] for r in worst_results]) if worst_results else 0
                }
    
    def _create_comparison_matrix_plot(self, comparison_matrix, configs, save_path):
        """创建比较矩阵图"""
        n_configs = len(configs)
        config_names = [config["name"] for config in configs]
        
        # 创建图形
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))
        
        # 子图1: F1差异矩阵
        im1 = ax1.imshow(comparison_matrix['f1_differences'], cmap='RdBu_r', 
                        vmin=-0.1, vmax=0.1, aspect='equal')
        ax1.set_title('F1 Score 差异矩阵\n(正值表示列方法优于行方法)', fontsize=14, pad=20)
        
        # 设置坐标轴
        ax1.set_xticks(range(n_configs))
        ax1.set_yticks(range(n_configs))
        ax1.set_xticklabels(config_names, rotation=45, ha='right')
        ax1.set_yticklabels(config_names)
        
        # 添加数值和显著性标记
        for i in range(n_configs):
            for j in range(n_configs):
                if i != j and not np.isnan(comparison_matrix['f1_differences'][i, j]):
                    # 显示F1差异值
                    value = comparison_matrix['f1_differences'][i, j]
                    ax1.text(j, i, f'{value:.3f}', 
                            ha='center', va='center', fontsize=9,
                            color='white' if abs(value) > 0.05 else 'black')
                    
                    # 添加显著性标记
                    significance = comparison_matrix['significance'][i, j]
                    if significance > 0:
                        stars = str(significance)+"*" #stars = '*' * significance
                        ax1.text(j, i + 0.3, stars, 
                                ha='center', va='center', fontsize=12, 
                                color='yellow', fontweight='bold')
        
        # 添加颜色条
        plt.colorbar(im1, ax=ax1, shrink=0.8)
        
        # 子图2: 显著性矩阵
        significance_cmap = plt.cm.get_cmap('YlOrRd', 4)
        im2 = ax2.imshow(comparison_matrix['significance'], cmap=significance_cmap, 
                        vmin=-0.5, vmax=3.5, aspect='equal')
        ax2.set_title('统计显著性矩阵\n(星号数量表示显著性水平)', fontsize=14, pad=20)
        
        # 设置坐标轴
        ax2.set_xticks(range(n_configs))
        ax2.set_yticks(range(n_configs))
        ax2.set_xticklabels(config_names, rotation=45, ha='right')
        ax2.set_yticklabels(config_names)
        
        # 添加显著性说明
        for i in range(n_configs):
            for j in range(n_configs):
                if i != j:
                    significance = comparison_matrix['significance'][i, j]
                    if significance >= 0:
                        stars = '*' * int(significance)
                        color = 'black' if significance == 0 else 'white'
                        ax2.text(j, i, stars, 
                                ha='center', va='center', fontsize=14, 
                                color=color, fontweight='bold')
        
        # 添加颜色条和标签
        cbar = plt.colorbar(im2, ax=ax2, shrink=0.8, ticks=[0, 1, 2, 3])
        cbar.ax.set_yticklabels(['不显著', 'p<0.05', 'p<0.01', 'p<0.001'])
        
        # 调整布局
        plt.tight_layout()
        
        # 保存图像
        plt.savefig(save_path, dpi=300, bbox_inches='tight', format='jpg')
        plt.close(fig)
        print(f"✓ 比较矩阵图已保存: {save_path}")
    
    def _save_multi_comparison_excel(self, all_results, configs, save_path):
        """保存多比较结果到Excel"""
        print(f"保存多比较结果到Excel: {save_path}")
        
        with pd.ExcelWriter(save_path, engine='openpyxl') as writer:
            # 1. 汇总表
            self._create_summary_sheet(writer, all_results, configs)
            
            # 2. 各对比较的详细结果
            for comparison_key, results in all_results.items():
                self._create_comparison_detail_sheets(writer, comparison_key, results)
            
            # 3. 最佳/最差图像汇总
            self._create_extreme_images_sheet(writer, all_results)
            
            # 4. 配置信息表
            self._create_config_info_sheet(writer, configs)
        
        print(f"✓ 多比较结果Excel已保存: {save_path}")
    
    def _create_summary_sheet(self, writer, all_results, configs):
        """创建汇总表"""
        summary_data = []
        config_names = [config["name"] for config in configs]
        
        for comparison_key, results in all_results.items():
            tag1, tag2 = comparison_key.split('_vs_')
            
            if 'f1' in results['t_test_results']:
                f1_result = results['t_test_results']['f1']
                
                summary_data.append({
                    '比较对': comparison_key,
                    f'{tag1} F1均值': f1_result[f'mean_{tag1}'],
                    f'{tag2} F1均值': f1_result[f'mean_{tag2}'],
                    'F1均值差异': f1_result['mean_difference'],
                    'p值': f1_result['p_value'],
                    '显著性': f1_result['significance_text'],
                    '样本量': f1_result['sample_size'],
                    f'{tag1}优于{tag2}的图像数': len([r for r in results['best_results'] if r['f1_diff'] > 0]),
                    f'{tag2}优于{tag1}的图像数': len([r for r in results['worst_results'] if r['f1_diff'] < 0])
                })
        
        if summary_data:
            summary_df = pd.DataFrame(summary_data)
            summary_df.to_excel(writer, sheet_name='比较汇总', index=False)
    
    def _create_comparison_detail_sheets(self, writer, comparison_key, results):
        """创建各对比较的详细工作表"""
        # 原始数据
        results['df'].to_excel(writer, sheet_name=f'{comparison_key}_数据', index=False)
        
        # 统计摘要
        summary_data = []
        for metric, result in results['t_test_results'].items():
            summary_data.append({
                '指标': metric.upper(),
                f"{results['config1']['name']}均值": result[f"mean_{results['config1']['name']}"],
                f"{results['config2']['name']}均值": result[f"mean_{results['config2']['name']}"],
                '均值差异': result['mean_difference'],
                't统计量': result['t_statistic'],
                'p值': result['p_value'],
                '显著性': result['significance_text'],
                '样本量': result['sample_size']
            })
        
        summary_df = pd.DataFrame(summary_data)
        summary_df.to_excel(writer, sheet_name=f'{comparison_key}_统计', index=False)
    
    def _create_extreme_images_sheet(self, writer, all_results):
        """创建最佳/最差图像汇总表"""
        extreme_data = []
        
        for comparison_key, results in all_results.items():
            tag1, tag2 = comparison_key.split('_vs_')
            
            # 最佳图像 (tag1优于tag2)
            for i, img_result in enumerate(results['best_results'][:5]):  # 前5个
                extreme_data.append({
                    '比较对': comparison_key,
                    '类型': f'{tag1}优于{tag2}',
                    '排名': i + 1,
                    '患者ID': img_result['patientID'],
                    '视频ID': img_result['videoId'],
                    '帧ID': img_result['frameId'],
                    f'{tag1} F1': img_result['f1_tag1'],
                    f'{tag2} F1': img_result['f1_tag2'],
                    'F1差异': img_result['f1_diff'],
                    f'{tag1}图像路径': img_result['tag1_image_path'],
                    f'{tag2}图像路径': img_result['tag2_image_path']
                })
            
            # 最差图像 (tag2优于tag1)
            for i, img_result in enumerate(results['worst_results'][:5]):  # 前5个
                extreme_data.append({
                    '比较对': comparison_key,
                    '类型': f'{tag2}优于{tag1}',
                    '排名': i + 1,
                    '患者ID': img_result['patientID'],
                    '视频ID': img_result['videoId'],
                    '帧ID': img_result['frameId'],
                    f'{tag1} F1': img_result['f1_tag1'],
                    f'{tag2} F1': img_result['f1_tag2'],
                    'F1差异': img_result['f1_diff'],
                    f'{tag1}图像路径': img_result['tag1_image_path'],
                    f'{tag2}图像路径': img_result['tag2_image_path']
                })
        
        if extreme_data:
            extreme_df = pd.DataFrame(extreme_data)
            extreme_df.to_excel(writer, sheet_name='极端情况图像', index=False)
    
    def _create_config_info_sheet(self, writer, configs):
        """创建配置信息表"""
        config_data = []
        
        for i, config in enumerate(configs):
            config_data.append({
                '配置名称': config["name"],
                '预计算': config.get("precomputed", False),
                '输入模式': config.get("input_mode", config.get("input_mode_for_display", "N/A")),
                '二值化': config.get("binarize", True),
                '归一化方法': config["norm_method"].__name__ if not config.get("precomputed", False) else "N/A",
                '结果路径模板': config.get("result_path_template", "N/A")
            })
        
        config_df = pd.DataFrame(config_data)
        config_df.to_excel(writer, sheet_name='配置信息', index=False)
    
    def _create_significance_summary(self, comparison_matrix, configs, save_path):
        """创建显著性差异汇总图"""
        n_configs = len(configs)
        config_names = [config["name"] for config in configs]
        
        # 计算每个配置的"胜率"（显著优于其他配置的次数）
        win_counts = np.zeros(n_configs)
        for i in range(n_configs):
            for j in range(n_configs):
                if i != j and comparison_matrix['significance'][i, j] > 0:
                    if comparison_matrix['f1_differences'][i, j] > 0:  # j优于i
                        win_counts[j] += 1
                    else:  # i优于j
                        win_counts[i] += 1
        
        # 创建图形
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # 子图1: 胜率条形图
        colors = plt.cm.viridis(win_counts / max(win_counts) if max(win_counts) > 0 else 0.5)
        bars = ax1.bar(range(n_configs), win_counts, color=colors)
        ax1.set_title('各配置显著优于其他配置的次数', fontsize=14, pad=20)
        ax1.set_xlabel('配置')
        ax1.set_ylabel('显著优胜次数')
        ax1.set_xticks(range(n_configs))
        ax1.set_xticklabels(config_names, rotation=45, ha='right')
        
        # 在条形上添加数值
        for bar, count in zip(bars, win_counts):
            ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1,
                    f'{int(count)}', ha='center', va='bottom')
        
        # 子图2: 平均F1差异热力图
        avg_differences = np.zeros(n_configs)
        for i in range(n_configs):
            valid_differences = []
            for j in range(n_configs):
                if i != j and not np.isnan(comparison_matrix['f1_differences'][i, j]):
                    valid_differences.append(comparison_matrix['f1_differences'][i, j])
            avg_differences[i] = np.mean(valid_differences) if valid_differences else 0
        
        im = ax2.imshow([avg_differences], cmap='RdBu_r', aspect='auto',
                       vmin=-0.05, vmax=0.05)
        ax2.set_title('各配置相对于其他配置的平均F1差异', fontsize=14, pad=20)
        ax2.set_yticks([0])
        ax2.set_yticklabels(['平均差异'])
        ax2.set_xticks(range(n_configs))
        ax2.set_xticklabels(config_names, rotation=45, ha='right')
        
        # 添加数值
        for i, diff in enumerate(avg_differences):
            ax2.text(i, 0, f'{diff:.4f}', ha='center', va='center', 
                    color='white' if abs(diff) > 0.025 else 'black', fontweight='bold')
        
        # 添加颜色条
        plt.colorbar(im, ax=ax2, shrink=0.8)
        
        # 调整布局
        plt.tight_layout()
        
        # 保存图像
        plt.savefig(save_path, dpi=300, bbox_inches='tight', format='jpg')
        plt.close(fig)
        print(f"✓ 显著性差异汇总图已保存: {save_path}")


# class ComprehensiveMulComparison:
#     """生成彩色图片"""
    
    def __init__(self, config, model_manager, image_loader, norm_calculator):
        self.config = config
        self.model_manager = model_manager
        self.image_loader = image_loader
        self.norm_calculator = norm_calculator
        self.statistical_analyzer = StatisticalAnalyzer()
        self.visualizer = ResultVisualizer()
        self.saver = ResultSaver()
    
    # ... 之前的其他方法保持不变 ...
    
    def inference(self, configs, save_path="", block_cath=False, threshold=0.85):    

        """
        收集所有配置的预测结果
        
        参数:
            configs: 配置列表
            threshold: 分割阈值
            block_cath: 是否屏蔽导管区域
            
        返回:
            dict: 包含所有配置预测结果的字典
        """
        # 步骤1: 收集所有配置的预测结果
        print("步骤1: 收集所有配置的预测结果...")
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
        
        # 计算总图像数量
        total_images = self._count_annotated_images(patient_names)
        print(f"总图像数量: {total_images}")
        
        # 存储所有配置的预测结果
        all_predictions = {config["name"]: {} for config in configs}
        ground_truths = {}
        
        # 处理所有图像
        with tqdm(total=total_images, desc="收集预测结果") as progress_bar:
            for patient_id in patient_names:
                patient_results = self._process_patient_predictions(
                    patient_id, configs, model, threshold, block_cath, progress_bar
                )
                
                # # 合并结果
                # for config_name, predictions in patient_results.items():
                #     if config_name != "ground_truth":
                #         all_predictions[config_name].update(predictions)
                # ground_truths.update(patient_results.get("ground_truth", {}))

                # break#由于快速测试，后续要删除

        # # 步骤2: 生成比较矩阵图
        # print("步骤2: 计算每张图像的F1分数...")
        # config_names = [config["name"] for config in configs]
        
        # # 计算每张图像的F1分数
        # print("计算每张图像的F1分数...")

        # # 获取所有图像ID
        # image_ids = list(all_predictions["ground_truth"].keys())
        # f1_scores = {}
        # for config_name in config_names:
        #     f1_scores[config_name] = {}

        #     for image_id in tqdm(image_ids, desc=f"计算 {config_name} 的F1"):
        #         if image_id in all_predictions[config_name] and image_id in all_predictions["ground_truth"]:
        #             pred = all_predictions[config_name][image_id]
        #             ###
        #             # # 获取ground truth
        #             # gt_tensor = self._load_and_process_ground_truth(patient_id, video_id, frame_id, block_cath)
        #             # gt = gt_tensor[0, 0].detach().cpu().numpy()
        #             ###
        #             gt = all_predictions["ground_truth"][image_id]
                    
        #             # 计算F1分数
        #             f1 = self._calculate_f1_score(pred, gt)
        #             f1_scores[config_name][image_id] = f1
        # print("计算完成")
        
    def _process_patient_predictions(self, patient_id, configs, model, threshold, block_cath, progress_bar):
        """处理单个患者的所有配置预测结果"""
        patient_gt_path = os.path.join(self.config.dataset_path_gt, patient_id, "ground_truth")
        if not os.path.exists(patient_gt_path):
            return {config["name"]: {} for config in configs}
        
        video_names = [name for name in os.listdir(patient_gt_path) 
                      if os.path.isdir(os.path.join(patient_gt_path, name))]
        
        all_predictions = {config["name"]: {} for config in configs}
        ground_truths = {}
        
        for video_id in video_names:
            if "CATH" not in video_id:  # 只处理非导管视频
                video_results = self._process_video_predictions(
                    patient_id, video_id, configs, model, threshold, block_cath, progress_bar
                )
                
        #         # 合并结果
        #         for config_name, predictions in video_results.items():
        #             if config_name != "ground_truth":
        #                 all_predictions[config_name].update(predictions)
        #         ground_truths.update(video_results.get("ground_truth", {}))
        
        # # 添加ground truth到结果中
        # all_predictions["ground_truth"] = ground_truths

        # #########################################
        
        # return all_predictions
    
    # def _process_video_predictions(self, patient_id, video_id, configs, model, threshold, block_cath, progress_bar):
    #     """处理单个视频的所有配置预测结果"""
    #     video_gt_path = os.path.join(self.config.dataset_path_gt, patient_id, "ground_truth", video_id)
    #     if not os.path.exists(video_gt_path):
    #         return {config["name"]: {} for config in configs}
        
    #     # 计算归一化参数（如果需要）
    #     normalization_params = {}
    #     for config in configs:
    #         if not config.get("precomputed", False):
    #             mean, std = self._get_normalization_params(config, patient_id, video_id)
    #             normalization_params[config["name"]] = (mean, std)
        
    #     # all_predictions = {config["name"]: {} for config in configs}
    #     # ground_truths = {}
    #     frame_ids = os.listdir(video_gt_path)
        
    #     progress_bar.set_postfix(patient=patient_id, video=video_id)
        
    #     for frame_id in frame_ids:
    #         try:
    #             frame_predictions = self._process_single_image_predictions(
    #                 patient_id, video_id, frame_id, configs, model, threshold, 
    #                 block_cath, normalization_params
    #             )
                
    #             # # 合并结果
    #             # for config_name, prediction in frame_predictions.items():
    #             #     if config_name != "ground_truth":
    #             #         image_id = f"{patient_id}_{video_id}_{frame_id}"
    #             #         all_predictions[config_name][image_id] = prediction
                
    #             # 保存ground truth
    #             # if "ground_truth" in frame_predictions:
    #             #     image_id = f"{patient_id}_{video_id}_{frame_id}"
    #             #     ground_truths[image_id] = frame_predictions["ground_truth"]
                    
    #         except Exception as e:
    #             print(f"处理错误 {patient_id}/{video_id}/{frame_id}: {str(e)}")
            
    #         progress_bar.update(1)
        
    #     # 添加ground truth到结果中
    #     # all_predictions["ground_truth"] = ground_truths
        
    #     # return all_predictions
    
    def _process_video_predictions(self, patient_id, video_id, configs, model, threshold, block_cath, progress_bar):
        """处理单个视频的所有配置预测结果"""
        video_gt_path = os.path.join(self.config.dataset_path_gt, patient_id, "ground_truth", video_id)
        if not os.path.exists(video_gt_path):
            return {config["name"]: {} for config in configs}
        
        # 计算归一化参数（如果需要）
        normalization_params = {}
        for config in configs:
            if not config.get("precomputed", False):
                mean, std = self._get_normalization_params(config, patient_id, video_id)
                normalization_params[config["name"]] = (mean, std)
        
        all_predictions = {config["name"]: {} for config in configs}
        ground_truths = {}
        frame_ids = os.listdir(video_gt_path)
        
        progress_bar.set_postfix(patient=patient_id, video=video_id)
        
        for frame_id in frame_ids:
            # try:
                frame_predictions = self._process_single_image_predictions(
                    patient_id, video_id, frame_id, configs, model, threshold, 
                    block_cath, normalization_params
                )
                
        #         # 合并结果
        #         for config_name, prediction in frame_predictions.items():
        #             if config_name != "ground_truth":
        #                 image_id = f"{patient_id}_{video_id}_{frame_id}"
        #                 all_predictions[config_name][image_id] = prediction
                
        #         # 保存ground truth
        #         if "ground_truth" in frame_predictions:
        #             image_id = f"{patient_id}_{video_id}_{frame_id}"
        #             ground_truths[image_id] = frame_predictions["ground_truth"]
                    
        #     except Exception as e:
        #         print(f"处理错误 {patient_id}/{video_id}/{frame_id}: {str(e)}")
            
        #     progress_bar.update(1)
                progress_bar.update(1)
        
        # # 添加ground truth到结果中
        # all_predictions["ground_truth"] = ground_truths
        
        # return all_predictions

    def _process_single_image_predictions(self, patient_id, video_id, frame_id, configs, 
                                        model, threshold, block_cath, normalization_params):
        """处理单张图像的所有配置预测结果"""
        frame_predictions = {}
        
        # 获取ground truth
        gt_tensor = self._load_and_process_ground_truth(patient_id, video_id, frame_id, block_cath)
        gt_numpy = gt_tensor[0, 0].detach().cpu().numpy()
        frame_predictions["ground_truth"] = gt_numpy
        
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
            pred = self._get_prediction2(config, model, img_norm, threshold, patient_id, video_id, frame_id)
            
        #     # 处理导管区域对预测的影响
        #     if block_cath:
        #         pred, _ = self._apply_catheter_mask(pred, pred, gt_tensor)  # 第二个参数不使用
            
        #     # 转换为numpy数组
        #     pred_numpy = pred[0, 0].detach().cpu().numpy()
        #     frame_predictions[config_name] = pred_numpy
        
        # return frame_predictions
    
    # def _create_color_comparison_matrix_plot(self, all_predictions, configs, save_path):
    #     """
    #     创建彩色比较矩阵图
        
    #     参数:
    #         all_predictions: 所有配置的预测结果
    #         configs: 配置列表
    #         save_path: 保存路径
    #     """
    #     n_configs = len(configs)
    #     config_names = [config["name"] for config in configs]
        
    #     # 获取所有图像ID
    #     image_ids = list(all_predictions["ground_truth"].keys())
    #     if not image_ids:
    #         print("警告: 没有找到有效的图像数据")
    #         return
        
    #     print(f"处理 {len(image_ids)} 张图像生成比较矩阵...")
        
    #     # 创建图形 - 调整大小以适应N*N矩阵
    #     fig_size = max(4 * n_configs, 12)  # 根据配置数量调整大小
    #     fig, axes = plt.subplots(n_configs, n_configs, figsize=(fig_size, fig_size))
        
    #     # 如果只有一张图，调整axes的形状
    #     if n_configs == 1:
    #         axes = np.array([[axes]])
        
    #     # 计算每对配置的最佳差异图像
    #     best_images_matrix = self._find_best_difference_images(all_predictions, configs, image_ids)
        
    #     # 绘制矩阵图
    #     for i in range(n_configs):
    #         for j in range(n_configs):
    #             ax = axes[i, j]
                
    #             if i == j:
    #                 # 对角线显示配置名称
    #                 ax.text(0.5, 0.5, config_names[i], 
    #                        ha='center', va='center', fontsize=16, fontweight='bold',
    #                        transform=ax.transAxes)
    #                 ax.set_facecolor('#f0f0f0')
    #             else:
    #                 # 非对角线显示彩色MASK图
    #                 image_id = best_images_matrix[i][j]
    #                 if image_id and image_id in all_predictions["ground_truth"]:
    #                     self._plot_color_mask(ax, all_predictions, config_names[i], config_names[j], image_id)
                    
    #                 # 设置标题
    #                 if i == 0:  # 第一行显示列标题
    #                     ax.set_title(config_names[j], fontsize=12, pad=10)
    #                 if j == 0:  # 第一列显示行标题
    #                     ax.set_ylabel(config_names[i], fontsize=12, rotation=90, labelpad=10)
                
    #             ax.set_xticks([])
    #             ax.set_yticks([])
    #             ax.spines['top'].set_visible(True)
    #             ax.spines['right'].set_visible(True)
    #             ax.spines['bottom'].set_visible(True)
    #             ax.spines['left'].set_visible(True)
        
    #     # 添加总标题
    #     plt.suptitle('配置比较矩阵图\n(第i行j列: 配置i比配置j效果最好的图像)', 
    #                 fontsize=20, y=0.95)
        
    #     # 添加图例说明
    #     legend_text = "颜色说明:\n- 红色: 配置i的MASK\n- 绿色: 配置j的MASK\n- 蓝色: 真值MASK\n- 黄色: i和j重叠\n- 洋红: i和真值重叠\n- 青色: j和真值重叠\n- 白色: 三者重叠"
    #     fig.text(0.02, 0.02, legend_text, fontsize=10, 
    #             bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgray"))
        
    #     # 调整布局
    #     plt.tight_layout()
    #     plt.subplots_adjust(top=0.90, bottom=0.15)
        
    #     # 保存图像
    #     plt.savefig(save_path, dpi=300, bbox_inches='tight', format='jpg')
    #     plt.close(fig)
    #     print(f"✓ 彩色比较矩阵图已保存: {save_path}")
    
    # def _find_best_difference_images(self, all_predictions, configs, image_ids):
    #     """
    #     为每对配置找到F1差异最大的图像
        
    #     参数:
    #         all_predictions: 所有配置的预测结果
    #         configs: 配置列表
    #         image_ids: 图像ID列表
            
    #     返回:
    #         list: 最佳差异图像矩阵
    #     """
    #     n_configs = len(configs)
    #     config_names = [config["name"] for config in configs]
        
    #     # 初始化矩阵
    #     best_images_matrix = [[None for _ in range(n_configs)] for _ in range(n_configs)]
    #     best_differences = [[-float('inf') for _ in range(n_configs)] for _ in range(n_configs)]
        
    #     # 计算每张图像的F1分数
    #     print("计算每张图像的F1分数...")
    #     f1_scores = {}
    #     for config_name in config_names:
    #         f1_scores[config_name] = {}
    #         for image_id in tqdm(image_ids, desc=f"计算 {config_name} 的F1"):
    #             if image_id in all_predictions[config_name] and image_id in all_predictions["ground_truth"]:
    #                 pred = all_predictions[config_name][image_id]
    #                 gt = all_predictions["ground_truth"][image_id]
                    
    #                 # 计算F1分数
    #                 f1 = self._calculate_f1_score(pred, gt)
    #                 f1_scores[config_name][image_id] = f1
        
    #     # 为每对配置找到最佳差异图像
    #     print("寻找每对配置的最佳差异图像...")
    #     for i in range(n_configs):
    #         for j in range(n_configs):
    #             if i != j:
    #                 config_i = config_names[i]
    #                 config_j = config_names[j]
                    
    #                 best_image_id = None
    #                 best_diff = -float('inf')
                    
    #                 for image_id in image_ids:
    #                     if (image_id in f1_scores[config_i] and 
    #                         image_id in f1_scores[config_j]):
                            
    #                         f1_i = f1_scores[config_i][image_id]
    #                         f1_j = f1_scores[config_j][image_id]
    #                         diff = f1_i - f1_j
                            
    #                         # 只考虑config_i比config_j好的情况
    #                         if diff > 0 and diff > best_diff:
    #                             best_diff = diff
    #                             best_image_id = image_id
                    
    #                 if best_image_id:
    #                     best_images_matrix[i][j] = best_image_id
    #                     best_differences[i][j] = best_diff
                        
    #                     # 打印最佳差异信息
    #                     if best_diff > 0:
    #                         print(f"{config_i} 优于 {config_j}: 图像 {best_image_id}, F1差异: {best_diff:.4f}")
        
    #     return best_images_matrix
    
    def _calculate_f1_score(self, pred, gt):
        """
        计算F1分数
        
        参数:
            pred: 预测结果 (0-1之间的值)
            gt: 真实标签 (0-1之间的值)
            
        返回:
            float: F1分数
        """
        # 二值化预测结果
        pred_binary = (pred > 0.5).astype(np.float32)
        gt_binary = (gt > 0.5).astype(np.float32)
        
        # 计算TP, FP, FN
        tp = np.sum(pred_binary * gt_binary)
        fp = np.sum(pred_binary * (1 - gt_binary))
        fn = np.sum((1 - pred_binary) * gt_binary)
        
        # 计算precision和recall
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        
        # 计算F1分数
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
        
        return f1
    
    # def _plot_color_mask(self, ax, all_predictions, config_i, config_j, image_id):
    #     """
    #     在指定坐标轴上绘制彩色MASK图
        
    #     参数:
    #         ax: matplotlib坐标轴
    #         all_predictions: 所有配置的预测结果
    #         config_i: 第i个配置名称
    #         config_j: 第j个配置名称
    #         image_id: 图像ID
    #     """
    #     # 获取三个MASK
    #     mask_i = all_predictions[config_i].get(image_id)
    #     mask_j = all_predictions[config_j].get(image_id)
    #     mask_gt = all_predictions["ground_truth"].get(image_id)
        
    #     if mask_i is None or mask_j is None or mask_gt is None:
    #         ax.text(0.5, 0.5, "数据缺失", ha='center', va='center', transform=ax.transAxes)
    #         return
        
    #     # 二值化MASK
    #     # mask_i_binary = (mask_i > 0.5).astype(np.uint8)
    #     # mask_j_binary = (mask_j > 0.5).astype(np.uint8)
    #     # mask_gt_binary = (mask_gt > 0.5).astype(np.uint8)
    #     mask_i_binary = mask_i.astype(np.uint8)
    #     mask_j_binary = mask_j.astype(np.uint8)
    #     mask_gt_binary = mask_gt.astype(np.uint8)
        
    #     # 创建彩色图像 (R: config_i, G: config_j, B: ground_truth)
    #     color_mask = np.zeros((mask_i.shape[0], mask_i.shape[1], 3), dtype=np.uint8)
    #     # color_mask[:, :, 0] = mask_i_binary * 255  # 红色通道
    #     # color_mask[:, :, 1] = mask_j_binary * 255  # 绿色通道
    #     # color_mask[:, :, 2] = mask_gt_binary * 255  # 蓝色通道
    #     color_mask[:, :, 0] = (mask_i_binary+mask_gt_binary) * 127  # 红色通道
    #     color_mask[:, :, 1] = (mask_j_binary+mask_gt_binary) * 127  # 绿色通道
    #     color_mask[:, :, 2] = mask_gt_binary * 255  # 蓝色通道
        
    #     # 显示彩色图像
    #     ax.imshow(color_mask)
        
    #     # 计算并显示F1差异
    #     f1_i = self._calculate_f1_score(mask_i, mask_gt)
    #     f1_j = self._calculate_f1_score(mask_j, mask_gt)
    #     f1_diff = f1_i - f1_j
        
    #     # 在图像上方显示F1差异
    #     ax.set_title(f'F1差异: {f1_diff:.3f}', fontsize=10, pad=5)
        
    #     # 在图像下方显示图像ID（缩短显示）
    #     # short_image_id = os.path.basename(image_id) if len(image_id) > 30 else image_id
    #     # ax.set_xlabel(short_image_id, fontsize=8, rotation=45, ha='right')
    #     short_image_id = os.path.basename(image_id) if len(image_id) > 30 else image_id
    #     ax.set_xlabel(short_image_id.split(".png")[0], fontsize=8, ha='right')

def main(): 
    """主函数"""
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
    configs = [
        {
            "name": "1.masks", 
            "precomputed": True,
            "result_path_template": os.path.join("../DeNVeR.006/log_6/outputs", "{videoId}", "1.masks", "{frameId}"),
            "input_mode_for_display": "orig",
            "binarize": True
        },
        {
            "name": "5.refine", 
            "precomputed": True,
            "result_path_template": os.path.join("../DeNVeR.006/log_6/outputs", "{videoId}", "5.refine", "{frameId}"),
            "input_mode_for_display": "orig",
            "binarize": True
        },
        {
            "name": "orig",
            "precomputed": False,
            "input_mode": "orig",
            "norm_method": norm_calculator.calculate_mean_variance,
            "binarize": True
        },
    ]
    
    # 创建综合比较器
    comparator = ComprehensiveMulComparison(config, model_manager, image_loader, norm_calculator)
    
    # 生成彩色比较矩阵图
    comparator.inference(
        configs, config.root_path + "/", block_cath, threshold
    )
    


if __name__ == "__main__":
    main()
