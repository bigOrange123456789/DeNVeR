import os
import yaml
import json
import datetime
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
from PIL import Image
from tqdm import tqdm
import torch
from torchvision import transforms
import torch.backends.cudnn as cudnn

# Import custom modules
from nir.new import startDecouple1, startDecouple3
from free_cos.newTrain import initCSV, save2CVS, getIndicators
from nir.analysisOri import getModel
from free_cos.ModelSegment import ModelSegment
from free_cos.main import mainFreeCOS
# from preprocess.mySkeleton import getCatheter


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
            self.dataset_path_gt = config["my"]["datasetPath"]  # 真值标签路径
            self.dataset_path = os.path.join("..", "DeNVeR.008", "log_8", "dataset_decouple")  # 解耦数据路径
            
        print(f"真值标签的路径: {self.dataset_path_gt}")
        print(f"解耦数据的路径: {self.dataset_path}")


class ImageLoader:
    """图像加载器类"""
    
    IMAGE_PATHS = {
        "orig": "{dataset_path}/{patient_id}/decouple/{video_id}/orig/{frame_id}",
        "fluid": "{dataset_path}/{patient_id}/decouple/{video_id}/recon_non2/{frame_id}",
        "fluid2": "{dataset_path}/{patient_id}/decouple/{video_id}/C.recon_non2/{frame_id}",
        "noRigid1": "{dataset_path}/{patient_id}/decouple/{video_id}/A.rigid.main_non1/{frame_id}",
        "noRigid2": "{dataset_path}/{patient_id}/decouple/{video_id}/A.rigid.main_non2/{frame_id}",
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
            raise ValueError(f"未知的标签类型: {tag}")
    
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
        
        for sub_path in config[:2]:  # 前两个元素是路径
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


class Evaluator:
    """评估器类"""
    
    def __init__(self, config, model_manager, image_loader, norm_calculator):
        self.config = config
        self.model_manager = model_manager
        self.image_loader = image_loader
        self.norm_calculator = norm_calculator
    
    def evaluate_single_method(self, tag="orig", threshold=0.85):
        """评估单个方法"""
        print(f"评估方法: {tag}")
        
        param_path = "../DeNVeR_in/models_config/freecos_Seg.pt"
        model = self.model_manager.init_model(param_path)
        
        patient_names = self._get_patient_names()
        total_images = self._count_total_images(patient_names)
        
        metrics = self._evaluate_all_images(tag, model, patient_names, total_images, threshold)
        self._print_results(tag, metrics, total_images)
        
        return metrics
    
    def _get_patient_names(self):
        """获取患者名称列表"""
        return [name for name in os.listdir(self.config.dataset_path) 
                if os.path.isdir(os.path.join(self.config.dataset_path, name))]
    
    def _count_total_images(self, patient_names):
        """计算总图像数量"""
        total_count = 0
        for patient_id in patient_names:
            gt_path = os.path.join(self.config.dataset_path, patient_id, "ground_truth")
            if not os.path.exists(gt_path):
                continue
                
            video_names = [name for name in os.listdir(gt_path) 
                          if os.path.isdir(os.path.join(gt_path, name))]
            for video_id in video_names:
                if "CATH" not in video_id:
                    video_path = os.path.join(gt_path, video_id)
                    total_count += len(os.listdir(video_path))
        return total_count
    
    def _evaluate_all_images(self, tag, model, patient_names, total_images, threshold):
        """评估所有图像"""
        sum_recall, sum_precision, sum_f1 = 0, 0, 0
        
        with tqdm(total=total_images) as progress_bar:
            for patient_id in patient_names:
                patient_results = self._process_patient(
                    tag, model, patient_id, progress_bar, threshold
                )
                sum_recall += patient_results["recall"]
                sum_precision += patient_results["precision"]
                sum_f1 += patient_results["f1"]
        
        return self._calculate_final_metrics(sum_recall, sum_precision, sum_f1, total_images)
    
    def _process_patient(self, tag, model, patient_id, progress_bar, threshold):
        """处理单个患者的所有视频"""
        gt_path = os.path.join(self.config.dataset_path, patient_id, "ground_truth")
        if not os.path.exists(gt_path):
            return {"recall": 0, "precision": 0, "f1": 0}
            
        video_names = [name for name in os.listdir(gt_path) 
                      if os.path.isdir(os.path.join(gt_path, name))]
        
        patient_recall, patient_precision, patient_f1 = 0, 0, 0
        
        for video_id in video_names:
            if "CATH" not in video_id:
                video_results = self._process_video(
                    tag, model, patient_id, video_id, progress_bar, threshold
                )
                patient_recall += video_results["recall"]
                patient_precision += video_results["precision"]
                patient_f1 += video_results["f1"]
        
        return {"recall": patient_recall, "precision": patient_precision, "f1": patient_f1}
    
    def _process_video(self, tag, model, patient_id, video_id, progress_bar, threshold):
        """处理单个视频的所有帧"""
        video_gt_path = os.path.join(self.config.dataset_path, patient_id, "ground_truth", video_id)
        if not os.path.exists(video_gt_path):
            return {"recall": 0, "precision": 0, "f1": 0}
        
        if tag != "pred":
            mean, std = self.norm_calculator.calculate_mean_variance(
                tag, self.model_manager.transform, patient_id, video_id
            )
        
        video_recall, video_precision, video_f1 = 0, 0, 0
        frame_ids = os.listdir(video_gt_path)
        
        for frame_id in frame_ids:
            metrics = self._process_single_frame(
                tag, model, patient_id, video_id, frame_id, threshold, 
                mean if tag != "pred" else None, std if tag != "pred" else None
            )
            video_recall += metrics["recall"]
            video_precision += metrics["precision"]
            video_f1 += metrics["f1"]
            progress_bar.update(1)
        
        return {"recall": video_recall, "precision": video_precision, "f1": video_f1}
    
    def _process_single_frame(self, tag, model, patient_id, video_id, frame_id, threshold, mean, std):
        """处理单帧图像"""
        # 获取预测结果
        if tag == "pred":
            pred = self._get_precomputed_prediction(patient_id, video_id, frame_id)
        else:
            pred = self._get_model_prediction(tag, model, patient_id, video_id, frame_id, mean, std, threshold)
        
        # 获取真实标签
        gt = self._load_ground_truth(patient_id, video_id, frame_id)
        
        # 计算指标
        indicators = getIndicators(
            pred[0, 0].detach().cpu() * 255,
            gt[0, 0].detach().cpu() * 255
        )
        
        return {
            "recall": indicators["recall"],
            "precision": indicators["precision"],
            "f1": indicators["f1"]
        }
    
    def _get_precomputed_prediction(self, patient_id, video_id, frame_id):
        """获取预计算的预测结果"""
        pred_path = os.path.join(
            self.config.dataset_path, patient_id, "decouple", video_id, 
            "A.mask.main_nr2.cf", "filter", frame_id
        )
        pred = Image.open(pred_path).convert('L')
        return self.model_manager.transform(pred).unsqueeze(0).cuda()
    
    def _get_model_prediction(self, tag, model, patient_id, video_id, frame_id, mean, std, threshold):
        """获取模型预测结果"""
        img = self.image_loader.load_image(tag, patient_id, video_id, frame_id)
        img_normalized = (img - mean) / std
        pred = model(img_normalized)["pred"]
        pred[pred > threshold] = 1
        pred[pred <= threshold] = 0
        return pred
    
    def _load_ground_truth(self, patient_id, video_id, frame_id):
        """加载真实标签"""
        gt_path = os.path.join(
            self.config.dataset_path, patient_id, "ground_truth", video_id, frame_id
        )
        gt = Image.open(gt_path).convert('L')
        gt_tensor = self.model_manager.transform(gt).unsqueeze(0).cuda()
        gt_tensor[gt_tensor > 0.5] = 1
        gt_tensor[gt_tensor <= 0.5] = 0
        return gt_tensor
    
    def _calculate_final_metrics(self, sum_recall, sum_precision, sum_f1, total_images):
        """计算最终指标"""
        # 处理张量维度
        if hasattr(sum_f1, 'dim') and not sum_f1.dim() == 0:
            sum_f1 = sum_f1[0]
        if hasattr(sum_precision, 'dim') and not sum_precision.dim() == 0:
            sum_precision = sum_precision[0]
        
        f1 = sum_f1 / total_images
        precision = sum_precision / total_images
        recall = sum_recall / total_images
        
        return {"f1": f1, "precision": precision, "recall": recall}
    
    def _print_results(self, tag, metrics, total_images):
        """打印评估结果"""
        print(f"{tag} 评估结果 (共{total_images}张图像):")
        print(f"F1: {metrics['f1']}")
        print(f"Precision: {metrics['precision']}") 
        print(f"Recall: {metrics['recall']}")


class StatisticalAnalyzer:
    """统计分析器"""
    
    @staticmethod
    def perform_paired_t_test(df, tag1, tag2):
        """
        对DataFrame中的两种方法的指标进行配对t检验
        
        Args:
            df: 包含指标数据的DataFrame
            tag1: 第一种方法名称
            tag2: 第二种方法名称
            
        Returns:
            dict: 包含所有指标检验结果的字典
        """
        print(f"\n执行配对t检验: {tag1} vs {tag2}")
        
        tag1_columns = [col for col in df.columns if col.startswith(tag1 + '_')]
        tag2_columns = [col for col in df.columns if col.startswith(tag2 + '_')]
        
        metrics = [col.replace(tag1 + '_', '') for col in tag1_columns]
        tag2_metrics = [col.replace(tag2 + '_', '') for col in tag2_columns]
        
        # 使用指标交集
        metrics = list(set(metrics) & set(tag2_metrics))
        results = {}
        
        for metric in metrics:
            tag1_col = f"{tag1}_{metric}"
            tag2_col = f"{tag2}_{metric}"
            
            data1 = df[tag1_col].values
            data2 = df[tag2_col].values
            
            # 清理数据
            mask = ~(np.isnan(data1) | np.isnan(data2))
            data1_clean = data1[mask]
            data2_clean = data2[mask]
            
            if len(data1_clean) < 2:
                print(f"警告：{metric} 指标的有效样本数不足，无法进行t检验")
                continue
            
            # 执行t检验
            t_stat, p_value = stats.ttest_rel(data1_clean, data2_clean)
            
            # 计算描述性统计
            mean1, mean2 = np.mean(data1_clean), np.mean(data2_clean)
            std1, std2 = np.std(data1_clean, ddof=1), np.std(data2_clean, ddof=1)
            mean_diff = mean2 - mean1
            std_diff = np.std(data2_clean - data1_clean, ddof=1)
            
            # 计算置信区间
            n = len(data1_clean)
            se = std_diff / np.sqrt(n)
            ci_lower = mean_diff - 1.96 * se
            ci_upper = mean_diff + 1.96 * se
            
            # 判断显著性
            significance, significance_text = StatisticalAnalyzer._get_significance_level(p_value)
            
            results[metric] = {
                't_statistic': t_stat,
                'p_value': p_value,
                f'mean_{tag1}': mean1,
                f'mean_{tag2}': mean2,
                'mean_difference': mean_diff,
                f'std_{tag1}': std1,
                f'std_{tag2}': std2,
                'std_difference': std_diff,
                'sample_size': n,
                'confidence_interval_lower': ci_lower,
                'confidence_interval_upper': ci_upper,
                'significance': significance,
                'significance_text': significance_text
            }
            
            StatisticalAnalyzer._print_single_test_result(metric, tag1, tag2, results[metric])
        
        return results
    
    @staticmethod
    def _get_significance_level(p_value):
        """根据p值确定显著性水平"""
        if p_value < 0.001:
            return "***", "非常显著"
        elif p_value < 0.01:
            return "**", "很显著"
        elif p_value < 0.05:
            return "*", "显著"
        else:
            return "ns", "不显著"
    
    @staticmethod
    def _print_single_test_result(metric, tag1, tag2, result):
        """打印单个指标的检验结果"""
        print(f"\n=== {metric.upper()} 指标配对t检验结果 ===")
        print(f"{tag1}均值: {result[f'mean_{tag1}']:.4f} ± {result[f'std_{tag1}']:.4f}")
        print(f"{tag2}均值: {result[f'mean_{tag2}']:.4f} ± {result[f'std_{tag2}']:.4f}")
        print(f"均值差异: {result['mean_difference']:.4f} (95% CI: [{result['confidence_interval_lower']:.4f}, {result['confidence_interval_upper']:.4f}])")
        print(f"t 统计量: {result['t_statistic']:.4f}")
        print(f"p 值: {result['p_value']:.4f} {result['significance']}")
        print(f"统计显著性: {result['significance_text']}")


class ResultVisualizer:
    """结果可视化器"""
    
    @staticmethod
    def save_comparison_figure(results, K, tag1, tag2, save_path, result_type="Best"):
        """
        保存对比图为JPG文件
        
        Args:
            results: 结果数据列表
            K: 显示图像数量
            tag1: 第一种方法名称
            tag2: 第二种方法名称
            save_path: 保存路径
            result_type: "Best" 或 "Worst"，表示最好或最差结果
        """
        print(f"创建{result_type}对比图，包含 {K} 张图像...")
        
        fig_width = max(4 * K, 12)
        fig_height = 16
        fig, axes = plt.subplots(6, K, figsize=(fig_width, fig_height))
        
        if K == 1:
            axes = axes.reshape(6, 1)
        
        plt.rcParams.update({'font.size': 10})
        
        for i, result in enumerate(results):
            ResultVisualizer._plot_single_comparison(axes, i, result, tag1, tag2)
        
        title = ResultVisualizer._get_figure_title(result_type, K, tag1, tag2)
        plt.suptitle(title, fontsize=16, y=0.98)
        plt.tight_layout()
        plt.subplots_adjust(top=0.93)
        
        plt.savefig(save_path, dpi=300, bbox_inches='tight', format='jpg')
        print(f"✓ {result_type}对比图像已保存: {save_path}")
        plt.close(fig)
    
    @staticmethod
    def _plot_single_comparison(axes, index, result, tag1, tag2):
        """绘制单个对比结果"""
        # 第1行: tag1输入图像
        axes[0, index].imshow(result['img_tag1'], cmap='gray')
        axes[0, index].set_title(f"{tag1} Input", fontsize=10)
        axes[0, index].axis('off')
        
        # 第2行: tag1分割结果
        axes[1, index].imshow(result['pred_tag1'], cmap='gray')
        axes[1, index].set_title(f"{tag1} Seg\nF1: {result['f1_tag1']:.4f}", fontsize=10)
        axes[1, index].axis('off')
        
        # 第3行: tag2输入图像
        axes[2, index].imshow(result['img_tag2'], cmap='gray')
        axes[2, index].set_title(f"{tag2} Input", fontsize=10)
        axes[2, index].axis('off')
        
        # 第4行: tag2分割结果
        axes[3, index].imshow(result['pred_tag2'], cmap='gray')
        axes[3, index].set_title(f"{tag2} Seg\nF1: {result['f1_tag2']:.4f}", fontsize=10)
        axes[3, index].axis('off')
        
        # 第5行: ground truth
        axes[4, index].imshow(result['gt'], cmap='gray')
        axes[4, index].set_title("Ground Truth", fontsize=10)
        axes[4, index].axis('off')
        
        # 第6行: 文件名和F1差异
        ResultVisualizer._add_filename_and_difference(axes[5, index], result)
    
    @staticmethod
    def _add_filename_and_difference(ax, result):
        """添加文件名和F1差异信息"""
        f1_diff_text = f"F1差异: {result['f1_diff']:.4f}"
        ax.text(0.5, 0.7, f"{result['videoId']}/{result['frameId']}", 
               horizontalalignment='center', verticalalignment='center',
               transform=ax.transAxes, fontsize=8)
        ax.text(0.5, 0.3, f1_diff_text,
               horizontalalignment='center', verticalalignment='center',
               transform=ax.transAxes, fontsize=8)
        ax.set_title("文件名和F1差异", fontsize=10)
        ax.axis('off')
    
    @staticmethod
    def _get_figure_title(result_type, K, tag1, tag2):
        """获取图像标题"""
        if result_type == "Best":
            return f'Top {K} Images Where {tag1} Outperforms {tag2} (Best F1 Difference)'
        else:
            return f'Top {K} Images Where {tag2} Outperforms {tag1} (Worst F1 Difference)'


class ResultSaver:
    """结果保存器"""
    
    @staticmethod
    def save_results_to_excel(df, t_test_results, tag1, tag2, save_path):
        """
        将数据和统计检验结果保存到Excel文件
        
        Args:
            df: 原始数据DataFrame
            t_test_results: t检验结果字典
            tag1: 第一种方法名称
            tag2: 第二种方法名称
            save_path: 保存路径
        """
        print(f"保存统计结果到Excel: {save_path}")
        
        with pd.ExcelWriter(save_path, engine='openpyxl') as writer:
            # 1. 原始数据
            df.to_excel(writer, sheet_name='原始数据', index=False)
            
            # 2. 统计摘要
            summary_df = ResultSaver._create_summary_dataframe(t_test_results, tag1, tag2)
            summary_df.to_excel(writer, sheet_name='统计摘要', index=False)
            
            # 3. 差异数据
            diff_df = ResultSaver._create_difference_dataframe(df, t_test_results, tag1, tag2)
            diff_df.to_excel(writer, sheet_name='差异数据', index=False)
            
            # 4. 方法比较
            comparison_df = ResultSaver._create_comparison_dataframe(t_test_results, tag1, tag2)
            comparison_df.to_excel(writer, sheet_name='方法比较', index=False)
            
            # 5. 图像路径
            path_df = ResultSaver._create_path_dataframe(df)
            path_df.to_excel(writer, sheet_name='图像路径', index=False)
        
        print(f"✓ 统计结果已保存: {save_path}")
        ResultSaver._print_excel_sheet_info()
    
    @staticmethod
    def _create_summary_dataframe(t_test_results, tag1, tag2):
        """创建统计摘要DataFrame"""
        summary_data = []
        for metric, result in t_test_results.items():
            summary_data.append({
                '指标': metric.upper(),
                f'{tag1}均值': result[f'mean_{tag1}'],
                f'{tag1}标准差': result[f'std_{tag1}'],
                f'{tag2}均值': result[f'mean_{tag2}'],
                f'{tag2}标准差': result[f'std_{tag2}'],
                '均值差异': result['mean_difference'],
                '差异标准差': result['std_difference'],
                't统计量': result['t_statistic'],
                'p值': result['p_value'],
                '显著性': result['significance'],
                '样本量': result['sample_size'],
                '95%置信区间下限': result['confidence_interval_lower'],
                '95%置信区间上限': result['confidence_interval_upper']
            })
        return pd.DataFrame(summary_data)
    
    @staticmethod
    def _create_difference_dataframe(df, t_test_results, tag1, tag2):
        """创建差异数据DataFrame"""
        diff_data = df[['patientID', 'videoId', 'frameId', 'image_id', 
                       'tag1_image_path', 'tag2_image_path', 'ground_truth_path']].copy()
        
        for metric in t_test_results.keys():
            tag1_col = f"{tag1}_{metric}"
            tag2_col = f"{tag2}_{metric}"
            diff_col = f"{metric}_difference"
            
            if tag1_col in df.columns and tag2_col in df.columns:
                diff_data[tag1_col] = df[tag1_col]
                diff_data[tag2_col] = df[tag2_col]
                diff_data[diff_col] = df[tag2_col] - df[tag1_col]
        
        return diff_data
    
    @staticmethod
    def _create_comparison_dataframe(t_test_results, tag1, tag2):
        """创建方法比较DataFrame"""
        comparison_data = []
        for metric, result in t_test_results.items():
            comparison_data.append({
                '指标': metric.upper(),
                '检验方法': '配对t检验',
                f'{tag1}均值±标准差': f"{result[f'mean_{tag1}']:.4f} ± {result[f'std_{tag1}']:.4f}",
                f'{tag2}均值±标准差': f"{result[f'mean_{tag2}']:.4f} ± {result[f'std_{tag2}']:.4f}",
                '均值差异(95% CI)': f"{result['mean_difference']:.4f} ({result['confidence_interval_lower']:.4f} to {result['confidence_interval_upper']:.4f})",
                't值': f"{result['t_statistic']:.4f}",
                'p值': f"{result['p_value']:.4f}",
                '显著性': result['significance_text']
            })
        return pd.DataFrame(comparison_data)
    
    @staticmethod
    def _create_path_dataframe(df):
        """创建图像路径DataFrame"""
        return df[['patientID', 'videoId', 'frameId', 'image_id', 
                  'tag1_image_path', 'tag2_image_path', 'ground_truth_path']].copy()
    
    @staticmethod
    def _print_excel_sheet_info():
        """打印Excel工作表信息"""
        print("Excel文件包含以下工作表:")
        print("  1. 原始数据 - 所有图像的详细指标")
        print("  2. 统计摘要 - 详细的统计检验结果") 
        print("  3. 差异数据 - 两种方法的差异值")
        print("  4. 方法比较 - 简明的比较结果")
        print("  5. 图像路径 - 所有输入图像和ground truth的路径")


# 主比较分析类（由于代码长度限制，这里只展示重构后的主类结构）
class ComprehensiveComparison:
    """综合比较分析器"""
    
    def __init__(self, config, model_manager, image_loader, norm_calculator):
        self.config = config
        self.model_manager = model_manager
        self.image_loader = image_loader
        self.norm_calculator = norm_calculator
    
    #########################################
    def analyze(self, config1, config2, K=5, threshold=0.85, save_path="", block_cath=False):
        """
        对两种配置进行全面的比较分析
        
        参数:
            config1: 第一种配置的字典
            config2: 第二种配置的字典  
            K: 对比图中显示的图像数量
            threshold: 分割阈值
            save_path: 保存路径
            block_cath: 是否屏蔽导管区域
        
        返回:
            tuple: (数据DataFrame, t检验结果, 最佳结果列表, 最差结果列表)
        """
        tag1 = config1["name"]
        tag2 = config2["name"]
        
        print(f"开始全面的比较分析: {tag1} vs {tag2}")
        print(f"模式: {'屏蔽标签中的导管' if block_cath else '不屏蔽标签中的导管'}")
        print("=" * 60)
        
        # 设置保存路径
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        mode_suffix = "_hasCath" if block_cath else "_nonCath"
        
        save_img_path_best = f"{save_path}{tag1}_vs_{tag2}_best{mode_suffix}_{timestamp}.jpg"
        save_img_path_worst = f"{save_path}{tag1}_vs_{tag2}_worst{mode_suffix}_{timestamp}.jpg"
        save_excel_path = f"{save_path}{tag1}_vs_{tag2}_statistical_analysis{mode_suffix}_{timestamp}.xlsx"
        
        # 步骤1: 收集所有图像的指标数据
        print("步骤1/4: 收集所有图像的指标数据...")
        df, best_results, worst_results = self._collect_all_metrics(
            config1, config2, K, threshold, block_cath
        )
        
        # 步骤2: 执行统计检验
        print("步骤2/4: 执行统计检验...")
        statistical_analyzer = StatisticalAnalyzer()
        t_test_results = statistical_analyzer.perform_paired_t_test(df, tag1, tag2)
        
        # 步骤3: 保存对比图像
        print("步骤3/4: 保存对比图像...")
        visualizer = ResultVisualizer()
        visualizer.save_comparison_figure(best_results, K, tag1, tag2, save_img_path_best, "Best")
        visualizer.save_comparison_figure(worst_results, K, tag1, tag2, save_img_path_worst, "Worst")
        
        # 步骤4: 保存Excel统计结果
        print("步骤4/4: 保存Excel统计结果...")
        saver = ResultSaver()
        saver.save_results_to_excel(df, t_test_results, tag1, tag2, save_excel_path)
        
        print("\n" + "=" * 60)
        print("分析完成!")
        print(f"✓ 最佳对比图像已保存: {save_img_path_best}")
        print(f"✓ 最差对比图像已保存: {save_img_path_worst}") 
        print(f"✓ 统计结果已保存: {save_excel_path}")
        
        return df, t_test_results, best_results, worst_results

    def _collect_all_metrics(self, config1, config2, K=5, threshold=0.85, block_cath=False):
        """
        收集所有图像的指标数据
        
        参数:
            config1: 第一种配置
            config2: 第二种配置
            K: 返回的最佳/最差结果数量
            threshold: 分割阈值
            block_cath: 是否屏蔽导管区域
        
        返回:
            tuple: (所有结果的DataFrame, 最佳结果列表, 最差结果列表)
        """
        # 检查哪些配置需要模型
        need_model1 = not config1.get("precomputed", False)
        need_model2 = not config2.get("precomputed", False)
        
        # 初始化模型（如果需要）
        if need_model1 or need_model2:
            param_path = "../DeNVeR_in/models_config/freecos_Seg.pt"
            model = self.model_manager.init_model(param_path)
        else:
            model = None
        
        # 获取配置信息
        tag1 = config1["name"]
        tag2 = config2["name"]
        
        # 获取所有患者
        patient_names = self._get_patient_names()
        
        # 计算总图像数量
        total_images = self._count_annotated_images(patient_names)
        print(f"总图像数量: {total_images}")
        
        # 存储结果
        all_results = []
        comparison_results = []
        
        # 处理所有图像
        with tqdm(total=total_images, desc="处理图像") as progress_bar:
            for patient_id in patient_names:
                patient_results = self._process_patient_metrics(
                    patient_id, config1, config2, tag1, tag2, model, 
                    threshold, block_cath, progress_bar
                )
                all_results.extend(patient_results["all_results"])
                comparison_results.extend(patient_results["comparison_results"])
        
        # 转换为DataFrame
        df = pd.DataFrame(all_results)
        
        # 提取有效结果用于对比
        valid_results = [r for r in comparison_results if r.get('has_ground_truth', True)]
        
        # 找出最佳和最差结果
        best_results = sorted(valid_results, key=lambda x: x['f1_diff'], reverse=True)[:K]
        worst_results = sorted(valid_results, key=lambda x: x['f1_diff'])[:K]
        
        return df, best_results, worst_results

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

    def _process_patient_metrics(self, patient_id, config1, config2, tag1, tag2, 
                            model, threshold, block_cath, progress_bar):
        """处理单个患者的所有图像指标"""
        patient_gt_path = os.path.join(self.config.dataset_path_gt, patient_id, "ground_truth")
        if not os.path.exists(patient_gt_path):
            return {"all_results": [], "comparison_results": []}
        
        video_names = [name for name in os.listdir(patient_gt_path) 
                    if os.path.isdir(os.path.join(patient_gt_path, name))]
        
        all_results = []
        comparison_results = []
        
        for video_id in video_names:
            if "CATH" not in video_id:  # 只处理非导管视频
                video_results = self._process_video_metrics(
                    patient_id, video_id, config1, config2, tag1, tag2,
                    model, threshold, block_cath, progress_bar
                )
                all_results.extend(video_results["all_results"])
                comparison_results.extend(video_results["comparison_results"])
        
        return {"all_results": all_results, "comparison_results": comparison_results}

    def _process_video_metrics(self, patient_id, video_id, config1, config2, tag1, tag2,
                            model, threshold, block_cath, progress_bar):
        """处理单个视频的所有帧指标"""
        video_gt_path = os.path.join(self.config.dataset_path_gt, patient_id, "ground_truth", video_id)
        if not os.path.exists(video_gt_path):
            return {"all_results": [], "comparison_results": []}
        
        # 计算归一化参数（如果需要）
        mean_tag1, std_tag1 = self._get_normalization_params(config1, patient_id, video_id)
        mean_tag2, std_tag2 = self._get_normalization_params(config2, patient_id, video_id)
        
        all_results = []
        comparison_results = []
        frame_ids = os.listdir(video_gt_path)
        
        progress_bar.set_postfix(patient=patient_id, video=video_id, mode="annotated")
        
        for frame_id in frame_ids:
            try:
                result = self._process_single_image(
                    patient_id, video_id, frame_id, config1, config2, tag1, tag2,
                    mean_tag1, std_tag1, mean_tag2, std_tag2, model, threshold, block_cath
                )
                all_results.append(result["all_result"])
                if result.get("comparison_result"):
                    comparison_results.append(result["comparison_result"])
                    
            except Exception as e:
                print(f"处理错误 {patient_id}/{video_id}/{frame_id}: {str(e)}")
            
            progress_bar.update(1)
        
        return {"all_results": all_results, "comparison_results": comparison_results}

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

    def _get_prediction(self, config, model, img_norm, threshold, patient_id, video_id, frame_id):
        """获取预测结果"""
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
                if config.get("binarize", True):
                    pred_tensor[pred_tensor > threshold] = 1
                    pred_tensor[pred_tensor <= threshold] = 0
                return pred_tensor
            else:
                raise FileNotFoundError(f"预计算预测结果不存在: {pred_path}")
        else:
            # 使用模型进行推理
            pred = model(img_norm)["pred"]
            if config.get("binarize", True):
                pred[pred > threshold] = 1
                pred[pred <= threshold] = 0
            return pred

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
        """应用导管掩码到预测结果"""
        # 重新计算导管掩码
        mask_cath = torch.zeros_like(gt_tensor)
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
    #########################################


def denoising():
# if False:#
# if __name__ == "__main__":
    script_path = os.path.abspath(__file__)
    ROOT1 = os.path.dirname(script_path)
    file_path = os.path.join(ROOT1, "../",'confs/newConfig.yaml')
    
    # 进度文件路径
    progress_file = os.path.join(ROOT1, "progress_newBatch.json")
    # 加载进度文件
    processed_videos = set()
    if os.path.exists(progress_file):
        try:
            with open(progress_file, 'r', encoding='utf-8') as f:
                progress_data = json.load(f)
                processed_videos = set(progress_data.get("processed_videos", []))
                print(f"加载进度文件，已处理 {len(processed_videos)} 个视频")
        except Exception as e:
            print(f"加载进度文件失败: {e}，将重新开始处理")
    
    # 打开并读取 YAML 文件
    with open(file_path, 'r', encoding='utf-8') as file:
        config = yaml.safe_load(file)
        print("notes:",config["my"]["notes"])
        rootPath = config["my"]["filePathRoot"]
        datasetPath0=config["my"]["datasetPath"]

        # datasetPath="../DeNVeR_in/xca_dataset"
        paramPath = "../DeNVeR_in/models_config/freecos_Seg.pt"
        patient_names = [name for name in os.listdir(datasetPath0)
                    if os.path.isdir(os.path.join(datasetPath0, name))]
        CountSum = 0
        for patientID in patient_names:
            patient_path = os.path.join(datasetPath0, patientID, "images")
            video_names = [name for name in os.listdir(patient_path)
                        if os.path.isdir(os.path.join(patient_path, name))]
            CountSum = CountSum + len(video_names)
        CountI = len(processed_videos)  # 从已处理的数量开始计数
        
        print(f"总视频数: {CountSum}, 已处理: {CountI}, 待处理: {CountSum - CountI}")
        
        for patientID in patient_names:
            patient_path = os.path.join(datasetPath0, patientID, "images")
            video_names = [name for name in os.listdir(patient_path)
                        if os.path.isdir(os.path.join(patient_path, name))]
            for videoId in video_names:
                # 生成唯一标识符
                video_key = f"{patientID}/{videoId}"
                
                # 如果已经处理过，则跳过
                if video_key in processed_videos:
                    continue
                
                inpath = os.path.join(datasetPath0, patientID, "images", videoId)
                outpath = os.path.join(datasetPath0, patientID, "decouple", videoId)#数据集路径
                outpath = os.path.join(datasetPath0, patientID, "decouple", videoId)#本地路径
                print("outpath:",outpath)
                os.makedirs(outpath, exist_ok=True)
                
                startDecouple1(videoId, paramPath, inpath, outpath)  # 去除刚体层
                # startDecouple1(videoId, paramPath, inpath, outpath)  # 去除刚体层
                # startDecouple3(videoId, paramPath, inpath, outpath)  # 获取流体层
                    
                # 处理成功，更新进度
                CountI += 1
                processed_videos.add(video_key)
                    
                # 更新进度文件
                progress_data = {"processed_videos": list(processed_videos)}
                with open(progress_file, 'w', encoding='utf-8') as f:
                    json.dump(progress_data, f, ensure_ascii=False, indent=2)    
                print(f"{CountI}/{CountSum} {videoId} - 已完成")


def main():
    """主函数"""
    # 初始化配置
    config = Config()
    model_manager = ModelManager()
    image_loader = ImageLoader(config.dataset_path, model_manager.transform)
    norm_calculator = NormalizationCalculator(config.dataset_path, image_loader)
    
    # 设置参数
    topK = 10
    threshold = 0.5
    block_cath = True
    
    # 定义比较配置
    config1 = {
        "name": "orig",
        "precomputed": False,
        "input_mode": "orig",
        "norm_method": norm_calculator.calculate_mean_variance,
        "binarize": True
    }

    config2 = {
        "name": "refine", 
        "precomputed": True,
        "result_path_template": os.path.join("../DeNVeR.006/log_6/outputs", "{videoId}", "5.refine", "{frameId}"),
        "input_mode_for_display": "orig",
        "binarize": True
    }
    
    # 执行比较
    comparison = ComprehensiveComparison(config, model_manager, image_loader, norm_calculator)
    comparison.analyze(config1, config2, topK, threshold, config.root_path + "/", block_cath)


if __name__ == "__main__":
    main()