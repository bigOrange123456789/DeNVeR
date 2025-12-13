import os
import numpy as np
from itertools import chain
from PIL import Image
import torch
from torch.utils.data import DataLoader
import torchvision.transforms as T
import re
import torch.nn as nn

#########################################################################

import torch
import numpy as np
from collections import defaultdict

class GradientMonitor:
    def __init__(self, model):
        self.model = model
        self.gradient_stats = defaultdict(list)
        
        # 修复lambda闭包问题
        for name, param in model.named_parameters():
            if param.requires_grad:
                # 使用局部变量捕获name，避免lambda闭包问题
                param.register_hook(
                    lambda grad, n=name: self._grad_hook(grad, n)
                )
        
        log_path = f"./grad_analysis.txt"
        os.makedirs(os.path.dirname(log_path), exist_ok=True)
        self.f = open(log_path, "w", encoding="utf-8")
    def close(self):
        self.f.close()
    def _grad_hook(self, grad, name):
        """梯度hook，收集梯度统计信息"""
        if grad is not None:
            self.gradient_stats[name].append({
                'mean': grad.abs().mean().item(),
                'std': grad.std().item(),
                'norm': grad.norm().item(),
                'zero_ratio': (grad.abs() < 1e-10).float().mean().item(),  # 更精确的零值检测
                'max': grad.abs().max().item(),
                'min': grad.abs().min().item()
            })
        return grad
    
    def clear_stats(self):
        """清除历史统计"""
        self.gradient_stats.clear()
    
    def analyze_old(self, epoch, threshold=1e-7, print_all=False):
        """分析梯度健康状况
        
        Args:
            epoch: 当前epoch
            threshold: 梯度消失阈值
            print_all: 是否打印所有层（包括非layer层）
        """
        # print(f"\n=== 第 {epoch} 轮梯度分析 ===")
        
        layer_stats = {}
        all_layer_names = []
        
        for name, stats_list in self.gradient_stats.items():
            if not stats_list:
                continue
                
            latest = stats_list[-1]
            grad_mean = latest['mean']
            zero_ratio = latest['zero_ratio']
            grad_norm = latest['norm']
            grad_max = latest['max']
            grad_min = latest['min']
            
            # 尝试提取层号，支持多种命名模式
            layer = -1
            patterns = [
                r'layers\.(\d+)',        # layers.0
                r'layer(\d+)',           # layer0
                r'blocks\.(\d+)',        # blocks.0
                r'encoder\.(\d+)',       # encoder.0
                r'decoder\.(\d+)',       # decoder.0
            ]
            
            for pattern in patterns:
                match = re.search(pattern, name)
                if match:
                    layer = int(match.group(1))
                    break
            
            if layer >= 0:
                layer_stats[layer] = {
                    'name': name,
                    'mean': grad_mean,
                    'zero': zero_ratio,
                    'norm': grad_norm,
                    'max': grad_max,
                    'min': grad_min
                }
            
            all_layer_names.append((name, grad_mean, zero_ratio, grad_norm))
        
        # 按层排序输出（如果有层号）
        if True:#layer_stats:
            print("\n[按层分析]")
            vanishing_layers = []
            
            for layer in sorted(layer_stats.keys()):
                stats = layer_stats[layer]
                is_vanishing = stats['mean'] < threshold
                status = "✓" if not is_vanishing else "✗ 消失"
                
                if is_vanishing:
                    vanishing_layers.append(layer)
                
                print(f"Layer {layer:2d} ({stats['name']}):")
                print(f"  mean={stats['mean']:.2e} "
                      f"zeros={stats['zero']:.1%} "
                      f"norm={stats['norm']:.2e} {status}")
                print(f"  range=[{stats['min']:.2e}, {stats['max']:.2e}]")
            
            # 计算梯度衰减因子
            if True:#if len(layer_stats) > 1:
                decay_factors = []
                layers = sorted(layer_stats.keys())
                for i in range(1, len(layers)):
                    prev_norm = layer_stats[layers[i-1]]['norm']
                    curr_norm = layer_stats[layers[i]]['norm']
                    if prev_norm > 1e-10:
                        factor = curr_norm / prev_norm
                        decay_factors.append(factor)
                
                # print("decay_factors",decay_factors)
                # exit(0)
                if decay_factors:
                    avg_decay = np.mean(decay_factors)
                    print(f"\n平均梯度衰减因子: {avg_decay:.3f}")
                    
                    if avg_decay < 0.1:
                        print("⚠️ 严重梯度消失！每层损失90%以上梯度")
                    elif avg_decay < 0.5:
                        print("⚠️ 中度梯度消失")
                    elif avg_decay > 10:
                        print("⚠️ 可能梯度爆炸")
                    
                    # 输出逐层衰减
                    print("\n逐层梯度范数衰减:")
                    for i in range(1, len(layers)):
                        prev_norm = layer_stats[layers[i-1]]['norm']
                        curr_norm = layer_stats[layers[i]]['norm']
                        if prev_norm > 1e-10:
                            factor = curr_norm / prev_norm
                            status = "↓消失" if factor < 0.1 else "↓中度" if factor < 0.5 else "正常" if factor < 10 else "↑爆炸"
                            print(f"  Layer {layers[i-1]} → {layers[i]}: {factor:.3f} ({status})")
        
        # 输出所有参数（如果print_all为True）
        # if False:
        if print_all or not layer_stats:
            print("\n[所有参数梯度统计]")
            for name, mean, zero, norm in sorted(all_layer_names, key=lambda x: x[2], reverse=True):
                status = "消失" if mean < threshold else "正常"
                print(f"{name:40s}: mean={mean:.2e} zeros={zero:.1%} norm={norm:.2e} ({status})")
        
        # 总体统计
        if all_layer_names:
            means = [m for _, m, _, _ in all_layer_names]
            zeros = [z for _, _, z, _ in all_layer_names]
            
            print(f"\n[总体统计]")
            print(f"梯度均值范围: [{min(means):.2e}, {max(means):.2e}]")
            print(f"零梯度比例: 平均={np.mean(zeros):.1%}, 最大={max(zeros):.1%}")
            
            vanishing_count = sum(1 for m in means if m < threshold)
            print(f"梯度消失参数比例: {vanishing_count}/{len(means)} ({vanishing_count/len(means):.1%})")
        
        return vanishing_layers if 'vanishing_layers' in locals() else []
    
    def analyze_old2(self, epoch, threshold=1e-7, print_all=False):
        """分析梯度健康状况

        Args:
            epoch: 当前epoch
            threshold: 梯度消失阈值
            print_all: 是否打印所有层（包括非layer层）
        """
        # 1. 预计算：每层参数里绝对值小于 threshold 的个数
        tiny_counter = {}
        for name, param in self.model.named_parameters():
            if param.grad is None:
                continue
            g = param.grad.data.view(-1)          # 拉平
            tiny_cnt = (g.abs() < threshold).sum().item()
            total_cnt = g.numel()
            tiny_counter[name] = (tiny_cnt, total_cnt) #较小的数量,总数

        layer_stats = {}
        all_layer_names = []

        for name, stats_list in self.gradient_stats.items():
            if not stats_list:
                continue
            latest = stats_list[-1]
            grad_mean = latest['mean']
            zero_ratio = latest['zero_ratio']
            grad_norm = latest['norm']
            grad_max = latest['max']
            grad_min = latest['min']

            # 提取层号
            layer = -1
            patterns = [
                r'layers\.(\d+)',
                r'layer(\d+)',
                r'blocks\.(\d+)',
                r'encoder\.(\d+)',
                r'decoder\.(\d+)',
            ]
            for pattern in patterns:
                match = re.search(pattern, name)
                if match:
                    layer = int(match.group(1))
                    break

            if layer >= 0:
                layer_stats[layer] = {
                    'name': name,
                    'mean': grad_mean,
                    'zero': zero_ratio,
                    'norm': grad_norm,
                    'max': grad_max,
                    'min': grad_min
                }

            all_layer_names.append((name, grad_mean, zero_ratio, grad_norm))

        # 按层排序输出
        if layer_stats:
            print("\n[按层分析]")
            vanishing_layers = []

            for layer in sorted(layer_stats.keys()):
                stats = layer_stats[layer]
                is_vanishing = stats['mean'] < threshold
                status = "✓" if not is_vanishing else "✗ 消失"
                if is_vanishing:
                    vanishing_layers.append(layer)

                tiny_cnt, total_cnt = tiny_counter.get(stats['name'], (0, 1))
                tiny_ratio = tiny_cnt / max(total_cnt, 1)

                print(f"Layer {layer:2d} ({stats['name']}):")
                print(f"  mean={stats['mean']:.2e} "
                    f"zeros={stats['zero']:.1%} "
                    f"norm={stats['norm']:.2e} {status}")
                print(f"  range=[{stats['min']:.2e}, {stats['max']:.2e}] "
                    f"tiny<{threshold:.0e}: {tiny_cnt}/{total_cnt} ({tiny_ratio:.1%})")

            # 梯度衰减因子
            if len(layer_stats) > 1:
                decay_factors = []
                layers = sorted(layer_stats.keys())
                for i in range(1, len(layers)):
                    prev_norm = layer_stats[layers[i-1]]['norm']
                    curr_norm = layer_stats[layers[i]]['norm']
                    if prev_norm > 1e-10:
                        factor = curr_norm / prev_norm
                        decay_factors.append(factor)
                if decay_factors:
                    avg_decay = np.mean(decay_factors)
                    print(f"\n平均梯度衰减因子: {avg_decay:.3f}")
                    if avg_decay < 0.1:
                        print("⚠️ 严重梯度消失！每层损失90%以上梯度")
                    elif avg_decay < 0.5:
                        print("⚠️ 中度梯度消失")
                    elif avg_decay > 10:
                        print("⚠️ 可能梯度爆炸")

                    print("\n逐层梯度范数衰减:")
                    for i in range(1, len(layers)):
                        prev_norm = layer_stats[layers[i-1]]['norm']
                        curr_norm = layer_stats[layers[i]]['norm']
                        if prev_norm > 1e-10:
                            factor = curr_norm / prev_norm
                            status = "↓消失" if factor < 0.1 else "↓中度" if factor < 0.5 else "正常" if factor < 10 else "↑爆炸"
                            print(f"  Layer {layers[i-1]} → {layers[i]}: {factor:.3f} ({status})")

        # 所有参数模式
        if print_all or not layer_stats:
            print("\n[所有参数梯度统计]")
            for name, mean, zero, norm in sorted(all_layer_names, key=lambda x: x[2], reverse=True):
                tiny_cnt, total_cnt = tiny_counter.get(name, (0, 1))
                tiny_ratio = tiny_cnt / max(total_cnt, 1) #梯度较小的参数的比例
                status = "消失" if mean < threshold else "正常"
                print(f"{name:40s}: mean={mean:.2e} zeros={zero:.1%} norm={norm:.2e} tiny<{threshold:.0e}:{tiny_cnt}/{total_cnt}({tiny_ratio:.1%}) ({status})")

        # 总体统计
        if all_layer_names:
            means = [m for _, m, _, _ in all_layer_names]
            zeros = [z for _, _, z, _ in all_layer_names]
            print(f"\n[总体统计]")
            print(f"梯度均值范围: [{min(means):.2e}, {max(means):.2e}]")
            print(f"零梯度比例: 平均={np.mean(zeros):.1%}, 最大={max(zeros):.1%}")

            vanishing_count = sum(1 for m in means if m < threshold)
            print(f"梯度消失参数比例: {vanishing_count}/{len(means)} ({vanishing_count/len(means):.1%})")

        return vanishing_layers if 'vanishing_layers' in locals() else []

    def analyze(self, epoch, threshold=1e-7, print_all=False):
        """分析梯度健康状况，结果写入本地 txt 文件"""
        import os
        import re
        import numpy as np

        # log_path = f"./grad_analysis_epoch_{epoch}.txt"
        # log_path = f"./grad_analysis.txt"
        # os.makedirs(os.path.dirname(log_path), exist_ok=True)
        # f = open(log_path, "w", encoding="utf-8")

        def writeln(msg=""):
            self.f.write(msg + "\n")

        writeln(f"\n=== 第 {epoch} 轮梯度分析 ===")
        # 1. 预计算 tiny 梯度
        tiny_counter = {}
        for name, param in self.model.named_parameters():
            if param.grad is None:
                continue
            g = param.grad.data.view(-1)
            tiny_cnt = (g.abs() < threshold).sum().item()
            total_cnt = g.numel()
            tiny_counter[name] = (tiny_cnt, total_cnt)

        layer_stats = {}
        all_layer_names = []

        for name, stats_list in self.gradient_stats.items():
            if not stats_list:
                continue
            latest = stats_list[-1]
            grad_mean = latest['mean']
            zero_ratio = latest['zero_ratio']
            grad_norm = latest['norm']
            grad_max = latest['max']
            grad_min = latest['min']

            layer = -1
            patterns = [
                r'layers\.(\d+)', r'layer(\d+)', r'blocks\.(\d+)',
                r'encoder\.(\d+)', r'decoder\.(\d+)'
            ]
            for pat in patterns:
                m = re.search(pat, name)
                if m:
                    layer = int(m.group(1))
                    break

            if layer >= 0:
                layer_stats[layer] = {
                    'name': name, 'mean': grad_mean, 'zero': zero_ratio,
                    'norm': grad_norm, 'max': grad_max, 'min': grad_min
                }
            all_layer_names.append((name, grad_mean, zero_ratio, grad_norm))

        # ------------ 按层分析 ------------
        if layer_stats:
            writeln("[按层分析]")
            vanishing_layers = []
            for layer in sorted(layer_stats.keys()):
                st = layer_stats[layer]
                is_vanishing = st['mean'] < threshold
                status = "✓" if not is_vanishing else "✗ 消失"
                if is_vanishing:
                    vanishing_layers.append(layer)

                tiny_cnt, total_cnt = tiny_counter.get(st['name'], (0, 1))
                tiny_ratio = tiny_cnt / max(total_cnt, 1)

                writeln(f"Layer {layer:2d} ({st['name']}):")
                writeln(f"  mean={st['mean']:.2e}  "
                        f"zeros={st['zero']:.1%}  "
                        f"norm={st['norm']:.2e} {status}")
                writeln(f"  range=[{st['min']:.2e}, {st['max']:.2e}]  "
                        f"tiny<{threshold:.0e}: {tiny_cnt}/{total_cnt} ({tiny_ratio:.1%})")

            # 衰减因子
            if len(layer_stats) > 1:
                decay_factors = []
                layers = sorted(layer_stats.keys())
                for i in range(1, len(layers)):
                    prev = layer_stats[layers[i-1]]['norm']
                    curr = layer_stats[layers[i]]['norm']
                    if prev > 1e-10:
                        decay_factors.append(curr / prev)
                if decay_factors:
                    avg = np.mean(decay_factors)
                    writeln(f"\n平均梯度衰减因子: {avg:.3f}")
                    if avg < 0.1:
                        writeln("⚠️ 严重梯度消失！每层损失90%以上梯度")
                    elif avg < 0.5:
                        writeln("⚠️ 中度梯度消失")
                    elif avg > 10:
                        writeln("⚠️ 可能梯度爆炸")

                    writeln("\n逐层梯度范数衰减:")
                    for i in range(1, len(layers)):
                        prev = layer_stats[layers[i-1]]['norm']
                        curr = layer_stats[layers[i]]['norm']
                        if prev > 1e-10:
                            factor = curr / prev
                            status = "↓消失" if factor < 0.1 else "↓中度" if factor < 0.5 else "正常" if factor < 10 else "↑爆炸"
                            writeln(f"  Layer {layers[i-1]} → {layers[i]}: {factor:.3f} {status}")

        # ------------ 所有参数模式 ------------
        myCount=0
        if print_all or not layer_stats:
            writeln("\n[所有参数梯度统计]")
            for name, mean, zero, norm in sorted(all_layer_names, key=lambda x: x[2], reverse=True):
                tiny_cnt, total_cnt = tiny_counter.get(name, (0, 1))
                tiny_ratio = tiny_cnt / max(total_cnt, 1)
                # status = "消失" if mean < threshold else "正常"
                status = "低梯度参数超1%" if tiny_ratio > 0.40 else ""#"正常"
                if tiny_ratio > 0.40:
                    myCount = myCount +1
                writeln(f"{name:40s}: mean={mean:.2e}; zeros={zero:.1%}; norm={norm:.2e}; "
                        f"tiny<{threshold:.0e}:{tiny_cnt}/{total_cnt}\t({tiny_ratio:.1%}); \t{status}")

        # ------------ 总体统计 ------------
        if all_layer_names:
            means = [m for _, m, _, _ in all_layer_names]
            zeros = [z for _, _, z, _ in all_layer_names]
            writeln("\n[总体统计]")
            writeln(f"梯度均值范围: [{min(means):.2e}, {max(means):.2e}]")
            writeln(f"零梯度比例: 平均={np.mean(zeros):.1%}, 最大={max(zeros):.1%}")
            if False:
                vanishing_count = sum(1 for m in means if m < threshold)
                writeln(f"梯度消失参数比例: {vanishing_count}/{len(means)} ({vanishing_count/len(means):.1%})")
            writeln(f"梯度消失参数比例: {myCount}/{len(means)} ({myCount/len(means):.1%})")

        # f.close()
        return vanishing_layers if 'vanishing_layers' in locals() else []

