import os
import cv2
import numpy as np
from pathlib import Path
import argparse
import sys

def create_placeholder_image(width, height, method_name="unknown"):
    """创建占位图片显示'no data'文字"""
    # 创建黑色背景
    img = np.zeros((height, width, 3), dtype=np.uint8)
    
    # 添加文字
    font = cv2.FONT_HERSHEY_SIMPLEX
    text = f"no data: {method_name}"
    
    # 计算文字大小和位置
    font_scale = 0.6
    thickness = 2
    text_size = cv2.getTextSize(text, font, font_scale, thickness)[0]
    
    # 确保文字不会超出图片边界
    max_width = width - 20
    if text_size[0] > max_width:
        font_scale = font_scale * max_width / text_size[0]
        text_size = cv2.getTextSize(text, font, font_scale, thickness)[0]
    
    text_x = (width - text_size[0]) // 2
    text_y = (height + text_size[1]) // 2
    
    cv2.putText(img, text, (text_x, text_y), font, font_scale, (255, 255, 255), thickness)
    
    return img

def get_video_structure(original_path):
    """从原视频文件夹获取所有videoId和frameId的结构"""
    video_structure = {}
    original_path = Path(original_path)
    
    for video_dir in original_path.iterdir():
        if video_dir.is_dir():
            video_id = video_dir.name
            frames = []
            
            # 获取所有图片文件
            for frame_file in video_dir.glob("*.*"):
                if frame_file.suffix.lower() in ['.jpg', '.jpeg', '.png', '.bmp', '.tiff']:
                    frames.append({
                        'frame_id': frame_file.stem,
                        'extension': frame_file.suffix
                    })
            
            # 按frameId排序
            frames.sort(key=lambda x: x['frame_id'])
            video_structure[video_id] = frames
    
    return video_structure

def find_frame_file(method_path, frame_id):
    """在方法文件夹中查找指定frameId的图片文件"""
    method_path = Path(method_path)
    
    if not method_path.exists():
        return None
    
    # 尝试不同的图片格式
    for ext in ['.jpg', '.jpeg', '.png', '.bmp', '.tiff']:
        frame_file = method_path / f"{frame_id}{ext}"
        if frame_file.exists():
            return frame_file
    
    return None

def merge_video_frames_with_layout(root_paths, output_path, layout):
    """根据布局合并多个视频文件夹的帧为多行多列"""
    
    # 获取视频结构（以原视频为基准）
    original_path = root_paths[0]
    video_structure = get_video_structure(original_path)
    
    if not video_structure:
        print("错误：在原视频路径中未找到任何视频数据")
        return
    
    # 创建输出目录
    output_path = Path(output_path)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # 获取方法名称（用于显示在占位图片上）
    method_names = []
    for i, path in enumerate(root_paths):
        if i == 0:
            method_names.append("original")
        else:
            method_names.append(Path(path).name)
    
    # 处理每个视频
    for video_id, frames in video_structure.items():
        print(f"处理视频: {video_id}, 帧数: {len(frames)}")
        
        # 创建视频输出目录
        video_output_dir = output_path / video_id
        video_output_dir.mkdir(parents=True, exist_ok=True)
        
        # 处理每一帧
        for frame_info in frames:
            frame_id = frame_info['frame_id']
            
            # 首先读取原视频帧作为基准尺寸
            original_frame_file = find_frame_file(Path(original_path) / video_id, frame_id)
            if original_frame_file:
                original_img = cv2.imread(str(original_frame_file))
                if original_img is not None:
                    base_height, base_width = original_img.shape[:2]
                else:
                    base_height, base_width = 480, 640
            else:
                base_height, base_width = 480, 640
            
            # 根据布局构建图片矩阵
            rows = []
            for row_indices in layout:
                row_images = []
                for method_index in row_indices:
                    if method_index < len(root_paths):
                        method_path = root_paths[method_index]
                        method_name = method_names[method_index]
                        frame_file = find_frame_file(Path(method_path) / video_id, frame_id)
                        
                        if frame_file and frame_file.exists():
                            # 读取图片
                            img = cv2.imread(str(frame_file))
                            if img is not None:
                                row_images.append(img)
                            else:
                                # 图片读取失败，创建占位图片
                                placeholder = create_placeholder_image(base_width, base_height, method_name)
                                row_images.append(placeholder)
                        else:
                            # 图片不存在，创建占位图片
                            placeholder = create_placeholder_image(base_width, base_height, method_name)
                            row_images.append(placeholder)
                    else:
                        # 索引超出范围，创建空白占位
                        placeholder = create_placeholder_image(base_width, base_height, "invalid index")
                        row_images.append(placeholder)
                
                # 水平合并该行的所有图片
                if row_images:
                    merged_row = np.hstack(row_images)
                    rows.append(merged_row)
            
            # 垂直合并所有行
            if rows:
                try:
                    merged_frame = np.vstack(rows)
                    
                    # 保存合并后的图片
                    output_file = video_output_dir / f"{frame_id}.jpg"
                    cv2.imwrite(str(output_file), merged_frame)
                    
                except Exception as e:
                    print(f"合并帧 {frame_id} 时出错: {e}")
                    continue
        
        print(f"视频 {video_id} 处理完成")

def main():
    # 定义布局：二维数组，每个元素代表方法在root_paths中的索引
    # 例如：[[0, 1], [2, 3]] 表示：
    # 第一行：原视频(0) 和 方法1(1)
    # 第二行：方法2(2) 和 方法3(3)
    layout = [
        [0, 2, 4, 6],  # 第一行：原视频、方法1、方法2
        [1, 3, 5, 7]      # 第二行：方法3、方法4
    ]
    
    root_paths = [
        "outputs/xca_dataset_sub1_copy/images",
        "outputs/xca_dataset_sub1_copy/ground_truth",

        "outputs/xca_dataset_sub1_decouple/A.rigid.main_non1",
        "outputs/xca_dataset_sub1_result/_015_01_noRigid1",


        "outputs/xca_dataset_sub1_decouple/A-01-epoch1000.rigid.main_non1",
        "outputs/xca_dataset_sub1_result/_015_03_noRigid1(b1000)",

        "outputs/xca_dataset_sub1_decouple/A-01-epoch500.rigid.main_non1",
        "outputs/xca_dataset_sub1_result/_015_04_noRigid1(b500)",

    ]



    layout = [
        [0,1,2],
        [3,4,5],
        [6,7,8]
        # [0, 2, 4, 6],  # 第一行：原视频、方法1、方法2
        # [1, 3, 5, 7]      # 第二行：方法3、方法4
    ]
    
    root_paths = [
        "outputs/xca_dataset_sub1_copy/images",
        "outputs/xca_dataset_sub1_copy/images",
        # "outputs/xca_dataset_sub1_copy/ground_truth",
        "outputs/xca_dataset_sub1_result/_013_05_orig", # 训练批次:4000epoch

        # 无平滑
        "outputs/xca_dataset_sub1_decouple/A-01-epoch1000.rigid.main",#刚体层
        "outputs/xca_dataset_sub1_decouple/A-01-epoch1000.rigid.main_non1",#去刚层
        "outputs/xca_dataset_sub1_result/_015_03_noRigid1(b1000)",#分割结果


        # 有平滑
        "outputs/xca_dataset_sub1_decouple/A-02-smooth.rigid.main",#刚体层
        "outputs/xca_dataset_sub1_decouple/A-02-smooth.rigid.main_non1",#去刚层
        "outputs/xca_dataset_sub1_result/_016_01_noRigid1(b1000)",#分割结果

        # "outputs/xca_dataset_sub1_decouple/A-01-epoch500.rigid.main_non1",
        # "outputs/xca_dataset_sub1_result/_015_04_noRigid1(b500)",

    ]

    layout = [
        [0,1,2],
        [3,4,5],
        [6,7,8]
        # [0, 2, 4, 6],  # 第一行：原视频、方法1、方法2
        # [1, 3, 5, 7]      # 第二行：方法3、方法4
    ]
    
    root_paths = [
        "outputs/xca_dataset_sub1_copy/images",
        "outputs/xca_dataset_sub1_copy/images",
        # "outputs/xca_dataset_sub1_copy/ground_truth",
        "outputs/xca_dataset_sub1_result/_013_05_orig", # 训练批次:4000epoch

        # # 无平滑
        # "outputs/xca_dataset_sub1_decouple/A-01-epoch1000.rigid.main",#刚体层
        # "outputs/xca_dataset_sub1_decouple/A-01-epoch1000.rigid.main_non1",#去刚层
        # "outputs/xca_dataset_sub1_result/_015_03_noRigid1(b1000)",#分割结果

        # 有平滑、主刚体层
        "outputs/xca_dataset_sub1_decouple/A-02-smooth.rigid.main",#刚体层
        "outputs/xca_dataset_sub1_decouple/A-02-smooth.rigid.main_non1",#去刚层
        "outputs/xca_dataset_sub1_result/_016_01_noRigid1(b1000)",#分割结果

        # 有平滑、全部刚体层
        "outputs/xca_dataset_sub1_decouple/A-02-smooth.rigid",#全刚体层
        "outputs/xca_dataset_sub1_decouple/A-02-smooth.rigid.non1",#去全刚层
        "outputs/xca_dataset_sub1_result/_016_01_noRigidAll1(b1000)",#分割结果

    ]

    output_path = "outputs/xca_dataset_sub1_merge"


    layout = [
        [0, 2, 4, 6, 8],  
        [1, 3, 5, 7, 9],  
    ]
    
    root_paths = [

        "outputs/xca_dataset_sub2_copy/images",#原视频
        "outputs/xca_dataset_sub1_copy/ground_truth",
        
        "outputs/xca_dataset_sub2_decouple/A-02-e2000.rigid.main",        # 主刚体层
        "outputs/xca_dataset_sub2_decouple/A-02-e2000.rigid.main_non1",   

        # 有平滑、全部刚体层
        "outputs/xca_dataset_sub1_decouple/A-02-smooth.rigid",#全刚体层
        "outputs/xca_dataset_sub1_decouple/A-02-smooth.rigid.non1",#去全刚层
        
        "outputs/xca_dataset_sub2_decouple/Brigid",        # 两阶段刚体层（两阶段的刚体看起来比主刚体要好）
        "outputs/xca_dataset_sub2_decouple/B.rigid_non", 
        
        # "outputs/xca_dataset_sub2_decouple/B.soft",        # 两阶段软体层
        # "outputs/xca_dataset_sub2_decouple/B.soft_non",

        "outputs/xca_dataset_sub2_decouple/B.soft",        # 两阶段软体层
        "outputs/xca_dataset_sub2_decouple/B.recon_non",

    ]

    output_path = "outputs/xca_dataset_sub2_merge"

    # root_paths_config = [

    #     "",
    #     "outputs/xca_dataset_sub2_copy/images",#原视频
    #     # "outputs/xca_dataset_sub1_copy/ground_truth",#真值
        
    #     "outputs/xca_dataset_sub2_decouple/A-02-e2000.rigid.main",        # 主刚体层
    #     # {
    #     #     "type": "div", 
    #     #     "name": "method3-method4", 
    #     #     "paths": [
    #     #         "outputs/xca_dataset_sub2_copy/images",
    #     #         "outputs/xca_dataset_sub2_decouple/A-02-e2000.rigid.main",
    #     #     ]
    #     # },
    #     "outputs/xca_dataset_sub2_decouple/A-02-e2000.rigid.main_non1",   
        

    #     # 有平滑、全部刚体层
    #     "outputs/xca_dataset_sub1_decouple/A-02-smooth.rigid",#全刚体层
    #     # {
    #     #     "type": "div", 
    #     #     "name": "method3-method4", 
    #     #     "paths": [
    #     #         "outputs/xca_dataset_sub2_copy/images",
    #     #         "outputs/xca_dataset_sub1_decouple/A-02-smooth.rigid",
    #     #     ]
    #     # },
    #     "outputs/xca_dataset_sub1_decouple/A-02-smooth.rigid.non1",#去全刚层
        
    #     "outputs/xca_dataset_sub2_decouple/Brigid",        # 两阶段刚体层（两阶段的刚体看起来比主刚体要好）
    #     # {
    #     #     "type": "div", 
    #     #     "name": "method3-method4", 
    #     #     "paths": [
    #     #         "outputs/xca_dataset_sub2_copy/images",
    #     #         "outputs/xca_dataset_sub1_decouple/Brigid",
    #     #     ]
    #     # },
    #     "outputs/xca_dataset_sub2_decouple/B.rigid_non", 
        
    #     # "outputs/xca_dataset_sub2_decouple/B.soft",        # 两阶段软体层
    #     # "outputs/xca_dataset_sub2_decouple/B.soft_non",

    #     "outputs/xca_dataset_sub2_decouple/B.soft",        # 两阶段软体层
    #     # {
    #     #     "type": "mul", 
    #     #     "name": "method3-method4", 
    #     #     "paths": [
    #     #         'outputs/xca_dataset_sub2_decouple/B.soft',
    #     #         'outputs/xca_dataset_sub2_decouple/Brigid',
    #     #     ]
    #     # },
    #     "outputs/xca_dataset_sub2_decouple/B.recon_non",

    # ]
    # layout = [
    #     [0, 2, 4, 6, 8],  
    #     [1, 3, 5, 7, 9],  
    # ]

    print("开始合并视频帧...")
    print(f"输入路径: {root_paths}")
    print(f"输出路径: {output_path}")
    print(f"布局: {layout}")
    
    merge_video_frames_with_layout(
        root_paths=root_paths,
        output_path=output_path,
        layout=layout
    )

if __name__ == "__main__":
    main()
