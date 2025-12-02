# import os
import cv2
import numpy as np
from pathlib import Path
# import argparse
# import sys

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

def parse_root_paths(root_paths_config):
    """
    解析root_paths配置，支持以下格式：
    1. 字符串路径：直接作为方法路径
    2. add(path1, path2)：两个路径的图片相加
    3. sub(path1, path2)：两个路径的图片相减
    ...
    """
    parsed_methods = []
    
    for item in root_paths_config:
        if isinstance(item, str):
            # 单个路径
            parsed_methods.append({
                'type': 'single',
                'name': Path(item).name,
                'paths': [item]
            })
        elif isinstance(item, dict):
            # 已经解析过的字典格式
            parsed_methods.append(item)
        else:
            # 假设是字符串，尝试解析add()或sub()格式
            item_str = str(item).strip()
            # 默认为单个路径
            parsed_methods.append({
                'type': 'single',
                'name': Path(item_str).name,
                'paths': [item_str]
            })
    
    return parsed_methods


def add_frame_info_to_image(image, frame_id, video_id):
    """
    在图片左上角添加帧信息
    第一行: Frame: frame_id
    第二行: Video: video_id
    """
    # 复制图片以避免修改原图
    img_with_text = image.copy()
    
    # 设置字体和颜色
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.6
    thickness = 2
    color = (255, 255, 255)  # 白色文字
    text_color_bg = (0, 0, 0)  # 黑色背景
    
    # 创建两行文字
    text_line1 = f"Frame: {frame_id}"
    text_line2 = f"Video: {video_id}"
    
    # # 计算文字大小
    # text_size1 = cv2.getTextSize(text_line1, font, font_scale, thickness)[0]
    # text_size2 = cv2.getTextSize(text_line2, font, font_scale, thickness)[0]
    
    # # 获取最大宽度
    # text_width = max(text_size1[0], text_size2[0])
    # text_height = text_size1[1] + text_size2[1] + 5  # 5像素的行间距
    
    # # 计算背景矩形位置
    # padding = 5
    # bg_x1, bg_y1 = padding, padding
    # bg_x2, bg_y2 = padding + text_width + 2*padding, padding + text_height + 2*padding
    
    # # 添加半透明背景
    # overlay = img_with_text.copy()
    # cv2.rectangle(overlay, (bg_x1, bg_y1), (bg_x2, bg_y2), text_color_bg, -1)
    
    # # 设置透明度并合并
    # alpha = 0.7
    # img_with_text = cv2.addWeighted(overlay, alpha, img_with_text, 1-alpha, 0)
    
    # # 添加第一行文字
    # text_y1 = padding + text_size1[1] + 2
    # cv2.putText(img_with_text, text_line1, (padding+2, text_y1), font, font_scale, color, thickness)
    
    # # 添加第二行文字
    # text_y2 = text_y1 + text_size2[1] + 5
    # cv2.putText(img_with_text, text_line2, (padding+2, text_y2), font, font_scale, color, thickness)
    
    # return img_with_text

def load_and_process_image(method_info, video_id, frame_id, base_width, base_height):
    """
    根据方法信息加载并处理图片
    """
    method_type = method_info['type']
    method_name = method_info['name']
    paths = method_info['paths']
    
    # 加载原始图片
    images = []
    for path in paths:
        frame_file = find_frame_file(Path(path) / video_id, frame_id)
        if frame_file and frame_file.exists():
            img = cv2.imread(str(frame_file))
            # print(img.shape)
            # print(type(img))
            # exit(0)
            img = img.astype(np.float16)/255
            if img is not None:
                # 调整到基准尺寸
                if img.shape[:2] != (base_height, base_width):
                    img = cv2.resize(img, (base_width, base_height))
                # img2 = add_frame_info_to_image(img, frame_id, video_id)
                images.append(img)
            else:
                return None, f"read failed: {Path(path).name}"
        else:
            return None, f"not found: {Path(path).name}"
    
    # 根据方法类型处理图片
    result=None
    if method_type == 'single':
        if images:
            result = images[0]

    elif method_type == 'add':
        if len(images) == 2:
            result = cv2.addWeighted(images[0], 1.0, images[1], 1.0, 0)

    elif method_type == 'sub':
        if len(images) == 2:
            result = images[0]- images[1]
    
    elif method_type == 'mul':
        if len(images) == 2:
            result = images[0] * images[1]

    elif method_type == 'm*0.5':
        if len(images) == 1:
            result = images[0] *0.5#* images[1]

    result = np.clip(result, 0, 1)
    result = (result*255).astype(np.uint8)
    return result, method_name

def merge_video_frames_with_layout(root_paths_config, output_path, layout):
    """根据布局合并多个视频文件夹的帧为多行多列"""
    
    # 解析方法配置
    methods = parse_root_paths(root_paths_config)
    print(methods)
    
    if not methods:
        print("错误：未找到有效的方法配置")
        return
    
    # 获取第一个方法作为基准（用于获取视频结构）
    base_path = methods[0]['paths'][0]
    video_structure = get_video_structure(base_path)
    
    if not video_structure:
        print("错误：在基准路径中未找到任何视频数据")
        return
    
    # 创建输出目录
    output_path = Path(output_path)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # 处理每个视频
    for video_id, frames in video_structure.items():
        print(f"处理视频: {video_id}, 帧数: {len(frames)}")
        
        # 创建视频输出目录
        video_output_dir = output_path / video_id
        video_output_dir.mkdir(parents=True, exist_ok=True)
        
        # 获取基准图片尺寸
        base_width, base_height = 480, 640
        for frame_info in frames:
            frame_id = frame_info['frame_id']
            frame_file = find_frame_file(Path(base_path) / video_id, frame_id)
            if frame_file and frame_file.exists():
                img = cv2.imread(str(frame_file))
                if img is not None:
                    base_height, base_width = img.shape[:2]
                    break
        
        # 处理每一帧
        for frame_info in frames:
            frame_id = frame_info['frame_id']
            
            # 根据布局构建图片矩阵
            rows = []
            for row_indices in layout:
                row_images = []
                for method_index in row_indices:
                    if method_index < len(methods):
                        method_info = methods[method_index]
                        method_name = method_info['name']
                        
                        # 加载并处理图片
                        img, processed_name = load_and_process_image(
                            method_info, video_id, frame_id, base_width, base_height
                        )
                        
                        if img is not None:
                            row_images.append(img)
                        else:
                            # 图片不存在或处理失败，创建占位图片
                            placeholder = create_placeholder_image(base_width, base_height, processed_name)
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

# 示例使用方式
def main():


    root_paths_config = [

        "outputs/xca_dataset_sub2_copy/images",#原视频
        "outputs/xca_dataset_sub2_copy/images",#原视频
        # "outputs/xca_dataset_sub1_copy/ground_truth",#真值
        "outputs/xca_dataset_sub2_result/_017_07_orig(sub2)",   
        
        "outputs/xca_dataset_sub2_decouple/A-02-e2000.rigid.main",        # 主刚体层
        # {
        #     "type": "div", 
        #     "name": "method3-method4", 
        #     "paths": [
        #         "outputs/xca_dataset_sub2_copy/images",
        #         "outputs/xca_dataset_sub2_decouple/A-02-e2000.rigid.main",
        #     ]
        # },
        "outputs/xca_dataset_sub2_decouple/A-02-e2000.rigid.main_non1",  
        "outputs/xca_dataset_sub2_result/_017_02_nr(b2000)",   

        

        # 有平滑、全部刚体层
        "outputs/xca_dataset_sub1_decouple/A-02-smooth.rigid",#全刚体层
        # {
        #     "type": "div", 
        #     "name": "method3-method4", 
        #     "paths": [
        #         "outputs/xca_dataset_sub2_copy/images",
        #         "outputs/xca_dataset_sub1_decouple/A-02-smooth.rigid",
        #     ]
        # },
        "outputs/xca_dataset_sub1_decouple/A-02-smooth.rigid.non1",#去全刚层
        "outputs/xca_dataset_sub2_result/_017_02_nr(b2000)",   
        
        "outputs/xca_dataset_sub2_decouple/Brigid",        # 两阶段刚体层（两阶段的刚体看起来比主刚体要好）
        # {
        #     "type": "div", 
        #     "name": "method3-method4", 
        #     "paths": [
        #         "outputs/xca_dataset_sub2_copy/images",
        #         "outputs/xca_dataset_sub1_decouple/Brigid",
        #     ]
        # },
        "outputs/xca_dataset_sub2_decouple/B.rigid_non", 
        "outputs/xca_dataset_sub2_result/_017_05_rigid.non(doubleStage)",   
        
        # "outputs/xca_dataset_sub2_decouple/B.soft",        # 两阶段软体层
        # "outputs/xca_dataset_sub2_decouple/B.soft_non",

        # "outputs/xca_dataset_sub2_decouple/B.soft",        # 两阶段软体层
        {
            "type": "mul", 
            "name": "method3-method4", 
            "paths": [
                'outputs/xca_dataset_sub2_decouple/B.soft',
                'outputs/xca_dataset_sub2_decouple/Brigid',
            ]
        },
        "outputs/xca_dataset_sub2_decouple/B.recon_non",
        # {
        #     "type": "m*0.5", 
        #     "name": "method3-method4", 
        #     "paths": [
        #         'outputs/xca_dataset_sub2_decouple/B.recon_non',
        #     ]
        # },
        "outputs/xca_dataset_sub2_result/_017_06_recon.non(doubleStage)",   

    ]
    layout = [
        [0, 3, 6, 9, 12],  
        [1, 4, 7, 10, 13],  
        [2, 5, 8, 11, 14],  
    ]

    
    output_path = "outputs/merge"
    
    print("开始合并视频帧...")
    print(f"方法配置: {root_paths_config}")
    print(f"输出路径: {output_path}")
    print(f"布局: {layout}")
    
    merge_video_frames_with_layout(
        root_paths_config=root_paths_config,
        output_path=output_path,
        layout=layout
    )

if __name__ == "__main__":
    main()