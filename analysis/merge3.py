# import os
import cv2
import numpy as np
from pathlib import Path
# import argparse
# import sys


def create_placeholder_barChart(width, height, method_name="unknown",
                             dice=0.0, pr=0.0, recall=0.0):
    """
    返回一张 BGR 图像：白底，上方居中显示方法名，下方画三指标柱状图
    指标范围 0~1
    """
    img = np.full((height, width, 3), 255, dtype=np.uint8)

    # 1. 顶部写方法名
    font = cv2.FONT_HERSHEY_SIMPLEX
    if False:
        txt = f"method: {method_name}"
        fs, th = 0.6, 2
        (tw, thg), _ = cv2.getTextSize(txt, font, fs, th)
        cv2.putText(img, txt, ((width - tw) // 2, 30),
                    font, fs, (0, 0, 0), th, cv2.LINE_AA)

    # 2. 柱状图区域
    margin = 40          # 左右留空
    bar_w = 40           # 单柱像素宽
    gap = 30             # 柱间空隙
    bottom_y = height - 40
    max_h = height - 100  # 图最大高度

    xs = []              # 每柱中心 x
    n = 3
    total_w = n * bar_w + (n - 1) * gap
    start_x = (width - total_w) // 2
    for i in range(n):
        xs.append(start_x + i * (bar_w + gap) + bar_w // 2)

    vals = [dice, pr, recall]
    # print("vals",vals)
    colors = [(39, 174, 96),   # Dice 绿
              (33, 150, 243),  # PR   蓝
              (255, 152, 0)]   # Rec  橙
    labels = ["Dice", "Precision", "Recall"]

    # 画柱
    for x, v, c in zip(xs, vals, colors):
        h = int(v * max_h)
        top = bottom_y - h
        cv2.rectangle(img,
                      (x - bar_w // 2, top),
                      (x + bar_w // 2, bottom_y),
                      c, -1, cv2.LINE_AA)
        # 顶端写数值
        # txt_val = f"{v:.2f}"
        # (tw2, _), _ = cv2.getTextSize(txt_val, font, 0.45, 1)
        # cv2.putText(img, txt_val, (x - tw2 // 2, top - 5),
        #             font, 0.45, (0, 0, 0), 1, cv2.LINE_AA)
        font_scale = 0.7 #0.45
        txt_val = f"{v:.2f}"
        (tw2, th2), _ = cv2.getTextSize(txt_val, font, font_scale, 1)
        cv2.putText(img, txt_val,
                    (x - tw2 // 2, top - 5),
                    font, font_scale, (0, 0, 0), 1, cv2.LINE_AA)

    # 画坐标轴
    if False:cv2.line(img, (margin, bottom_y), (width - margin, bottom_y), (0, 0, 0), 2)
    # y 轴刻度
    if False:
     for y_tick in [0.0, 0.5, 1.0]:
        y_pos = bottom_y - int(y_tick * max_h)
        cv2.line(img, (margin - 5, y_pos), (margin, y_pos), (0, 0, 0), 1)
        txt = f"{y_tick:.1f}"
        (tw3, _), _ = cv2.getTextSize(txt, font, 0.4, 1)
        cv2.putText(img, txt, (margin - tw3 - 8, y_pos + 4),
                    font, 0.4, (0, 0, 0), 1, cv2.LINE_AA)

    # 底部标签
    for x, lab in zip(xs, labels):
        (tw4, _), _ = cv2.getTextSize(lab, font, 0.5, 1)
        cv2.putText(img, lab, (x - tw4 // 2, bottom_y + 20),
                    font, 0.5, (0, 0, 0), 1, cv2.LINE_AA)

    return img
def create_placeholder_image(width, height, method_name="unknown"):
    """创建占位图片显示'no data'文字"""
    # 创建黑色背景
    img = np.zeros((height, width, 3), dtype=np.uint8)+255
    
    if True:
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
        
        cv2.putText(img, text, (text_x, text_y), font, font_scale, (0, 0, 0), thickness)
    
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

tempSave={}
def getMetric(excel_path,VideoId,frameId):
    import pandas as pd
    # ========== 1. 一次性把文件读成 DataFrame ==========
    # excel_path = 'your_file.xlsx'          # 换成自己的路径
    if excel_path in tempSave:
        df = tempSave[excel_path]
    else:
        df = pd.read_excel(excel_path)         # 默认会把第一行当表头
        tempSave[excel_path] = df
    row = df[(df["frameId"] == frameId+".png") & (df["videoId"] == VideoId)]
    if row.empty:
        return None, None, None
    else:
        data = row.iloc[0][["dice", "precision", "recall"]].tolist()
        # print("data",data)
        # exit(0)
        return data[0], data[1], data[2]
    # ========== 2. 建立“联合主键”索引，保证查找 O(1) ==========
    df = df.set_index(['videoId', 'frameId'], verify_integrity=True)
    # verify_integrity=True 会自动检查 (VideoId,frameId) 是否唯一

    # ========== 3. 封装一个查询函数 ==========
    def query(VideoId, frameId):
        row = df.loc[(VideoId, frameId)]   # 精确匹配
        return float(row['dice']), float(row['precision']), float(row['recall'])

    # ========== 4. 使用示例 ==========
    # try:
    #     dice, pr, sn = query(VideoId='A001', frameId=123)
    #     print(f'dice={dice}, pr={pr}, sn={sn}')
    # except KeyError:
    #     print('(VideoId, frameId) 组合不存在')
    return query(VideoId=VideoId, frameId=frameId)

def load_and_process_image(method_info, video_id, frame_id, base_width, base_height):
    """
    根据方法信息加载并处理图片
    """
    method_type = method_info['type']
    if 'name' in method_info:
        method_name = method_info['name']
    else:
        method_name="unknown"
    paths = method_info['paths']

    # print("method_type",method_type)
    if method_type == "metric" :
        # print("video_id, frame_id",video_id, frame_id)
        dice, pr , sn =getMetric(paths[0],video_id,frame_id)
        # print("dice, pr , sn",dice, pr , sn)
        # exit(0)
        return None, {"dice":dice, "precision":pr, "recall":sn}#"dice:"+str(dice)+"pr:"+str(pr)+"sn:"+str(sn)
    
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
    
    elif method_type == 'div':
        if len(images) == 2:
            result = images[0] / ( images[1] + 10**-10 )

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
    # print(methods)
    
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
                        # method_name = method_info['name']
                        
                        # 加载并处理图片
                        img, processed_name = load_and_process_image(
                            method_info, video_id, frame_id, base_width, base_height
                        )
                        
                        # print("processed_name",
                        #       processed_name,type(processed_name),
                        #       isinstance(processed_name,str))
                        if img is not None:
                            row_images.append(img)
                        elif isinstance(processed_name,str): # or processed_name["dice"]:
                            # 图片不存在或处理失败，创建占位图片
                            placeholder = create_placeholder_image(base_width, base_height, processed_name)
                            row_images.append(placeholder)
                        else:
                            # placeholder = create_placeholder_image(base_width, base_height, processed_name)
                            # print(processed_name["dice"],processed_name["dice"] is None)
                            if processed_name["dice"] is None:
                                placeholder = create_placeholder_image(base_width, base_height, "unkonwn")
                            else:
                                placeholder = create_placeholder_barChart(base_width, base_height, method_name="unknown",
                                dice=processed_name["dice"], 
                                pr=processed_name["precision"], 
                                recall=processed_name["recall"])#"dice":dice, "precision":pr, "recall"
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



        
from analysis.merge3_json import root_paths_config, layout 
# 示例使用方式
def main():
    print(len(root_paths_config))
    
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