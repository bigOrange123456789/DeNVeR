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

def merge_video_frames(root_paths, output_path):
    """合并多个视频文件夹的帧"""
    
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
            frame_images = []
            
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
            
            # 收集所有方法的帧
            for i, method_path in enumerate(root_paths):
                method_name = method_names[i]
                frame_file = find_frame_file(Path(method_path) / video_id, frame_id)
                
                if frame_file and frame_file.exists():
                    # 读取图片
                    img = cv2.imread(str(frame_file))
                    if img is not None:
                        frame_images.append(img)
                    else:
                        # 图片读取失败，创建占位图片
                        placeholder = create_placeholder_image(base_width, base_height, method_name)
                        frame_images.append(placeholder)
                else:
                    # 图片不存在，创建占位图片
                    placeholder = create_placeholder_image(base_width, base_height, method_name)
                    frame_images.append(placeholder)
            
            # 水平合并所有图片
            try:
                merged_frame = np.hstack(frame_images)
                
                # 保存合并后的图片
                output_file = video_output_dir / f"{frame_id}.jpg"
                cv2.imwrite(str(output_file), merged_frame)
                
            except Exception as e:
                print(f"合并帧 {frame_id} 时出错: {e}")
                continue
        
        print(f"视频 {video_id} 处理完成")

# def main():
#     parser = argparse.ArgumentParser(description='合并多个视频文件夹的帧')
#     parser.add_argument('--root_paths', nargs='+', required=True, 
#                        help='视频文件夹路径数组，第一个为原视频路径')
#     parser.add_argument('--output_path', required=True, 
#                        help='输出路径')
    
#     args = parser.parse_args()
    
#     if len(args.root_paths) < 1:
#         print("错误：至少需要提供一个视频文件夹路径")
#         sys.exit(1)
    
#     # 检查原视频路径是否存在
#     if not os.path.exists(args.root_paths[0]):
#         print(f"错误：原视频路径不存在: {args.root_paths[0]}")
#         sys.exit(1)
    
#     print("开始合并视频帧...")
#     print(f"输入路径: {args.root_paths}")
#     print(f"输出路径: {args.output_path}")
    
#     try:
#         merge_video_frames(
#             root_paths=args.root_paths,
#             output_path=args.output_path
#         )
#         print("所有视频帧合并完成！")
#     except Exception as e:
#         print(f"处理过程中发生错误: {e}")
#         sys.exit(1)

# 简化调用函数
def simple_merge():
    """简化调用示例"""
    # 定义输入路径（第一个是原视频路径）
    root_paths = [
        "/path/to/original/videos",
        "/path/to/method1/videos", 
        "/path/to/method2/videos",
        "/path/to/method3/videos"
    ]
    
    output_path = "/path/to/merged/output"
    
    # 调用合并函数
    merge_video_frames(root_paths, output_path)

# if __name__ == "__main__":
#     main()

def main():
    # parser = argparse.ArgumentParser(description='合并多个视频文件夹的帧')
    # parser.add_argument('--root_paths', nargs='+', required=True, 
    #                    help='视频文件夹路径数组，第一个为原视频路径')
    # parser.add_argument('--output_path', required=True, 
    #                    help='输出路径')
    # parser.add_argument('--no_text', action='store_true',
    #                    help='不在缺失数据位置显示文字，仅显示全黑')
    
    # args = parser.parse_args()
    
    # if len(args.root_paths) < 1:
    #     print("错误：至少需要提供一个视频文件夹路径")
    #     sys.exit(1)
    
    # # 检查输入路径是否存在
    # for path in args.root_paths:
    #     if not os.path.exists(path):
    #         print(f"错误：路径不存在: {path}")
    #         sys.exit(1)
    
    # print("开始合并视频帧...")
    # print(f"输入路径: {args.root_paths}")
    # print(f"输出路径: {args.output_path}")
    # print(f"显示文字: {not args.no_text}")
    
    # try:
    #     merge_video_frames(
    #         root_paths=args.root_paths,
    #         output_path=args.output_path,
    #         show_text=not args.no_text
    #     )
    #     print("所有视频帧合并完成！")
    # except Exception as e:
    #     print(f"处理过程中发生错误: {e}")
    #     sys.exit(1)
    
    root_paths=[
        "outputs/xca_dataset_decouple/A.rigid.main",
        "outputs/xca_dataset_copy/images",
        "outputs/xca_dataset_copy/ground_truth"#_CATH",
        # "xca_dataset_decouple/A.rigid.main_non2",
    ]
    output_path="outputs/xca_dataset_merge"



    root_paths=[
        "outputs/xca_dataset_sub1_decouple/A.rigid.main_non1",
        "outputs/xca_dataset_sub1_copy/images",
        "outputs/xca_dataset_sub1_copy/ground_truth"#_CATH",
        # "xca_dataset_decouple/A.rigid.main_non2",
    ]
    output_path="outputs/xca_dataset_sub1_merge"

    # no_text=False
    print("开始合并视频帧...")
    print(f"输入路径: {root_paths}")
    print(f"输出路径: {output_path}")
    # print(f"显示文字: {not no_text}")
    merge_video_frames(
            root_paths=root_paths,
            output_path=output_path,
            # show_text=True#not no_text
        )


if __name__ == "__main__":
    main()