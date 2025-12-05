import cv2
import os
import glob
import argparse

def images_to_video_advanced(image_folder, output_video, fps=30, 
                           file_pattern="*.jpg", start_number=0, 
                           zero_padding=5, codec='mp4v'):
    """
    增强版图片序列转视频函数
    
    参数:
    image_folder: 图片文件夹路径
    output_video: 输出视频文件路径
    fps: 帧率
    file_pattern: 文件匹配模式
    start_number: 起始编号
    zero_padding: 零填充位数
    codec: 视频编码器
    """
    
    # 生成文件模式
    pattern = f"{start_number:0{zero_padding}d}.{file_pattern.split('.')[-1]}"
    images = sorted(glob.glob(os.path.join(image_folder, file_pattern)))
    
    if not images:
        print(f"在 {image_folder} 中未找到 {file_pattern} 图片文件！")
        return False
    
    # 读取第一张图片获取尺寸
    frame = cv2.imread(images[0])
    if frame is None:
        print(f"无法读取第一张图片: {images[0]}")
        return False
        
    height, width = frame.shape[:2]
    
    # 创建视频写入器
    fourcc = cv2.VideoWriter_fourcc(*codec)
    video = cv2.VideoWriter(output_video, fourcc, fps, (width, height))
    
    if not video.isOpened():
        print("无法创建视频文件！")
        return False
    
    print(f"开始处理 {len(images)} 张图片...")
    print(f"视频尺寸: {width}x{height}, 帧率: {fps}fps")
    
    success_count = 0
    # 处理每张图片
    for i, image_path in enumerate(images):
        img = cv2.imread(image_path)
        if img is not None:
            # 确保图片尺寸一致
            if img.shape[:2] != (height, width):
                img = cv2.resize(img, (width, height))
            video.write(img)
            success_count += 1
        
        # 显示进度
        if (i + 1) % 100 == 0:
            print(f"已处理 {i + 1}/{len(images)} 张图片")
    
    # 释放资源
    video.release()
    
    print(f"处理完成！成功处理 {success_count}/{len(images)} 张图片")
    print(f"视频已保存为: {output_video}")
    return True

def main():
    parser = argparse.ArgumentParser(description='将图片序列转换为MP4视频')
    parser.add_argument('--input', '-i', required=True, help='输入图片文件夹路径')
    parser.add_argument('--output', '-o', default='output.mp4', help='输出视频文件路径')
    parser.add_argument('--fps', type=float, default=30, help='视频帧率')
    parser.add_argument('--pattern', default='*.jpg', help='文件匹配模式')
    
    args = parser.parse_args()
    
    # 调用函数
    success = images_to_video_advanced(
        image_folder=args.input,
        output_video=args.output,
        fps=args.fps,
        file_pattern=args.pattern
    )
    
    if success:
        print("转换成功！")
    else:
        print("转换失败！")

def start(videoId):
# if __name__ == "__main__":
    pathPre = "outputs/merge/"
    # videoId = "CVAI-1207LAO44_CRA29" # "CVAI-2177RAO32_CRA28"
    # 直接运行示例
    images_to_video_advanced(
        image_folder=pathPre+videoId,#"./images",
        output_video=pathPre+videoId+".mp4",#"my_video.mp4",
        fps=10#25
    )
    
    # 或者使用命令行参数
    # main()

start("CVAI-1207LAO44_CRA29")
start("CVAI-1253LAO0_CAU29")
start("CVAI-2174LAO42_CRA18")
