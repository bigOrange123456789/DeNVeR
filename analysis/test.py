import os
from PIL import Image, ImageTk
import tkinter as tk

def load_images_from_folders(folder_paths, image_names):
    """
    从多个文件夹加载图片
    :param folder_paths: 包含图片的文件夹路径列表
    :param image_names: 图片文件名列表
    :return: 图片列表的列表，每个子列表对应一个文件夹中的图片
    """
    all_images = []
    for folder in folder_paths:
        folder_images = []
        for image_name in image_names:
            image_path = os.path.join(folder, image_name)
            if os.path.exists(image_path):
                image = Image.open(image_path)
                folder_images.append(image)
            else:
                print(f"Warning: {image_path} does not exist.")
        all_images.append(folder_images)
    return all_images

def display_images(all_images):
    """
    在tkinter窗口中显示图片
    :param all_images: 图片列表的列表
    """
    root = tk.Tk()
    root.title("Image Comparison")

    # 设置窗口大小
    window_width = 800
    window_height = 600
    root.geometry(f"{window_width}x{window_height}")

    # 计算每行的图片数量
    num_images_per_row = len(all_images[0])
    image_width = int(3*window_width // num_images_per_row)
    image_height = image_width#window_height // len(all_images)

    for row, folder_images in enumerate(all_images):
        for col, image in enumerate(folder_images):
            # 调整图片大小以适应窗口
            image = image.resize((image_width, image_height))
            photo = ImageTk.PhotoImage(image)

            # 创建一个标签来显示图片
            label = tk.Label(root, image=photo)
            label.image = photo  # 保持对PhotoImage的引用
            label.grid(row=row, column=col, padx=5, pady=5)

    root.mainloop()

if __name__ == "__main__":
    # 设置文件夹路径和图片文件名
    folder_paths = [
        # "./log/CVAI-2828RAO2_CRA32/src",
        "./log/CVAI-2828RAO2_CRA32/1.masks",
        # "./log/CVAI-2828RAO2_CRA32/2.2.planar",
        "./log/CVAI-2828RAO2_CRA32/3.parallel",
        # "./log/CVAI-2828RAO2_CRA32/4.deform",
        "./log/CVAI-2828RAO2_CRA32/5.refine",
        "./log/CVAI-2828RAO2_CRA32/gt",
    ]
    image_names = []
    for i in range(10):
        fileName=f"{(i+56):05}" + ".png"
        image_names.append(fileName)

    # 加载图片
    all_images = load_images_from_folders(folder_paths, image_names)

    # 显示图片
    display_images(all_images)


'''
export PATH="~/anaconda3/bin:$PATH"
source activate DNVR
python ./analysis/test.py
'''









