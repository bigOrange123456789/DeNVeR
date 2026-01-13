import os
import shutil
def get_subfolder_names(folder_path):
    """
    获取指定文件夹中的所有子文件夹名称
    :param folder_path: 指定文件夹的路径
    :return: 子文件夹名称列表
    """
    subfolder_names = []  # 用于存储子文件夹名称
    try:
        # 遍历指定文件夹中的所有项
        for item in os.listdir(folder_path):
            # 构造完整的路径
            item_path = os.path.join(folder_path, item)
            # 检查是否是文件夹
            if os.path.isdir(item_path):
                subfolder_names.append(item)
    except FileNotFoundError:
        print(f"错误：指定的路径 '{folder_path}' 不存在。")
    except PermissionError:
        print(f"错误：没有权限访问路径 '{folder_path}'。")
    except Exception as e:
        print(f"发生错误：{e}")
    
    return subfolder_names

# 示例使用
NUMBER=0
NUMBER2=0
my_list=[]
folder_path = "xca_dataset"#input("请输入文件夹路径：")
gapMin=999
for item in os.listdir(folder_path):
    path_gt_user=folder_path+"/"+item+"/ground_truth"#/CVAI-1207LAO44_CRA29"
    videoId_list=os.listdir(path_gt_user)
    for videoId in videoId_list:
        if len(videoId.split("CATH"))==1:
            path_gt_video=os.path.join(path_gt_user,videoId)
            gt_id_max=0
            # print("path_gt_video",path_gt_video)
            for name_gt_img in os.listdir(path_gt_video):
                gt_id=int(name_gt_img.split(".")[0])
                if gt_id>gt_id_max:gt_id_max=gt_id
                
            NUMBER=NUMBER+1
            source_path = os.path.join(folder_path+"/"+item+"/images",videoId)
            src_id_max=0
            for name_src_img in os.listdir(source_path):
                src_id=int(name_src_img.split(".")[0])
                if src_id>src_id_max:src_id_max=src_id
            # print(source_path)
            # print("gt_id_max",gt_id_max,"src_id_max",src_id_max)
            if src_id_max-gt_id_max<gapMin:gapMin=src_id_max-gt_id_max
            if False:
              if gapMin==2:
                print("videoId",videoId)
                exit(0)
            NUMBER2=NUMBER2+len(os.listdir(source_path))
            source_path2=source_path.split("xca_dataset/")[1]
            # print(source_path2)
            my_list.append(source_path2)
print("视频数量",NUMBER)
print("图片数量",NUMBER2)
print("gapMin",gapMin)

##############################################################################################################
def copy_folder_contents(src_folder, dest_folder):
    """
    复制一个文件夹中的所有内容到另一个文件夹
    :param src_folder: 源文件夹路径
    :param dest_folder: 目标文件夹路径
    """
    # 确保源文件夹存在
    if not os.path.exists(src_folder):
        print(f"源文件夹 {src_folder} 不存在，请检查路径！")
        return

    # 如果目标文件夹不存在，则创建它
    if not os.path.exists(dest_folder):
        os.makedirs(dest_folder)

    if os.path.isfile(src_folder): # 如果是文件，则直接复制
        src_item_path = src_folder 
        dest_item_path = dest_folder 
        shutil.copy2(src_item_path, dest_item_path)  # copy2 会保留文件的元数据
        return
    # 遍历源文件夹中的所有内容
    for item in os.listdir(src_folder):
        # 构建完整的文件/文件夹路径
        src_item_path = os.path.join(src_folder, item)
        dest_item_path = os.path.join(dest_folder, item)

        if not os.path.exists(dest_item_path):
            # 如果是文件，则直接复制
            if os.path.isfile(src_item_path):
                shutil.copy2(src_item_path, dest_item_path)  # copy2 会保留文件的元数据
            # 如果是文件夹，则递归复制
            elif os.path.isdir(src_item_path):
                shutil.copytree(src_item_path, dest_item_path)


# 示例使用
NUMBER=0
NUMBER2=0
my_list=[]
folder_path = "xca_dataset" # input("请输入文件夹路径：")
folder2_path = "xca_dataset_sim" 
for item in os.listdir(folder_path):
    path_gt_user=folder_path+"/"+item+"/ground_truth"#/CVAI-1207LAO44_CRA29"
    path_gt_user2=folder2_path+"/"+item+"/ground_truth"#/CVAI-1207LAO44_CRA29"
    # print("path_gt_user",path_gt_user)
    # print("path_gt_user2",path_gt_user2)
    copy_folder_contents(path_gt_user, path_gt_user2)
    # exit(0)
    videoId_list=os.listdir(path_gt_user)
    for videoId in videoId_list:
        if len(videoId.split("CATH"))==1:
            path_gt_video=os.path.join(path_gt_user,videoId)
            gt_id_max=0
            # print("path_gt_video",path_gt_video)
            for name_gt_img in os.listdir(path_gt_video):
                gt_id=int(name_gt_img.split(".")[0])
                if gt_id>gt_id_max:gt_id_max=gt_id
            print("gt_id_max",gt_id_max)
            
                
            NUMBER=NUMBER+1
            source_path = os.path.join(folder_path+"/"+item+"/images",videoId)
            dest_path = os.path.join(folder2_path+"/"+item+"/images",videoId)
            for i in range(gt_id_max+1+gapMin):
                imgfileName=str(i).zfill(5)+".png"
                img_path_src=os.path.join(source_path,imgfileName)
                # img_path_des=os.path.join(dest_path  ,imgfileName)
                # img_path_des=os.path.join(dest_path  ,imgfileName)
                # print(img_path_src,img_path_des)
                copy_folder_contents(img_path_src, dest_path)
            # exit(0)

            # src_id_max=0
            # for name_src_img in os.listdir(source_path):
            #     src_id=int(name_src_img.split(".")[0])
            #     if src_id>src_id_max:src_id_max=src_id
            # # print(source_path)
            # # print("gt_id_max",gt_id_max,"src_id_max",src_id_max)
            # if src_id_max-gt_id_max<gapMin:gapMin=src_id_max-gt_id_max
            # if False:
            #   if gapMin==2:
            #     print("videoId",videoId)
            #     exit(0)
            # NUMBER2=NUMBER2+len(os.listdir(source_path))
            # source_path2=source_path.split("xca_dataset/")[1]
            # # print(source_path2)
            # my_list.append(source_path2)

