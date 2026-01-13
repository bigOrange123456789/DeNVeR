import os
import shutil

def copy_file_contents(src_folder, dest_folder,newName):
    # 确保源文件夹存在
    if not os.path.exists(src_folder):
        print(f"源文件夹 {src_folder} 不存在，请检查路径！")
        return
    # 如果目标文件夹不存在，则创建它
    if not os.path.exists(dest_folder):
        os.makedirs(dest_folder)
    if os.path.isfile(src_folder): # 如果是文件，则直接复制
        src_item_path = src_folder 
        dest_item_path = os.path.join( dest_folder,newName ) 
        shutil.copy2(src_item_path, dest_item_path)  # copy2 会保留文件的元数据

folder_path = "xca_dataset" 
folder2_path = "xca_dataset_copy" 
for userId in os.listdir(folder_path):
    path_gt_user=folder_path+"/"+userId+"/ground_truth"#/CVAI-1207LAO44_CRA29"
    path_gt_user2=folder2_path+"/"+userId+"/ground_truth"#/CVAI-1207LAO44_CRA29"
    videoId_list=os.listdir(path_gt_user)
    for videoId in videoId_list:
            path_gt_video=os.path.join(path_gt_user,videoId)
            
            if len(videoId.split("CATH"))==1:
                source_path = os.path.join(folder_path+"/"+userId+"/images",videoId)
                dest_path = os.path.join(folder2_path+"/images",videoId)
                for frameId in os.listdir(source_path):
                    copy_file_contents(
                        os.path.join(source_path, frameId), 
                        dest_path, 
                        frameId
                    )
                source_path = os.path.join(folder_path+"/"+userId+"/ground_truth",videoId)
                dest_path = os.path.join(folder2_path+"/ground_truth",videoId)
                for frameId in os.listdir(source_path):
                    copy_file_contents(
                        os.path.join(source_path, frameId), 
                        dest_path, 
                        frameId
                    )
                source_path = os.path.join(folder_path+"/"+userId+"/ground_truth",videoId+"CATH")
                dest_path = os.path.join(folder2_path+"/ground_truth_CATH",videoId+"CATH")
                for frameId in os.listdir(source_path):
                    copy_file_contents(
                        os.path.join(source_path, frameId), 
                        dest_path, 
                        frameId
                    )
                


