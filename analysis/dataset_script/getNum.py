import os

folder_path = "xca_dataset" 
folder2_path = "xca_dataset_copy" 
len1=0
len2=0
for userId in os.listdir(folder_path):
    path_gt_user=folder_path+"/"+userId+"/ground_truth"#/CVAI-1207LAO44_CRA29"
    path_gt_user2=folder2_path+"/"+userId+"/ground_truth"#/CVAI-1207LAO44_CRA29"
    videoId_list=os.listdir(path_gt_user)
    for videoId in videoId_list:
            path_gt_video=os.path.join(path_gt_user,videoId)
            if len(videoId.split("CATH"))==1:
                source_path = os.path.join(folder_path+"/"+userId+"/images",videoId) #图像
                len1=len1+len(os.listdir(source_path))
                source_path = os.path.join(folder_path+"/"+userId+"/ground_truth",videoId) #标签
                len2=len2+len(os.listdir(source_path))
print("len1",len1)
print("len2",len2)
                


