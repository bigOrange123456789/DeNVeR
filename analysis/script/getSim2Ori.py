import os
def getSim2Ori(folder_path = "xca_dataset"):
    NUMBER=0
    NUMBER2=0
    my_list=[]
    gapMin=999 #距离末尾的最近距离
    gapPre=999 #距离0号帧的最近距离
    for item in os.listdir(folder_path):
        path_gt_user=folder_path+"/"+item+"/ground_truth"#/CVAI-1207LAO44_CRA29"
        videoId_list=os.listdir(path_gt_user)
        for videoId in videoId_list:
            if len(videoId.split("CATH"))==1:
                path_gt_video=os.path.join(path_gt_user,videoId)
                gt_id_max=0
                gt_id_min=999
                for name_gt_img in os.listdir(path_gt_video):
                    gt_id=int(name_gt_img.split(".")[0])
                    if gt_id>gt_id_max:gt_id_max=gt_id
                    if gt_id<gt_id_min:gt_id_min=gt_id
                if gt_id_min<gapPre:
                    gapPre=gt_id_min
                NUMBER=NUMBER+1
                source_path = os.path.join(folder_path+"/"+item+"/images",videoId)
                src_id_max=0
                for name_src_img in os.listdir(source_path):
                    src_id=int(name_src_img.split(".")[0])
                    if src_id>src_id_max:
                        src_id_max=src_id
                if src_id_max-gt_id_max<gapMin:
                    gapMin=src_id_max-gt_id_max
                NUMBER2=NUMBER2+len(os.listdir(source_path))
                source_path2=source_path.split("xca_dataset/")[1]
                my_list.append(source_path2)
    # print("视频数量",NUMBER)
    # print("图片数量",NUMBER2)
    # print("gapMin",gapMin)
    # print("gapPre",gapPre)
    ##############################################################################################################
    sim2ori={}
    folder_path = "xca_dataset" # input("请输入文件夹路径：")
    for item in os.listdir(folder_path):
        path_gt_user=folder_path+"/"+item+"/ground_truth"#/CVAI-1207LAO44_CRA29"
        videoId_list=os.listdir(path_gt_user)
        for videoId in videoId_list:
            if len(videoId.split("CATH"))==1:
                path_gt_video=os.path.join(path_gt_user,videoId)
                gt_id_max=0
                gt_id_min=999
                for name_gt_img in os.listdir(path_gt_video):
                    gt_id=int(name_gt_img.split(".")[0])
                    if gt_id>gt_id_max:gt_id_max=gt_id
                    if gt_id<gt_id_min:gt_id_min=gt_id
                sim2ori[videoId]={
                    "start" : gt_id_min - 8,
                    "len" : gt_id_max + 1 + gapMin-(gt_id_min - 8)
                }
    return sim2ori

##############################################################################################################
import json
def save_json(data, path):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=4) 

if __name__ == "__main__":
    sim2ori = getSim2Ori()
    save_json(sim2ori, "sim2ori.json")
