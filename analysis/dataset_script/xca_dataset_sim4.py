import os
import shutil
def get_subfolder_names(folder_path):
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
gapMin=999 #距离末尾的最近距离
gapPre=999 #距离0号帧的最近距离
frameIdMax=-1
frameIdMin=999
frameIdMax0=-1
frameIdMin0=999
frameIdMax1=-1
frameIdMin1=999
gapLenMin=999
gapLenMax=-1
videoMax=-1
videoMin=999
for item in os.listdir(folder_path):
    path_gt_user=folder_path+"/"+item+"/ground_truth"#/CVAI-1207LAO44_CRA29"
    videoId_list=os.listdir(path_gt_user)
    for videoId in videoId_list:
        if len(videoId.split("CATH"))==1:
            path_gt_video=os.path.join(path_gt_user,videoId)
            gt_id_max=0
            gt_id_min=999
            # print("path_gt_video",path_gt_video)
            for name_gt_img in os.listdir(path_gt_video):
                gt_id=int(name_gt_img.split(".")[0])
                if gt_id>gt_id_max:gt_id_max=gt_id
                if gt_id<gt_id_min:gt_id_min=gt_id
            if gt_id_min<gapPre:
                gapPre=gt_id_min
            # print(gt_id_min)
                
            NUMBER=NUMBER+1
            source_path = os.path.join(folder_path+"/"+item+"/images",videoId)
            l = len(os.listdir(source_path))#视频长度
            if videoMax<l:videoMax=l
            if videoMin>l:videoMin=l
            src_id_max=0
            # src_id_min=999
            for name_src_img in os.listdir(source_path):
                src_id=int(name_src_img.split(".")[0])
                if src_id>src_id_max:
                    src_id_max=src_id
                # if src_id<src_id_min:
                #     src_id_min=src_id
            # print(source_path)
            # print("gt_id_max",gt_id_max,"src_id_max",src_id_max)
            if src_id_max-gt_id_max<gapMin:
                gapMin=src_id_max-gt_id_max
            if False:
              if gapMin==2:
                print("videoId",videoId)
                exit(0)
            NUMBER2=NUMBER2+len(os.listdir(source_path))
            source_path2=source_path.split("xca_dataset/")[1]
            # print(source_path2)
            my_list.append(source_path2)


            src_gt_path = os.path.join(folder_path +"/"+item+"/ground_truth",videoId)
            for i in os.listdir(src_gt_path):
                i=i.split(".")[0]
                i=int(i)
                if i>frameIdMax:frameIdMax=i
                if i<frameIdMin:frameIdMin=i
            if True:
                i0=os.listdir(src_gt_path)[0]
                i0=i0.split(".")[0]
                i0=int(i0)
                if i0>frameIdMax0:frameIdMax0=i0
                if i0<frameIdMin0:frameIdMin0=i0
            if True:
                i1=os.listdir(src_gt_path)[-1]
                i1=i1.split(".")[0]
                i1=int(i1)
                if i1>frameIdMax1:frameIdMax1=i1
                if i1<frameIdMin1:frameIdMin1=i1
                # print(i)
            # print(os.listdir(src_gt_path))
            gapLen = i1-i0+1
            if gapLen<gapLenMin:gapLenMin=gapLen
            if gapLen>gapLenMax:gapLenMax=gapLen
            framelen=22#总帧数
            if (i1-i0+1)>framelen:
                print(videoId,"err1")
                exit(0)
            if not (i1>=framelen-1 or l>=framelen):
                print(videoId,"err2")
                exit(0)
print("framelen",framelen)
print("视频数量",NUMBER)
print("图片数量",NUMBER2)
print("gapMin",gapMin)
print("gapPre",gapPre)
print("gapLenMin",gapLenMin)
print("gapLenMax",gapLenMax)

print("关键帧编号最小最大数值:",frameIdMin,frameIdMax)
print("关键帧编号0最小最大数值:",frameIdMin0,frameIdMax0)
print("关键帧编号1最小最大数值:",frameIdMin1,frameIdMax1)
print("视频长度  最小最大数值:",videoMin,videoMax)
# exit(0)
##############################################################################################################
def copy_folder_contents(src_folder, dest_folder):
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

# 示例使用
NUMBER=0
NUMBER2=0
my_list=[]
folder_path = "xca_dataset" # input("请输入文件夹路径：")
folder2_path = "xca_dataset_sim4" 
for item in os.listdir(folder_path):
    path_gt_user=folder_path+"/"+item+"/ground_truth"#/CVAI-1207LAO44_CRA29"
    path_gt_user2=folder2_path+"/"+item+"/ground_truth"#/CVAI-1207LAO44_CRA29"
    # print("path_gt_user",path_gt_user)
    # print("path_gt_user2",path_gt_user2)
    # copy_folder_contents(path_gt_user, path_gt_user2) #完全复制测试数据集
    # exit(0)
    videoId_list=os.listdir(path_gt_user)
    for videoId in videoId_list:
        # if len(videoId.split("CATH"))==1:
            path_gt_video=os.path.join(path_gt_user,videoId)
            gt_id_max=0
            gt_id_min=999
            # print("path_gt_video",path_gt_video)
            for name_gt_img in os.listdir(path_gt_video):
                gt_id=int(name_gt_img.split(".")[0])
                if gt_id>gt_id_max:gt_id_max=gt_id
                if gt_id<gt_id_min:gt_id_min=gt_id
            # print("gt_id_max",gt_id_max)
            # print("gt_id_min",gt_id_min)

            startPos=0#起始帧位置
            # framelen=40#总帧数
            src_gt_path = os.path.join(folder_path +"/"+item+"/ground_truth",videoId)
            i1 = int(os.listdir(src_gt_path)[-1].split(".")[0])#末尾关键帧的编号
            # if i1>=framelen: startPos=i1-framelen
            if i1>=framelen: startPos=i1-framelen+1

                
            NUMBER=NUMBER+1
            if len(videoId.split("CATH"))==1:
                source_path = os.path.join(folder_path+"/"+item+"/images",videoId)
                dest_path = os.path.join(folder2_path+"/"+item+"/images",videoId)
                # startPos = gt_id_min-8 # 起点是 第一关键帧的前8帧
                # for i in range(gt_id_min-8,gt_id_max+1+gapMin):
                for i in range(startPos, startPos+framelen):
                    imgfileName=str(i).zfill(5)+".png"
                    img_path_src=os.path.join(source_path,imgfileName)
                    # img_path_des=os.path.join(dest_path  ,imgfileName)
                    # img_path_des=os.path.join(dest_path  ,imgfileName)
                    # print(img_path_src, dest_path)
                    newFileName=str(i-startPos).zfill(5)+".png"
                    if True:
                        copy_file_contents(img_path_src, dest_path,newFileName)
            src_gt_path = os.path.join(folder_path +"/"+item+"/ground_truth",videoId)
            des_gt_path = os.path.join(folder2_path+"/"+item+"/ground_truth",videoId)
            for name_gt_img in os.listdir(src_gt_path):
                gt_id=int(name_gt_img.split(".")[0])
                imgfileName = str(gt_id).zfill(5)+".png"
                newFileName = str(gt_id-startPos).zfill(5)+".png"
                img_path_src=os.path.join(src_gt_path,imgfileName)
                copy_file_contents(img_path_src, des_gt_path, newFileName)
            # exit(0)


