from script.getSim2Ori import getSim2Ori
import shutil
sim2ori = getSim2Ori()
def my_predict(userId, videoId):
    folderName = "A.mask.main_nr2" # "A.mask_nr2"
    path_source = os.path.join("xca_dataset",      userId, "decouple", videoId, folderName,"filter")
    path_target = os.path.join("xca_dataset_sim2", userId, "decouple", videoId, folderName,"filter")
    os.makedirs(path_target, exist_ok=True)
    sim2ori0 = sim2ori[videoId]
    for i in range(sim2ori0["len"]):
        i1 = i +sim2ori0["start"]
        i2 = i
        name1 = str(i1).zfill(5)+".png"
        name2 = str(i2).zfill(5)+".png"
        shutil.copy2(
            os.path.join(path_source, name1), 
            os.path.join(path_target, name2))  
    ##################################################
    path_source = os.path.join("xca_dataset",      userId, "decouple", videoId, folderName,"binary")
    path_target = os.path.join("xca_dataset_sim2", userId, "decouple", videoId, folderName,"binary")
    os.makedirs(path_target, exist_ok=True)
    sim2ori0 = sim2ori[videoId]
    for i in range(sim2ori0["len"]):
        i1 = i +sim2ori0["start"]
        i2 = i
        name1 = str(i1).zfill(5)+".png"
        name2 = str(i2).zfill(5)+".png"
        shutil.copy2(
            os.path.join(path_source, name1), 
            os.path.join(path_target, name2))  
def my(userId, videoId):
    folderName = "recon_non2"#"A.rigid.main_non1" #"A.mask.main_nr2" # "A.mask_nr2"
    path_source = os.path.join("xca_dataset",      userId, "decouple", videoId, folderName)
    path_target = os.path.join("xca_dataset_sim2", userId, "decouple", videoId, folderName)
    os.makedirs(path_target, exist_ok=True)
    sim2ori0 = sim2ori[videoId]
    for i in range(sim2ori0["len"]):
        i1 = i +sim2ori0["start"]
        i2 = i
        name1 = str(i1).zfill(5)+".png"
        name2 = str(i2).zfill(5)+".png"
        shutil.copy2(
            os.path.join(path_source, name1), 
            os.path.join(path_target, name2))  
import os
def traverse(folder_path = "xca_dataset_sim2"):
    for userId in os.listdir(folder_path):
        path = os.path.join(folder_path, userId, "ground_truth")
        for videoId in os.listdir(path):
            if len(videoId.split("CATH"))==1:
                my(userId, videoId) #source_path = os.path.join(path,videoId) #print("source_path",source_path)
if __name__ == "__main__":
    traverse()
