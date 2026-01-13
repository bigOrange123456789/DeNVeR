import os
import shutil
folderOut = "fistImg"
os.makedirs(folderOut, exist_ok=True) 
folder_path = "xca_dataset" 
for item in os.listdir(folder_path):
    path=folder_path+"/"+item+"/decouple" 
    users=os.listdir(path)
    users2=[]
    for user0 in users:
        if len(user0.split("CATH"))==1:
            path1 = os.path.join(path,user0, "recon_non2", "00000.png")
            path2 = os.path.join(folderOut, user0+".png")
            shutil.copy2(path1, path2)
            print(path1)
