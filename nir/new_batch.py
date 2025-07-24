import os
from nir.new import startDecouple1,startDecouple2

if __name__ == "__main__":
    datasetPath="../DeNVeR_in/xca_dataset"
    paramPath = "../DeNVeR_in/models_config/freecos_Seg.pt"
    patient_names = [name for name in os.listdir(datasetPath)
                 if os.path.isdir(os.path.join(datasetPath, name))]
    CountSum = 0
    for patientID in patient_names:
        patient_path = os.path.join(datasetPath, patientID, "images")
        video_names = [name for name in os.listdir(patient_path)
                     if os.path.isdir(os.path.join(patient_path, name))]
        CountSum = CountSum + len(video_names)
    CountI = 0
    for patientID in patient_names:
        patient_path = os.path.join(datasetPath, patientID, "images")
        video_names = [name for name in os.listdir(patient_path)
                     if os.path.isdir(os.path.join(patient_path, name))]
        for videoId in video_names:
            CountI = CountI + 1
            print(str(CountI)+"/"+str(CountSum),videoId)
            inpath = os.path.join(datasetPath, patientID, "images", videoId)
            outpath = os.path.join(datasetPath, patientID, "decouple", videoId)
            os.makedirs(outpath, exist_ok=True)
            startDecouple1(videoId, paramPath, inpath, outpath)  # 去除刚体层
            startDecouple2(videoId, paramPath, inpath, outpath)  # 获取流体层