import os
import cv2
import csv
import torch
import pandas as pd
from deepface import DeepFace
from mtcnn import MTCNN

# 設定裝置
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 設定電影名稱（只要改這裡就好）
movie_name = "MoviePowderPresentsHisGirlFriday_512kb"

# 設定路徑
face_dir = f"labeled/{movie_name}"  # 含 _man 和 _woman 的角色資料夾
detect_folder = f"output_picture2/{movie_name}"
output_file_men = f"csv/{movie_name}_men.csv"
output_file_women = f"csv/{movie_name}_women.csv"

# 建立角色性別對應表
man_names = [f for f in os.listdir(face_dir) if f.endswith("_man")]
woman_names = [f for f in os.listdir(face_dir) if f.endswith("_woman")]

# 收集圖片檔案
jpg_files = [f for f in os.listdir(detect_folder) if f.lower().endswith('.jpg')]

# 初始化人臉偵測器
detector = MTCNN()
csv_list_men = []
csv_list_women = []

# 處理每張圖片
for filename in jpg_files:
    img_path = os.path.join(detect_folder, filename)
    img = cv2.imread(img_path)

    if img is None:
        continue

    faces = detector.detect_faces(img)
    if not faces:
        continue

    for result in faces:
        x, y, w, h = result['box']
        crop_img = img[y:y + h, x:x + w]
        crop_img = cv2.resize(crop_img, (160, 160))

        # 人臉比對辨識人物
        try:
            df = DeepFace.find(img_path=crop_img, db_path=face_dir, enforce_detection=False,
                               detector_backend='retinaface', model_name='Facenet512')
            df_first_element = pd.DataFrame(df[0])
        except ValueError as e:
            print(f"Error analyzing face: {e}")
            continue

        if df_first_element.empty:
            continue

        name = df_first_element.loc[0].values[0].split('\\')[-2]

        # 表情偵測
        try:
            emotion = DeepFace.analyze(crop_img, actions=['emotion'], enforce_detection=False)
            macro_emotion = emotion[0]['dominant_emotion']
        except Exception as e:
            print(f"Error detecting emotion: {e}")
            continue

        row = [name, macro_emotion, filename]

        if name in man_names:
            csv_list_men.append(row)
        elif name in woman_names:
            csv_list_women.append(row)

# 儲存 CSV
def save_csv(path, data):
    with open(path, mode='w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(["person", "emotion", "time"])
        writer.writerows(data)

save_csv(output_file_men, csv_list_men)
save_csv(output_file_women, csv_list_women)

print(f"男角色 CSV 已儲存至 {output_file_men}，共 {len(csv_list_men)} 筆資料")
print(f"女角色 CSV 已儲存至 {output_file_women}，共 {len(csv_list_women)} 筆資料")