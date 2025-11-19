import pandas as pd
import re
import os

# 設定檔案名稱
movie_name = "Gone With The Wind Part 2"
#設定一秒的張數
fps = 4

# 檔案路徑
csv_file_men = f"../csv/{movie_name}_men.csv"
csv_file_women = f"../csv/{movie_name}_women.csv"
txt_file = f"../Text2emotion/output/{movie_name}.txt"
output_csv_men = f"csv/{movie_name}_men.csv"
output_csv_women = f"csv/{movie_name}_women.csv"

# 時間轉換
def time_to_seconds(time_str):
    minutes, seconds = map(float, time_str.split(':'))
    return int(minutes * 60 + seconds)

# 解析 TXT 對話
dialogues = []
with open(txt_file, "r", encoding="utf-8") as file:
    for line in file:
        match = re.match(r"\[(\d{2}:\d{2}\.\d{2}) --> (\d{2}:\d{2}\.\d{2})\] (Speaker_\d+): (.+)", line)
        if match:
            start_time, end_time, speaker, dialogue = match.groups()
            dialogues.append((time_to_seconds(start_time), time_to_seconds(end_time), speaker, dialogue))

# 處理 men/women
def process_emotion_csv(csv_path, output_path):
    csv_data = pd.read_csv(csv_path)
    csv_data['time'] = csv_data['time'].str.replace('.jpg', '', regex=False).astype(int) / fps
    output_data = []

    for start_seconds, end_seconds, speaker, dialogue in dialogues:
        matching_rows = csv_data[(csv_data['time'] >= start_seconds) & (csv_data['time'] <= end_seconds)]
        if matching_rows.empty:
            continue

        first_row = matching_rows.iloc[0]
        last_row = matching_rows.iloc[-1]

        emotion1_data = csv_data[(csv_data['time'] >= start_seconds - 3) & (csv_data['time'] < start_seconds)]
        emotion1_data = emotion1_data.sort_values(by='time', ascending=False)
        emotion1 = emotion1_data.iloc[0]['emotion'] if not emotion1_data.empty else first_row['emotion']
        time1 = emotion1_data.iloc[0]['time'] if not emotion1_data.empty else first_row['time']

        emotion2_data = csv_data[(csv_data['time'] > end_seconds) & (csv_data['time'] <= end_seconds + 5)]
        emotion2_data = emotion2_data.sort_values(by='time', ascending=True)
        emotion2 = emotion2_data.iloc[0]['emotion'] if not emotion2_data.empty else last_row['emotion']
        time2 = emotion2_data.iloc[0]['time'] if not emotion2_data.empty else last_row['time']

        output_data.append({
            "person": first_row['person'],
            "emotion1": emotion1,
            "time1": time1,
            "dialogue": dialogue,
            "emotion2": emotion2,
            "time2": time2
        })

    output_df = pd.DataFrame(output_data)
    output_df.to_csv(output_path, index=False, encoding='utf-8')
    print(f"✅ 已輸出 {output_path}，共 {len(output_df)} 筆對話")

# 執行
process_emotion_csv(csv_file_men, output_csv_men)
process_emotion_csv(csv_file_women, output_csv_women)
