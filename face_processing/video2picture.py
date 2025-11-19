import cv2
import os

# 設定電影名稱
movie_name = "Gone With The Wind Part 2"
interval = 0.25  # 每幾秒擷取一張

input_video = f'Text2emotion/test_video/{movie_name}.mp4'
output_dir = f'output_picture2/{movie_name}'

# 建立或清空輸出資料夾
if not os.path.exists(output_dir):
    os.makedirs(output_dir)
else:
    for file_name in os.listdir(output_dir):
        file_path = os.path.join(output_dir, file_name)
        if os.path.isfile(file_path):
            os.remove(file_path)

# 開啟影片
vidcap = cv2.VideoCapture(input_video)
fps = vidcap.get(cv2.CAP_PROP_FPS)
total_frames = int(vidcap.get(cv2.CAP_PROP_FRAME_COUNT))
duration = total_frames / fps
print(f"🎬 影片長度: {duration:.2f} 秒, FPS: {fps:.2f}")

frame_count = 0
img_count = 1
next_capture_time = 0.0  # 下一次該擷取的時間點

while True:
    success, frame = vidcap.read()
    if not success:
        break

    # 影片目前時間（根據幀數 & fps 推算）
    time_sec = frame_count / fps

    if time_sec >= next_capture_time:
        save_path = os.path.join(output_dir, f"{img_count}.jpg")
        cv2.imwrite(save_path, frame)

        error = time_sec - next_capture_time
        print(f"📸 擷取第 {img_count} 張圖：實際時間 {time_sec:.2f}s，預期 {next_capture_time:.2f}s，誤差 {error:.3f}s")
        
        img_count += 1
        next_capture_time += interval  # 下一次擷取時間

    frame_count += 1

vidcap.release()
print("擷取完成")
