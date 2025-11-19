# Dataset Generation / 資料集生成

This folder contains scripts for constructing the final multimodal emotion dataset.
It integrates:
- Facial emotion data extracted from frames (from MTCNN + DeepFace)
- Dialogue text and timestamps transcribed by Whisper
- Matching emotion before/after each dialogue turn

此資料夾用於生成最終的多模態情緒資料集，整合以下來源：
- 由 MTCNN + DeepFace 擷取的影格人物情緒資料
- Whisper 轉寫的台詞內容與時間戳
- 對話發生前後的情緒變化（emotion1 → emotion2）匹配

Scripts:
- `dataset_generate.py`：整合 men/women CSV 與台詞，生成最終訓練用 CSV。
