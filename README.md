# Emotional-AI：電影多模態情緒建模系統
訓練一個能理解角色情緒變化的 AI

本專案為我的碩士論文「訓練一個情緒型 AI」，  
透過電影中的臉部表情、語音台詞與記憶分類建立多模態情緒資料集，  
並以 Transformer（RoBERTa、DeBERTa）模型預測角色在劇情中的情緒變化。

---

## 📌 系統流程（Pipeline Overview）

本系統會自動從電影中完成以下步驟：

1. 找出角色人臉並分析表情（MTCNN + DeepFace）  
2. 將電影語音轉換成文字並取得時間軸（Whisper）  
3. 整合多模態訊息成可訓練的資料集  
4. 使用 Transformer 模型預測角色情緒變化  

---

## 🗂 資料夾說明（Folder Structure）

### 📁 **face_processing/** — 臉部偵測與表情分析
透過 MTCNN 找出角色臉部位置，再使用 DeepFace 進行角色辨識與情緒分析。  
可輸出每個影格的人臉表情（Neutral / Sad / Happy / Angry）。

---

### 📁 **speech_processing/** — 語音轉文字
使用 Whisper 將電影對白轉成文字，並輸出成 LRC / TXT / Excel 格式。  
提供台詞內容與時間軸資訊。

---

### 📁 **dataset_generation/** — 多模態資料整合
將以下資訊合併成最終可訓練的 CSV：

- 台詞文字  
- 語音時間軸  
- 對話前後的表情（emotion1 → emotion2）  
- 記憶類型（感官 / 短期 / 長期）  

形成可供 Transformer 模型使用的多模態資料。

---

### 📁 **transformer_model/** — Transformer 情緒預測模型
使用 RoBERTa 與 DeBERTa-v3-large 進行微調，模型輸入包含：

- 台詞內容  
- 角色原始情緒（emotion1）  
- 記憶類型  

模型預測：

角色聽完這句話後，情緒會變成什麼？（emotion2）

---

