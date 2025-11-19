# NLP Model / Transformers 模型訓練

This folder contains scripts for fine-tuning transformer-based models
(RoBERTa and DeBERTa) for emotion prediction based on dialogue context
and multimodal features.

Features included:
- Preprocessing structured CSV datasets
- Encoding text using HuggingFace Transformers
- Fine-tuning Microsoft DeBERTa-v3-large
- Evaluation of emotion prediction performance

此資料夾包含用於對 Transformer 模型
（RoBERTa、DeBERTa）進行情緒分類訓練的腳本。

功能內容：
- 處理整合後的多模態 CSV 資料
- 使用 HuggingFace Transformers 進行文字編碼
- 微調 Microsoft DeBERTa-v3-large 模型
- 執行情緒預測結果評估（Accuracy、F1-score 等）
