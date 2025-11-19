import os
import torch
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
from transformers import (
    AutoTokenizer, RobertaForSequenceClassification, Trainer, TrainingArguments, EarlyStoppingCallback
)
from datasets import Dataset
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.utils.class_weight import compute_class_weight
from sklearn.model_selection import train_test_split
import torch.nn.functional as F
from transformers import RobertaConfig, RobertaForSequenceClassification

# 1. 讀取資料
df = pd.read_csv("Gone With The Wind/Gone_With_The_Wind_actor_train.csv")
file = "Gone_With_The_Wind_actor"
MODEL = "roberta-large"
# 設定要過濾的情緒（可空）
#emotion_labels = ["neutral", "fear", "angry", "happy", "sad", "surprise", "disgust"] 
#下面的remove_emotion可以設定把其中幾個移除
remove_emotion1 = ["fear", "disgust", "surprise"]  # 若不想過濾就設為 []
remove_emotion2 = ["fear", "disgust", "surprise"]  # 若不想過濾就設為 []



#  根據設定過濾資料
if remove_emotion1:
    df = df[~df["emotion1"].isin(remove_emotion1)]
if remove_emotion2:
    df = df[~df["emotion2"].isin(remove_emotion2)]
# 自動建立保留的情緒列表（根據 emotion2）
emotion_labels = sorted(df["emotion2"].unique().tolist())


# 2. 編碼標籤
emotion_encoder = LabelEncoder()
emotion_encoder.fit(emotion_labels)
df["emotion2_label"] = emotion_encoder.transform(df["emotion2"])

# 3. 分層切分資料集
train_df, test_df = train_test_split(df, test_size=0.2, stratify=df["emotion2_label"], random_state=42)
train_dataset = Dataset.from_pandas(train_df)
test_dataset = Dataset.from_pandas(test_df)

# 4. 初始化 tokenizer
MODEL_NAME = f"{MODEL}"
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

# 5. 處理輸入格式
def preprocess_function(examples):
    memory_map = {0: "sensory-memory", 1: "short-term memory", 2: "long-term memory"}
    inputs = [
        f"The character initially felt {e}. After hearing the dialogue — '{d}' — which was processed as a {memory_map[m]}, they displayed a new emotion. Choose from: neutral, fear, angry, happy, sad, surprise, disgust."
        for e, m, d in zip(examples["emotion1"], examples["memory"], examples["dialogue"])
    ]
    encodings = tokenizer(inputs, padding="max_length", truncation=True, max_length=256)
    encodings["labels"] = examples["emotion2_label"]
    return encodings

train_dataset = train_dataset.map(preprocess_function, batched=True)
test_dataset = test_dataset.map(preprocess_function, batched=True)

# 6. 計算 class weights
class_weights = compute_class_weight(
    class_weight="balanced",
    classes=np.unique(df["emotion2_label"]),
    y=df["emotion2_label"]
)
class_weights_tensor = torch.tensor(class_weights, dtype=torch.float)

# 7. 自定義 Trainer
class WeightedTrainer(Trainer):
    def __init__(self, *args, label_smoothing=0.1, class_weights=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.label_smoothing = label_smoothing
        self.class_weights = class_weights.to(self.args.device)

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        labels = inputs.pop("labels").to(torch.long)
        outputs = model(**inputs)
        logits = outputs.logits
        num_classes = logits.size(-1)

        with torch.no_grad():
            true_dist = torch.zeros_like(logits)
            true_dist.fill_(self.label_smoothing / (num_classes - 1))
            true_dist.scatter_(1, labels.unsqueeze(1), 1.0 - self.label_smoothing)

        log_probs = torch.nn.functional.log_softmax(logits, dim=-1)
        weights = self.class_weights[labels].unsqueeze(1)
        loss = torch.nn.functional.kl_div(log_probs, true_dist, reduction="none").sum(dim=1)
        weighted_loss = (loss * weights.squeeze()).mean()

        return (weighted_loss, outputs) if return_outputs else weighted_loss

# 8. 評估指標
def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)
    return {
        "accuracy": accuracy_score(labels, preds),
        "f1": f1_score(labels, preds, average="weighted"),
        "precision": precision_score(labels, preds, average="weighted", zero_division=0),
        "recall": recall_score(labels, preds, average="weighted", zero_division=0),
    }

# 9. 訓練參數
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
output_dir = f"./saved_model/{MODEL}/{file}/saved_model_{file}_{timestamp}"

training_args = TrainingArguments(
    output_dir=output_dir,
    evaluation_strategy="epoch",
    save_strategy="epoch",
    metric_for_best_model="f1",
    greater_is_better=True,
    load_best_model_at_end=True,
    save_total_limit=1,
    learning_rate=3e-6,
    gradient_accumulation_steps=2,
    per_device_train_batch_size=4,
    per_device_eval_batch_size=4,
    num_train_epochs=12,
    warmup_ratio=0.2,
    weight_decay=0.05,
    max_grad_norm=0.8,  # ✅ 調整為更穩定
    lr_scheduler_type="cosine",
    fp16=True,
    logging_dir="./logs",
    logging_steps=10,
)

# 10. 初始化模型與 Trainer
model = RobertaForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=len(emotion_labels))

trainer = WeightedTrainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=test_dataset,
    compute_metrics=compute_metrics,
    tokenizer=tokenizer,
    callbacks=[EarlyStoppingCallback(early_stopping_patience=3)],
    class_weights=class_weights_tensor,
)

# 11. 訓練並保存模型
train_result = trainer.train()
best_model_path = trainer.state.best_model_checkpoint
model.save_pretrained(best_model_path)
tokenizer.save_pretrained(best_model_path)
print(f"✅ 訓練完成，最佳模型已保存至 {best_model_path}")

# 12. 繪圖
log_history = trainer.state.log_history
epochs, eval_acc, eval_f1, eval_precision, eval_recall = [], [], [], [], []

for entry in log_history:
    if "eval_accuracy" in entry:
        epochs.append(entry["epoch"])
        eval_acc.append(entry["eval_accuracy"])
        eval_f1.append(entry["eval_f1"])
        eval_precision.append(entry["eval_precision"])
        eval_recall.append(entry["eval_recall"])

max_acc = max(eval_acc)
max_f1 = max(eval_f1)
max_precision = max(eval_precision)
max_recall = max(eval_recall)
print(f"📊 最佳指標表現：")
print(f"✅ Accuracy:  {max_acc:.4f}")
print(f"✅ F1 Score:  {max_f1:.4f}")
print(f"✅ Precision: {max_precision:.4f}")
print(f"✅ Recall:    {max_recall:.4f}")

plt.figure(figsize=(10, 6))
plt.plot(epochs, eval_acc, label="Accuracy", marker='o')
plt.plot(epochs, eval_f1, label="F1 Score", marker='s')
plt.plot(epochs, eval_precision, label="Precision", marker='^')
plt.plot(epochs, eval_recall, label="Recall", marker='v')
plt.xlabel("Epoch")
plt.ylabel("Score")
plt.title("Validation Metrics per Epoch")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()


