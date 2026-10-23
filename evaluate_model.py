
import os
import pandas as pd
import torch
import matplotlib.pyplot as plt

from transformers import AutoTokenizer, AutoModelForSequenceClassification
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay,
)

# --------------------------------------------------
# CONFIGURATION
# --------------------------------------------------
MODEL_CHECKPOINT = "./checkpoints"
TEST_CSV = "./test.csv"
MAX_LENGTH = 512

LABELS = {
    0: "Human-written",
    1: "Machine-generated",
}

# Same language prefixes as your Gradio application
def add_language_prefix(language, code):
    prefixes = {
        "python": "# language: python\n",
        "c++": "// language: cpp\n",
        "java": "// language: java\n",
    }

    return prefixes.get(str(language).lower(), "") + str(code).strip()


# --------------------------------------------------
# LOAD MODEL
# --------------------------------------------------
if not os.path.isdir(MODEL_CHECKPOINT):
    raise FileNotFoundError(
        f"Fine-tuned model not found at {MODEL_CHECKPOINT}. "
        "Train the model first and save its weights there. "
        "Do not evaluate randomly initialized classification weights."
    )

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

tokenizer = AutoTokenizer.from_pretrained(MODEL_CHECKPOINT)
model = AutoModelForSequenceClassification.from_pretrained(
    MODEL_CHECKPOINT
)
model.to(device)
model.eval()

print("Model loaded on:", device)


# --------------------------------------------------
# LOAD LABELED TEST DATA
# Required CSV columns: code, label, language
# label 0 = Human, label 1 = Machine-generated
# --------------------------------------------------
df = pd.read_csv(TEST_CSV)

required_columns = {"code", "label", "language"}
missing = required_columns - set(df.columns)

if missing:
    raise ValueError(f"Missing CSV columns: {missing}")

df = df.dropna(subset=["code", "label", "language"]).copy()
df = df[df["code"].astype(str).str.strip() != ""]

df["label"] = pd.to_numeric(df["label"], errors="raise").astype(int)

if not set(df["label"].unique()).issubset({0, 1}):
    raise ValueError("Labels must be 0 (Human) or 1 (Machine-generated).")

if set(df["label"].unique()) != {0, 1}:
    raise ValueError("Test set must contain examples from both classes.")

y_true = df["label"].tolist()
y_pred = []
probabilities_list = []

# --------------------------------------------------
# PREDICTION
# --------------------------------------------------
with torch.inference_mode():
    for row in df.itertuples(index=False):
        text = add_language_prefix(row.language, row.code)

        inputs = tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            max_length=MAX_LENGTH,
        )
        inputs = {k: v.to(device) for k, v in inputs.items()}

        logits = model(**inputs).logits
        probs = torch.softmax(logits, dim=-1)[0]

        prediction = int(torch.argmax(probs).item())

        y_pred.append(prediction)
        probabilities_list.append(probs.cpu().tolist())


# --------------------------------------------------
# METRICS
# --------------------------------------------------
accuracy = accuracy_score(y_true, y_pred)

precision, recall, f1, _ = precision_recall_fscore_support(
    y_true,
    y_pred,
    labels=[0, 1],
    zero_division=0,
)

macro_p, macro_r, macro_f1, _ = precision_recall_fscore_support(
    y_true,
    y_pred,
    average="macro",
    zero_division=0,
)

print("\n========== MODEL EVALUATION ==========")
print(f"Test samples : {len(y_true)}")
print(f"Accuracy     : {accuracy:.4f} ({accuracy * 100:.2f}%)")
print(f"Macro Precision: {macro_p:.4f}")
print(f"Macro Recall   : {macro_r:.4f}")
print(f"Macro F1-score : {macro_f1:.4f}")

print("\nClass-wise report:")
print(
    classification_report(
        y_true,
        y_pred,
        labels=[0, 1],
        target_names=["Human-written", "Machine-generated"],
        digits=4,
        zero_division=0,
    )
)

# --------------------------------------------------
# CONFUSION MATRIX
# --------------------------------------------------
cm = confusion_matrix(y_true, y_pred, labels=[0, 1])

display = ConfusionMatrixDisplay(
    confusion_matrix=cm,
    display_labels=["Human-written", "Machine-generated"],
)

display.plot(cmap="Blues", values_format="d")
plt.title("Code Detector - Confusion Matrix")
plt.tight_layout()
plt.savefig("confusion_matrix.png", dpi=300)
plt.close()

# --------------------------------------------------
# SAVE PREDICTIONS
# --------------------------------------------------
df["predicted_label"] = y_pred
df["predicted_class"] = [
    LABELS[p] for p in y_pred
]
df["correct"] = [
    actual == predicted
    for actual, predicted in zip(y_true, y_pred)
]
df["human_probability"] = [
    p[0] for p in probabilities_list
]
df["machine_probability"] = [
    p[1] for p in probabilities_list
]

df.to_csv("evaluation_predictions.csv", index=False)

# Save a metrics summary for your report
with open("evaluation_metrics.txt", "w") as f:
    f.write(f"Test samples: {len(y_true)}\n")
    f.write(f"Accuracy: {accuracy:.4f}\n")
    f.write(f"Macro Precision: {macro_p:.4f}\n")
    f.write(f"Macro Recall: {macro_r:.4f}\n")
    f.write(f"Macro F1-score: {macro_f1:.4f}\n\n")
    f.write(
        classification_report(
            y_true,
            y_pred,
            labels=[0, 1],
            target_names=["Human-written", "Machine-generated"],
            digits=4,
            zero_division=0,
        )
    )

print("\nSaved files:")
print("- confusion_matrix.png")
print("- evaluation_predictions.csv")
print("- evaluation_metrics.txt")
