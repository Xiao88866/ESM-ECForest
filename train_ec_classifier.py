import os
import pandas as pd
import numpy as np
from enzyllm_predictor import EnzyLLMPredictor
from joblib import dump
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.multiclass import OneVsRestClassifier
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.metrics import f1_score, precision_score, recall_score, classification_report
import gc
import torch

# 1. Prepare output directory for embeddings
embedding_dir = "train_data/embeddings"
os.makedirs(embedding_dir, exist_ok=True)

# 2. Load training data and predictor (using ESM2 8M)
print("Loading training data...")
df = pd.read_csv("train_data/uniprot_enzyme_binary.csv")
# Filter only enzyme sequences (is_enzyme=1) for EC classification
df = df[df['is_enzyme'] == 1].copy()
print(f"Using enzyme sequences for EC training: {len(df)} samples")

predictor = EnzyLLMPredictor(esm2_model_name="esm2_t6_8M_UR50D", device="cpu")

print("No sequence length filtering applied. Training will use every sequence provided.")

# Helper: parse EC field into list[str]
def parse_ec_list(ec_value: str):
    if pd.isna(ec_value):
        return []
    s = str(ec_value)
    # split by common delimiters
    for sep in [";", ",", "|", "/", " "]:
        s = s.replace(sep, " ")
    parts = [p.strip() for p in s.split() if len(p.strip()) > 0]
    return parts

# 3. Generate embeddings and multilabel targets
print("Generating embeddings and parsing multilabel ECs...")
batch_size = 10
embeddings = []
labels_multi: list[list[str]] = []
sample_ids: list[str] = []
ec_numbers_raw: list[str] = []

for i in range(0, len(df), batch_size):
    batch = df.iloc[i:i+batch_size]
    batch_seqs = batch['sequence'].tolist()
    batch_ecs_raw = batch['ec_number'].tolist()
    batch_ec_lists = [parse_ec_list(v) for v in batch_ecs_raw]

    try:
        batch_embs = predictor.encode_sequences(batch_seqs)
        embeddings.extend(batch_embs)
        labels_multi.extend(batch_ec_lists)
        sample_ids.extend(batch['id'].tolist())
        ec_numbers_raw.extend(batch_ecs_raw)
        print(f"Processed {min(i+batch_size, len(df))}/{len(df)} sequences")
    except RuntimeError as e:
        if "out of memory" in str(e).lower() or "cannot allocate memory" in str(e).lower():
            print(f"Memory error at batch starting {i}. Skipping this batch.")
            continue
        else:
            raise e

    del batch_embs
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

embeddings = np.array(embeddings)
print(f"Final embeddings shape: {embeddings.shape}")

# Remove samples with zero EC labels (edge cases) and warn
valid_idx = [idx for idx, lbls in enumerate(labels_multi) if len(lbls) > 0]
if len(valid_idx) < len(labels_multi):
    print(f"Warning: {len(labels_multi) - len(valid_idx)} samples had no EC labels; they will be dropped from EC training.")
embeddings = embeddings[valid_idx]
labels_multi = [labels_multi[i] for i in valid_idx]
sample_ids = [sample_ids[i] for i in valid_idx]
ec_numbers_raw = [ec_numbers_raw[i] for i in valid_idx]

# 4. Binarize multilabels
mlb = MultiLabelBinarizer()
Y = mlb.fit_transform(labels_multi)
print(f"Number of EC classes (multilabel): {len(mlb.classes_)}")

# 5. 8:1:1 split by indices (multilabel stratify is not supported by default)
print("Splitting dataset into train/val/test (8:1:1, random)...")
idx = np.arange(len(embeddings))
idx_train, idx_temp = train_test_split(idx, test_size=0.2, random_state=42, shuffle=True)
idx_val, idx_test = train_test_split(idx_temp, test_size=0.5, random_state=42, shuffle=True)

X_train, Y_train = embeddings[idx_train], Y[idx_train]
X_val, Y_val = embeddings[idx_val], Y[idx_val]
X_test, Y_test = embeddings[idx_test], Y[idx_test]

print(f"Shapes: train={X_train.shape}, val={X_val.shape}, test={X_test.shape}")

# Save embeddings and metadata for downstream use (e.g. UMAP)
ec_emb_path = os.path.join(embedding_dir, "ec_embeddings.npy")
ec_meta_path = os.path.join(embedding_dir, "ec_metadata.csv")
ec_split_path = os.path.join(embedding_dir, "ec_split.npz")
np.save(ec_emb_path, embeddings)
pd.DataFrame({
    "id": sample_ids,
    "ec_number": ec_numbers_raw,
    "ec_labels_parsed": ["; ".join(lbls) for lbls in labels_multi],
}).to_csv(ec_meta_path, index=False)
np.savez(ec_split_path, idx_train=idx_train, idx_val=idx_val, idx_test=idx_test)
print(f"Saved embeddings to {ec_emb_path} (shape {embeddings.shape})")
print(f"Saved metadata to {ec_meta_path}")
print(f"Saved train/val/test indices to {ec_split_path}")

# 6. Train One-vs-Rest RandomForest classifier
print("Training One-vs-Rest Random Forest classifier (multilabel)...")
base_rf = RandomForestClassifier(
    n_estimators=400,
    random_state=42,
    n_jobs=-1,
    max_depth=40,
    max_samples=0.8,
    min_samples_split=3,
    min_samples_leaf=1,
    max_features='sqrt',
    bootstrap=True,
)
clf = OneVsRestClassifier(base_rf)
clf.fit(X_train, Y_train)

# 7. Evaluate multilabel performance with threshold=0.7 for reporting
def evaluate_split(X, Y_true, split_name: str):
    if len(Y_true) == 0:
        print(f"{split_name}: no samples")
        return
    # predict probabilities; OneVsRest with RF supports predict_proba
    Y_prob = clf.predict_proba(X)
    Y_pred = (Y_prob >= 0.7).astype(int)
    micro_p = precision_score(Y_true, Y_pred, average='micro', zero_division=0)
    micro_r = recall_score(Y_true, Y_pred, average='micro', zero_division=0)
    micro_f1 = f1_score(Y_true, Y_pred, average='micro', zero_division=0)
    macro_p = precision_score(Y_true, Y_pred, average='macro', zero_division=0)
    macro_r = recall_score(Y_true, Y_pred, average='macro', zero_division=0)
    macro_f1 = f1_score(Y_true, Y_pred, average='macro', zero_division=0)
    print(f"{split_name} micro P/R/F1: {micro_p:.4f} / {micro_r:.4f} / {micro_f1:.4f}")
    print(f"{split_name} macro P/R/F1: {macro_p:.4f} / {macro_r:.4f} / {macro_f1:.4f}")

print("Evaluating...")
evaluate_split(X_val, Y_val, "Validation")
evaluate_split(X_test, Y_test, "Test")

# 8. Save the classifier and label binarizer (use previous paths for compatibility)
dump(clf, "train_data/ec_classifier.joblib")
dump(mlb, "train_data/ec_label_encoder.joblib")

print("Training complete. Multilabel classifier and binarizer saved.")