import os
import pandas as pd
import numpy as np
from enzyllm_predictor import EnzyLLMPredictor
from joblib import dump
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
import gc
import torch


def train_enzyme_classifier(data_csv: str = "train_data/uniprot_enzyme_binary.csv",
                            out_path: str = "train_data/enzyme_classifier.joblib",
                            device: str = "cpu",
                            embedding_dir: str = "train_data/embeddings"):
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    os.makedirs(embedding_dir, exist_ok=True)

    print("Loading binary enzyme dataset...")
    df = pd.read_csv(data_csv)
    # Expect columns: id, sequence, is_enzyme, ec_number(optional)
    if not {"sequence", "is_enzyme"}.issubset(df.columns):
        raise ValueError("Input CSV must contain 'sequence' and 'is_enzyme' columns")

    predictor = EnzyLLMPredictor(device=device)

    print("Encoding sequences with ESM2...")
    batch_size = 16
    features, labels, sample_ids = [], [], []
    for i in range(0, len(df), batch_size):
        batch = df.iloc[i:i+batch_size]
        seqs = batch["sequence"].tolist()
        xs = predictor.encode_sequences(seqs)
        features.extend(xs)
        labels.extend(batch["is_enzyme"].tolist())
        sample_ids.extend(batch["id"].tolist())
        print(f"Processed {min(i+batch_size, len(df))}/{len(df)}")
        del xs
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    X = np.array(features)
    y = np.array(labels)

    # 8:1:1 split train, val, test
    idx = np.arange(len(X))
    idx_train, idx_temp = train_test_split(
        idx, test_size=0.2, random_state=42, stratify=y
    )
    idx_val, idx_test = train_test_split(
        idx_temp, test_size=0.5, random_state=42, stratify=y[idx_temp]
    )
    X_train, y_train = X[idx_train], y[idx_train]
    X_val, y_val = X[idx_val], y[idx_val]
    X_test, y_test = X[idx_test], y[idx_test]

    enzyme_emb_path = os.path.join(embedding_dir, "enzyme_embeddings.npy")
    enzyme_meta_path = os.path.join(embedding_dir, "enzyme_metadata.csv")
    enzyme_split_path = os.path.join(embedding_dir, "enzyme_split.npz")
    np.save(enzyme_emb_path, X)
    meta_cols = ["id", "is_enzyme"]
    if "ec_number" in df.columns:
        meta_cols.append("ec_number")
    meta_df = df[meta_cols].copy()
    meta_df = meta_df.set_index("id").loc[sample_ids].reset_index()
    meta_df.to_csv(enzyme_meta_path, index=False)
    np.savez(enzyme_split_path, idx_train=idx_train, idx_val=idx_val, idx_test=idx_test)
    print(f"Saved embeddings to {enzyme_emb_path} (shape {X.shape})")
    print(f"Saved metadata to {enzyme_meta_path}")
    print(f"Saved train/val/test indices to {enzyme_split_path}")

    print(f"Shapes: train={X_train.shape}, val={X_val.shape}, test={X_test.shape}")

    print("Training RandomForest for enzyme classification...")
    clf = RandomForestClassifier(
        n_estimators=400,
        random_state=42,
        n_jobs=-1,
        max_depth=40,
        max_samples=0.8,
        min_samples_split=3,
        min_samples_leaf=1,
        max_features='sqrt',
        bootstrap=True,
        class_weight='balanced'
    )
    clf.fit(X_train, y_train)

    # Validation performance
    y_val_pred = clf.predict(X_val)
    val_acc = accuracy_score(y_val, y_val_pred)
    print(f"Validation accuracy: {val_acc:.4f}")
    print(classification_report(y_val, y_val_pred, target_names=["non-enzyme", "enzyme"]))

    # Test performance
    y_test_pred = clf.predict(X_test)
    test_acc = accuracy_score(y_test, y_test_pred)
    print(f"Test accuracy: {test_acc:.4f}")
    print(classification_report(y_test, y_test_pred, target_names=["non-enzyme", "enzyme"]))

    dump(clf, out_path)
    print(f"Saved enzyme classifier to {out_path}")

    return val_acc, test_acc


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="train_data/uniprot_enzyme_binary.csv")
    parser.add_argument("--out", default="train_data/enzyme_classifier.joblib")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--embedding-dir", default="train_data/embeddings")
    args = parser.parse_args()
    train_enzyme_classifier(args.data, args.out, args.device, args.embedding_dir)
