import os
import argparse
import re
import numpy as np
import pandas as pd
from joblib import load
from enzyllm_predictor import EnzyLLMPredictor

# Paths for trained models
EC_CLASSIFIER_PATH = "train_data/ec_classifier.joblib"
ENZYME_CLASSIFIER_PATH = "train_data/enzyme_classifier.joblib"
LABEL_ENCODER_PATH = "train_data/ec_label_encoder.joblib"


def parse_fasta(fasta_path: str) -> list[tuple[str, str]]:
    pairs = []
    with open(fasta_path, "r", encoding="utf-8", errors="replace") as f:
        seq_id = None
        seq_lines = []
        for line in f:
            line = line.rstrip("\n\r")
            if line.startswith(">"):
                if seq_id is not None:
                    pairs.append((seq_id, "".join(seq_lines)))
                header = line[1:].strip()
                token = header.split()[0] if header else ""
                m = re.match(r"^(?:sp|tr)\|([A-Za-z0-9]+)\|", token)
                if m:
                    seq_id = m.group(1)
                else:
                    seq_id = token if token else f"seq_{len(pairs)}"
                seq_lines = []
            else:
                seq_lines.append(line.strip())
        if seq_id is not None:
            pairs.append((seq_id, "".join(seq_lines)))
    return pairs


def _run_predictions(
    ids: list,
    sequences: list[str],
    label_encoder,
    predictor,
    batch_size: int,
    threshold: float,
) -> list[dict]:
    total = len(sequences)
    print("Step 1: Encoding all sequences...")
    all_embeddings = []
    for i in range(0, total, batch_size):
        batch_seqs = sequences[i : i + batch_size]
        batch_embs = predictor.encode_sequences(batch_seqs)
        all_embeddings.append(batch_embs)
        if (i + batch_size) % (batch_size * 10) == 0 or i + batch_size >= total:
            print(f"  Encoded {min(i + batch_size, total)}/{total} sequences")
    all_embeddings = np.vstack(all_embeddings)
    print(f"Encoding complete. Embedding shape: {all_embeddings.shape}")

    print("Step 2: Batch predicting enzyme/non-enzyme...")
    enzyme_preds = predictor.enzyme_classifier.predict(all_embeddings)
    enzyme_probs = None
    if hasattr(predictor.enzyme_classifier, "predict_proba"):
        enzyme_probs = predictor.enzyme_classifier.predict_proba(all_embeddings)

    print("Step 3: Batch predicting EC numbers...")
    ec_probs_all = None
    if predictor.classifier is not None:
        ec_probs_all = predictor.classifier.predict_proba(all_embeddings)

    print("Step 4: Assembling results...")
    results = []
    classes = getattr(label_encoder, "classes_", None)
    for idx in range(total):
        seq_id = ids[idx] if idx < len(ids) else idx
        seq = sequences[idx]

        is_enzyme = bool(enzyme_preds[idx])
        enzyme_prob_val = None
        if enzyme_probs is not None:
            enzyme_prob_val = float(enzyme_probs[idx][1] if is_enzyme else enzyme_probs[idx][0])

        predicted_ec = "non-enzyme"
        ec_prob_value = None
        ml_ec_str = ""
        if is_enzyme and ec_probs_all is not None and classes is not None:
            prob = ec_probs_all[idx]
            if len(prob) == len(classes):
                top_idx = int(np.argmax(prob))
                predicted_ec = str(classes[top_idx])
                ec_prob_value = float(prob[top_idx])
                idxs = np.where(prob >= threshold)[0]
                if len(idxs) > 0:
                    ml_ec = [str(classes[i]) for i in idxs]
                    ml_ec_str = ",".join(ml_ec)

        result = {
            "id": seq_id,
            "sequence_length": len(seq),
            "predicted_is_enzyme": is_enzyme,
            "enzyme_probability": enzyme_prob_val,
            "predicted_ec": predicted_ec,
            "ec_probability": ec_prob_value,
            "predicted_ec_multilabel": ml_ec_str,
        }
        results.append(result)
    return results


def predict_for_fasta(fasta_path: str, batch_size: int = 32, threshold: float = 0.7) -> None:
    if not os.path.exists(fasta_path):
        print(f"Error: FASTA not found -> {fasta_path}")
        return
    for p in (EC_CLASSIFIER_PATH, ENZYME_CLASSIFIER_PATH, LABEL_ENCODER_PATH):
        if not os.path.exists(p):
            print(f"Error: Not found -> {p}")
            return

    print(f"\nProcessing FASTA: {fasta_path}")
    pairs = parse_fasta(fasta_path)
    if not pairs:
        print(f"Error: No sequences in FASTA -> {fasta_path}")
        return
    ids = [p[0] for p in pairs]
    sequences = [p[1] for p in pairs]

    label_encoder = load(LABEL_ENCODER_PATH)
    predictor = EnzyLLMPredictor(
        classifier_path=EC_CLASSIFIER_PATH,
        enzyme_classifier_path=ENZYME_CLASSIFIER_PATH,
    )
    print(f"Using device: {predictor.device}")
    print(f"Batch size: {batch_size}")
    print(f"Total sequences: {len(sequences)}")

    results = _run_predictions(ids, sequences, label_encoder, predictor, batch_size, threshold)
    out_path = os.path.splitext(fasta_path)[0] + "_predictions.csv"
    pd.DataFrame(results).to_csv(out_path, index=False)
    print(f"Saved predictions to: {out_path}")


def main():
    parser = argparse.ArgumentParser(description="Run EC predictions for FASTA files.")
    parser.add_argument("fasta_files", nargs="+", help="FASTA files (.fasta or .fa)")
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size (default: 32)")
    parser.add_argument("--threshold", type=float, default=0.7, help="Multi-label probability threshold (default: 0.7)")
    args = parser.parse_args()

    print(f"Running predictions for {len(args.fasta_files)} FASTA file(s)...")
    print(f"Using probability threshold: {args.threshold}")
    for path in args.fasta_files:
        predict_for_fasta(path, batch_size=args.batch_size, threshold=args.threshold)
    print("\nAll predictions complete.")


if __name__ == "__main__":
    main()
