import torch
import esm
import numpy as np

class EnzyLLMPredictor:
    def __init__(self, esm2_model_name="esm2_t6_8M_UR50D", classifier_path=None, enzyme_classifier_path=None, device=None, max_length=1000):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.esm_model, self.esm_alphabet = esm.pretrained.esm2_t6_8M_UR50D()
        self.esm_model = self.esm_model.to(self.device)
        self.batch_converter = self.esm_alphabet.get_batch_converter()
        self.max_length = max_length
        if classifier_path:
            from joblib import load
            self.classifier = load(classifier_path)
        else:
            self.classifier = None
        if enzyme_classifier_path:
            from joblib import load
            self.enzyme_classifier = load(enzyme_classifier_path)
        else:
            self.enzyme_classifier = None

    def _chunk_sequence(self, sequence):
        """Split a long sequence into non-overlapping chunks within the model token limit."""
        if len(sequence) <= self.max_length:
            return [sequence]
        chunks = []
        for start in range(0, len(sequence), self.max_length):
            chunks.append(sequence[start:start + self.max_length])
        return chunks

    def encode_sequence(self, sequence):
        seq_chunks = self._chunk_sequence(sequence)
        if len(seq_chunks) == 1:
            batch_labels, batch_strs, batch_tokens = self.batch_converter([("protein1", seq_chunks[0])])
            batch_tokens = batch_tokens.to(self.device)
            with torch.no_grad():
                results = self.esm_model(batch_tokens, repr_layers=[6], return_contacts=False)
            token_representations = results["representations"][6]
            cls_embedding = token_representations[0, 0, :].cpu().numpy()
            return cls_embedding
        chunk_embeddings = []
        for idx, chunk in enumerate(seq_chunks):
            batch_labels, batch_strs, batch_tokens = self.batch_converter([(f"protein_chunk_{idx}", chunk)])
            batch_tokens = batch_tokens.to(self.device)
            with torch.no_grad():
                results = self.esm_model(batch_tokens, repr_layers=[6], return_contacts=False)
            token_representations = results["representations"][6]
            chunk_cls = token_representations[0, 0, :].cpu().numpy()
            chunk_embeddings.append(chunk_cls)
        return np.mean(np.stack(chunk_embeddings, axis=0), axis=0)

    def encode_sequences(self, sequences):
        short_indices = [i for i, s in enumerate(sequences) if len(s) <= self.max_length]
        long_indices = [i for i, s in enumerate(sequences) if len(s) > self.max_length]

        embeddings = [None] * len(sequences)

        if len(short_indices) > 0:
            short_batch = [(f"protein{i}", sequences[i]) for i in short_indices]
            batch_labels, batch_strs, batch_tokens = self.batch_converter(short_batch)
            batch_tokens = batch_tokens.to(self.device)
            try:
                with torch.no_grad():
                    results = self.esm_model(batch_tokens, repr_layers=[6], return_contacts=False)
                token_representations = results["representations"][6]
                cls_embeddings = token_representations[:, 0, :].cpu().numpy()
                for pos, idx in enumerate(short_indices):
                    embeddings[idx] = cls_embeddings[pos]
            except RuntimeError as e:
                if "out of memory" in str(e).lower():
                    print("Short batch OOM, falling back to per-sequence processing for short sequences")
                    for idx in short_indices:
                        embeddings[idx] = self.encode_sequence(sequences[idx])
                else:
                    raise e

        for idx in long_indices:
            embeddings[idx] = self.encode_sequence(sequences[idx])

        return np.array(embeddings)

    def predict_ec(self, sequence):
        embedding = self.encode_sequence(sequence)
        if self.classifier:
            ec_pred = self.classifier.predict([embedding])[0]
            return ec_pred
        else:
            return embedding  

    def predict_batch(self, sequences):
        return [self.predict_ec(seq) for seq in sequences]

    def predict_enzyme(self, sequence):
        embedding = self.encode_sequence(sequence)
        if self.enzyme_classifier is None:
            return None, None
        pred = self.enzyme_classifier.predict([embedding])[0]
        prob = None
        if hasattr(self.enzyme_classifier, "predict_proba"):
            prob = self.enzyme_classifier.predict_proba([embedding])[0]
        return pred, prob

    def predict_enzyme_batch(self, sequences):
        embeddings = self.encode_sequences(sequences)
        if self.enzyme_classifier is None:
            return None, None
        preds = self.enzyme_classifier.predict(embeddings)
        probs = None
        if hasattr(self.enzyme_classifier, "predict_proba"):
            probs = self.enzyme_classifier.predict_proba(embeddings)
        return preds, probs

    def predict_complete(self, sequence, ec_label_encoder=None):
        enzyme_pred, enzyme_prob = self.predict_enzyme(sequence)
        if enzyme_pred == 0:
            return {
                "is_enzyme": False,
                "enzyme_probability": None if enzyme_prob is None else float(enzyme_prob[0]),
                "ec_number": "non-enzyme",
                "ec_probability": None
            }
        embedding = self.encode_sequence(sequence)
        if self.classifier is None or ec_label_encoder is None:
            return {
                "is_enzyme": True if enzyme_pred is None else bool(enzyme_pred),
                "enzyme_probability": None if enzyme_prob is None else float(enzyme_prob[1]),
                "ec_number": None,
                "ec_probability": None
            }
        ec_label = self.classifier.predict([embedding])[0]
        ec_number = ec_label_encoder.inverse_transform([ec_label])[0]
        ec_prob = None
        if hasattr(self.classifier, "predict_proba"):
            ec_prob = self.classifier.predict_proba([embedding])[0]
        return {
            "is_enzyme": True,
            "enzyme_probability": None if enzyme_prob is None else float(enzyme_prob[1]),
            "ec_number": ec_number,
            "ec_probability": None if ec_prob is None else float(np.max(ec_prob))
        }
