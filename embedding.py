import pandas as pd
import torch
import numpy as np
import pickle
from transformers import AutoTokenizer, AutoModel
from esm import pretrained
from sklearn.decomposition import PCA
from tqdm import tqdm  # For displaying progress bars

class DrugEmbeddingGenerator:
    def __init__(self, device="cpu"):
        self.device = torch.device(device)

        # Initialize ChemBERTa
        print("Initializing ChemBERTa...")
        token = "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"
        self.smile_tokenizer = AutoTokenizer.from_pretrained(
            "seyonec/PubChem10M_SMILES_BPE_450k",
            use_auth_token=token
        )
        self.smile_model = AutoModel.from_pretrained(
            "seyonec/PubChem10M_SMILES_BPE_450k",
            use_auth_token=token
        ).to(self.device)

        # Initialize ESM model
        print("Initializing ESM model...")
        self.esm_model, self.esm_alphabet = pretrained.esm2_t33_650M_UR50D()
        self.esm_model = self.esm_model.to(self.device)
        self.esm_batch_converter = self.esm_alphabet.get_batch_converter()

        self.max_smile_len = 510

    def generate_smile_embedding(self, smiles_list):
        print("Generating SMILES embeddings...")
        embeddings = []
        for smi in tqdm(smiles_list, desc="Processing SMILES"):
            if not isinstance(smi, str) or pd.isna(smi) or smi.strip() == "":
                embeddings.append(torch.zeros(self.smile_model.config.hidden_size, device=self.device))
                continue

            inputs = self.smile_tokenizer(
                smi,
                return_tensors="pt",
                padding="max_length",
                truncation=True,
                max_length=self.max_smile_len
            ).to(self.device)

            with torch.no_grad():
                outputs = self.smile_model(**inputs)

            # Use the mean of the last hidden state vectors as SMILES representation
            embedding = outputs.last_hidden_state.mean(dim=1).squeeze(0)
            embeddings.append(embedding)

        if len(embeddings) == 0:
            # Return empty tensor, will be converted to target dimension later
            return torch.empty(0, self.smile_model.config.hidden_size, device=self.device)
        return torch.stack(embeddings)

    def generate_biotech_embedding(self, sequence_list):
        print("Generating biotech embeddings...")
        embeddings = []
        for seq in tqdm(sequence_list, desc="Processing Biotech Sequences"):
            if not isinstance(seq, str) or pd.isna(seq) or seq.strip() == "":
                # ESM2 base version outputs 1280 dimensions, set to 0
                embeddings.append(torch.zeros(1280, device=self.device))
                continue

            # Clean FASTA format, remove headers starting with '>'
            lines = seq.strip().splitlines()
            cleaned_lines = [line.strip() for line in lines if not line.startswith(">")]
            seq_cleaned = "".join(cleaned_lines).replace(" ", "").strip()
            if not seq_cleaned:
                embeddings.append(torch.zeros(1280, device=self.device))
                continue

            batch_data = [("protein", seq_cleaned)]
            labels, seqs, tokens = self.esm_batch_converter(batch_data)
            tokens = tokens.to(self.device)

            with torch.no_grad():
                results = self.esm_model(tokens, repr_layers=[33])
                token_reps = results["representations"][33]

            # Remove CLS (token 0) and PAD (assuming length matches seq_cleaned), 
            # take average of (1: len(seq_cleaned)+1)
            seq_embedding = token_reps[0, 1: len(seq_cleaned) + 1].mean(dim=0)
            embeddings.append(seq_embedding)

        if len(embeddings) == 0:
            return torch.empty(0, 1280, device=self.device)
        return torch.stack(embeddings)

    def reduce_dimension(self, embeddings, target_dim):
        print(f"Reducing dimensions to {target_dim} using PCA...")
        n_samples, n_features = embeddings.shape

        if n_samples < 2:
            # Too few samples for PCA, use linear projection instead
            linear_proj = torch.nn.Linear(n_features, target_dim)
            linear_proj.eval()
            with torch.no_grad():
                projected = linear_proj(embeddings.cpu()).to(self.device)
            return projected

        feasible_dim = min(n_samples - 1, n_features)
        if feasible_dim < 1:
            print("Warning: Not enough samples for PCA. Returning original embeddings.")
            return embeddings

        n_components = min(feasible_dim, target_dim)
        pca = PCA(n_components=n_components)
        reduced = pca.fit_transform(embeddings.cpu().numpy())

        # If PCA output dimension is smaller than target_dim, pad with zeros
        if reduced.shape[1] < target_dim:
            padding = target_dim - reduced.shape[1]
            reduced = np.pad(reduced, ((0, 0), (0, padding)), mode="constant")

        return torch.tensor(reduced, device=self.device)

    def combine_embeddings(self, df, target_dim=768):
        """
        Generate embeddings separately for small molecule and biotech based on drug_type,
        then perform dimensionality reduction and combination. The returned order 
        corresponds one-to-one with the row order of df.
        """
        print("Combining embeddings for all drugs...")
        # Group by type
        small_mol_df = df[df["drug_type"] == "small molecule"]
        biotech_df = df[df["drug_type"] == "biotech"]

        print("Processing small molecule drugs...")
        smile_emb = self.generate_smile_embedding(small_mol_df["smiles"].tolist())
        print("Processing biotech drugs...")
        biotech_emb = self.generate_biotech_embedding(biotech_df["sequence"].tolist())

        # If any group is empty, construct empty tensor with shape (0, original_dim)
        if smile_emb.size(0) == 0:
            smile_emb = torch.empty(0, self.smile_model.config.hidden_size, device=self.device)
        if biotech_emb.size(0) == 0:
            biotech_emb = torch.empty(0, 1280, device=self.device)

        # Align both groups in feature dimension
        dim_smile = smile_emb.shape[1]
        dim_biotech = biotech_emb.shape[1]
        max_dim = max(dim_smile, dim_biotech)
        if dim_smile < max_dim:
            smile_emb = torch.nn.functional.pad(smile_emb, (0, max_dim - dim_smile))
        if dim_biotech < max_dim:
            biotech_emb = torch.nn.functional.pad(biotech_emb, (0, max_dim - dim_biotech))

        # Combine all embeddings
        all_emb = []
        if smile_emb.size(0) > 0:
            all_emb.append(smile_emb)
        if biotech_emb.size(0) > 0:
            all_emb.append(biotech_emb)
        if len(all_emb) == 0:
            return torch.zeros(0, target_dim, device=self.device)

        all_emb = torch.cat(all_emb, dim=0)
        final_emb = self.reduce_dimension(all_emb, target_dim=target_dim)

        # Restore according to original grouping
        offset = 0
        if smile_emb.size(0) > 0:
            smile_emb_final = final_emb[offset : offset + smile_emb.size(0)]
            offset += smile_emb.size(0)
        else:
            smile_emb_final = torch.empty(0, target_dim, device=self.device)

        if biotech_emb.size(0) > 0:
            biotech_emb_final = final_emb[offset : offset + biotech_emb.size(0)]
            offset += biotech_emb.size(0)
        else:
            biotech_emb_final = torch.empty(0, target_dim, device=self.device)

        # Construct final drug_embeddings, filling according to original DataFrame row order
        drug_embeddings = torch.zeros((len(df), target_dim), device=self.device)
        if smile_emb_final.size(0) > 0:
            drug_embeddings[small_mol_df.index] = smile_emb_final
        if biotech_emb_final.size(0) > 0:
            drug_embeddings[biotech_df.index] = biotech_emb_final

        return drug_embeddings


def main(csv_path="drug_info.csv", output_path="drug_embedding.pkl", device="cpu", target_dim=768):
    print("Reading input CSV file...")
    df = pd.read_csv(csv_path, sep=",")

    # Ensure CSV file contains "drugbank_id" column, otherwise KeyError will be raised
    if "drugbank_id" not in df.columns:
        raise ValueError("'drugbank_id' column not found in CSV file, please verify input file is correct.")

    generator = DrugEmbeddingGenerator(device=device)

    print("Starting embedding generation process...")
    with torch.no_grad():
        embeddings = generator.combine_embeddings(df, target_dim=target_dim)
        # Transfer embedding from GPU back to CPU for subsequent use or storage
        embeddings = embeddings.cpu().numpy()

    print(f"Generated embeddings shape: {embeddings.shape}")

    # Convert result to DataFrame and set drugbank_id as index
    # Assume row order in df corresponds to embeddings
    embedding_df = pd.DataFrame(embeddings, index=df["drugbank_id"].values)

    print(f"Saving embeddings to {output_path}...")
    # You can choose to use to_pickle or pickle.dump, both have equivalent effects
    embedding_df.to_pickle(output_path)
    print("Process completed successfully.")


if __name__ == "__main__":
    main(
        csv_path="drug_info.csv",
        output_path="drug_embedding.pkl",
        device="cpu",
        target_dim=768
    )