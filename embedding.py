import os
import random
import string
import numpy as np
import pandas as pd
import torch
import pickle
import scipy.io as sio
import ast
import difflib
from tqdm import tqdm
from sklearn.decomposition import PCA
from transformers import AutoTokenizer, AutoModel
from esm import pretrained

# Set random seeds for reproducibility
np.random.seed(42)
random.seed(42)
torch.manual_seed(42)


class DrugEmbeddingGenerator:
    def __init__(self, device="cpu"):
        self.device = torch.device(device)

        # Initialize ChemBERTa
        print("Initializing ChemBERTa...")
        # Note: You need to provide your own Hugging Face token
        token = "YOUR_HUGGINGFACE_TOKEN_HERE"  # Replace with your token
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
            return torch.empty(0, self.smile_model.config.hidden_size, device=self.device)
        return torch.stack(embeddings)

    def generate_biotech_embedding(self, sequence_list):
        print("Generating biotech embeddings...")
        embeddings = []
        for seq in tqdm(sequence_list, desc="Processing Biotech Sequences"):
            if not isinstance(seq, str) or pd.isna(seq) or seq.strip() == "":
                # ESM2 base version outputs 1280 dimensions
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

            # Remove CLS and PAD tokens, take average
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
        then perform dimensionality reduction and combination.
        """
        print("Combining embeddings for all drugs...")
        # Group by type
        small_mol_df = df[df["drug_type"] == "small molecule"]
        biotech_df = df[df["drug_type"] == "biotech"]

        print("Processing small molecule drugs...")
        smile_emb = self.generate_smile_embedding(small_mol_df["smiles"].tolist())
        print("Processing biotech drugs...")
        biotech_emb = self.generate_biotech_embedding(biotech_df["sequence"].tolist())

        # Handle empty groups
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

        # Construct final drug_embeddings
        drug_embeddings = torch.zeros((len(df), target_dim), device=self.device)
        if smile_emb_final.size(0) > 0:
            drug_embeddings[small_mol_df.index] = smile_emb_final
        if biotech_emb_final.size(0) > 0:
            drug_embeddings[biotech_df.index] = biotech_emb_final

        return drug_embeddings


# Disease embedding functions
def get_text_embedding(text, tokenizer, model, device):
    """
    Generate text embedding using BioBERT with mean pooling
    """
    inputs = tokenizer(text, padding=True, truncation=True, max_length=128, return_tensors="pt")
    input_ids = inputs["input_ids"].to(device)
    attention_mask = inputs["attention_mask"].to(device)
    with torch.no_grad():
        outputs = model(input_ids, attention_mask=attention_mask)
        last_hidden_state = outputs.last_hidden_state
    mask_expanded = attention_mask.unsqueeze(-1).expand(last_hidden_state.size()).float()
    sum_embeddings = torch.sum(last_hidden_state * mask_expanded, dim=1)
    sum_mask = torch.clamp(mask_expanded.sum(dim=1), min=1e-9)
    mean_embeddings = sum_embeddings / sum_mask
    return mean_embeddings[0]


# Utility functions
def generate_random_drugbank_id():
    """Generate random drugbank ID"""
    return "RAND" + ''.join(random.choices(string.ascii_uppercase + string.digits, k=6))


def load_mim_titles(file_path):
    mim_dict = {}
    with open(file_path, "r", encoding="utf-8") as f:
        f.readline()  # Skip header
        for line in f:
            parts = line.strip().split("\t")
            if len(parts) >= 3:
                mim_dict[parts[1].strip()] = parts[2].strip()
    return mim_dict


def clean_disease_code(raw_code):
    return raw_code.replace("[", "").replace("]", "").replace("'", "").replace('"', "").strip()


def code_to_mim_number(code):
    return code.lstrip("D")


def clean_drug_name(raw_name):
    """Clean drug name string"""
    try:
        parsed = ast.literal_eval(raw_name)
        if isinstance(parsed, list) and len(parsed) > 0:
            return str(parsed[0]).strip()
    except Exception:
        pass
    return raw_name.replace("[", "").replace("]", "").replace("'", "").strip()


# Main processing function
def main():
    # Configuration
    device = "cpu"  # or "cuda" if you have GPU
    target_dim = 768
    
    # File paths
    csv_path = "drug_info.csv"
    mim_titles_file = "mimTitles.txt"
    mat_file = "lrssl.mat"
    drug_emb_output = "drug_embedding.pkl"
    
    # Note: You need to provide your own Hugging Face token
    hf_token = "YOUR_HUGGINGFACE_TOKEN_HERE"  # Replace with your token
    
    # Step 1: Generate drug embeddings
    print("Step 1: Generating drug embeddings...")
    df = pd.read_csv(csv_path, sep=",")
    
    if "drugbank_id" not in df.columns:
        raise ValueError("'drugbank_id' column not found in CSV file")
    
    generator = DrugEmbeddingGenerator(device=device)
    
    with torch.no_grad():
        embeddings = generator.combine_embeddings(df, target_dim=target_dim)
        embeddings = embeddings.cpu().numpy()
    
    print(f"Generated drug embeddings shape: {embeddings.shape}")
    
    # Save drug embeddings
    embedding_df = pd.DataFrame(embeddings, index=df["drugbank_id"].values)
    embedding_df.to_pickle(drug_emb_output)
    print(f"Saved drug embeddings to {drug_emb_output}")
    
    # Step 2: Process disease embeddings and update MAT file
    print("\nStep 2: Processing disease embeddings...")
    
    # Initialize BioBERT for disease embeddings
    model_name = "dmis-lab/biobert-base-cased-v1.1"
    disease_tokenizer = AutoTokenizer.from_pretrained(model_name, token=hf_token)
    disease_model = AutoModel.from_pretrained(model_name, token=hf_token)
    disease_model.to(device)
    disease_model.eval()
    
    # Load OMIM titles
    mim_dict = load_mim_titles(mim_titles_file)
    
    # Load MAT file
    mat_data = sio.loadmat(mat_file)
    
    # Process disease embeddings
    if 'Wdname' not in mat_data:
        raise Exception("'Wdname' not found in MAT file!")
    
    Wdname = mat_data['Wdname']
    disease_codes = [clean_disease_code(str(Wdname[i, 0])) for i in range(Wdname.shape[0])]
    
    embeddings_list = []
    for code in disease_codes:
        mim_num = code_to_mim_number(code)
        text_input = mim_dict.get(mim_num, code)
        emb = get_text_embedding(text_input, disease_tokenizer, disease_model, device).cpu().numpy()
        embeddings_list.append(emb)
    
    disease_embed = np.vstack(embeddings_list)
    mat_data["disease_embed"] = disease_embed
    
    # Process drug names and map to DrugBank IDs
    if "Wrname" not in mat_data:
        raise Exception("'Wrname' not found in MAT file!")
    
    drug_names = [clean_drug_name(str(x)) for x in mat_data["Wrname"].flatten()]
    drug_info_df = pd.read_csv(csv_path)
    
    # Build drug name to ID mapping
    drug_name_to_id = {}
    for name, db_id in zip(drug_info_df["name"], drug_info_df["drugbank_id"]):
        name_clean = clean_drug_name(str(name)).lower().strip()
        if pd.isna(db_id) or not isinstance(db_id, str):
            new_id = generate_random_drugbank_id()
        else:
            new_id = db_id.strip().upper()
        drug_name_to_id[name_clean] = new_id
    
    # Map drug names to IDs with fuzzy matching
    mapped_drug_info = []
    for name in drug_names:
        key = name.lower().strip()
        if key in drug_name_to_id:
            mapped_drug_info.append((name, drug_name_to_id[key]))
        else:
            close_matches = difflib.get_close_matches(key, list(drug_name_to_id.keys()), n=1, cutoff=0.8)
            if close_matches:
                matched = close_matches[0]
                print(f"Info: '{name}' fuzzy matched to '{matched}', ID: {drug_name_to_id[matched]}")
                mapped_drug_info.append((name, drug_name_to_id[matched]))
            else:
                random_id = generate_random_drugbank_id()
                print(f"Warning: '{name}' not found, generated random ID: {random_id}")
                mapped_drug_info.append((name, random_id))
    
    mapped_drug_ids = [x[1] for x in mapped_drug_info]
    mat_data["Wrname"] = np.array(mapped_drug_ids, dtype=object).reshape(-1, 1)
    
    # Load pretrained embeddings and map to drugs
    pretrained_embeddings = pd.read_pickle(drug_emb_output)
    pretrained_embeddings.index = pretrained_embeddings.index.astype(str).str.strip().str.upper()
    
    drug_feats = []
    for drug_id in mapped_drug_ids:
        if drug_id and drug_id in pretrained_embeddings.index:
            feat = pretrained_embeddings.loc[drug_id].values
        else:
            feat = np.random.randn(target_dim)
        drug_feats.append(feat.reshape(1, -1))
    
    drug_embed = np.concatenate(drug_feats, axis=0)
    mat_data["drug_embed"] = drug_embed
    
    # Save updated MAT file
    sio.savemat(mat_file, mat_data)
    
    # Print results
    print(f"\nFinal disease_embed shape: {disease_embed.shape}")
    print(f"Final drug_embed shape: {drug_embed.shape}")
    print(f"Final Wrname shape: {mat_data['Wrname'].shape}")
    print(f"MAT file updated successfully!")


if __name__ == '__main__':
    main()
