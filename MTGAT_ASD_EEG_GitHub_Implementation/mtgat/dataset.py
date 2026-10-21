import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

class EpochTensorDataset(Dataset):
    # NPZ: node_features [S,T,K,N,F], adjacency [S,T,K,N,N].
    # CSV required columns: sample_id, participant_id, asd_label.
    def __init__(self, npz_path, labels_csv):
        z = np.load(npz_path)
        self.x = z["node_features"].astype(np.float32)
        self.a = z["adjacency"].astype(np.float32)
        self.labels = pd.read_csv(labels_csv).reset_index(drop=True)
        required = {"sample_id", "participant_id", "asd_label"}
        if not required.issubset(self.labels.columns):
            raise ValueError(f"CSV requires {sorted(required)}")
        if len(self.labels) != len(self.x):
            raise ValueError("CSV rows must match tensor sample order and count")
    def __len__(self): return len(self.labels)
    def __getitem__(self, i):
        r = self.labels.iloc[i]
        out = {"node_features": torch.from_numpy(self.x[i]),
               "adjacency": torch.from_numpy(self.a[i]),
               "asd_label": torch.tensor(int(r.asd_label), dtype=torch.long),
               "participant_id": str(r.participant_id), "sample_id": str(r.sample_id)}
        if "paradigm_id" in self.labels and pd.notna(r.get("paradigm_id")):
            out["paradigm_id"] = torch.tensor(int(r.paradigm_id), dtype=torch.long)
        return out
