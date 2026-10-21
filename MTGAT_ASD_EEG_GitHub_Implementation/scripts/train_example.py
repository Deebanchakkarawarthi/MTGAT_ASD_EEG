# Train on preassembled sequence tensors, not raw extractor output.
import argparse, json, yaml, torch, numpy as np
from torch.utils.data import DataLoader, Subset
from mtgat.dataset import EpochTensorDataset
from mtgat.splits import participant_split
from mtgat.model import MTGAT
from mtgat.train import fit, run_epoch, collate_batch
from mtgat.seed import set_seed
from mtgat.metrics import classification_metrics

def main():
    p=argparse.ArgumentParser(); p.add_argument("--config",default="configs/default.yaml")
    p.add_argument("--npz",required=True); p.add_argument("--labels",required=True); a=p.parse_args()
    c=yaml.safe_load(open(a.config,encoding="utf-8")); set_seed(c["training"]["seed"])
    ds=EpochTensorDataset(a.npz,a.labels); tr,va,te=participant_split(ds.labels.participant_id.to_numpy(),seed=c["training"]["seed"])
    def loader(idx,shuffle):
        return DataLoader(Subset(ds,idx),batch_size=c["training"]["batch_size"],shuffle=shuffle,collate_fn=collate_batch)
    model=MTGAT(hidden_dim=c["training"]["hidden_dim"],num_classes=c["training"]["num_classes"],dropout=c["training"]["dropout"])
    device="cuda" if torch.cuda.is_available() else "cpu"
    fit(model,loader(tr,True),loader(va,False),c["training"]["epochs"],c["training"]["learning_rate"],c["training"]["patience"],device)
    model.load_state_dict(torch.load("outputs/model/best_model.pt",map_location=device,weights_only=True))
    _,y,pred=run_epoch(model,loader(te,False),None,device); metrics=classification_metrics(y,pred)
    with open("outputs/model/test_metrics.json","w") as f: json.dump(metrics,f,indent=2)
    np.save("outputs/model/test_probabilities.npy",pred); np.save("outputs/model/test_labels.npy",y)
    print(json.dumps(metrics,indent=2))
if __name__=="__main__": main()
