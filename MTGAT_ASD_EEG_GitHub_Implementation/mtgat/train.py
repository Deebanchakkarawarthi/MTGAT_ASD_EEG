from pathlib import Path
import numpy as np
import torch
from torch import nn

def collate_batch(batch):
    keys = set.intersection(*(set(x) for x in batch))
    out = {}
    for k in keys:
        vals = [x[k] for x in batch]
        out[k] = torch.stack(vals) if torch.is_tensor(vals[0]) else vals
    return out

def run_epoch(model, loader, optimizer=None, device="cpu"):
    train = optimizer is not None
    model.train(train); criterion = nn.CrossEntropyLoss()
    losses, ys, probs = [], [], []
    for b in loader:
        x, a, y = b["node_features"].to(device), b["adjacency"].to(device), b["asd_label"].to(device)
        paradigm = b.get("paradigm_id")
        if paradigm is not None: paradigm = paradigm.to(device)
        with torch.set_grad_enabled(train):
            out = model(x, a, paradigm)
            loss = criterion(out["asd_logits"], y)
            if train:
                optimizer.zero_grad(); loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); optimizer.step()
        losses.append(float(loss.detach().cpu()))
        ys.extend(y.detach().cpu().tolist())
        probs.extend(torch.softmax(out["asd_logits"],-1).detach().cpu().numpy())
    return float(np.mean(losses)), np.asarray(ys), np.asarray(probs)

def fit(model, train_loader, val_loader, epochs=50, lr=1e-3, patience=8, device="cpu", output_dir="outputs/model"):
    outdir = Path(output_dir); outdir.mkdir(parents=True, exist_ok=True)
    model.to(device); opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    best, wait, history = float("inf"), patience, []
    for epoch in range(1, epochs+1):
        tl, _, _ = run_epoch(model, train_loader, opt, device)
        vl, y, p = run_epoch(model, val_loader, None, device)
        history.append({"epoch":epoch,"train_loss":tl,"val_loss":vl})
        if vl < best:
            best, wait = vl, patience
            torch.save(model.state_dict(), outdir/"best_model.pt")
            np.save(outdir/"val_probabilities.npy", p); np.save(outdir/"val_labels.npy", y)
        else:
            wait -= 1
            if wait <= 0: break
    import pandas as pd
    pd.DataFrame(history).to_csv(outdir/"training_history.csv",index=False)
    return history
