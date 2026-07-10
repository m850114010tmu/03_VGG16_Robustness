import json
import numpy as np
import pandas as pd
import torch
import torchvision.transforms.functional as TF
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import accuracy_score

from config2 import (DEVICE, OUTPUT_DIR, NORM_STATS_PATH, IMAGE_SIZE,
                     NUM_CLASSES, IN_CHANNELS, BATCH_SIZE,
                     seed_everything, SEED, version_model_path)
from model2 import SamVGG16Ex2

with open(NORM_STATS_PATH) as f:
    _s = json.load(f); MEAN, STD = _s["mean"], _s["std"]


def base_tensor(fp):
    img = Image.open(fp).convert("L").resize((IMAGE_SIZE, IMAGE_SIZE))
    return TF.to_tensor(img)                       # [1,H,W] in [0,1]


def perturb(t, kind):
    if kind == "clean":    return t
    if kind == "brighter": return torch.clamp(t + 0.20, 0, 1)
    if kind == "darker":   return torch.clamp(t - 0.20, 0, 1)
    if kind == "contrast": return torch.clamp((t - t.mean()) * 1.6 + t.mean(), 0, 1)
    if kind == "noise":    return torch.clamp(t + torch.randn_like(t) * 0.10, 0, 1)
    if kind == "combined": return torch.clamp((t + 0.15) + torch.randn_like(t) * 0.08, 0, 1)
    raise ValueError(kind)


class ValDS(Dataset):
    def __init__(self, df, kind):
        self.fp = df["filepath"].tolist(); self.y = df["label"].tolist(); self.kind = kind
    def __len__(self): return len(self.fp)
    def __getitem__(self, i):
        t = perturb(base_tensor(self.fp[i]), self.kind)
        return TF.normalize(t, [MEAN], [STD]), self.y[i]


@torch.no_grad()
def acc_of(model, df, kind):
    seed_everything(SEED)                          
    loader = DataLoader(ValDS(df, kind), batch_size=BATCH_SIZE, shuffle=False)
    ys, ps = [], []
    for x, y in loader:
        out = model(x.to(DEVICE))
        ps.append(out.argmax(1).cpu().numpy()); ys.append(y.numpy())
    return accuracy_score(np.concatenate(ys), np.concatenate(ps)) * 100


def load(name):
    ck = torch.load(version_model_path(name), map_location=DEVICE, weights_only=False)
    m = SamVGG16Ex2(NUM_CLASSES, IN_CHANNELS, dropout=ck.get("dropout", 0.5)).to(DEVICE)
    m.load_state_dict(ck["model_state"]); m.eval(); return m


def main():
    val = pd.read_csv(OUTPUT_DIR / "val_split.csv")
    kinds = ["clean", "brighter", "darker", "contrast", "noise", "combined"]
    rows = []
    for name in ["ver1", "ver2", "ver3"]:
        try:
            m = load(name)
        except FileNotFoundError:
            continue
        r = {"model": name}
        for k in kinds:
            r[k] = round(acc_of(m, val, k), 2)
        r["worst_drop"] = round(r["clean"] - min(r[k] for k in kinds if k != "clean"), 2)
        rows.append(r)
        del m
        if DEVICE.type == "cuda": torch.cuda.empty_cache()

    df = pd.DataFrame(rows)
    print("\n" + "=" * 80)
    print("  ROBUSTNESS UNDER SIMULATED TEST-TIME PERTURBATION (val set)")
    print("=" * 80)
    print(df.to_string(index=False))
    df.to_csv(OUTPUT_DIR / "robustness_check.csv", index=False)

    best = df.sort_values("worst_drop").iloc[0]["model"]
    print(f"\n  -> Most robust: {best}")
    print(f"     consider re-saving the best version as report2.pth.\n")


if __name__ == "__main__":
    main()