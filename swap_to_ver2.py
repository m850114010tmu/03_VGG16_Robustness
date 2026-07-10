import torch
from pathlib import Path

BASE_DIR   = Path(r"C:\Users\sammy\Exam3Ex2")
OUTPUT_DIR = BASE_DIR / "outputs"

SRC  = OUTPUT_DIR / "report2_ver2.pth"
DST1 = OUTPUT_DIR / "report2.pth"
DST2 = BASE_DIR   / "report2.pth"

def main():
    if not SRC.exists():
        print(f"  [ERROR] Source not found: {SRC}")
        return

    ckpt = torch.load(SRC, map_location="cpu", weights_only=False)
    print(f"  Source   : {SRC}")
    print(f"  Version  : {ckpt.get('version', '?')}")
    print(f"  Epoch    : {ckpt.get('epoch', '?')}")
    print(f"  Val Acc  : {ckpt.get('val_acc', '?')}")
    print(f"  Val F1   : {ckpt.get('val_f1_macro', '?')}")

    for dst in (DST1, DST2):
        torch.save(ckpt, dst)
        print(f"  -> Saved : {dst}")

    print("\n  report2.pth is now ver2.\n")

if __name__ == "__main__":
    main()
