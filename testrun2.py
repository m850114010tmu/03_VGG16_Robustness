import os
import sys
import json
import time
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from pathlib import Path
from PIL import Image
from torchvision import transforms

# --- Paths -----------------------------------------
SCRIPT_DIR  = Path(__file__).resolve().parent
MODEL_PATH  = SCRIPT_DIR / "report2.pth"
DATA_DIR    = SCRIPT_DIR / "gen_data_exam"
TEST_DIR    = DATA_DIR / "test2"
OUTPUT_CSV  = SCRIPT_DIR / "clsn2_ans.csv"
NORM_JSON   = SCRIPT_DIR / "outputs" / "norm_stats.json"

# --- Class mapping ------------------------
CLASS_NAMES = ["class 1", "class 2", "class 3", "class 4"]
NUM_CLASSES = len(CLASS_NAMES)
IN_CHANNELS = 1
IMAGE_SIZE  = 128
BATCH_SIZE  = 64

# --- VGG-16-BN model --
VGG16_CFG = [64, 64, "M",
             128, 128, "M",
             256, 256, 256, "M",
             512, 512, 512, "M",
             512, 512, 512, "M"]


def make_feature_layers(cfg, in_channels):
    layers = []
    c = in_channels
    for v in cfg:
        if v == "M":
            layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
        else:
            layers += [
                nn.Conv2d(c, v, kernel_size=3, padding=1),
                nn.BatchNorm2d(v),
                nn.ReLU(inplace=True),
            ]
            c = v
    return nn.Sequential(*layers)


class SamVGG16Ex2(nn.Module):
    """VGG-16 (with BatchNorm) for grayscale 4-class classification."""
    def __init__(self, num_classes=NUM_CLASSES,
                 in_channels=IN_CHANNELS, dropout=0.5):
        super().__init__()
        self.features   = make_feature_layers(VGG16_CFG, in_channels)
        self.avgpool    = nn.AdaptiveAvgPool2d((7, 7))
        self.classifier = nn.Sequential(
            nn.Linear(512 * 7 * 7, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout),
            nn.Linear(4096, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = x.flatten(1)
        return self.classifier(x)


# --- VGG-16 weight table -----------------------------------------------------
def print_weight_table(model):
    """Print a detailed layer-by-layer parameter count (VGG-16 weight table)."""
    print("\n" + "=" * 80)
    print("  VGG-16 LAYER-BY-LAYER WEIGHT TABLE")
    print("=" * 80)
    print(f"  {'Layer':<42s} {'Output Shape':<22s} {'Params':>10s}")
    print("  " + "-" * 76)

    total = 0
    # Features
    block, conv_id = 1, 0
    for i, layer in enumerate(model.features):
        if isinstance(layer, nn.Conv2d):
            conv_id += 1
            w = layer.weight
            p = sum(pp.numel() for pp in layer.parameters())
            out_c = w.shape[0]
            print(f"  Conv2d-{conv_id:<2d}  (block {block})  "
                  f"3x3, {w.shape[1]:>3d}->{out_c:>3d}, pad=1"
                  f"{'':>6s}{p:>10,d}")
            total += p
        elif isinstance(layer, nn.BatchNorm2d):
            p = sum(pp.numel() for pp in layer.parameters())
            print(f"  BatchNorm2d  (block {block})"
                  f"{'':>26s}{p:>10,d}")
            total += p
        elif isinstance(layer, nn.MaxPool2d):
            print(f"  MaxPool2d    (block {block})  "
                  f"2x2, stride 2{'':>16s}{'0':>10s}")
            block += 1

    # AvgPool
    print(f"  AdaptiveAvgPool2d -> 512x7x7{'':>20s}{'0':>10s}")

    # Classifier
    for j, layer in enumerate(model.classifier):
        if isinstance(layer, nn.Linear):
            p = sum(pp.numel() for pp in layer.parameters())
            print(f"  Linear  {layer.in_features:>5d} -> {layer.out_features:<5d}"
                  f"{'':>22s}{p:>10,d}")
            total += p
        elif isinstance(layer, nn.Dropout):
            print(f"  Dropout(p={layer.p}){'':>28s}{'0':>10s}")

    print("  " + "-" * 76)
    print(f"  {'TOTAL':>42s}{'':>22s}{total:>10,d}")
    trainable = sum(pp.numel() for pp in model.parameters() if pp.requires_grad)
    print(f"  {'Trainable':>42s}{'':>22s}{trainable:>10,d}")
    print(f"  Model size (FP32):  {total * 4 / 1e6:.1f} MB")
    print("=" * 80)


# --- Main ---------------------------------------------------------------------
def main():
    # -- 1. Device ---------------------------------------------------------
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    gpu_name = torch.cuda.get_device_name(0) if device.type == "cuda" else "CPU"
    print(f"\n  Device : {gpu_name}")

    # -- 2. Load model -----------------------------------------------------
    if not MODEL_PATH.exists():
        print(f"  [ERROR] Model not found: {MODEL_PATH}")
        print(f"          Ensure report2.pth is next to this script.")
        sys.exit(1)

    checkpoint = torch.load(MODEL_PATH, map_location=device, weights_only=False)
    dropout = checkpoint.get("dropout", 0.5)
    model = SamVGG16Ex2(num_classes=NUM_CLASSES, in_channels=IN_CHANNELS,
                        dropout=dropout).to(device)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()

    print(f"  Model loaded   : {MODEL_PATH}")
    print(f"  Version        : {checkpoint.get('version', '?')}")
    print(f"  Trained epoch  : {checkpoint.get('epoch', '?')}")
    print(f"  Val accuracy   : {checkpoint.get('val_acc', '?')}")
    print(f"  Val F1 (macro) : {checkpoint.get('val_f1_macro', '?')}")
    print(f"  Image size     : {checkpoint.get('image_size', IMAGE_SIZE)}")
    print(f"  Input channels : {checkpoint.get('in_channels', IN_CHANNELS)}")
    print(f"  Seed           : {checkpoint.get('seed', '?')}")
    print(f"  Classes        : {checkpoint.get('class_names', CLASS_NAMES)}")

    # -- 3. Print VGG-16 weight table --------------------------------------
    print_weight_table(model)

    # -- 4. Load normalization stats ---------------------------------------
    if not NORM_JSON.exists():
        print(f"  [WARNING] {NORM_JSON} not found, using fallback stats.")
        mean_val, std_val = 0.0992, 0.2109
    else:
        with open(NORM_JSON, "r", encoding="utf-8") as f:
            stats = json.load(f)
        mean_val, std_val = stats["mean"], stats["std"]
    print(f"  Norm stats     : mean={mean_val:.4f}  std={std_val:.4f}")

    # -- 5. Build CLEAN test transform (NO noise/jitter -- per professor) --
    test_transform = transforms.Compose([
        transforms.Grayscale(num_output_channels=1),
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[mean_val], std=[std_val]),
    ])

    # -- 6. Collect test images --------------------------------------------
    if not TEST_DIR.exists():
        print(f"  [ERROR] Test folder not found: {TEST_DIR}")
        sys.exit(1)

    image_files = sorted(
        f for f in os.listdir(TEST_DIR)
        if f.lower().endswith((".png", ".jpg", ".jpeg", ".bmp"))
    )
    print(f"  Test images    : {len(image_files)}")

    if len(image_files) == 0:
        print("  [ERROR] No images found in test2 folder.")
        sys.exit(1)

    # -- 7. Inference ------------------------------------------------------
    results = []
    use_amp = (device.type == "cuda")
    latencies = []

    print(f"\n  Running inference on {len(image_files)} images ...")

    with torch.no_grad():
        for fname in image_files:
            img_path = TEST_DIR / fname
            image = Image.open(img_path).convert("L")   # force grayscale
            tensor = test_transform(image).unsqueeze(0).to(device)

            if device.type == "cuda":
                torch.cuda.synchronize()
            t0 = time.perf_counter()

            with torch.amp.autocast(device.type, enabled=use_amp):
                logits = model(tensor)

            if device.type == "cuda":
                torch.cuda.synchronize()
            latencies.append(time.perf_counter() - t0)

            pred_idx   = logits.argmax(dim=1).item()
            pred_class = CLASS_NAMES[pred_idx]
            file_stem  = Path(fname).stem

            results.append({
                "filename":   file_stem,
                "prediction": pred_class,
            })

    # -- 8. Save CSV -------------------------------------------------------
    df = pd.DataFrame(results)
    df.to_csv(OUTPUT_CSV, index=False)
    print(f"\n  Predictions saved -> {OUTPUT_CSV}")
    print(f"  Total predictions : {len(df)}")

    # -- 9. Prediction distribution ----------------------------------------
    print("\n  Prediction distribution:")
    for cls_name in CLASS_NAMES:
        count = len(df[df["prediction"] == cls_name])
        pct = count / len(df) * 100
        print(f"    {cls_name} : {count:>3d}  ({pct:.1f}%)")

    # -- 10. Inference statistics ------------------------------------------
    lat = np.array(latencies) * 1000   # ms
    print(f"\n  Inference speed:")
    print(f"    Mean   : {lat.mean():.2f} ms/image")
    print(f"    Median : {np.median(lat):.2f} ms/image")
    print(f"    Std    : {lat.std():.2f} ms")
    print(f"    Total  : {lat.sum()/1000:.2f} s for {len(image_files)} images")
    if device.type == "cuda":
        peak_mb = torch.cuda.max_memory_allocated() / 1e6
        print(f"    Peak VRAM : {peak_mb:.1f} MB")

    # -- 11. CSV preview ---------------------------------------------------
    print(f"\n  CSV preview (first 10 rows):")
    print(df.head(10).to_string(index=False))

    print(f"\n  Done.\n")


if __name__ == "__main__":
    main()
