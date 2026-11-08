import csv
from collections import Counter
from pathlib import Path

import torch
import torch.nn as nn
from sklearn.metrics import classification_report, confusion_matrix
from torch.utils.data import DataLoader
from torchvision import datasets, models, transforms

MODEL_PATH = Path("models/resnet18_pets.pt")
REPORTS = Path("reports")


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tf = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])
    test_set = datasets.OxfordIIITPet("data", split="test", transform=tf)
    loader = DataLoader(test_set, batch_size=32, num_workers=2)
    names = test_set.classes

    model = models.resnet18(weights=None)
    model.fc = nn.Linear(model.fc.in_features, len(names))
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    model.to(device).eval()

    preds, labels = [], []
    with torch.no_grad():
        for x, y in loader:
            out = model(x.to(device))
            preds += out.argmax(1).cpu().tolist()
            labels += y.tolist()

    report = classification_report(
        labels, preds, target_names=names, output_dict=True, zero_division=0
    )
    rows = [
        (n, report[n]["precision"], report[n]["recall"], report[n]["f1-score"], int(report[n]["support"]))
        for n in names
    ]

    REPORTS.mkdir(exist_ok=True)
    with open(REPORTS / "per_class.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["class", "precision", "recall", "f1", "support"])
        for r in rows:
            w.writerow([r[0], round(r[1], 3), round(r[2], 3), round(r[3], 3), r[4]])

    print("\nWeakest 5 classes (by recall):")
    for r in sorted(rows, key=lambda r: r[2])[:5]:
        print(f"  {r[0]:30s} recall {r[2]:.2f}  precision {r[1]:.2f}  n={r[4]}")

    cm = confusion_matrix(labels, preds)
    pairs = Counter()
    for i in range(len(names)):
        for j in range(len(names)):
            if i != j and cm[i][j] > 0:
                pairs[(i, j)] = int(cm[i][j])

    lines = [f"{names[i]} -> predicted as {names[j]}: {c} times" for (i, j), c in pairs.most_common(10)]
    (REPORTS / "top_confusions.txt").write_text("\n".join(lines))
    print("\nMost common mistakes:")
    print("\n".join("  " + l for l in lines))


if __name__ == "__main__":
    main()