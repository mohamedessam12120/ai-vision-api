import json
from pathlib import Path

import torch
import torch.nn as nn
from sklearn.metrics import f1_score, precision_score, recall_score
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, models, transforms

SEED = 42
BATCH_SIZE = 32
EPOCHS = 10
PATIENCE = 3
MODEL_PATH = Path("models/resnet18_pets.pt")
METRICS_PATH = Path("reports/metrics.json")


def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss, preds, labels = 0.0, [], []
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            out = model(x)
            total_loss += criterion(out, y).item() * x.size(0)
            preds += out.argmax(1).cpu().tolist()
            labels += y.cpu().tolist()
    acc = sum(p == t for p, t in zip(preds, labels)) / len(labels)
    return total_loss / len(labels), acc, preds, labels


def main():
    torch.manual_seed(SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_amp = device.type == "cuda"
    print("device:", device)

    normalize = transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    train_tf = transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        normalize,
    ])
    eval_tf = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        normalize,
    ])

    # Same images, two transforms: augmentation on train only
    train_full = datasets.OxfordIIITPet("data", split="trainval", transform=train_tf)
    val_full = datasets.OxfordIIITPet("data", split="trainval", transform=eval_tf)
    test_set = datasets.OxfordIIITPet("data", split="test", transform=eval_tf)

    g = torch.Generator().manual_seed(SEED)
    perm = torch.randperm(len(train_full), generator=g).tolist()
    n_val = int(0.2 * len(perm))
    val_idx, train_idx = perm[:n_val], perm[n_val:]

    kw = dict(batch_size=BATCH_SIZE, num_workers=2, pin_memory=use_amp)
    train_loader = DataLoader(Subset(train_full, train_idx), shuffle=True, **kw)
    val_loader = DataLoader(Subset(val_full, val_idx), **kw)
    test_loader = DataLoader(test_set, **kw)
    print(f"train: {len(train_idx)} | val: {len(val_idx)} | test: {len(test_set)}")

    model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
    model.fc = nn.Linear(model.fc.in_features, len(train_full.classes))
    model.to(device)

    criterion = nn.CrossEntropyLoss()
    head = list(model.fc.parameters())
    head_ids = {id(p) for p in head}
    backbone = [p for p in model.parameters() if id(p) not in head_ids]
    optimizer = torch.optim.AdamW(
        [{"params": backbone, "lr": 1e-4}, {"params": head, "lr": 1e-3}],
        weight_decay=1e-4,
    )
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    MODEL_PATH.parent.mkdir(exist_ok=True)
    METRICS_PATH.parent.mkdir(exist_ok=True)

    best_acc, bad_epochs, epochs_run = 0.0, 0, 0
    for epoch in range(1, EPOCHS + 1):
        model.train()
        run_loss = 0.0
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            with torch.autocast(device_type=device.type, enabled=use_amp):
                loss = criterion(model(x), y)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            run_loss += loss.item() * x.size(0)
        train_loss = run_loss / len(train_idx)
        val_loss, val_acc, _, _ = evaluate(model, val_loader, criterion, device)
        epochs_run = epoch
        print(f"epoch {epoch:02d} | train loss {train_loss:.3f} | "
              f"val loss {val_loss:.3f} | val acc {val_acc:.3f}")

        if val_acc > best_acc:
            best_acc, bad_epochs = val_acc, 0
            torch.save(model.state_dict(), MODEL_PATH)
        else:
            bad_epochs += 1
            if bad_epochs >= PATIENCE:
                print("early stopping")
                break

    # Test set is used once, with the best weights
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    _, test_acc, preds, labels = evaluate(model, test_loader, criterion, device)
    metrics = {
        "best_val_accuracy": round(best_acc, 4),
        "test_accuracy": round(test_acc, 4),
        "test_precision_macro": round(precision_score(labels, preds, average="macro", zero_division=0), 4),
        "test_recall_macro": round(recall_score(labels, preds, average="macro", zero_division=0), 4),
        "test_f1_macro": round(f1_score(labels, preds, average="macro", zero_division=0), 4),
        "epochs_run": epochs_run,
    }
    METRICS_PATH.write_text(json.dumps(metrics, indent=2))
    print(metrics)


if __name__ == "__main__":
    main()