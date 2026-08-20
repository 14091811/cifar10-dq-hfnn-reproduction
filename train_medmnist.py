"""Train matched CNN/DWT-QPA models on an official MedMNIST split."""

import argparse
import json
import random
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import roc_auc_score
from torch.utils.data import DataLoader
from torchvision import transforms

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))

from dq_hfnn.local_spa import (
    TwoBlockClassicalCNN,
    TwoBlockDirectionalDWTClassicalD8CNN,
    TwoBlockDirectionalDWTQuantumD8CNN,
    TwoBlockDirectionalDWTClassicalD16CNN,
    TwoBlockDirectionalDWTQuantumD16CNN,
    TwoBlockDirectionalDWTClassicalD24CNN,
    TwoBlockDirectionalDWTQuantumD24CNN,
    TwoBlockDirectionalDWTClassicalD64CNN,
    TwoBlockDirectionalDWTQuantumD64CNN,
    TwoBlockDirectionalDWTClassicalGroupedD64CNN,
    TwoBlockDirectionalDWTQuantumGroupedD64CNN,
    TwoBlockDirectionalDWTClassicalPartialD16CNN,
    TwoBlockDirectionalDWTQuantumPartialD16CNN,
    TwoBlockDirectionalDWTClassicalRoPED16CNN,
    TwoBlockDirectionalDWTQuantumRoPED16CNN,
    TwoBlockDirectionalDWTClassicalBucketD16CNN,
    TwoBlockDirectionalDWTQuantumBucketD16CNN,
)
from dq_hfnn.run_io import create_run_directory, write_json


MODELS = {
    "two_block_classical_cnn": TwoBlockClassicalCNN,
    "two_block_dwt_directional_classical_d8_cnn": TwoBlockDirectionalDWTClassicalD8CNN,
    "two_block_dwt_directional_qpa_d8_cnn": TwoBlockDirectionalDWTQuantumD8CNN,
    "two_block_dwt_directional_classical_d16_cnn": TwoBlockDirectionalDWTClassicalD16CNN,
    "two_block_dwt_directional_qpa_d16_cnn": TwoBlockDirectionalDWTQuantumD16CNN,
    "two_block_dwt_directional_classical_d24_cnn": TwoBlockDirectionalDWTClassicalD24CNN,
    "two_block_dwt_directional_qpa_d24_cnn": TwoBlockDirectionalDWTQuantumD24CNN,
    "two_block_dwt_directional_classical_d64_cnn": TwoBlockDirectionalDWTClassicalD64CNN,
    "two_block_dwt_directional_qpa_d64_cnn": TwoBlockDirectionalDWTQuantumD64CNN,
    "two_block_dwt_directional_classical_grouped_d64_cnn": TwoBlockDirectionalDWTClassicalGroupedD64CNN,
    "two_block_dwt_directional_qpa_grouped_d64_cnn": TwoBlockDirectionalDWTQuantumGroupedD64CNN,
    "two_block_dwt_directional_classical_partial_d16_cnn": TwoBlockDirectionalDWTClassicalPartialD16CNN,
    "two_block_dwt_directional_qpa_partial_d16_cnn": TwoBlockDirectionalDWTQuantumPartialD16CNN,
    "two_block_dwt_directional_classical_rope_d16_cnn": TwoBlockDirectionalDWTClassicalRoPED16CNN,
    "two_block_dwt_directional_qpa_rope_d16_cnn": TwoBlockDirectionalDWTQuantumRoPED16CNN,
    "two_block_dwt_directional_classical_bucket_d16_cnn": TwoBlockDirectionalDWTClassicalBucketD16CNN,
    "two_block_dwt_directional_qpa_bucket_d16_cnn": TwoBlockDirectionalDWTQuantumBucketD16CNN,
}


def configure_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_dataset_class(name):
    from medmnist import INFO
    import medmnist

    return getattr(medmnist, INFO[name]["python_class"])


def evaluate(model, loader, device, num_classes, loss_fn):
    model.eval()
    confusion = torch.zeros(num_classes, num_classes, dtype=torch.long)
    loss_sum = 0.0
    all_labels, all_probabilities = [], []
    with torch.no_grad():
        for images, labels in loader:
            labels = labels.reshape(-1).long()
            logits = model(images.to(device))
            probabilities = logits.softmax(dim=1).cpu()
            predictions = probabilities.argmax(dim=1)
            loss_sum += loss_fn(logits, labels.to(device)).item() * labels.numel()
            confusion += torch.bincount(
                labels * num_classes + predictions,
                minlength=num_classes ** 2,
            ).reshape(num_classes, num_classes)
            all_labels.append(labels.cpu())
            all_probabilities.append(probabilities)

    total = confusion.sum().float()
    true_positive = confusion.diag().float()
    precision = true_positive / confusion.sum(dim=0).clamp_min(1)
    recall = true_positive / confusion.sum(dim=1).clamp_min(1)
    f1 = 2 * precision * recall / (precision + recall).clamp_min(1e-12)
    specificity = (
        total - confusion.sum(dim=0).float() - confusion.sum(dim=1).float() + true_positive
    ) / (total - confusion.sum(dim=1).float()).clamp_min(1)
    labels_np = torch.cat(all_labels).numpy()
    probabilities_np = torch.cat(all_probabilities).numpy()
    try:
        auc = float(
            roc_auc_score(labels_np, probabilities_np[:, 1])
            if num_classes == 2
            else roc_auc_score(labels_np, probabilities_np, multi_class="ovr", average="macro")
        )
    except ValueError:
        auc = None
    return {
        "loss": loss_sum / total.item(),
        "accuracy": (true_positive.sum() / total).item(),
        "macro_precision": precision.mean().item(),
        "macro_recall": recall.mean().item(),
        "macro_specificity": specificity.mean().item(),
        "macro_f1": f1.mean().item(),
        "auc": auc,
        "confusion_matrix": confusion.tolist(),
    }


def metric_line(metrics):
    auc = "NA" if metrics["auc"] is None else f"{metrics['auc'] * 100:.2f}%"
    return (
        f"Accuracy={metrics['accuracy'] * 100:.2f}% "
        f"Precision={metrics['macro_precision'] * 100:.2f}% "
        f"Recall={metrics['macro_recall'] * 100:.2f}% "
        f"Specificity={metrics['macro_specificity'] * 100:.2f}% "
        f"F1={metrics['macro_f1'] * 100:.2f}% AUC={auc}"
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args()
    cfg = json.loads(args.config.read_text(encoding="utf-8"))
    configure_seed(cfg["seed"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    run_dir, experiment_name, started_at = create_run_directory(ROOT, cfg["run_name"], cfg["seed"])
    write_json(run_dir / "config.json", cfg)

    transform = transforms.Compose([
        transforms.Resize((32, 32)),
        transforms.ToTensor(),
        transforms.Lambda(lambda image: image.repeat(3, 1, 1)),
    ])
    dataset_class = load_dataset_class(cfg["dataset"])
    root = ROOT / cfg.get("data_root", "medmnist_data")
    train_set = dataset_class(split="train", root=str(root), transform=transform, download=False)
    val_set = dataset_class(split="val", root=str(root), transform=transform, download=False)
    test_set = dataset_class(split="test", root=str(root), transform=transform, download=False)
    num_classes = int(cfg["num_classes"])
    loader_args = dict(batch_size=cfg["batch_size"], num_workers=cfg["num_workers"], pin_memory=device.type == "cuda")
    train_loader = DataLoader(train_set, shuffle=True, generator=torch.Generator().manual_seed(cfg["seed"]), **loader_args)
    val_loader = DataLoader(val_set, shuffle=False, **loader_args)
    test_loader = DataLoader(test_set, shuffle=False, **loader_args)

    model_type = cfg["model_type"]
    if model_type not in MODELS:
        raise ValueError(f"Unsupported model_type: {model_type}")
    model = MODELS[model_type](num_classes=num_classes, hidden_dim=cfg["hidden_dim"]).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg["learning_rate"], weight_decay=cfg["weight_decay"])
    loss_fn = torch.nn.CrossEntropyLoss()
    best_f1, best_metrics, stale_epochs, history = -1.0, None, 0, []

    print(f"Experiment={experiment_name} Seed={cfg['seed']} Device={device}", flush=True)
    print(f"Dataset={cfg['dataset']} train={len(train_set)} val={len(val_set)} test={len(test_set)}", flush=True)
    print(f"Model={model_type} params={sum(p.numel() for p in model.parameters()):,}", flush=True)
    for epoch in range(1, cfg["epochs"] + 1):
        started = time.perf_counter()
        model.train()
        loss_sum = correct = total = 0
        for images, labels in train_loader:
            labels = labels.reshape(-1).long().to(device)
            optimizer.zero_grad(set_to_none=True)
            logits = model(images.to(device))
            loss = loss_fn(logits, labels)
            loss.backward()
            optimizer.step()
            loss_sum += loss.item() * labels.numel()
            correct += (logits.argmax(1) == labels).sum().item()
            total += labels.numel()
        val_metrics = evaluate(model, val_loader, device, num_classes, loss_fn)
        record = {
            "epoch": epoch,
            "train_loss": loss_sum / total,
            "train_accuracy": correct / total,
            **{f"validation_{key}": value for key, value in val_metrics.items() if key != "confusion_matrix"},
        }
        history.append(record)
        is_best = val_metrics["macro_f1"] > best_f1
        if is_best:
            best_f1 = val_metrics["macro_f1"]
            best_metrics = dict(val_metrics)
            stale_epochs = 0
            torch.save({"epoch": epoch, "state": model.state_dict()}, run_dir / "best.pt")
        else:
            stale_epochs += 1
        print(
            f"Ep{epoch}: TrLoss={record['train_loss']:.4f} TrAcc={record['train_accuracy'] * 100:.2f}% | "
            f"Val {metric_line(val_metrics)} | Time={time.perf_counter() - started:.1f}s"
            + (" (Best)" if is_best else ""),
            flush=True,
        )
        if stale_epochs >= cfg["early_stopping_patience"]:
            print(f"EarlyStop: patience={cfg['early_stopping_patience']}", flush=True)
            break

    checkpoint = torch.load(run_dir / "best.pt", map_location=device)
    model.load_state_dict(checkpoint["state"])
    test_metrics = evaluate(model, test_loader, device, num_classes, loss_fn)
    best_metrics.pop("confusion_matrix")
    result = {
        "protocol": "official_medmnist_train_val_test",
        "selection_metric": "validation_macro_f1",
        "best_epoch": checkpoint["epoch"],
        **{f"best_validation_{key}": value for key, value in best_metrics.items()},
        **{f"test_{key}": value for key, value in test_metrics.items()},
        "started_at": started_at.isoformat(),
        "completed_at": datetime.now().astimezone().isoformat(),
    }
    write_json(run_dir / "history.json", history)
    write_json(run_dir / "result.json", result)
    print("=" * 96)
    print(f"BEST VALIDATION | Epoch={checkpoint['epoch']} | {metric_line(best_metrics)}")
    print(f"FINAL TEST      | {metric_line(test_metrics)}")
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
