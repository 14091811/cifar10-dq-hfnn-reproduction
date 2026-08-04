import argparse
import json
import os
import random
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))
from dq_hfnn.model import DQHFNN
from dq_hfnn.frequency_qpa_model import FrequencyQPANet, WideTwoLevelFrequencyQPANet
from dq_hfnn.run_io import create_run_directory, write_json


def residual_direction_loss(quantum_logits, classical_logits, labels):
    """Align quantum evidence with the correction missing from the CNN logits."""
    target = F.one_hot(labels, num_classes=quantum_logits.shape[1]).to(
        quantum_logits.dtype
    ) - classical_logits.detach().softmax(dim=1)
    centered_quantum = quantum_logits - quantum_logits.mean(dim=1, keepdim=True)
    return (1.0 - F.cosine_similarity(centered_quantum, target, dim=1)).mean()


def seed_worker(worker_id):
    """Seed Python and NumPy from the DataLoader's deterministic torch seed."""
    worker_seed = torch.initial_seed() % 2**32
    random.seed(worker_seed)
    np.random.seed(worker_seed)


def configure_reproducibility(seed, enabled):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if enabled:
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        torch.use_deterministic_algorithms(True)


class TeeStream:
    def __init__(self, terminal, log_file):
        self.terminal = terminal
        self.log_file = log_file

    def write(self, value):
        self.terminal.write(value)
        self.log_file.write(value)
        return len(value)

    def flush(self):
        self.terminal.flush()
        self.log_file.flush()

    def isatty(self):
        return self.terminal.isatty()


class RunLogger:
    def __init__(self, path):
        self.handle = Path(path).open("w", encoding="utf-8", buffering=1)
        self.original_stdout = sys.stdout
        self.original_stderr = sys.stderr
        sys.stdout = TeeStream(self.original_stdout, self.handle)
        sys.stderr = TeeStream(self.original_stderr, self.handle)

    def __call__(self, message):
        print(message, flush=True)

    def close(self):
        sys.stdout = self.original_stdout
        sys.stderr = self.original_stderr
        self.handle.close()


def evaluate(model, loader, device, num_classes, loss_fn):
    model.eval(); confusion = torch.zeros(num_classes, num_classes, dtype=torch.long); loss_sum = 0.0
    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device); logits = model(images); predictions = logits.argmax(1).cpu()
            loss_sum += loss_fn(logits, labels.to(device)).item() * labels.numel()
            confusion += torch.bincount(labels * num_classes + predictions, minlength=num_classes ** 2).reshape(num_classes, num_classes)
    total = confusion.sum().float(); true_positive = confusion.diag().float()
    precision = true_positive / confusion.sum(dim=0).clamp_min(1)
    recall = true_positive / confusion.sum(dim=1).clamp_min(1)
    f1 = 2 * precision * recall / (precision + recall).clamp_min(1e-12)
    specificity = (total - confusion.sum(dim=0).float() - confusion.sum(dim=1).float() + true_positive) / (total - confusion.sum(dim=1).float()).clamp_min(1)
    metrics = {
        "loss": loss_sum / total.item(),
        "accuracy": (true_positive.sum() / total).item(),
        "macro_precision": precision.mean().item(),
        "macro_recall": recall.mean().item(),
        "macro_f1": f1.mean().item(),
        "macro_specificity": specificity.mean().item(),
        "confusion_matrix": confusion.tolist(),
    }
    return metrics


def main():
    parser = argparse.ArgumentParser(); parser.add_argument("--config", type=Path, required=True); args = parser.parse_args()
    cfg = json.loads(args.config.read_text())
    configure_reproducibility(cfg["seed"], cfg.get("deterministic", False))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    run_dir, experiment_name, started_at = create_run_directory(
        ROOT, cfg["run_name"], cfg["seed"]
    )
    logger = RunLogger(run_dir / "train.log")
    write_json(run_dir / "config.json", cfg)
    status = {
        "status": "running",
        "experiment": experiment_name,
        "seed": cfg["seed"],
        "started_at": started_at.isoformat(),
        "last_epoch": 0,
        "best_accuracy": None,
        "best_epoch": None,
    }
    write_json(run_dir / "status.json", status)
    logger(f"RunDir={run_dir}")
    logger(
        f"Experiment={experiment_name} | Seed={cfg['seed']} | Device={device} | "
        f"Config={args.config}"
    )
    mean, std = (0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)
    train_tf = transforms.Compose([transforms.RandomCrop(32, padding=4), transforms.RandomHorizontalFlip(), transforms.ToTensor(), transforms.Normalize(mean, std)])
    eval_tf = transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean, std)])
    root = ROOT / "data" / "cifar10"
    source = datasets.CIFAR10(root, train=True, download=True, transform=train_tf)
    val_source = datasets.CIFAR10(root, train=True, download=False, transform=eval_tf)
    test = datasets.CIFAR10(root, train=False, download=True, transform=eval_tf)
    if cfg.get("selection_protocol") == "test_per_epoch":
        train, val = source, test
    else:
        indices = torch.randperm(len(source), generator=torch.Generator().manual_seed(cfg["seed"])).tolist()
        split = round(len(indices) * cfg["validation_ratio"])
        train, val = Subset(source, indices[split:]), Subset(val_source, indices[:split])
    loader_kwargs = dict(
        batch_size=cfg["batch_size"],
        num_workers=cfg["num_workers"],
        pin_memory=device.type == "cuda",
    )
    train_generator = torch.Generator().manual_seed(cfg["seed"] + 10_000)
    train_loader = DataLoader(
        train,
        shuffle=True,
        generator=train_generator,
        worker_init_fn=seed_worker,
        **loader_kwargs,
    )
    val_loader = DataLoader(val, shuffle=False, **loader_kwargs)
    test_loader = DataLoader(test, shuffle=False, **loader_kwargs)
    if cfg.get("model_type", "dq_hfnn") in {"frequency_qpa", "wide_twolevel_frequency_qpa"}:
        model_class = (
            WideTwoLevelFrequencyQPANet
            if cfg["model_type"] == "wide_twolevel_frequency_qpa"
            else FrequencyQPANet
        )
        model = model_class(
            num_classes=cfg["num_classes"],
            hidden_dim=cfg["hidden_dim"],
            mode=cfg.get("frequency_qpa_mode", "torchquantum"),
            entangled=cfg.get("frequency_qpa_entangled", True),
        ).to(device)
    else:
        model = DQHFNN(cfg["num_classes"], cfg["hidden_dim"], cfg["total_pairs"], cfg["random_pair_ratio"], cfg["circuit_variant"], cfg["seed"], cfg.get("pair_source", "pixels"), cfg.get("pairing_layout", "compact"), cfg.get("evaluation_pairing", "fixed"), cfg.get("vectorize_class_circuits", False), cfg.get("quantum_enabled", True), cfg.get("local_tap_index", 6), cfg.get("local_grid_size", 4), cfg.get("frequency_tap_index", 3), cfg.get("frequency_num_groups", 4), cfg.get("frequency_alpha_max", 0.5), cfg.get("frequency_alpha_init", 0.1), cfg.get("measurement_basis", "z"), cfg.get("quantum_auxiliary_weight", 0.0), cfg.get("fusion_alpha_max", 0.5), cfg.get("fusion_alpha_init", 0.1), cfg.get("channel_attention", "none"), cfg.get("fca_num_groups", 16), cfg.get("fca_num_circuits", 4), cfg.get("author_dq_branch_enabled", True), cfg.get("first_pool", "maxpool"), cfg.get("frequency_beta")).to(device)
    quantum_module = (
        getattr(model, "local_frequency_head", None)
        if getattr(model, "local_frequency_head", None) is not None
        else getattr(model, "quantum_logit_head", None)
        if getattr(model, "quantum_logit_head", None) is not None
        else getattr(model, "trunk_frequency_modulator", None)
    )
    if quantum_module is not None and "quantum_weight_decay" in cfg:
        quantum_parameters = list(quantum_module.parameters())
        if model.local_frequency_head is not None:
            quantum_parameters += list(model.fuzzy_projection.parameters())
        quantum_parameter_ids = {id(parameter) for parameter in quantum_parameters}
        classical_parameters = [
            parameter
            for parameter in model.parameters()
            if parameter.requires_grad and id(parameter) not in quantum_parameter_ids
        ]
        optim = torch.optim.SGD(
            [
                {
                    "params": classical_parameters,
                    "weight_decay": cfg["weight_decay"],
                },
                {
                    "params": quantum_parameters,
                    "weight_decay": cfg["quantum_weight_decay"],
                    "lr": cfg.get("quantum_learning_rate", cfg["learning_rate"]),
                },
            ],
            lr=cfg["learning_rate"],
            momentum=cfg["momentum"],
        )
        logger(
            f"OptimizerWeightDecay: classical={cfg['weight_decay']} "
            f"quantum={cfg['quantum_weight_decay']} "
            f"quantum_lr={cfg.get('quantum_learning_rate', cfg['learning_rate'])}"
        )
    else:
        optim = torch.optim.SGD(model.parameters(), lr=cfg["learning_rate"], momentum=cfg["momentum"], weight_decay=cfg["weight_decay"])
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optim, cfg["milestones"], cfg["gamma"])
    loss_fn, best, history = torch.nn.CrossEntropyLoss(), -1.0, []
    metrics_file = (run_dir / "metrics.jsonl").open("w", encoding="utf-8", buffering=1)
    for epoch in range(1, cfg["epochs"] + 1):
        epoch_start = time.perf_counter()
        model.train(); loss_sum = auxiliary_loss_sum = correct = total = 0
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device); optim.zero_grad(set_to_none=True)
            if model.has_quantum_auxiliary:
                logits, classical_logits, quantum_logits = model.hybrid_components(images)
                if cfg.get("quantum_auxiliary_objective", "class_ce") == "residual_direction":
                    auxiliary_loss = residual_direction_loss(
                        quantum_logits, classical_logits, labels
                    )
                else:
                    auxiliary_loss = loss_fn(quantum_logits, labels)
                loss = loss_fn(logits, labels) + model.quantum_auxiliary_weight * auxiliary_loss
                auxiliary_loss_sum += auxiliary_loss.item() * labels.numel()
            else:
                logits = model(images)
                loss = loss_fn(logits, labels)
            loss.backward()
            if cfg.get("max_grad_norm") is not None:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=cfg["max_grad_norm"])
            optim.step()
            loss_sum += loss.item() * labels.numel(); correct += (logits.argmax(1) == labels).sum().item(); total += labels.numel()
        val_metrics = evaluate(model, val_loader, device, cfg["num_classes"], loss_fn); scheduler.step()
        record = {"epoch": epoch, "train_loss": loss_sum / total, "train_accuracy": correct / total, **{f"validation_{key}": value for key, value in val_metrics.items() if key != "confusion_matrix"}}
        if model.has_quantum_auxiliary:
            record["train_quantum_loss"] = auxiliary_loss_sum / total
            record["fusion_alpha"] = model.fusion_alpha.item()
        if cfg.get("first_pool", "maxpool").startswith("qgwp"):
            modulator = model.trunk_frequency_modulator
            if modulator.fixed_beta is None:
                record["pool_alpha_mean"] = modulator.alpha.detach().mean().item()
            else:
                record["pool_beta"] = modulator.fixed_beta
        history.append(record)
        is_best = val_metrics["accuracy"] > best
        logger(
            f"Ep{epoch}: TrLoss={record['train_loss']:.4f} TrAcc={record['train_accuracy'] * 100:.2f}% | "
            f"VLoss={val_metrics['loss']:.4f} VAcc={val_metrics['accuracy'] * 100:.2f}% "
            f"VP={val_metrics['macro_precision'] * 100:.2f}% VR={val_metrics['macro_recall'] * 100:.2f}% "
            f"VS={val_metrics['macro_specificity'] * 100:.2f}% VF1={val_metrics['macro_f1'] * 100:.2f}% | "
            + (
                f" QTrLoss={record['train_quantum_loss']:.4f} Alpha={record['fusion_alpha']:.4f}"
                if model.has_quantum_auxiliary
                else ""
            )
            + (
                f" PoolAlpha={record['pool_alpha_mean']:.4f}"
                if "pool_alpha_mean" in record
                else ""
            )
            + (
                f" PoolBeta={record['pool_beta']:.4f}"
                if "pool_beta" in record
                else ""
            )
            + f" | Time={time.perf_counter() - epoch_start:.1f}s"
            + (" (Best)" if is_best else "")
        )
        metrics_file.write(json.dumps(record) + "\n")
        if is_best:
            best = val_metrics["accuracy"]
            status["best_accuracy"] = best
            status["best_epoch"] = epoch
            torch.save({"epoch": epoch, "state": model.state_dict()}, run_dir / "best.pt")
        status["last_epoch"] = epoch
        write_json(run_dir / "status.json", status)
    metrics_file.close()
    checkpoint = torch.load(run_dir / "best.pt", map_location=device); model.load_state_dict(checkpoint["state"])
    test_metrics = evaluate(model, test_loader, device, cfg["num_classes"], loss_fn)
    selection_name = "best_test_accuracy" if cfg.get("selection_protocol") == "test_per_epoch" else "best_validation_accuracy"
    result = {"best_epoch": checkpoint["epoch"], selection_name: best, **{f"test_{key}": value for key, value in test_metrics.items()}}
    if model.feature_quantum_adapter or model.class_logit_residual:
        result["fusion_alpha"] = model.fusion_alpha.item()
        result["quantum_auxiliary_weight"] = model.quantum_auxiliary_weight
        result["quantum_auxiliary_objective"] = cfg.get(
            "quantum_auxiliary_objective", "class_ce"
        )
    if cfg.get("first_pool", "maxpool").startswith("qgwp"):
        modulator = model.trunk_frequency_modulator
        if modulator.fixed_beta is None:
            pool_alpha = modulator.alpha.detach()
            result["pool_alpha_mean"] = pool_alpha.mean().item()
            result["pool_alpha_min"] = pool_alpha.min().item()
            result["pool_alpha_max"] = pool_alpha.max().item()
        else:
            result["pool_beta"] = modulator.fixed_beta
    write_json(run_dir / "history.json", history)
    write_json(run_dir / "result.json", result)
    status["status"] = "completed"
    status["completed_at"] = datetime.now().astimezone().isoformat()
    write_json(run_dir / "status.json", status)
    logger(json.dumps(result, indent=2))
    logger.close()


if __name__ == "__main__":
    main()
