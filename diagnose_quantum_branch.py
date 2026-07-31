"""Evaluate whether a trained local-frequency model uses its quantum branch."""

import argparse
import json
import sys
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms


ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))

from dq_hfnn.model import DQHFNN
from dq_hfnn.run_io import resolve_latest_run, write_json


def build_model(cfg):
    return DQHFNN(
        cfg["num_classes"],
        cfg["hidden_dim"],
        cfg["total_pairs"],
        cfg["random_pair_ratio"],
        cfg["circuit_variant"],
        cfg["seed"],
        cfg.get("pair_source", "pixels"),
        cfg.get("pairing_layout", "compact"),
        cfg.get("evaluation_pairing", "fixed"),
        cfg.get("vectorize_class_circuits", False),
        cfg.get("quantum_enabled", True),
        cfg.get("local_tap_index", 6),
        cfg.get("local_grid_size", 4),
        cfg.get("frequency_tap_index", 3),
        cfg.get("frequency_num_groups", 4),
        cfg.get("frequency_alpha_max", 0.5),
        cfg.get("frequency_alpha_init", 0.1),
        cfg.get("measurement_basis", "z"),
        cfg.get("quantum_auxiliary_weight", 0.0),
        cfg.get("fusion_alpha_max", 0.5),
        cfg.get("fusion_alpha_init", 0.1),
        cfg.get("channel_attention", "none"),
        cfg.get("fca_num_groups", 16),
        cfg.get("fca_num_circuits", 4),
        cfg.get("author_dq_branch_enabled", True),
        cfg.get("first_pool", "maxpool"),
        cfg.get("frequency_beta"),
    )


def update_confusion(confusion, labels, logits, num_classes):
    predictions = logits.argmax(dim=1).cpu()
    confusion += torch.bincount(
        labels.cpu() * num_classes + predictions,
        minlength=num_classes ** 2,
    ).reshape(num_classes, num_classes)


def metrics_from_confusion(confusion):
    total = confusion.sum().float()
    true_positive = confusion.diag().float()
    precision = true_positive / confusion.sum(dim=0).clamp_min(1)
    recall = true_positive / confusion.sum(dim=1).clamp_min(1)
    f1 = 2 * precision * recall / (precision + recall).clamp_min(1e-12)
    return {
        "accuracy": (true_positive.sum() / total).item(),
        "macro_f1": f1.mean().item(),
    }


def main():
    parser = argparse.ArgumentParser()
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--run-dir", type=Path)
    source.add_argument("--experiment")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--num-workers", type=int)
    args = parser.parse_args()

    run_dir = args.run_dir
    if run_dir is None:
        run_dir = resolve_latest_run(ROOT, args.experiment, args.seed)
    run_dir = run_dir.resolve()
    cfg = json.loads((run_dir / "config.json").read_text(encoding="utf-8"))
    if cfg.get("pair_source") not in {
        "v215_local",
        "v216_grouped_local",
        "v217_relative_energy",
        "v219_simple_entangled",
        "v219_simple_noent",
        "v220_zz_entangled",
        "v220_zz_noent",
        "v223_trunk_entangled",
        "v223_trunk_noent",
        "v224_rawfreq_entangled",
        "v224_rawfreq_noent",
        "frequency_guided_pixels",
        "v226_fuzzy_pooled_quantum",
        "v227_spatial_fuzzy_quantum",
        "v228_superpixel9_uniform_quantum",
        "v230_directional_spatial_quantum",
        "v231_cnn_spatial_frequency_quantum",
        "v236_twolevel_haar_quantum",
        "v236_twolevel_haar_local_rx",
    } and not (
        cfg.get("first_pool", "maxpool").startswith(("qgwp", "qbwp", "qsrp"))
        or cfg.get("channel_attention", "none").startswith("v232_quantum_fca")
        or cfg.get("channel_attention", "none").startswith("v233_local_wavelet_quantum")
        or cfg.get("channel_attention", "none").startswith("v234_projected_wavelet_quantum")
        or cfg.get("channel_attention", "none").startswith("v235_hqnet_projected_wavelet_quantum")
    ):
        raise ValueError("This diagnostic requires a supported quantum-frequency model")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_model(cfg).to(device)
    checkpoint = torch.load(run_dir / "best.pt", map_location=device)
    model.load_state_dict(checkpoint["state"])
    model.eval()

    mean = (0.4914, 0.4822, 0.4465)
    std = (0.2470, 0.2435, 0.2616)
    transform = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize(mean, std)]
    )
    test = datasets.CIFAR10(
        ROOT / "data" / "cifar10", train=False, download=False, transform=transform
    )
    loader = DataLoader(
        test,
        batch_size=args.batch_size or cfg["batch_size"],
        num_workers=args.num_workers if args.num_workers is not None else cfg["num_workers"],
        pin_memory=device.type == "cuda",
    )

    has_softpool_base = (
        model.trunk_frequency_modulator is not None
        and hasattr(model.trunk_frequency_modulator, "base_probabilities")
    )
    modes = (
        ("normal", "base_softpool", "zero_q", "shuffle_q", "no_entanglement")
        if has_softpool_base
        else ("normal", "zero_q", "shuffle_q", "no_entanglement")
    )
    confusion = {
        mode: torch.zeros(cfg["num_classes"], cfg["num_classes"], dtype=torch.long)
        for mode in modes
    }
    quantum_head_confusion = (
        torch.zeros(cfg["num_classes"], cfg["num_classes"], dtype=torch.long)
        if model.feature_quantum_adapter or model.class_logit_residual
        else None
    )
    pool_gate_labels = getattr(
        model.trunk_frequency_modulator, "diagnostic_gate_labels", ()
    )
    feature_stats = {
        "samples": 0,
        "classical_l2_sum": 0.0,
        "quantum_l2_sum": 0.0,
        "quantum_to_classical_ratio_sum": 0.0,
        "cosine_sum": 0.0,
        "residual_direction_cosine_sum": 0.0,
        "detail_l2_sum": 0.0,
        "detail_to_maxpool_ratio_sum": 0.0,
        "gate_sum": torch.zeros(len(pool_gate_labels), dtype=torch.float64),
        "gate_square_sum": torch.zeros(len(pool_gate_labels), dtype=torch.float64),
        "gate_count": 0,
        "selection_entropy_sum": 0.0,
        "selection_max_sum": 0.0,
        "selection_count": 0,
        "refinement_probability_l1_sum": 0.0,
        "refinement_probability_count": 0,
        "refinement_pool_l2_sum": 0.0,
        "refinement_pool_to_base_ratio_sum": 0.0,
    }

    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)
            if model.standalone_quantum:
                normal_logits = model.quantum_class_scores(images)
                variants = [circuit.variant for circuit in model.circuits]
                for circuit in model.circuits:
                    circuit.variant = "no_ent"
                no_entanglement_logits = model.quantum_class_scores(images)
                for circuit, variant in zip(model.circuits, variants):
                    circuit.variant = variant
                logits = {
                    "normal": normal_logits,
                    "zero_q": torch.zeros_like(normal_logits),
                    "shuffle_q": normal_logits.roll(1, dims=0),
                    "no_entanglement": no_entanglement_logits,
                }
                classical = None
                quantum = normal_logits
            elif model.class_logit_residual:
                classical_features, local_features = model.classical.forward_with_local_features(
                    images, model.local_tap_index
                )
                classical_logits = model.classifier(classical_features)
                head = model.quantum_logit_head
                quantum_logits = head(local_features)
                was_entangled = head.entangled
                head.entangled = False
                no_entanglement_quantum = head(local_features)
                head.entangled = was_entangled
                alpha = model.fusion_alpha
                logits = {
                    "normal": classical_logits + alpha * quantum_logits,
                    "zero_q": classical_logits,
                    "shuffle_q": classical_logits + alpha * quantum_logits.roll(1, dims=0),
                    "no_entanglement": classical_logits + alpha * no_entanglement_quantum,
                }
                update_confusion(
                    quantum_head_confusion,
                    labels,
                    quantum_logits,
                    cfg["num_classes"],
                )
                classical = classical_logits
                quantum = alpha * quantum_logits
            elif model.feature_quantum_adapter:
                classical_features, local_features = model.classical.forward_with_local_features(
                    images, model.local_tap_index
                )
                classical_logits = model.classifier(classical_features)
                quantum_logits = model.quantum_class_scores(local_features)
                alpha = model.fusion_alpha
                variants = [circuit.variant for circuit in model.circuits]
                for circuit in model.circuits:
                    circuit.variant = "no_ent"
                no_entanglement_quantum = model.quantum_class_scores(local_features)
                for circuit, variant in zip(model.circuits, variants):
                    circuit.variant = variant
                logits = {
                    "normal": classical_logits + alpha * quantum_logits,
                    "zero_q": classical_logits,
                    "shuffle_q": classical_logits + alpha * quantum_logits.roll(1, dims=0),
                    "no_entanglement": classical_logits + alpha * no_entanglement_quantum,
                }
                update_confusion(
                    quantum_head_confusion,
                    labels,
                    quantum_logits,
                    cfg["num_classes"],
                )
                classical = classical_logits
                quantum = alpha * quantum_logits
            elif model.trunk_frequency_modulator is not None:
                modulator = model.trunk_frequency_modulator
                if hasattr(modulator, "components"):
                    pool_input = model.classical.features[0](images)
                    pool_baseline, pool_gates, pool_detail = modulator.components(
                        pool_input
                    )
                    detail_l2 = pool_detail.flatten(1).norm(dim=1)
                    maxpool_l2 = pool_baseline.flatten(1).norm(dim=1)
                    feature_stats["detail_l2_sum"] += detail_l2.sum().item()
                    feature_stats["detail_to_maxpool_ratio_sum"] += (
                        detail_l2 / maxpool_l2.clamp_min(1e-12)
                    ).sum().item()
                    reduced_gates = pool_gates.detach().double().cpu()
                    feature_stats["gate_sum"] += reduced_gates.sum(
                        dim=(0, 1, 2, 3)
                    )
                    feature_stats["gate_square_sum"] += reduced_gates.square().sum(
                        dim=(0, 1, 2, 3)
                    )
                    feature_stats["gate_count"] += reduced_gates[..., 0].numel()
                    if getattr(modulator, "is_probability_pool", False):
                        feature_stats["selection_entropy_sum"] += (
                            -(reduced_gates * reduced_gates.clamp_min(1e-12).log())
                            .sum(dim=-1)
                            .sum()
                            .item()
                        )
                        feature_stats["selection_max_sum"] += (
                            reduced_gates.max(dim=-1).values.sum().item()
                        )
                        feature_stats["selection_count"] += reduced_gates[..., 0].numel()
                    if hasattr(modulator, "base_probabilities"):
                        pool_patches = modulator.patches(pool_input)
                        base_probabilities = modulator.base_probabilities(pool_patches)
                        feature_stats["refinement_probability_l1_sum"] += (
                            (pool_gates - base_probabilities).abs().sum(dim=-1).sum().item()
                        )
                        feature_stats["refinement_probability_count"] += (
                            pool_gates[..., 0].numel()
                        )
                        channels_per_group = (
                            modulator.channels // modulator.num_groups
                        )
                        base_channel_probabilities = base_probabilities.repeat_interleave(
                            channels_per_group, dim=1
                        )
                        base_pool = (
                            pool_patches * base_channel_probabilities
                        ).sum(dim=-1)
                        refined_pool = pool_baseline + pool_detail
                        refinement_delta = refined_pool - base_pool
                        refinement_l2 = refinement_delta.flatten(1).norm(dim=1)
                        base_l2 = base_pool.flatten(1).norm(dim=1)
                        feature_stats["refinement_pool_l2_sum"] += (
                            refinement_l2.sum().item()
                        )
                        feature_stats["refinement_pool_to_base_ratio_sum"] += (
                            refinement_l2 / base_l2.clamp_min(1e-12)
                        ).sum().item()
                was_mode, was_entangled = modulator.mode, modulator.entangled
                features = {}
                runtime_modes = [
                    ("normal", "normal", was_entangled),
                    ("zero_q", "zero", was_entangled),
                    ("shuffle_q", "shuffle", was_entangled),
                    ("no_entanglement", "normal", False),
                ]
                if has_softpool_base:
                    runtime_modes.insert(1, ("base_softpool", "base", was_entangled))
                for mode, runtime_mode, entangled in runtime_modes:
                    modulator.mode = runtime_mode
                    modulator.entangled = entangled
                    features[mode] = model.classical(images)
                modulator.mode, modulator.entangled = was_mode, was_entangled
                base_dq_features = (
                    model.quantum_features(images)
                    if model.channel_attention != "none"
                    and model.author_dq_branch_enabled
                    else 0.0
                )
                logits = {
                    mode: model.classifier(value + base_dq_features)
                    for mode, value in features.items()
                }
                classical = features["zero_q"] + base_dq_features
                quantum = features["normal"] - classical
                if model.channel_attention != "none":
                    quantum = features["normal"] - features["zero_q"]
            else:
                if model.local_frequency_head is not None:
                    classical, local = model.classical.forward_with_local_features(
                        images, model.local_tap_index
                    )
                    quantum = model.quantum_features(images, local)
                else:
                    classical = model.classical(images)
                    quantum = model.quantum_features(images)
                logits = {
                    "normal": model.classifier(classical + quantum),
                    "zero_q": model.classifier(classical),
                    "shuffle_q": model.classifier(classical + quantum.roll(1, dims=0)),
                }

                if model.local_frequency_head is not None:
                    head = model.local_frequency_head
                    was_entangled = head.entangled
                    head.entangled = False
                    no_entanglement_quantum = model.quantum_features(images, local)
                    head.entangled = was_entangled
                else:
                    variants = [circuit.variant for circuit in model.circuits]
                    for circuit in model.circuits:
                        circuit.variant = "no_ent"
                    no_entanglement_quantum = model.quantum_features(images)
                    for circuit, variant in zip(model.circuits, variants):
                        circuit.variant = variant
                logits["no_entanglement"] = model.classifier(
                    classical + no_entanglement_quantum
                )

            for mode in modes:
                update_confusion(confusion[mode], labels, logits[mode], cfg["num_classes"])

            quantum_l2 = quantum.norm(dim=1)
            count = images.shape[0]
            feature_stats["samples"] += count
            feature_stats["quantum_l2_sum"] += quantum_l2.sum().item()
            if classical is not None:
                classical_l2 = classical.norm(dim=1)
                feature_stats["classical_l2_sum"] += classical_l2.sum().item()
                feature_stats["quantum_to_classical_ratio_sum"] += (
                    quantum_l2 / classical_l2.clamp_min(1e-12)
                ).sum().item()
                feature_stats["cosine_sum"] += F.cosine_similarity(
                    classical, quantum, dim=1
                ).sum().item()
                if model.class_logit_residual:
                    residual_target = F.one_hot(
                        labels.to(device), num_classes=cfg["num_classes"]
                    ).to(quantum_logits.dtype) - classical.detach().softmax(dim=1)
                    centered_quantum = quantum_logits - quantum_logits.mean(
                        dim=1, keepdim=True
                    )
                    feature_stats["residual_direction_cosine_sum"] += (
                        F.cosine_similarity(
                            centered_quantum, residual_target, dim=1
                        ).sum().item()
                    )

    mode_metrics = {
        mode: metrics_from_confusion(value) for mode, value in confusion.items()
    }
    baseline = mode_metrics["normal"]["accuracy"]
    for mode, values in mode_metrics.items():
        values["accuracy_delta_pp_vs_normal"] = 100.0 * (
            values["accuracy"] - baseline
        )

    samples = feature_stats.pop("samples")
    result = {
        "run_dir": str(run_dir),
        "checkpoint_epoch": checkpoint["epoch"],
        "integration": (
            "standalone_quantum_classifier"
            if model.standalone_quantum
            else "two_level_haar_quantum_logit_residual"
            if model.class_logit_residual
            else "cnn_spatial_frequency_logit_adapter"
            if model.feature_quantum_adapter
            else "raw_image_dense_residual"
            if cfg.get("pair_source", "").startswith("v224_rawfreq")
            else "quantum_fca_channel_attention"
            if cfg.get("channel_attention", "none").startswith("v232_quantum_fca")
            else "quantum_local_wavelet_attention"
            if cfg.get("channel_attention", "none").startswith("v233_local_wavelet_quantum")
            else "quantum_projected_wavelet_attention"
            if cfg.get("channel_attention", "none").startswith("v234_projected_wavelet_quantum")
            else "hqnet_rqc_projected_wavelet_attention"
            if cfg.get("channel_attention", "none").startswith("v235_hqnet_projected_wavelet_quantum")
            else "fixed_quantum_gated_haar_detail_pool"
            if cfg.get("first_pool", "maxpool") == "qgwp_fixed_quantum_local_rx"
            else "fixed_classical_gated_haar_detail_pool"
            if cfg.get("first_pool", "maxpool") == "qgwp_fixed_classical_matched"
            else "fixed_uniform_haar_detail_pool"
            if cfg.get("first_pool", "maxpool") == "qgwp_fixed_uniform"
            else "quantum_born_haar_pool_local_rx"
            if cfg.get("first_pool", "maxpool") == "qbwp_quantum_local_rx"
            else "quantum_born_haar_pool_entangled"
            if cfg.get("first_pool", "maxpool") == "qbwp_quantum_entangled"
            else "classical_matched_born_haar_pool"
            if cfg.get("first_pool", "maxpool") == "qbwp_classical_matched"
            else "soft_born_haar_pool_control"
            if cfg.get("first_pool", "maxpool") == "qbwp_softpool"
            else "quantum_refined_softpool_local_rx"
            if cfg.get("first_pool", "maxpool") == "qsrp_quantum_local_rx"
            else "quantum_refined_softpool_entangled"
            if cfg.get("first_pool", "maxpool") == "qsrp_quantum_entangled"
            else "classical_refined_softpool_matched"
            if cfg.get("first_pool", "maxpool") == "qsrp_classical_matched"
            else "quantum_gated_haar_detail_pool"
            if cfg.get("first_pool", "maxpool").startswith("qgwp_quantum")
            else "classical_gated_haar_detail_pool"
            if cfg.get("first_pool", "maxpool") == "qgwp_classical_matched"
            else "trunk_dense_residual"
            if model.trunk_frequency_modulator is not None
            else "terminal_feature_addition"
        ),
        "modes": mode_metrics,
        "features": (
            {"quantum_logit_l2_mean": feature_stats["quantum_l2_sum"] / samples}
            if model.standalone_quantum
            else {
                "classical_l2_mean": feature_stats["classical_l2_sum"] / samples,
                "quantum_l2_mean": feature_stats["quantum_l2_sum"] / samples,
                "quantum_to_classical_ratio_mean": feature_stats[
                    "quantum_to_classical_ratio_sum"
                ]
                / samples,
                "classical_quantum_cosine_mean": feature_stats["cosine_sum"] / samples,
            }
        ),
    }
    if model.trunk_frequency_modulator is not None:
        modulator = model.trunk_frequency_modulator
        if getattr(modulator, "is_probability_pool", False):
            result["modulation"] = {
                "gate_type": modulator.gate_type,
                "entangled": modulator.entangled,
            }
        else:
            if modulator.fixed_beta is None:
                alpha = modulator.alpha.detach().cpu()
                result["modulation"] = {
                    "alpha_mean": alpha.mean().item(),
                    "alpha_min": alpha.min().item(),
                    "alpha_max": alpha.max().item(),
                    "configured_alpha_max": modulator.alpha_max,
                }
            else:
                result["modulation"] = {"fixed_beta": modulator.fixed_beta}
        if feature_stats["gate_count"]:
            gate_mean = feature_stats["gate_sum"] / feature_stats["gate_count"]
            gate_variance = (
                feature_stats["gate_square_sum"] / feature_stats["gate_count"]
                - gate_mean.square()
            ).clamp_min(0.0)
            band_names = pool_gate_labels
            result["pool_detail"] = {
                "gate_mean": {
                    name: gate_mean[index].item()
                    for index, name in enumerate(band_names)
                },
                "gate_std": {
                    name: gate_variance[index].sqrt().item()
                    for index, name in enumerate(band_names)
                },
                "detail_l2_mean": feature_stats["detail_l2_sum"] / samples,
                "detail_to_maxpool_ratio_mean": feature_stats[
                    "detail_to_maxpool_ratio_sum"
                ]
                / samples,
            }
            if feature_stats["selection_count"]:
                result["pool_detail"]["selection_entropy_mean"] = (
                    feature_stats["selection_entropy_sum"]
                    / feature_stats["selection_count"]
                )
                result["pool_detail"]["selection_entropy_normalized_mean"] = (
                    feature_stats["selection_entropy_sum"]
                    / feature_stats["selection_count"]
                    / torch.log(torch.tensor(len(pool_gate_labels))).item()
                )
                result["pool_detail"]["selection_max_probability_mean"] = (
                    feature_stats["selection_max_sum"]
                    / feature_stats["selection_count"]
                )
            if feature_stats["refinement_probability_count"]:
                result["pool_refinement"] = {
                    "probability_l1_from_softpool_mean": (
                        feature_stats["refinement_probability_l1_sum"]
                        / feature_stats["refinement_probability_count"]
                    ),
                    "refined_pool_delta_l2_mean": (
                        feature_stats["refinement_pool_l2_sum"] / samples
                    ),
                    "refined_pool_delta_to_softpool_ratio_mean": (
                        feature_stats["refinement_pool_to_base_ratio_sum"]
                        / samples
                    ),
                }
    if model.feature_quantum_adapter or model.class_logit_residual:
        result["fusion"] = {
            "alpha": model.fusion_alpha.item(),
            "configured_alpha_max": model.fusion_alpha_max,
            "quantum_auxiliary_weight": model.quantum_auxiliary_weight,
        }
        result["quantum_head"] = metrics_from_confusion(quantum_head_confusion)
        if model.class_logit_residual:
            result["residual_alignment"] = {
                "quantum_residual_direction_cosine_mean": feature_stats[
                    "residual_direction_cosine_sum"
                ]
                / samples,
                "auxiliary_objective": cfg.get(
                    "quantum_auxiliary_objective", "class_ce"
                ),
            }
    if model.class_position_logits is not None:
        weights = model.class_position_logits.softmax(dim=1).detach().cpu()
        entropy = -(weights * weights.clamp_min(1e-12).log()).sum(dim=1)
        result["spatial_aggregation"] = {
            "mean_max_position_weight": weights.max(dim=1).values.mean().item(),
            "normalized_entropy_mean": (
                entropy / torch.log(torch.tensor(weights.shape[1], dtype=weights.dtype))
            ).mean().item(),
        }
    write_json(run_dir / "quantum_diagnostics.json", result)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
