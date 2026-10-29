import argparse
import json
import os
import time

import numpy as np
import torch

from createGraph import create_graph, rdSLIC
from loadData import data_read, data_reader, split_data
from model import VCASAHGCN, utils
from model.interaction_config import add_interaction_arguments, interaction_kwargs, InteractionConfig
from model.checkpoint_io import save_model_checkpoint
from model.research_config import ResearchConfig

try:
    from tqdm.auto import tqdm
except ImportError:
    tqdm = None


def make_progress(iterable, enabled=True, **kwargs):
    if enabled and tqdm is not None:
        return tqdm(iterable, **kwargs)
    return iterable


def log_message(message, progress_enabled=True):
    if progress_enabled and tqdm is not None:
        tqdm.write(str(message))
    else:
        print(message)


def safe_name(name):
    return "".join(ch if ch.isalnum() or ch in ["-", "_"] else "_" for ch in name)


def checkpoint_paths(save_path, dataset_names):
    root, ext = os.path.splitext(save_path)
    if ext == "":
        ext = ".pt"
        save_path = root + ext
    paths = {
        "primary": save_path,
        "macro_oa": root + "_best_macro_oa" + ext,
        "val_loss": root + "_best_val_loss" + ext,
        "last": root + "_last" + ext,
        "datasets": {
            name: root + "_best_{}_oa".format(safe_name(name)) + ext
            for name in dataset_names
        },
    }
    return paths


def save_checkpoint(model, path):
    save_model_checkpoint(model, path)


def normalize_maxmin(mx, axis=2):
    mx_min = mx.min()
    if mx_min < 0:
        mx = mx + abs(mx_min)

    if axis == 1:
        m_min = np.amin(mx, axis=1)
        m_max = np.amax(mx, axis=1)
        denom = np.maximum(m_max - m_min, 1e-12)
        for i in range(mx.shape[1]):
            mx[:, i] = (mx[:, i] - m_min) / denom
    elif axis == 0:
        m_min = np.amin(mx, axis=0)
        m_max = np.amax(mx, axis=0)
        denom = np.maximum(m_max - m_min, 1e-12)
        for i in range(mx.shape[0]):
            mx[i, :] = (mx[i, :] - m_min) / denom
    elif axis == 2:
        m_min = np.amin(mx)
        m_max = np.amax(mx)
        mx = (mx - m_min) / max(m_max - m_min, 1e-12)
    else:
        raise ValueError("axis must be 0, 1, or 2")
    return mx


def build_spatial_index(height, width):
    idx = np.ones((height, width))
    idx = np.where(idx == 1)
    idx_x = np.resize(idx[0], (height, width, 1))
    idx_y = np.resize(idx[1], (height, width, 1))
    return normalize_maxmin(np.concatenate((idx_x, idx_y), axis=2))


def to_tensor(array, device):
    return torch.from_numpy(array.astype(np.float32)).to(device)


def prepare_dataset(data_set_name, args, device, seed):
    print("\n==================== prepare {} ====================".format(data_set_name))
    data, data_gt = data_read.readData(data_set_name)
    data = np.asarray(data)
    data_gt = np.asarray(data_gt)
    height, width, bands = data.shape
    class_num = int(np.max(data_gt))
    wavelengths, fwhm = data_read.readWavelengths(data_set_name, bands)

    gt_reshape = np.reshape(data_gt, [-1])
    np.random.seed(seed)
    train_index, val_index, test_index = split_data.split_data(
        gt_reshape,
        class_num,
        args.train_ratio,
        args.val_ratio,
        args.train_num,
        args.val_num,
        args.samples_type,
    )
    train_samples_gt, test_samples_gt, val_samples_gt = create_graph.get_label(
        gt_reshape,
        train_index,
        val_index,
        test_index,
    )
    train_label_mask, test_label_mask, val_label_mask = create_graph.get_label_mask(
        train_samples_gt,
        test_samples_gt,
        val_samples_gt,
        data_gt,
        class_num,
    )

    train_gt = np.reshape(train_samples_gt, [height, width])
    test_gt = np.reshape(test_samples_gt, [height, width])
    val_gt = np.reshape(val_samples_gt, [height, width])
    data_reader.data_info(train_gt, val_gt, test_gt)

    slic_scale = args.pavia_superpixel_scale if data_set_name == "paviaU" else args.superpixel_scale
    if args.graph_mode == "static":
        print("LDA-SLIC for {} is processing".format(data_set_name))
        slic = rdSLIC.LDA_SLIC(data, train_gt, class_num - 1)
        tic = time.time()
        Q, _, A, _ = slic.simple_superpixel(scale=slic_scale)
        slic_time = time.time() - tic
        print("{} LDA-SLIC costs time: {:.2f}s".format(data_set_name, slic_time))
        Q = VCASAHGCN.dense_assignment_to_sparse_tensor(Q, device)
        A = to_tensor(A, device)
    else:
        print(
            "{} uses learned sparse assignments and dynamic graphs; skip LDA-SLIC".format(
                data_set_name
            )
        )
        Q = None
        A = None

    net_input = normalize_maxmin(np.array(data, np.float32))
    net_input = np.concatenate((net_input, build_spatial_index(height, width)), axis=2)

    item = {
        "name": data_set_name,
        "seed": int(seed),
        "height": height,
        "width": width,
        "bands": bands,
        "class_num": class_num,
        "Q": Q,
        "A": A,
        "superpixel_scale": slic_scale,
        "net_input": to_tensor(net_input, device),
        "wavelengths": to_tensor(wavelengths.reshape(-1), device),
        "fwhm": None if fwhm is None else to_tensor(fwhm.reshape(-1), device),
        "train_samples_gt": to_tensor(train_samples_gt, device),
        "test_samples_gt": to_tensor(test_samples_gt, device),
        "val_samples_gt": to_tensor(val_samples_gt, device),
        "train_gt_onehot": to_tensor(create_graph.label_to_one_hot(train_gt, class_num), device),
        "test_gt_onehot": to_tensor(create_graph.label_to_one_hot(test_gt, class_num), device),
        "val_gt_onehot": to_tensor(create_graph.label_to_one_hot(val_gt, class_num), device),
        "train_label_mask": to_tensor(train_label_mask, device),
        "test_label_mask": to_tensor(test_label_mask, device),
        "val_label_mask": to_tensor(val_label_mask, device),
        "zeros": torch.zeros([height * width], device=device).float(),
    }
    return item


def make_band_mixup_input(
        base_input,
        item,
        mixup_ratio,
        mixup_alpha,
        neighbor_count,
        mixup_prob=1.0,
        available_band_indices=None):
    bands = item["bands"]
    if mixup_ratio <= 0 or mixup_prob <= 0 or bands < 2:
        return base_input

    device = base_input.device
    if mixup_prob < 1.0 and float(torch.rand((), device=device).item()) >= mixup_prob:
        return base_input

    if available_band_indices is None:
        available = torch.arange(bands, device=device, dtype=torch.long)
    else:
        available = torch.as_tensor(available_band_indices, device=device, dtype=torch.long).reshape(-1)
        available = available[(available >= 0) & (available < bands)]
        available = torch.unique(available, sorted=True)
    if available.numel() < 2:
        return base_input

    mix_count = int(np.ceil(float(available.numel()) * mixup_ratio))
    mix_count = min(max(mix_count, 1), int(available.numel()))
    neighbor_count = min(max(int(neighbor_count), 1), int(available.numel()) - 1)

    wavelengths = item.get("wavelengths")
    if wavelengths is None:
        band_position = torch.arange(bands, device=device, dtype=base_input.dtype)
    else:
        band_position = torch.as_tensor(wavelengths, device=device, dtype=base_input.dtype).reshape(-1)
        if band_position.numel() != bands:
            band_position = torch.arange(bands, device=device, dtype=base_input.dtype)

    available_position = band_position[available]
    distance = torch.abs(available_position[:, None] - available_position[None, :])
    distance.fill_diagonal_(float("inf"))
    nearest_positions = torch.topk(distance, k=neighbor_count, largest=False).indices

    selected_positions = torch.randperm(int(available.numel()), device=device)[:mix_count]
    target_bands = available[selected_positions]
    neighbor_candidates = nearest_positions[selected_positions]
    neighbor_choices = torch.randint(0, neighbor_count, (mix_count,), device=device)
    source_positions = neighbor_candidates[torch.arange(mix_count, device=device), neighbor_choices]
    source_bands = available[source_positions]

    train_pixel_idx = (item["train_samples_gt"] != 0).nonzero(as_tuple=True)[0]
    if train_pixel_idx.numel() == 0:
        return base_input

    x = base_input.clone()
    x_flat = x.reshape(item["height"] * item["width"], -1)
    beta = torch.distributions.Beta(
        torch.tensor(mixup_alpha, device=device, dtype=base_input.dtype),
        torch.tensor(mixup_alpha, device=device, dtype=base_input.dtype),
    )
    lam = beta.sample([mix_count]).to(device=device, dtype=base_input.dtype)
    lam = torch.maximum(lam, 1.0 - lam)

    target_values = x_flat[train_pixel_idx[:, None], target_bands[None, :]].clone()
    source_values = x_flat[train_pixel_idx[:, None], source_bands[None, :]]
    x_flat[train_pixel_idx[:, None], target_bands[None, :]] = (
        lam.view(1, -1) * target_values + (1.0 - lam).view(1, -1) * source_values
    )
    return x


def make_train_input(
        item,
        drop_ratio,
        mixup_ratio=0.0,
        mixup_alpha=0.4,
        mixup_neighbors=1,
        mixup_prob=1.0):
    if drop_ratio <= 0 and mixup_ratio <= 0:
        return item["net_input"]

    bands = item["bands"]
    device = item["net_input"].device
    available = torch.arange(bands, device=device, dtype=torch.long)
    x = item["net_input"]

    if drop_ratio > 0:
        drop_count = int(np.floor(bands * drop_ratio))
        drop_count = min(max(drop_count, 1), bands - 1)
        dropped = torch.randperm(bands, device=device)[:drop_count]
        keep_mask = torch.ones(bands, device=device, dtype=torch.bool)
        keep_mask[dropped] = False
        available = available[keep_mask]
        x = x.clone()
        train_pixel_idx = (item["train_samples_gt"] != 0).nonzero(as_tuple=True)[0]
        x_flat = x.reshape(item["height"] * item["width"], -1)
        x_flat[train_pixel_idx[:, None], dropped[None, :]] = 0.0

    return make_band_mixup_input(
        x,
        item,
        mixup_ratio=mixup_ratio,
        mixup_alpha=mixup_alpha,
        neighbor_count=mixup_neighbors,
        mixup_prob=mixup_prob,
        available_band_indices=available,
    )


def supervised_loss(outputs, item, split, class_balance=False, graph_loss_weights=None,
                    cnn_loss_weight=1.0, graph_branch_loss_weight=1.0):
    y, y_hgnn, y_cnn = outputs[:3]
    onehot = item["{}_gt_onehot".format(split)]
    mask = item["{}_label_mask".format(split)]
    loss = (
        utils.compute_loss(y, onehot, mask, class_balance=class_balance)
        + graph_branch_loss_weight * utils.compute_loss(y_hgnn, onehot, mask, class_balance=class_balance)
        + cnn_loss_weight * utils.compute_loss(y_cnn, onehot, mask, class_balance=class_balance)
    )
    if len(outputs) > 3 and graph_loss_weights is not None:
        graph_info = outputs[3]
        loss = (
            loss
            + graph_loss_weights["compactness"] * graph_info.get("compactness_loss", 0.0)
            + graph_loss_weights["entropy"] * graph_info.get("assignment_entropy_loss", 0.0)
            + graph_loss_weights["balance"] * graph_info.get("assignment_balance_loss", 0.0)
        )
    return loss


def apply_stable_preset(args):
    if args.stable_preset == "none":
        return

    if args.stable_preset in ["xuzhou_k32", "xuzhou_k32_index"]:
        args.latent_channels = 32
        args.adapter_embed_dim = 16

    if args.stable_preset in ["xuzhou_index", "xuzhou_k32_index"]:
        args.adapter_attention_mode = "index"


def prediction_histogram(network_output, item, split):
    labels = item["{}_samples_gt".format(split)]
    valid = labels != 0
    if valid.sum() == 0:
        return []
    preds = torch.argmax(network_output, dim=1)[valid]
    counts = torch.bincount(preds, minlength=item["class_num"]).detach().cpu().numpy()
    total = max(int(counts.sum()), 1)
    top_indices = np.argsort(counts)[::-1][:3]
    return [
        (int(index + 1), int(counts[index]), round(float(counts[index]) / total, 4))
        for index in top_indices
        if counts[index] > 0
    ]


def evaluate_split(model, datasets, split):
    model.eval()
    losses = []
    oas = {}
    pred_hists = {}
    with torch.no_grad():
        for name, item in datasets.items():
            outputs = model(
                name,
                item["net_input"],
                item["Q"],
                item["A"],
                wavelengths=item["wavelengths"],
                fwhm=item["fwhm"],
                superpixel_scale=item["superpixel_scale"],
            )
            y = outputs[0]
            loss = utils.compute_loss(y, item["{}_gt_onehot".format(split)], item["{}_label_mask".format(split)])
            oa = utils.evaluate_performance(
                y,
                item["{}_samples_gt".format(split)],
                item["{}_gt_onehot".format(split)],
                item["zeros"],
            )
            losses.append(loss)
            oas[name] = float(oa)
            pred_hists[name] = prediction_histogram(y, item, split)
    return torch.stack(losses).mean(), oas, pred_hists


def build_parser():
    parser = argparse.ArgumentParser(description="Multi-dataset VCA-SAHGCN training")
    add_interaction_arguments(parser)
    parser.add_argument("--cnn-loss-weight", type=float, default=1.0)
    parser.add_argument("--graph-branch-loss-weight", type=float, default=1.0)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--datasets", type=str, default="paviaU")
    parser.add_argument("--skip-missing", action="store_true")
    parser.add_argument("--epochs", type=int, default=500)
    parser.add_argument("--steps-per-epoch", type=int, default=0)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--samples-type", type=str, default="ratio", choices=["ratio", "number"])
    parser.add_argument("--train-num", type=int, default=30)
    parser.add_argument("--val-num", type=int, default=10)
    parser.add_argument("--train-ratio", type=float, default=0.01)
    parser.add_argument("--val-ratio", type=float, default=0.005)
    parser.add_argument("--val-interval", type=int, default=5)
    parser.add_argument("--superpixel-scale", type=int, default=100)
    parser.add_argument("--pavia-superpixel-scale", type=int, default=50)
    parser.add_argument("--graph-mode", type=str, default="static", choices=["static", "learned"],
                        help="Use precomputed LDA-SLIC graphs or learn sparse assignments and graphs end to end.")
    parser.add_argument("--learned-assignment-dim", type=int, default=32)
    parser.add_argument("--learned-assignment-radius", type=int, default=1)
    parser.add_argument("--learned-graph-radius", type=int, default=1)
    parser.add_argument("--learned-assignment-temperature", type=float, default=1.0)
    parser.add_argument("--learned-assignment-spatial-sigma", type=float, default=1.0)
    parser.add_argument("--learned-graph-spatial-sigma", type=float, default=1.5)
    parser.add_argument("--graph-compactness-weight", type=float, default=0.01)
    parser.add_argument("--graph-entropy-weight", type=float, default=0.001)
    parser.add_argument("--graph-balance-weight", type=float, default=0.01)
    parser.add_argument("--latent-channels", type=int, default=128)
    parser.add_argument("--adapter-embed-dim", type=int, default=64)
    parser.add_argument("--locality-sigma", type=float, default=0.08)
    parser.add_argument("--locality-window", type=float, default=0.16)
    parser.add_argument("--adapter-attention-mode", type=str, default="wavelength",
                        choices=["wavelength", "index", "uniform"])
    parser.add_argument("--adapter-chunk-pixels", type=int, default=0,
                        help=(
                            "Split VCA value projection by pixel chunks to reduce peak GPU memory. "
                            "Use values like 32768 or 65536 for large full-image datasets."
                        ))
    parser.add_argument("--no-adapter-checkpoint", action="store_true",
                        help="Disable gradient checkpointing inside chunked VCA value projection.")
    parser.add_argument("--no-wavelength-attention", action="store_true",
                        help="Shortcut for --adapter-attention-mode index.")
    parser.add_argument("--stable-preset", type=str, default="none",
                        choices=["none", "xuzhou_k32", "xuzhou_index", "xuzhou_k32_index"],
                        help=(
                            "Non-destructive shortcut for Xuzhou-stable shared VCA settings. "
                            "xuzhou_k32 sets latent channels/embed dim to 32/16; "
                            "xuzhou_index uses index attention; xuzhou_k32_index applies both."
                        ))
    parser.add_argument("--wavelength-min", type=float, default=400.0)
    parser.add_argument("--wavelength-max", type=float, default=2500.0)
    parser.add_argument("--lr", type=float, default=5e-5)
    parser.add_argument("--head-lr", type=float, default=5e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-3)
    parser.add_argument("--train-drop-band-ratio", type=float, default=0.0)
    parser.add_argument("--train-band-mixup-ratio", type=float, default=0.0)
    parser.add_argument("--train-band-mixup-prob", type=float, default=1.0)
    parser.add_argument("--train-band-mixup-alpha", type=float, default=0.4)
    parser.add_argument("--train-band-mixup-neighbors", type=int, default=1)
    parser.add_argument("--no-balanced-loss", action="store_true")
    parser.add_argument("--norm-type", type=str, default="domain_bn",
                        choices=["domain_bn", "batch_norm", "group_norm"])
    parser.add_argument("--primary-checkpoint", type=str, default="macro_oa",
                        choices=["macro_oa", "val_loss"])
    parser.add_argument("--eval-checkpoint", type=str, default="primary")
    parser.add_argument("--no-progress", action="store_true")
    parser.add_argument("--step-progress", action="store_true")
    parser.add_argument("--save-path", type=str, default=None)
    parser.add_argument("--metrics-json", type=str, default=None)
    parser.add_argument("--experiment-name", type=str, default=None)
    return parser


def main(argv=None):
    return train(build_parser().parse_args(argv))


def train(args, run_record=None):
    if args.no_wavelength_attention:
        args.adapter_attention_mode = "index"
    apply_stable_preset(args)
    resolved = ResearchConfig.from_legacy(args)
    interaction_options = interaction_kwargs(args)
    interaction_config = InteractionConfig(**interaction_options)
    # The seed previously controlled splits only; also seed weights and training randomness.
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    device = torch.device(args.device)
    dataset_names = [name.strip() for name in args.datasets.split(",") if name.strip()]
    if len(dataset_names) == 0:
        raise ValueError("at least one dataset is required")

    datasets = {}
    for index, name in enumerate(dataset_names):
        try:
            datasets[name] = prepare_dataset(name, args, device, seed=args.seed + index)
        except Exception as exc:
            if not args.skip_missing:
                raise
            print("skip {} because it cannot be prepared: {}".format(name, exc))
    if len(datasets) == 0:
        raise RuntimeError("no dataset was prepared")

    if run_record is not None:
        run_record.record_data(datasets)

    class_counts = {name: item["class_num"] for name, item in datasets.items()}
    model = resolved.build_model(class_counts).to(device)
    model.training_config = vars(args).copy()
    model.training_config["research_config"] = resolved.to_dict()
    if run_record is not None:
        run_record.record_model(model)
        model.training_config["research_run"] = {"run_id": run_record.run_id,
                                                  "attempt": run_record.attempt.name,
                                                  "config_hash": resolved.config_hash}
    model.training_config["dataset_seeds"] = {
        name: int(item.get("seed", args.seed + index)) for index, (name, item) in enumerate(datasets.items())
    }

    head_params = list(model.heads.parameters()) + list(model.hgnn_heads.parameters()) + list(model.cnn_heads.parameters())
    optimizer = torch.optim.Adam(
        [
            {"params": model.backbone.parameters(), "lr": args.lr},
            {"params": head_params, "lr": args.head_lr},
        ],
        weight_decay=args.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=0.5,
        patience=20,
        threshold=0.0001,
        threshold_mode="rel",
    )

    save_path = args.save_path
    if save_path is None:
        os.makedirs(os.path.join(os.getcwd(), "weights"), exist_ok=True)
        checkpoint_name = (
            "vca_multidataset_learned_graph.pt"
            if args.graph_mode == "learned"
            else "vca_multidataset.pt"
        )
        if interaction_config.active:
            checkpoint_name = "vca_{}_{}_{}_{}_{}.pt".format(
                args.graph_mode, args.alignment_mode, args.interaction_mode,
                args.interaction_direction, args.graph_update_mode,
            )
        save_path = os.path.join(os.getcwd(), "weights", checkpoint_name)

    rng = np.random.default_rng(args.seed)
    names = list(datasets.keys())
    ckpt_paths = checkpoint_paths(save_path, names)
    steps_per_epoch = args.steps_per_epoch if args.steps_per_epoch > 0 else max(len(names) * 3, len(names))
    best_val_loss = None
    best_macro_oa = -1.0
    best_dataset_oa = {name: -1.0 for name in names}
    use_balanced_loss = not args.no_balanced_loss
    progress_enabled = not args.no_progress

    print("\n==================== train shared VCA-SAHGCN ====================")
    print("steps per epoch:", steps_per_epoch)
    print("validation interval:", args.val_interval)
    print("class-balanced loss:", use_balanced_loss)
    print("norm type:", args.norm_type)
    print("graph mode:", args.graph_mode)
    print("interaction configuration:", interaction_options)
    print("parameters:", sum(parameter.numel() for parameter in model.parameters()))
    if args.graph_mode == "learned":
        print("learned assignment dim:", args.learned_assignment_dim)
        print("learned assignment radius:", args.learned_assignment_radius)
        print("learned graph radius:", args.learned_graph_radius)
        print("graph compactness weight:", args.graph_compactness_weight)
        print("graph entropy weight:", args.graph_entropy_weight)
        print("graph balance weight:", args.graph_balance_weight)
    print("stable preset:", args.stable_preset)
    print("adapter attention mode:", args.adapter_attention_mode)
    print("adapter chunk pixels:", args.adapter_chunk_pixels)
    print("adapter checkpoint chunks:", not args.no_adapter_checkpoint)
    print("train drop band ratio:", args.train_drop_band_ratio)
    print("train band mixup ratio:", args.train_band_mixup_ratio)
    print("train band mixup prob:", args.train_band_mixup_prob)
    print("train band mixup alpha:", args.train_band_mixup_alpha)
    print("train band mixup neighbors:", args.train_band_mixup_neighbors)
    print("primary checkpoint:", args.primary_checkpoint)
    print("primary checkpoint path:", ckpt_paths["primary"])
    print("best macro-OA checkpoint path:", ckpt_paths["macro_oa"])
    print("best val-loss checkpoint path:", ckpt_paths["val_loss"])
    if progress_enabled and tqdm is None:
        print("tqdm is not installed; falling back to plain training logs.")

    epoch_iter = make_progress(
        range(args.epochs + 1),
        enabled=progress_enabled,
        desc="epochs",
        dynamic_ncols=True,
    )
    for epoch in epoch_iter:
        model.train()
        running_loss = 0.0
        sampled_names = rng.choice(names, size=steps_per_epoch, replace=True)
        step_iter = make_progress(
            sampled_names,
            enabled=progress_enabled and args.step_progress,
            desc="dataset steps",
            leave=False,
            dynamic_ncols=True,
        )
        for name in step_iter:
            item = datasets[name]
            optimizer.zero_grad()
            train_input = make_train_input(
                item,
                args.train_drop_band_ratio,
                mixup_ratio=args.train_band_mixup_ratio,
                mixup_alpha=args.train_band_mixup_alpha,
                mixup_neighbors=args.train_band_mixup_neighbors,
                mixup_prob=args.train_band_mixup_prob,
            )
            outputs = model(
                name,
                train_input,
                item["Q"],
                item["A"],
                wavelengths=item["wavelengths"],
                fwhm=item["fwhm"],
                superpixel_scale=item["superpixel_scale"],
                return_graph_info=args.graph_mode == "learned" or interaction_config.active,
            )
            loss = supervised_loss(
                outputs,
                item,
                "train",
                class_balance=use_balanced_loss,
                cnn_loss_weight=args.cnn_loss_weight,
                graph_branch_loss_weight=args.graph_branch_loss_weight,
                graph_loss_weights={
                    "compactness": args.graph_compactness_weight,
                    "entropy": args.graph_entropy_weight,
                    "balance": args.graph_balance_weight,
                },
            )
            loss.backward()
            optimizer.step()
            running_loss += float(loss.detach().cpu())

        avg_train_loss = running_loss / steps_per_epoch
        should_validate = (epoch % args.val_interval == 0) or (epoch == args.epochs)
        val_loss = None
        macro_oa = None
        val_oas = None
        val_pred_hists = None
        if should_validate:
            val_loss, val_oas, val_pred_hists = evaluate_split(model, datasets, "val")
            macro_oa = float(np.mean(list(val_oas.values())))
            scheduler.step(val_loss)
            if best_val_loss is None or val_loss < best_val_loss:
                best_val_loss = val_loss
                save_checkpoint(model, ckpt_paths["val_loss"])
                if args.primary_checkpoint == "val_loss":
                    save_checkpoint(model, ckpt_paths["primary"])
                log_message(
                    "save best val-loss model to {} (val loss={:.4f})".format(
                        ckpt_paths["val_loss"],
                        float(best_val_loss.detach().cpu()),
                    ),
                    progress_enabled,
                )

            if macro_oa > best_macro_oa:
                best_macro_oa = macro_oa
                save_checkpoint(model, ckpt_paths["macro_oa"])
                if args.primary_checkpoint == "macro_oa":
                    save_checkpoint(model, ckpt_paths["primary"])
                log_message(
                    "save best macro-OA model to {} (macro OA={:.4f})".format(
                        ckpt_paths["macro_oa"],
                        best_macro_oa,
                    ),
                    progress_enabled,
                )

            for dataset_name, dataset_oa in val_oas.items():
                if dataset_oa > best_dataset_oa[dataset_name]:
                    best_dataset_oa[dataset_name] = dataset_oa
                    save_checkpoint(model, ckpt_paths["datasets"][dataset_name])
                    log_message(
                        "save best {} OA model to {} ({} OA={:.4f})".format(
                            dataset_name,
                            ckpt_paths["datasets"][dataset_name],
                            dataset_name,
                            dataset_oa,
                        ),
                        progress_enabled,
                    )
        else:
            pass

        if hasattr(epoch_iter, "set_postfix"):
            postfix = {
                "train": "{:.4f}".format(avg_train_loss),
                "bestOA": "{:.4f}".format(best_macro_oa),
            }
            if val_loss is not None:
                postfix["val"] = "{:.4f}".format(float(val_loss.detach().cpu()))
                postfix["macroOA"] = "{:.4f}".format(macro_oa)
            else:
                postfix["val"] = "skip"
            epoch_iter.set_postfix(postfix)

        if should_validate:
            log_message(
                "epoch {}\ttrain loss={:.4f}\tval loss={:.4f}\tmacro OA={:.4f}\tval OA={}".format(
                    epoch,
                    avg_train_loss,
                    float(val_loss.detach().cpu()),
                    macro_oa,
                    {key: round(value, 4) for key, value in val_oas.items()},
                ),
                progress_enabled,
            )
            log_message("val top predicted classes: {}".format(val_pred_hists), progress_enabled)
        elif epoch % 10 == 0:
            log_message(
                "epoch {}\ttrain loss={:.4f}\tvalidation skipped".format(
                    epoch,
                    avg_train_loss,
                ),
                progress_enabled,
            )

    save_checkpoint(model, ckpt_paths["last"])
    log_message("save last model to {}".format(ckpt_paths["last"]), progress_enabled)

    print("\n==================== test each dataset head ====================")
    if args.eval_checkpoint in ckpt_paths and args.eval_checkpoint != "datasets":
        eval_path = ckpt_paths[args.eval_checkpoint]
    elif args.eval_checkpoint in ckpt_paths["datasets"]:
        eval_path = ckpt_paths["datasets"][args.eval_checkpoint]
    else:
        valid_eval_names = ["primary", "macro_oa", "val_loss", "last"] + list(ckpt_paths["datasets"].keys())
        raise ValueError("--eval-checkpoint must be one of {}".format(valid_eval_names))
    print("load eval checkpoint:", eval_path)
    model.load_state_dict(torch.load(eval_path, map_location=device))
    model.eval()
    test_metrics = {}
    interaction_diagnostics = {}
    with torch.no_grad():
        for name, item in datasets.items():
            outputs = model(
                name,
                item["net_input"],
                item["Q"],
                item["A"],
                wavelengths=item["wavelengths"],
                fwhm=item["fwhm"],
                superpixel_scale=item["superpixel_scale"],
                return_graph_info=interaction_config.active,
            )
            y = outputs[0]
            if len(outputs) > 3:
                interaction_diagnostics[name] = {
                    key: float(value.detach().cpu()) if torch.is_tensor(value) else value
                    for key, value in outputs[3].get("interaction", {}).items()
                }
            test_loss = utils.compute_loss(y, item["test_gt_onehot"], item["test_label_mask"])
            test_oa, test_aa, test_kappa, _ = utils.evaluate_performance2(
                y,
                item["test_samples_gt"],
                item["test_gt_onehot"],
                item["zeros"],
                np.array(item["class_num"]),
                np.array(item["height"]),
                np.array(item["width"]),
            )
            print(
                "{}\ttest loss={:.4f}\ttest OA={:.4f}\ttest AA={:.4f}\ttest kappa={:.4f}".format(
                    name,
                    float(test_loss.detach().cpu()),
                    test_oa,
                    test_aa,
                    test_kappa,
                )
            )
            test_metrics[name] = {
                "loss": float(test_loss.detach().cpu()),
                "oa": float(test_oa),
                "aa": float(test_aa),
                "kappa": float(test_kappa),
            }

    if args.metrics_json is not None:
        metrics_dir = os.path.dirname(os.path.abspath(args.metrics_json))
        if metrics_dir:
            os.makedirs(metrics_dir, exist_ok=True)
        payload = {
            "experiment": args.experiment_name,
            "script": "train_vca_multidataset.py",
            "model_config": model.export_config(),
            "parameter_count": sum(parameter.numel() for parameter in model.parameters()),
            "interaction_diagnostics": interaction_diagnostics,
            "datasets": names,
            "seed": args.seed,
            "config": {
                **vars(args),
                **interaction_options,
                "dataset_seeds": model.training_config["dataset_seeds"],
                "epochs": args.epochs,
                "steps_per_epoch": steps_per_epoch,
                "val_interval": args.val_interval,
                "latent_channels": args.latent_channels,
                "adapter_embed_dim": args.adapter_embed_dim,
                "adapter_attention_mode": args.adapter_attention_mode,
                "adapter_chunk_pixels": args.adapter_chunk_pixels,
                "adapter_checkpoint_chunks": not args.no_adapter_checkpoint,
                "stable_preset": args.stable_preset,
                "locality_sigma": args.locality_sigma,
                "locality_window": args.locality_window,
                "norm_type": args.norm_type,
                "graph_mode": args.graph_mode,
                "learned_assignment_dim": args.learned_assignment_dim,
                "learned_assignment_radius": args.learned_assignment_radius,
                "learned_graph_radius": args.learned_graph_radius,
                "learned_assignment_temperature": args.learned_assignment_temperature,
                "learned_assignment_spatial_sigma": args.learned_assignment_spatial_sigma,
                "learned_graph_spatial_sigma": args.learned_graph_spatial_sigma,
                "graph_compactness_weight": args.graph_compactness_weight,
                "graph_entropy_weight": args.graph_entropy_weight,
                "graph_balance_weight": args.graph_balance_weight,
                "train_drop_band_ratio": args.train_drop_band_ratio,
                "train_band_mixup_ratio": args.train_band_mixup_ratio,
                "train_band_mixup_prob": args.train_band_mixup_prob,
                "train_band_mixup_alpha": args.train_band_mixup_alpha,
                "train_band_mixup_neighbors": args.train_band_mixup_neighbors,
                "class_balanced_loss": use_balanced_loss,
                "primary_checkpoint": args.primary_checkpoint,
                "eval_checkpoint": args.eval_checkpoint,
            },
            "best": {
                "val_loss": None if best_val_loss is None else float(best_val_loss.detach().cpu()),
                "macro_oa": float(best_macro_oa),
                "dataset_oa": {name: float(value) for name, value in best_dataset_oa.items()},
            },
            "checkpoint_paths": ckpt_paths,
            "eval_checkpoint_path": eval_path,
            "test": test_metrics,
        }
        if run_record is not None:
            payload["research_run"] = model.training_config["research_run"]
        with open(args.metrics_json, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2, ensure_ascii=False)
        print("write metrics json:", args.metrics_json)


if __name__ == "__main__":
    main()
