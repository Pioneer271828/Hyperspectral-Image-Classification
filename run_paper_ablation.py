import argparse
import csv
import json
import os
import statistics
import subprocess
import sys
import time
from pathlib import Path
from interaction_ablation_presets import build_interaction_experiments
from model.interaction_config import interaction_kwargs


def parse_csv_list(text):
    return [item.strip() for item in str(text).split(",") if item.strip()]


def parse_int_list(text):
    return [int(item) for item in parse_csv_list(text)]


def parse_float_list(text):
    return [float(item) for item in parse_csv_list(text)]


def safe_name(text):
    return "".join(ch if ch.isalnum() or ch in ["-", "_"] else "_" for ch in str(text))


def fmt_float_tag(value):
    return ("{:.4g}".format(float(value))).replace("-", "m").replace(".", "p")


def mean_std(values):
    values = [float(value) for value in values]
    if len(values) == 0:
        return None, None
    if len(values) == 1:
        return values[0], 0.0
    return statistics.mean(values), statistics.pstdev(values)


def pct(mean_value, std_value):
    if mean_value is None:
        return ""
    return "{:.2f} +- {:.2f}".format(mean_value * 100.0, std_value * 100.0)


def build_multi_experiments(args):
    base_k = int(args.base_latent_channels)
    base_embed = int(args.base_adapter_embed_dim)
    experiments = [
        {
            "name": "vca_shared_domainbn_k{}".format(base_k),
            "latent_channels": base_k,
            "adapter_embed_dim": base_embed,
            "adapter_attention_mode": "wavelength",
            "norm_type": "domain_bn",
            "train_drop_band_ratio": 0.0,
        },
        {
            "name": "vca_no_wave_attention_k{}".format(base_k),
            "latent_channels": base_k,
            "adapter_embed_dim": base_embed,
            "adapter_attention_mode": "index",
            "norm_type": "domain_bn",
            "train_drop_band_ratio": 0.0,
        },
        {
            "name": "vca_k32_domainbn",
            "latent_channels": 32,
            "adapter_embed_dim": 16,
            "adapter_attention_mode": "wavelength",
            "norm_type": "domain_bn",
            "train_drop_band_ratio": 0.0,
        },
        {
            "name": "vca_k128_domainbn",
            "latent_channels": 128,
            "adapter_embed_dim": 64,
            "adapter_attention_mode": "wavelength",
            "norm_type": "domain_bn",
            "train_drop_band_ratio": 0.0,
        },
        {
            "name": "vca_batchnorm_k{}".format(base_k),
            "latent_channels": base_k,
            "adapter_embed_dim": base_embed,
            "adapter_attention_mode": "wavelength",
            "norm_type": "batch_norm",
            "train_drop_band_ratio": 0.0,
        },
    ]
    for ratio in parse_float_list(args.band_drop_ratios):
        experiments.append(
            {
                "name": "vca_banddrop{}p_k{}".format(int(round(ratio * 100)), base_k),
                "latent_channels": base_k,
                "adapter_embed_dim": base_embed,
                "adapter_attention_mode": "wavelength",
                "norm_type": "domain_bn",
                "train_drop_band_ratio": ratio,
            }
        )
    return experiments


def selected_experiments(args):
    multi = build_interaction_experiments(args) if args.suite == "interaction" else build_multi_experiments(args)
    requested = parse_csv_list(args.experiments)
    if requested == ["all"]:
        names = None
    else:
        names = set(requested)
        available = {item["name"] for item in multi}
        available.add("sahgcn_single_dataset")
        unknown = sorted(names - available)
        if unknown:
            raise ValueError("Unknown experiments: {}. Available: {}".format(unknown, sorted(available)))

    result = []
    if args.suite in ["vca", "full", "interaction"]:
        for item in multi:
            if names is None or item["name"] in names:
                result.append(("multi", item))
    if args.suite == "full" and (names is None or "sahgcn_single_dataset" in names):
        result.append(("single", {"name": "sahgcn_single_dataset"}))
    return result


def child_common_flags(args):
    flags = []
    if not args.child_progress:
        flags.append("--no-progress")
    return flags


def multi_command(args, root, experiment, seed, metrics_path, save_path):
    cmd = [
        args.python,
        str(root / "train_vca_multidataset.py"),
        "--device",
        args.device,
        "--datasets",
        args.datasets,
        "--epochs",
        str(args.epochs),
        "--steps-per-epoch",
        str(args.steps_per_epoch),
        "--val-interval",
        str(args.val_interval),
        "--seed",
        str(seed),
        "--samples-type",
        args.samples_type,
        "--train-ratio",
        str(args.train_ratio),
        "--val-ratio",
        str(args.val_ratio),
        "--train-num",
        str(args.train_num),
        "--val-num",
        str(args.val_num),
        "--latent-channels",
        str(experiment["latent_channels"]),
        "--adapter-embed-dim",
        str(experiment["adapter_embed_dim"]),
        "--adapter-attention-mode",
        experiment["adapter_attention_mode"],
        "--train-drop-band-ratio",
        str(experiment["train_drop_band_ratio"]),
        "--norm-type",
        experiment["norm_type"],
        "--primary-checkpoint",
        args.primary_checkpoint,
        "--eval-checkpoint",
        args.eval_checkpoint,
        "--save-path",
        str(save_path),
        "--metrics-json",
        str(metrics_path),
        "--experiment-name",
        experiment["name"],
    ]
    if args.suite == "interaction":
        cmd.extend(["--graph-mode", experiment["graph_mode"],
                    "--superpixel-scale", str(args.superpixel_scale),
                    "--pavia-superpixel-scale", str(args.pavia_superpixel_scale),
                    "--adapter-chunk-pixels", str(args.adapter_chunk_pixels),
                    "--cnn-loss-weight", str(args.cnn_loss_weight),
                    "--graph-branch-loss-weight", str(args.graph_branch_loss_weight)])
        for key, value in interaction_kwargs(experiment).items():
            cmd.extend(["--" + key.replace("_", "-"), str(value)])
    cmd.extend(child_common_flags(args))
    return cmd


def single_command(args, root, experiment, dataset_name, seed, metrics_path):
    cmd = [
        args.python,
        str(root / "main.py"),
        "--device",
        args.device,
        "--dataset",
        dataset_name,
        "--epochs",
        str(args.epochs),
        "--train-runs",
        "1",
        "--seeds",
        str(seed),
        "--samples-type",
        args.samples_type,
        "--train-ratio",
        str(args.train_ratio),
        "--val-ratio",
        str(args.val_ratio),
        "--train-num",
        str(args.train_num),
        "--val-num",
        str(args.val_num),
        "--metrics-json",
        str(metrics_path),
        "--experiment-name",
        experiment["name"],
        "--no-save-maps",
    ]
    cmd.extend(child_common_flags(args))
    return cmd


def run_child(cmd, root, log_path, dry_run=False, stream_output=False):
    log_path.parent.mkdir(parents=True, exist_ok=True)
    env = os.environ.copy()
    env["PYTHONUNBUFFERED"] = "1"
    with open(log_path, "w", encoding="utf-8") as log_file:
        log_file.write("COMMAND:\n{}\n\n".format(" ".join(cmd)))
        log_file.flush()
        if dry_run:
            return 0
        if stream_output:
            proc = subprocess.Popen(
                cmd,
                cwd=str(root),
                env=env,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
            )
            while True:
                chunk = proc.stdout.read(1)
                if chunk == "" and proc.poll() is not None:
                    break
                if chunk:
                    print(chunk, end="", flush=True)
                    log_file.write(chunk)
                    log_file.flush()
            return proc.wait()
        proc = subprocess.run(
            cmd,
            cwd=str(root),
            env=env,
            stdout=log_file,
            stderr=subprocess.STDOUT,
        )
        return proc.returncode


def collect_rows(metrics_dir):
    rows = []
    for path in sorted(metrics_dir.glob("*.json")):
        with open(path, "r", encoding="utf-8") as f:
            payload = json.load(f)
        experiment = payload.get("experiment") or path.stem
        script = payload.get("script")
        if script == "train_vca_multidataset.py":
            seed = payload.get("seed")
            config = payload.get("config", {})
            for dataset_name, metrics in payload.get("test", {}).items():
                rows.append(
                    {
                        "experiment": experiment,
                        "script": script,
                        "dataset": dataset_name,
                        "seed": seed,
                        "oa": metrics.get("oa"),
                        "aa": metrics.get("aa"),
                        "kappa": metrics.get("kappa"),
                        "loss": metrics.get("loss"),
                        "latent_channels": config.get("latent_channels"),
                        "adapter_embed_dim": config.get("adapter_embed_dim"),
                        "adapter_attention_mode": config.get("adapter_attention_mode"),
                        "norm_type": config.get("norm_type"),
                        "train_drop_band_ratio": config.get("train_drop_band_ratio"),
                        **interaction_kwargs(config),
                        "graph_mode": config.get("graph_mode", "static"),
                        "parameter_count": payload.get("parameter_count"),
                        "metrics_path": str(path),
                    }
                )
        elif script == "main.py":
            dataset_name = payload.get("dataset")
            config = payload.get("config", {})
            for run in payload.get("runs", []):
                rows.append(
                    {
                        "experiment": experiment,
                        "script": script,
                        "dataset": dataset_name,
                        "seed": run.get("seed"),
                        "oa": run.get("oa"),
                        "aa": run.get("aa"),
                        "kappa": run.get("kappa"),
                        "loss": run.get("loss"),
                        "latent_channels": config.get("vca_latent_channels"),
                        "adapter_embed_dim": config.get("vca_embed_dim"),
                        "adapter_attention_mode": config.get("vca_adapter_attention_mode"),
                        "norm_type": "single_dataset",
                        "train_drop_band_ratio": config.get("train_drop_band_ratio"),
                        "metrics_path": str(path),
                    }
                )
    return rows


def write_csv(path, rows, fieldnames):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in fieldnames})


def summarize_rows(rows):
    summary = []
    grouped = {}
    for row in rows:
        key = (row["experiment"], row["dataset"])
        grouped.setdefault(key, []).append(row)
    for (experiment, dataset_name), group in sorted(grouped.items()):
        oa_mean, oa_std = mean_std([row["oa"] for row in group])
        aa_mean, aa_std = mean_std([row["aa"] for row in group])
        kappa_mean, kappa_std = mean_std([row["kappa"] for row in group])
        summary.append(
            {
                "experiment": experiment,
                "dataset": dataset_name,
                "n": len(group),
                "oa_mean": oa_mean,
                "oa_std": oa_std,
                "aa_mean": aa_mean,
                "aa_std": aa_std,
                "kappa_mean": kappa_mean,
                "kappa_std": kappa_std,
            }
        )

    by_exp_seed = {}
    for row in rows:
        key = (row["experiment"], row["seed"])
        by_exp_seed.setdefault(key, []).append(float(row["oa"]))
    by_exp = {}
    for (experiment, seed), values in by_exp_seed.items():
        by_exp.setdefault(experiment, []).append(statistics.mean(values))
    for experiment, values in sorted(by_exp.items()):
        oa_mean, oa_std = mean_std(values)
        summary.append(
            {
                "experiment": experiment,
                "dataset": "macro",
                "n": len(values),
                "oa_mean": oa_mean,
                "oa_std": oa_std,
                "aa_mean": "",
                "aa_std": "",
                "kappa_mean": "",
                "kappa_std": "",
            }
        )
    return summary


def write_markdown(path, summary):
    lines = [
        "| Experiment | Dataset | n | OA (%) | AA (%) | Kappa (%) |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for row in summary:
        aa_cell = ""
        kappa_cell = ""
        if row["aa_mean"] != "":
            aa_cell = pct(row["aa_mean"], row["aa_std"])
        if row["kappa_mean"] != "":
            kappa_cell = pct(row["kappa_mean"], row["kappa_std"])
        lines.append(
            "| {} | {} | {} | {} | {} | {} |".format(
                row["experiment"],
                row["dataset"],
                row["n"],
                pct(row["oa_mean"], row["oa_std"]),
                aa_cell,
                kappa_cell,
            )
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_commands(path, commands):
    lines = []
    for item in commands:
        lines.append("# {}".format(item["name"]))
        lines.append(" ".join(item["cmd"]))
        lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description="Run paper-ready seed and ablation experiments.")
    parser.add_argument("--python", type=str, default=sys.executable)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--datasets", type=str, default="paviaU,xuzhou,WHUL")
    parser.add_argument("--seeds", type=str, default="0,2026,3407")
    parser.add_argument("--suite", type=str, default="full", choices=["vca", "full", "interaction"])
    parser.add_argument("--interaction-graph-mode", default="learned", choices=["static", "learned"])
    parser.add_argument("--interaction-norm-type", default="domain_bn", choices=["domain_bn", "batch_norm", "group_norm"])
    parser.add_argument("--interaction-strength", type=float, default=0.1)
    parser.add_argument("--interaction-hidden-dim", type=int, default=32)
    parser.add_argument("--interaction-chunk-pixels", type=int, default=16384)
    parser.add_argument("--graph-update-dim", type=int, default=32)
    parser.add_argument("--graph-update-mix", type=float, default=0.5)
    parser.add_argument("--cnn-loss-weight", type=float, default=1.0)
    parser.add_argument("--graph-branch-loss-weight", type=float, default=1.0)
    parser.add_argument("--superpixel-scale", type=int, default=100)
    parser.add_argument("--pavia-superpixel-scale", type=int, default=50)
    parser.add_argument("--adapter-chunk-pixels", type=int, default=32768)
    parser.add_argument("--experiments", type=str, default="all",
                        help="Comma list of experiment names, or all.")
    parser.add_argument("--epochs", type=int, default=500)
    parser.add_argument("--steps-per-epoch", type=int, default=3)
    parser.add_argument("--val-interval", type=int, default=10)
    parser.add_argument("--samples-type", type=str, default="ratio", choices=["ratio", "number"])
    parser.add_argument("--train-num", type=int, default=30)
    parser.add_argument("--val-num", type=int, default=10)
    parser.add_argument("--train-ratio", type=float, default=0.01)
    parser.add_argument("--val-ratio", type=float, default=0.005)
    parser.add_argument("--base-latent-channels", type=int, default=64)
    parser.add_argument("--base-adapter-embed-dim", type=int, default=32)
    parser.add_argument("--band-drop-ratios", type=str, default="0.1")
    parser.add_argument("--primary-checkpoint", type=str, default="macro_oa",
                        choices=["macro_oa", "val_loss"])
    parser.add_argument("--eval-checkpoint", type=str, default="primary")
    parser.add_argument("--out-dir", type=str, default=None)
    parser.add_argument("--child-progress", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--continue-on-error", action="store_true")
    args = parser.parse_args()

    root = Path(__file__).resolve().parent
    if args.out_dir is None:
        run_tag = time.strftime("%Y%m%d-%H%M%S", time.localtime())
        out_dir = root / "experiments" / ("paper_ablation_" + run_tag)
    else:
        out_dir = Path(args.out_dir).resolve()
    metrics_dir = out_dir / "metrics"
    logs_dir = out_dir / "logs"
    ckpt_dir = out_dir / "checkpoints"
    metrics_dir.mkdir(parents=True, exist_ok=True)
    logs_dir.mkdir(parents=True, exist_ok=True)
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    seeds = parse_int_list(args.seeds)
    dataset_names = parse_csv_list(args.datasets)
    experiments = selected_experiments(args)
    commands = []

    for kind, experiment in experiments:
        if kind == "multi":
            for seed in seeds:
                name = "{}_seed{}".format(experiment["name"], seed)
                metrics_path = metrics_dir / (safe_name(name) + ".json")
                save_path = ckpt_dir / (safe_name(name) + ".pt")
                cmd = multi_command(args, root, experiment, seed, metrics_path, save_path)
                commands.append({"name": name, "cmd": cmd, "log": logs_dir / (safe_name(name) + ".log")})
        elif kind == "single":
            for dataset_name in dataset_names:
                for seed in seeds:
                    name = "{}_{}_seed{}".format(experiment["name"], safe_name(dataset_name), seed)
                    metrics_path = metrics_dir / (safe_name(name) + ".json")
                    cmd = single_command(args, root, experiment, dataset_name, seed, metrics_path)
                    commands.append({"name": name, "cmd": cmd, "log": logs_dir / (safe_name(name) + ".log")})
        else:
            raise RuntimeError("unexpected experiment kind: {}".format(kind))

    write_commands(out_dir / "commands.txt", commands)
    print("output directory:", out_dir)
    print("planned runs:", len(commands))
    if args.dry_run:
        print("dry run: commands were written but not executed")

    for index, item in enumerate(commands, start=1):
        print("[{}/{}] {}".format(index, len(commands), item["name"]))
        code = run_child(
            item["cmd"],
            root,
            item["log"],
            dry_run=args.dry_run,
            stream_output=args.child_progress,
        )
        if code != 0:
            print("failed with exit code {}. log: {}".format(code, item["log"]))
            if not args.continue_on_error:
                raise SystemExit(code)

    rows = collect_rows(metrics_dir)
    run_fields = [
        "experiment",
        "script",
        "dataset",
        "seed",
        "oa",
        "aa",
        "kappa",
        "loss",
        "latent_channels",
        "adapter_embed_dim",
        "adapter_attention_mode",
        "norm_type",
        "train_drop_band_ratio",
        "graph_mode",
        "parameter_count",
        *interaction_kwargs({}).keys(),
        "metrics_path",
    ]
    write_csv(out_dir / "all_runs.csv", rows, run_fields)

    summary = summarize_rows(rows)
    summary_fields = [
        "experiment",
        "dataset",
        "n",
        "oa_mean",
        "oa_std",
        "aa_mean",
        "aa_std",
        "kappa_mean",
        "kappa_std",
    ]
    write_csv(out_dir / "summary.csv", summary, summary_fields)
    write_markdown(out_dir / "summary.md", summary)

    print("wrote:", out_dir / "all_runs.csv")
    print("wrote:", out_dir / "summary.csv")
    print("wrote:", out_dir / "summary.md")


if __name__ == "__main__":
    main()
