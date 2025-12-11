"""Run MAR inference locally following the demo notebook workflow.

This script mirrors the steps from ``demo/run_mar.ipynb`` so the model can be
exercised without relying on a notebook environment. It will download any
missing checkpoints, run the selected MAR variant, and write a grid of sampled
images to disk.
"""
from __future__ import annotations

import argparse
import importlib
from pathlib import Path
from typing import Iterable, TYPE_CHECKING
import sys

if TYPE_CHECKING:
    import torch
    from models.vae import AutoencoderKL


MODEL_CONFIGS = {
    "mar_base": {
        "download_fn": "download_pretrained_marb",
        "diffloss_d": 6,
        "diffloss_w": 1024,
    },
    "mar_large": {
        "download_fn": "download_pretrained_marl",
        "diffloss_d": 8,
        "diffloss_w": 1280,
    },
    "mar_huge": {
        "download_fn": "download_pretrained_marh",
        "diffloss_d": 12,
        "diffloss_w": 1536,
    },
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model-type",
        default="mar_huge",
        choices=MODEL_CONFIGS.keys(),
        help="Which MAR checkpoint to run.",
    )
    parser.add_argument(
        "--num-sampling-steps-diffloss",
        type=int,
        default=100,
        help="Number of sampling steps for the diffusion loss stage.",
    )
    parser.add_argument(
        "--num-ar-steps",
        type=int,
        default=64,
        help="Number of auto-regressive sampling steps.",
    )
    parser.add_argument(
        "--cfg-scale",
        type=float,
        default=4.0,
        help="Classifier-free guidance scale.",
    )
    parser.add_argument(
        "--cfg-schedule",
        choices=["linear", "constant"],
        default="constant",
        help="Classifier-free guidance schedule.",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=1.0,
        help="Sampling temperature for token generation.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Random seed for reproducible sampling.",
    )
    parser.add_argument(
        "--class-labels",
        type=int,
        nargs="+",
        default=[207, 360, 388, 113, 355, 980, 323, 979],
        help="Space-separated list of ImageNet class labels to condition on.",
    )
    parser.add_argument(
        "--samples-per-row",
        type=int,
        default=4,
        help="Number of samples per row when saving the image grid.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("sample.png"),
        help="Where to write the generated image grid.",
    )
    parser.add_argument(
        "--overwrite-downloads",
        action="store_true",
        help="Re-download checkpoints even if they already exist.",
    )
    parser.add_argument(
        "--progress",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Show sampling progress bars when supported.",
    )
    return parser.parse_args()


def ensure_checkpoints(model_type: str, overwrite: bool = False) -> None:
    download_module = importlib.import_module("util.download")
    download_module.download_pretrained_vae(overwrite=overwrite)
    getattr(download_module, MODEL_CONFIGS[model_type]["download_fn"])(
        overwrite=overwrite
    )


def format_labels(labels: Iterable[int], device: "torch.device") -> "torch.Tensor":
    import torch

    label_tensor = torch.tensor(list(labels), dtype=torch.long, device=device)
    if label_tensor.ndim == 0:
        return label_tensor.unsqueeze(0)
    return label_tensor


def load_models(
    model_type: str,
    num_sampling_steps_diffloss: int,
    device: "torch.device",
) -> tuple["torch.nn.Module", "AutoencoderKL"]:
    import torch
    from models import mar
    from models.vae import AutoencoderKL

    config = MODEL_CONFIGS[model_type]
    model = mar.__dict__[model_type](
        buffer_size=64,
        diffloss_d=config["diffloss_d"],
        diffloss_w=config["diffloss_w"],
        num_sampling_steps=str(num_sampling_steps_diffloss),
    ).to(device)
    checkpoint = torch.load(
        Path("pretrained_models/mar") / model_type / "checkpoint-last.pth",
        map_location="cpu",
    )
    model.load_state_dict(checkpoint["model_ema"])
    model.eval()

    vae = AutoencoderKL(
        embed_dim=16,
        ch_mult=(1, 1, 2, 2, 4),
        ckpt_path="pretrained_models/vae/kl16.ckpt",
    ).to(device)
    vae.eval()
    return model, vae


def main() -> None:
    args = parse_args()

    project_root = Path(__file__).resolve().parent.parent
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))

    torch_spec = importlib.util.find_spec("torch")
    if torch_spec is None:
        raise ModuleNotFoundError(
            "torch is required to run MAR inference. Please install PyTorch first."
        )
    torch = importlib.import_module("torch")

    torchvision_spec = importlib.util.find_spec("torchvision.utils")
    if torchvision_spec is None:
        raise ModuleNotFoundError(
            "torchvision is required for saving image grids. Please install torchvision."
        )
    save_image = importlib.import_module("torchvision.utils").save_image

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    try:
        import numpy as np  # type: ignore
    except ModuleNotFoundError:
        np = None

    torch.manual_seed(args.seed)
    if np is not None:
        np.random.seed(args.seed)
    else:
        print("NumPy is not available; skipping NumPy-specific seeding.")
    torch.set_grad_enabled(False)

    ensure_checkpoints(args.model_type, overwrite=args.overwrite_downloads)
    model, vae = load_models(
        args.model_type, args.num_sampling_steps_diffloss, device
    )

    labels = format_labels(args.class_labels, device)
    autocast_enabled = device.type == "cuda"

    with torch.no_grad(), torch.cuda.amp.autocast(enabled=autocast_enabled):
        sampled_tokens = model.sample_tokens(
            bsz=len(labels),
            num_iter=args.num_ar_steps,
            cfg=args.cfg_scale,
            cfg_schedule=args.cfg_schedule,
            labels=labels,
            temperature=args.temperature,
            progress=args.progress,
        )
        sampled_images = vae.decode(sampled_tokens / 0.2325)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    save_image(
        sampled_images,
        args.output,
        nrow=int(args.samples_per_row),
        normalize=True,
        value_range=(-1, 1),
    )
    print(f"Saved samples to {args.output.resolve()}")


if __name__ == "__main__":
    main()
