"""MAR-based mask inpainting utility.

This script loads a pre-trained MAR checkpoint and VAE tokenizer, then
reconstructs masked regions of an input image using iterative
autoregressive sampling. The workflow mirrors the standard MAR sampling
flow while keeping visible regions fixed.
"""
from __future__ import annotations

import argparse
import importlib
import math
import sys
from pathlib import Path
from typing import Iterable, TYPE_CHECKING

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
    parser.add_argument("--image", type=Path, required=True, help="Path to the input image (RGB).")
    parser.add_argument("--mask", type=Path, required=True, help="Binary mask image; white pixels will be inpainted.")
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
    parser.add_argument("--seed", type=int, default=0, help="Random seed for reproducible sampling.")
    parser.add_argument(
        "--class-label",
        type=int,
        default=None,
        help="Optional ImageNet class label to condition on. Defaults to unconditional inpainting when omitted.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("inpaint.png"),
        help="Where to write the reconstructed image.",
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
    getattr(download_module, MODEL_CONFIGS[model_type]["download_fn"])(overwrite=overwrite)


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


def format_mask(mask: "torch.Tensor", model: "torch.nn.Module") -> "torch.Tensor":
    """Resize mask to the MAR token grid and flatten."""
    import torch
    import torch.nn.functional as F

    resized = F.interpolate(
        mask.unsqueeze(0).unsqueeze(0),
        size=(model.seq_h, model.seq_w),
        mode="nearest",
    )
    return (resized.squeeze(0).squeeze(0).reshape(1, -1) > 0.5).to(torch.float32)


def sample_tokens_inpaint(
    model: "torch.nn.Module",
    tokens: "torch.Tensor",
    mask: "torch.Tensor",
    *,
    num_iter: int,
    cfg: float,
    cfg_schedule: str,
    labels: "torch.Tensor | None",
    temperature: float,
    progress: bool,
) -> "torch.Tensor":
    """Iteratively fill masked token positions while keeping visible regions fixed."""
    import numpy as np
    import torch

    bsz, seq_len, _ = tokens.shape
    device = tokens.device
    mask = mask.clone().to(device)
    initial_mask_counts = mask.sum(dim=1)

    masked_orders: list[torch.Tensor] = []
    for i in range(bsz):
        indices = torch.nonzero(mask[i], as_tuple=False).view(-1)
        if indices.numel() == 0:
            masked_orders.append(indices)
            continue
        perm = torch.randperm(indices.numel(), device=device)
        masked_orders.append(indices[perm])

    indices_range: Iterable[int] = range(num_iter)
    if progress:
        from tqdm import tqdm

        indices_range = tqdm(indices_range)

    mask_current = mask.clone()
    tokens_current = tokens.clone()

    for step in indices_range:
        class_embedding = (
            model.class_emb(labels)
            if labels is not None
            else model.fake_latent.repeat(bsz, 1)
        )

        if cfg != 1.0:
            tokens_input = torch.cat([tokens_current, tokens_current], dim=0)
            class_embedding = torch.cat(
                [class_embedding, model.fake_latent.repeat(bsz, 1)], dim=0
            )
            mask_input = torch.cat([mask_current, mask_current], dim=0)
        else:
            tokens_input = tokens_current
            mask_input = mask_current

        x = model.forward_mae_encoder(tokens_input, mask_input, class_embedding)
        z = model.forward_mae_decoder(x, mask_input)

        mask_ratio = np.cos(math.pi / 2.0 * (step + 1) / num_iter)
        target_lengths = torch.floor(initial_mask_counts * mask_ratio)
        remaining = mask_current.sum(dim=1)
        mask_len = torch.clamp(target_lengths, min=1)
        mask_len = torch.minimum(mask_len, torch.clamp(remaining - 1, min=0.0))
        mask_len = mask_len.to(dtype=torch.long)

        mask_next = torch.zeros_like(mask_current)
        for i in range(bsz):
            if mask_len[i] > 0 and masked_orders[i].numel() > 0:
                chosen = masked_orders[i][: mask_len[i]]
                mask_next[i, chosen] = 1

        if step >= num_iter - 1:
            mask_to_pred = mask_current.bool()[:bsz]
        else:
            mask_to_pred = torch.logical_xor(
                mask_current[:bsz].bool(), mask_next[:bsz].bool()
            )
        mask_current = mask_next

        if cfg != 1.0:
            mask_to_pred = torch.cat([mask_to_pred, mask_to_pred], dim=0)

        if mask_to_pred.sum() == 0:
            continue

        z_masked = z[mask_to_pred.nonzero(as_tuple=True)]
        if cfg_schedule == "linear":
            cfg_iter = 1 + (cfg - 1) * (1 - mask_ratio)
        elif cfg_schedule == "constant":
            cfg_iter = cfg
        else:
            raise NotImplementedError

        sampled_token_latent = model.diffloss.sample(z_masked, temperature, cfg_iter)
        if cfg != 1.0:
            sampled_token_latent, _ = sampled_token_latent.chunk(2, dim=0)
            mask_to_pred, _ = mask_to_pred.chunk(2, dim=0)

        tokens_current = tokens_current.clone()
        tokens_current[mask_to_pred.nonzero(as_tuple=True)] = sampled_token_latent

    return model.unpatchify(tokens_current)


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

    torchvision_spec = importlib.util.find_spec("torchvision.transforms")
    if torchvision_spec is None:
        raise ModuleNotFoundError(
            "torchvision is required for preprocessing. Please install torchvision first."
        )
    transforms = importlib.import_module("torchvision.transforms")

    try:
        import numpy as np  # type: ignore
    except ModuleNotFoundError:
        np = None

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    torch.manual_seed(args.seed)
    if np is not None:
        np.random.seed(args.seed)
    else:
        print("NumPy is not available; skipping NumPy-specific seeding.")
    torch.set_grad_enabled(False)

    ensure_checkpoints(args.model_type, overwrite=args.overwrite_downloads)
    model, vae = load_models(args.model_type, args.num_sampling_steps_diffloss, device)

    preprocess = transforms.Compose(
        [
            transforms.Lambda(lambda img: importlib.import_module("util.crop").center_crop_arr(img, 256)),
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
        ]
    )

    image = preprocess(args.image.open("rb").convert("RGB")).unsqueeze(0).to(device)

    mask_image = (
        transforms.Lambda(lambda img: importlib.import_module("util.crop").center_crop_arr(img, 256))
    )(args.mask.open("rb").convert("L"))
    mask_tensor = transforms.ToTensor()(mask_image).to(device)
    mask_tokens = format_mask(mask_tensor, model)

    posterior = vae.encode(image)
    latents = posterior.sample().mul_(0.2325)
    tokens = model.patchify(latents)

    tokens[:, mask_tokens.bool().squeeze(0)] = 0

    labels = None
    if args.class_label is not None:
        labels = torch.tensor([args.class_label], device=device, dtype=torch.long)

    autocast_enabled = device.type == "cuda"
    with torch.no_grad(), torch.cuda.amp.autocast(enabled=autocast_enabled):
        reconstructed_tokens = sample_tokens_inpaint(
            model,
            tokens,
            mask_tokens,
            num_iter=args.num_ar_steps,
            cfg=args.cfg_scale,
            cfg_schedule=args.cfg_schedule,
            labels=labels,
            temperature=args.temperature,
            progress=args.progress,
        )
        reconstructed = vae.decode(reconstructed_tokens / 0.2325)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    save_image = importlib.import_module("torchvision.utils").save_image
    save_image(
        reconstructed,
        args.output,
        normalize=True,
        value_range=(-1, 1),
    )
    print(f"Saved inpainted image to {args.output.resolve()}")


if __name__ == "__main__":
    main()
