"""Train the baseline brain tumour segmentation model.

Run from the project root, for example:

    python -m src.segmentation.train --data-dir data --download --epochs 2 --max-train 4 --max-val 2   # smoke test
    python -m src.segmentation.train --data-dir data --epochs 50                                         # real run
"""
import argparse
import time
from pathlib import Path

import torch
from monai.data import decollate_batch
from monai.inferers import sliding_window_inference
from monai.losses import DiceCELoss
from monai.metrics import DiceMetric
from monai.transforms import AsDiscrete, Compose
from monai.utils import set_determinism

from .data import NUM_CLASSES, get_loaders
from .model import build_model

CLASS_NAMES = ["edema", "non-enhancing", "enhancing"]


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data-dir", default="data")
    p.add_argument("--download", action="store_true", help="download the dataset (~7 GB)")
    p.add_argument("--epochs", type=int, default=50)
    p.add_argument("--batch-size", type=int, default=1)
    p.add_argument("--roi", type=int, default=96, help="cubic patch size")
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--val-every", type=int, default=2)
    p.add_argument("--cache-rate", type=float, default=0.0, help="0 = low RAM, 1 = cache all")
    p.add_argument("--num-workers", type=int, default=2)
    p.add_argument("--max-train", type=int, default=None, help="limit train cases (quick tests)")
    p.add_argument("--max-val", type=int, default=None, help="limit val cases (quick tests)")
    p.add_argument("--out-dir", default="checkpoints")
    p.add_argument("--seed", type=int, default=0)
    return p.parse_args()


@torch.no_grad()
def validate(model, loader, device, roi, dice_metric, post_pred, post_label):
    model.eval()
    dice_metric.reset()
    for batch in loader:
        images = batch["image"].to(device)
        labels = batch["label"].to(device)
        logits = sliding_window_inference(images, roi, sw_batch_size=2, predictor=model, overlap=0.5)
        preds = [post_pred(x) for x in decollate_batch(logits)]
        gts = [post_label(x) for x in decollate_batch(labels)]
        dice_metric(y_pred=preds, y=gts)
    per_class = dice_metric.aggregate()  # one value per foreground class
    return per_class.cpu().tolist()


def main():
    args = parse_args()
    set_determinism(seed=args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_amp = device.type == "cuda"
    print(f"Device: {device} | mixed precision: {use_amp}")

    roi = (args.roi,) * 3
    train_loader, val_loader = get_loaders(
        args.data_dir,
        roi_size=roi,
        batch_size=args.batch_size,
        download=args.download,
        cache_rate=args.cache_rate,
        num_workers=args.num_workers,
        max_train=args.max_train,
        max_val=args.max_val,
    )
    print(f"Train cases: {len(train_loader.dataset)} | Val cases: {len(val_loader.dataset)}")

    model = build_model().to(device)
    loss_fn = DiceCELoss(to_onehot_y=True, softmax=True)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    post_pred = Compose([AsDiscrete(argmax=True, to_onehot=NUM_CLASSES)])
    post_label = Compose([AsDiscrete(to_onehot=NUM_CLASSES)])
    dice_metric = DiceMetric(include_background=False, reduction="mean_batch")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    best = -1.0

    for epoch in range(1, args.epochs + 1):
        model.train()
        t0, running = time.time(), 0.0
        for step, batch in enumerate(train_loader, 1):
            images = batch["image"].to(device)
            labels = batch["label"].to(device)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type, enabled=use_amp):
                loss = loss_fn(model(images), labels)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            running += loss.item()
        scheduler.step()
        print(f"Epoch {epoch}/{args.epochs} | loss {running / step:.4f} | {time.time() - t0:.0f}s")

        if epoch % args.val_every == 0 or epoch == args.epochs:
            dice = validate(model, val_loader, device, roi, dice_metric, post_pred, post_label)
            mean = sum(dice) / len(dice)
            detail = ", ".join(f"{n} {d:.3f}" for n, d in zip(CLASS_NAMES, dice))
            print(f"  Val Dice: mean {mean:.3f} ({detail})")
            if mean > best:
                best = mean
                torch.save(model.state_dict(), out_dir / "best_model.pth")
                print(f"  Saved new best model (mean Dice {best:.3f})")

    torch.save(model.state_dict(), out_dir / "last_model.pth")
    print(f"Done. Best mean Dice: {best:.3f}")


if __name__ == "__main__":
    main()