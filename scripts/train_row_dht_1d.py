import argparse
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm

from datasets.shard_rows import SHARDRowDataset
from models.row_dht_1d import RowDHT1D


def train_one_epoch(model, loader, optimizer, device):
    model.train()
    total_loss = 0.0

    for batch in tqdm(loader, desc="train"):
        images = batch["image"].to(device)
        targets = batch["target"].to(device)

        logits = model(images)
        loss = F.binary_cross_entropy_with_logits(logits, targets)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * images.size(0)

    return total_loss / len(loader.dataset)


@torch.no_grad()
def validate(model, loader, device):
    model.eval()
    total_loss = 0.0

    for batch in tqdm(loader, desc="val"):
        images = batch["image"].to(device)
        targets = batch["target"].to(device)

        logits = model(images)
        loss = F.binary_cross_entropy_with_logits(logits, targets)

        total_loss += loss.item() * images.size(0)

    return total_loss / len(loader.dataset)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--processed_dir", default="data/processed/SHARD")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--out_dir", default="runs/row_dht_1d")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    dataset = SHARDRowDataset(
        processed_dir=args.processed_dir,
        image_size=(512, 512),
        sigma=3.0,
    )

    val_size = int(0.1 * len(dataset))
    train_size = len(dataset) - val_size

    train_set, val_set = random_split(
        dataset,
        [train_size, val_size],
        generator=torch.Generator().manual_seed(42),
    )

    train_loader = DataLoader(
        train_set,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
    )

    val_loader = DataLoader(
        val_set,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
    )

    model = RowDHT1D(out_height=512).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    best_val = float("inf")

    for epoch in range(1, args.epochs + 1):
        train_loss = train_one_epoch(model, train_loader, optimizer, device)
        val_loss = validate(model, val_loader, device)

        print(f"Epoch {epoch:03d} | train={train_loss:.5f} | val={val_loss:.5f}")

        checkpoint = {
            "epoch": epoch,
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "val_loss": val_loss,
        }

        torch.save(checkpoint, out_dir / "last.pt")

        if val_loss < best_val:
            best_val = val_loss
            torch.save(checkpoint, out_dir / "best.pt")
            print(f"Saved best checkpoint: {best_val:.5f}")


if __name__ == "__main__":
    main()