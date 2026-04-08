from pathlib import Path
import os
import sys
import torch
from torch.utils.data import DataLoader
import torch.nn as nn
from tqdm.auto import tqdm
import yaml
from models.resnet import ResNet18
from dataset_pipeline.cifar10 import get_datasets

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


CONFIG_PATH = PROJECT_ROOT / "configs" / "config.yaml"
with open(CONFIG_PATH, "r", encoding="utf-8") as f:
    config = yaml.safe_load(f)

num_epochs = config["training"]["epochs"]
batch_size = config["training"]["batch_size"]
learning_rate = config["training"]["learning_rate"]
workers = config["train_dataloader"]["num_workers"]
momentum = config["training"]["momentum"]
weight_decay = float(config["training"]["weight_decay"])
model_name = config["model"]["name"]
CHECKPOINT_PATH = PROJECT_ROOT / "results" / "checkpoints" / model_name


def train():
    torch.manual_seed(33)
    torch.cuda.manual_seed_all(33)
    torch.set_float32_matmul_precision("high")
    torch.backends.cudnn.benchmark = True

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    CHECKPOINT_PATH.parent.mkdir(parents=True, exist_ok=True)

    use_pin_memory = device.type == "cuda" and not (os.name == "nt" and workers > 0)

    train_dataset, _ = get_datasets()
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=workers,
        pin_memory=use_pin_memory,
        persistent_workers=workers > 0,
    )

    model = ResNet18(num_classes=10).to(device)
    should_compile = hasattr(torch, "compile") and device.type == "cuda"
    if should_compile:
        model = torch.compile(
            model,
            dynamic=False,
            backend="inductor",
            mode="max-autotune",
            fullgraph=True,
        )

    loss_fn = nn.CrossEntropyLoss(label_smoothing=0.01)

    optimizer = torch.optim.SGD(
        model.parameters(),
        lr=learning_rate,
        momentum=momentum,
        weight_decay=weight_decay,
    )

    scheduler_config = config.get("scheduler", {})
    scheduler_type = scheduler_config.get("type", "multistep")

    if scheduler_type == "step":
        scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer,
            step_size=scheduler_config.get("step_size", 20),
            gamma=scheduler_config.get("gamma", 0.1),
        )
    elif scheduler_type == "multistep":
        scheduler = torch.optim.lr_scheduler.MultiStepLR(
            optimizer,
            milestones=scheduler_config.get("milestones", [30, 45]),
            gamma=scheduler_config.get("gamma", 0.1),
        )
    else:
        scheduler = None

    prev_lr = optimizer.param_groups[0]["lr"]

    best_acc = 0.0
    for epoch in tqdm(range(num_epochs)):
        model.train()
        train_loss = 0
        train_correct = 0
        train_total = 0

        for X, y in train_loader:
            X = X.to(device, non_blocking=use_pin_memory)
            y = y.to(device, non_blocking=use_pin_memory)

            optimizer.zero_grad()
            outputs = model(X)
            loss = loss_fn(outputs, y)
            loss.backward()
            optimizer.step()

            train_loss += loss.item() * X.size(0)
            predicted = torch.argmax(outputs, dim=1)
            train_correct += (predicted == y).sum().item()
            train_total += y.size(0)

        train_loss = train_loss / train_total
        train_acc = train_correct / train_total
        current_lr = optimizer.param_groups[0]["lr"]

        print(
            f"Epoch [{epoch+1}/{num_epochs}], "
            f"Loss: {train_loss:.4f}, Accuracy: {train_acc:.6f}, LR: {current_lr:.5f}"
        )

        if scheduler is not None:
            scheduler.step()
            new_lr = optimizer.param_groups[0]["lr"]
            if new_lr != prev_lr:
                print(f"LR changed → {new_lr:.5f}")
                prev_lr = new_lr

        if train_acc > best_acc:
            best_acc = train_acc
            model_to_save = model._orig_mod if hasattr(model, "_orig_mod") else model
            torch.save(model_to_save.state_dict(), CHECKPOINT_PATH)
            print(f"New best model saved with accuracy: {best_acc:.6f}")


if __name__ == "__main__":
    train()
