"""Small, explicit training loop with gradient accumulation."""

from __future__ import annotations

from dataclasses import dataclass
import json
import math
from pathlib import Path
from typing import Optional
import time

import torch

from .torch_model import TorchD2NN


@dataclass(frozen=True)
class TrainConfig:
    epochs: int = 20
    effective_batch_size: int = 32
    micro_batch_size: int = 8
    learning_rate: float = 1.0e-2
    max_batches_per_epoch: Optional[int] = None
    num_workers: int = 0
    log_every: int = 10
    seed: int = 20261012

    @property
    def accumulation_steps(self) -> int:
        if self.effective_batch_size <= 0 or self.micro_batch_size <= 0:
            raise ValueError("batch sizes must be positive")
        return int(math.ceil(self.effective_batch_size / self.micro_batch_size))


def _move_batch(amplitude: torch.Tensor, labels: torch.Tensor, device: torch.device):
    if amplitude.ndim == 4 and amplitude.shape[1] == 1:
        amplitude = amplitude[:, 0]
    if amplitude.ndim != 3:
        raise ValueError("amplitude batch must have shape (B, H, W) or (B, 1, H, W)")
    return (
        amplitude.to(device=device, dtype=torch.float64),
        labels.to(device=device, dtype=torch.long),
    )


def train_one_epoch(
    model: TorchD2NN,
    data_loader,
    optimizer: torch.optim.Optimizer,
    train_config: TrainConfig,
    device: torch.device,
    epoch: Optional[int] = None,
) -> float:
    model.train()
    optimizer.zero_grad(set_to_none=True)
    running_loss = 0.0
    processed_micro_batches = 0
    accumulation_counter = 0

    for batch_index, (amplitude, labels) in enumerate(data_loader):
        if (
            train_config.max_batches_per_epoch is not None
            and batch_index >= train_config.max_batches_per_epoch
        ):
            break
        amplitude, labels = _move_batch(amplitude, labels, device)
        output = model(amplitude)
        loss = model.loss(output.sk, labels)
        (loss / train_config.accumulation_steps).backward()
        running_loss += float(loss.detach().cpu())
        processed_micro_batches += 1
        accumulation_counter += 1

        if accumulation_counter >= train_config.accumulation_steps:
            optimizer.step()
            model.wrap_phases_()
            optimizer.zero_grad(set_to_none=True)
            accumulation_counter = 0

        if (
            train_config.log_every > 0
            and processed_micro_batches % train_config.log_every == 0
        ):
            epoch_text = f"epoch={epoch} " if epoch is not None else ""
            print(
                f"{epoch_text}micro_batch={processed_micro_batches} "
                f"running_loss={running_loss / processed_micro_batches:.6f}",
                flush=True,
            )

    if accumulation_counter != 0:
        optimizer.step()
        model.wrap_phases_()
        optimizer.zero_grad(set_to_none=True)

    if processed_micro_batches == 0:
        raise RuntimeError("data_loader produced no batches")
    return running_loss / processed_micro_batches


@torch.no_grad()
def evaluate(model: TorchD2NN, data_loader, device: torch.device) -> dict:
    model.eval()
    total_examples = 0
    total_loss = 0.0
    correct = 0
    eta_sum = 0.0

    for amplitude, labels in data_loader:
        amplitude, labels = _move_batch(amplitude, labels, device)
        output = model(amplitude)
        loss = model.loss(output.sk, labels)
        batch_size = int(labels.shape[0])
        total_examples += batch_size
        total_loss += float(loss.detach().cpu()) * batch_size
        correct += int((output.prediction == labels).sum().detach().cpu())
        eta_sum += float(output.eta.sum().detach().cpu())

    if total_examples == 0:
        raise RuntimeError("data_loader produced no examples")
    return {
        "loss": total_loss / total_examples,
        "accuracy": correct / total_examples,
        "eta": eta_sum / total_examples,
        "examples": total_examples,
    }


def fit(
    model: TorchD2NN,
    train_loader,
    validation_loader,
    train_config: TrainConfig,
    device: torch.device,
    output_dir: Optional[str] = None,
    checkpoint_name: str = "last_checkpoint.pt",
) -> list:
    torch.manual_seed(train_config.seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(train_config.seed)

    optimizer = torch.optim.Adam(model.parameters(), lr=train_config.learning_rate)
    history = []
    best_validation_loss = float("inf")
    best_checkpoint_name = checkpoint_name.replace("_last.pt", "_best.pt")
    if best_checkpoint_name == checkpoint_name:
        best_checkpoint_name = "best_checkpoint.pt"
    output_path = Path(output_dir) if output_dir else None
    if output_path is not None:
        output_path.mkdir(parents=True, exist_ok=True)

    for epoch_index in range(train_config.epochs):
        start = time.perf_counter()
        train_loss = train_one_epoch(
            model,
            train_loader,
            optimizer,
            train_config,
            device,
            epoch=epoch_index + 1,
        )
        validation = evaluate(model, validation_loader, device)
        elapsed = time.perf_counter() - start
        record = {
            "epoch": epoch_index + 1,
            "train_loss": train_loss,
            "validation_loss": validation["loss"],
            "validation_accuracy": validation["accuracy"],
            "validation_eta": validation["eta"],
            "seconds": elapsed,
        }
        history.append(record)
        print(
            f"epoch={record['epoch']} "
            f"train_loss={record['train_loss']:.6f} "
            f"val_loss={record['validation_loss']:.6f} "
            f"val_acc={record['validation_accuracy']:.4f} "
            f"eta={record['validation_eta']:.6f} "
            f"seconds={record['seconds']:.2f}"
        )

        if output_path is not None:
            checkpoint = {
                "epoch": record["epoch"],
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "simulation_config": model.config,
                "optimization_branch": model.optimization_branch,
                "train_config": train_config.__dict__,
                "history": history,
            }
            torch.save(checkpoint, output_path / checkpoint_name)
            if validation["loss"] < best_validation_loss:
                best_validation_loss = validation["loss"]
                torch.save(checkpoint, output_path / best_checkpoint_name)
            with (output_path / "history.json").open(
                "w", encoding="utf-8"
            ) as handle:
                json.dump(history, handle, ensure_ascii=False, indent=2)

    return history
