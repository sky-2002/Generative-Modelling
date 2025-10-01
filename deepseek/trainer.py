import os, time, math, gc, datetime
import torch
import wandb
from tqdm import tqdm
import numpy as np
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset


class Trainer:
    def __init__(
        self,
        model,
        optimizer,
        train_bin_path,
        val_bin_path,
        batch_size,
        block_size,
        max_iters,
        eval_interval=500,
        eval_iters=100,
        lr=3e-4,
        weight_decay=0.01,
        warmup_iters=1000,
        lr_decay_iters=50000,
        min_lr=1e-5,
        checkpoint_dir="checkpoints",
        use_mixed_precision=True,
        project_name="deepseek-tiny-stories",
        run_name="test-trainer",
    ):

        self.model = model
        self.optimizer = optimizer
        self.train_data = self.load_data(train_bin_path)
        self.val_data = self.load_data(val_bin_path)
        self.batch_size = batch_size
        self.block_size = block_size
        self.max_iters = max_iters
        self.eval_interval = eval_interval
        self.eval_iters = eval_iters
        self.lr = lr
        self.weight_decay = weight_decay
        self.warmup_iters = warmup_iters
        self.lr_decay_iters = lr_decay_iters
        self.min_lr = min_lr
        self.checkpoint_dir = checkpoint_dir
        self.use_mixed_precision = use_mixed_precision

        os.makedirs(checkpoint_dir, exist_ok=True)

        # Device and multi-GPU
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        if torch.cuda.device_count() > 1:
            print(f"Using {torch.cuda.device_count()} GPUs with DataParallel")
            self.model = torch.nn.DataParallel(self.model)
        self.model = self.model.to(self.device)

        self.scaler = torch.amp.GradScaler("cuda") if use_mixed_precision else None

        # Wandb init
        wandb.init(
            project=project_name,
            config={
                "batch_size": batch_size,
                "max_iters": max_iters,
                "lr": lr,
                "weight_decay": weight_decay,
                "warmup_iters": warmup_iters,
                "lr_decay_iters": lr_decay_iters,
                "min_lr": min_lr,
                "block_size": block_size,
            },
            name=run_name,
            resume="allow",
        )

        self.start_iter = 0
        self.best_val_loss = float("inf")

    def load_data(self, path, dtype=np.uint16):
        data = np.memmap(path, dtype=dtype, mode="r")
        return torch.from_numpy(data.copy())

    def get_lr(self, it):
        if it < self.warmup_iters:
            return self.lr * it / self.warmup_iters
        if it > self.lr_decay_iters:
            return self.min_lr
        decay_ratio = (it - self.warmup_iters) / (
            self.lr_decay_iters - self.warmup_iters
        )
        coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
        return self.min_lr + coeff * (self.lr - self.min_lr)

    def get_batch(self, split="train"):
        data = self.train_data if split == "train" else self.val_data
        # batch_size = self.model.config.max_batch_size
        ix = torch.randint(len(data) - self.block_size, (self.batch_size,))
        x = torch.stack([data[i : i + self.block_size].long() for i in ix])
        y = torch.stack([data[i + 1 : i + 1 + self.block_size].long() for i in ix])
        return x.to(self.device), y.to(self.device)

    @torch.no_grad()
    def estimate_loss(self):
        self.model.eval()
        out = {}
        for split in ["train", "val"]:
            losses = []
            for _ in range(self.eval_iters):
                x, y = self.get_batch(split)
                with torch.amp.autocast("cuda", enabled=self.scaler is not None):
                    logits = self.model(x)
                    loss = F.cross_entropy(logits.view(-1, logits.size(-1)), y.view(-1))
                losses.append(loss.item())
            out[split] = sum(losses) / len(losses)
        self.model.train()
        return out

    def save_checkpoint(self, iter_num, val_loss):
        ckpt = {
            "model": self.model.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "iter_num": iter_num,
            "val_loss": val_loss,
        }
        path = os.path.join(self.checkpoint_dir, f"ckpt_{iter_num}.pt")
        torch.save(ckpt, path)
        if val_loss < self.best_val_loss:
            self.best_val_loss = val_loss
            torch.save(ckpt, os.path.join(self.checkpoint_dir, "best.pt"))
            print(f"New best model saved (val_loss={val_loss:.4f})")

    def train(self):
        print("Starting training...")
        start_time = time.time()
        x, y = self.get_batch("train")

        # Wrap iteration loop with tqdm
        pbar = tqdm(
            range(self.start_iter, self.max_iters), desc="Training", dynamic_ncols=True
        )

        for it in pbar:
            # Adjust LR
            lr = self.get_lr(it)
            for pg in self.optimizer.param_groups:
                pg["lr"] = lr

            # Forward + backward
            if self.scaler is not None:
                with torch.amp.autocast("cuda"):
                    logits = self.model(x)
                    loss = F.cross_entropy(logits.view(-1, logits.size(-1)), y.view(-1))
                self.scaler.scale(loss).backward()
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                logits = self.model(x)
                loss = F.cross_entropy(logits.view(-1, logits.size(-1)), y.view(-1))
                loss.backward()
                self.optimizer.step()

            self.optimizer.zero_grad(set_to_none=True)
            x, y = self.get_batch("train")

            # Logging
            if it % self.eval_interval == 0:
                losses = self.estimate_loss()
                elapsed = time.time() - start_time
                print(
                    f"Iter {it}: train {losses['train']:.4f}, val {losses['val']:.4f}, "
                    f"lr {lr:.2e}, time {elapsed:.1f}s"
                )
                wandb.log(
                    {
                        "train_loss": losses["train"],
                        "val_loss": losses["val"],
                        "lr": lr,
                        "iter": it,
                    }
                )
                self.save_checkpoint(it, losses["val"])

            if it % 50 == 0:
                gc.collect()
                if self.device == "cuda":
                    torch.cuda.empty_cache()

        print(f"Training finished in {time.time() - start_time:.1f}s")

    def setup_dataloader(self, split="train"):
        """Builds a DataLoader that guarantees one pass over dataset"""
        data = self.train_data if split == "train" else self.val_data
        xs, ys = [], []
        for i in range(len(data) - self.block_size):
            xs.append(data[i : i + self.block_size].long())
            ys.append(data[i + 1 : i + 1 + self.block_size].long())
        dataset = TensorDataset(torch.stack(xs), torch.stack(ys))
        loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)
        return loader

    def train_one_epoch(self):
        print("Starting one-epoch training...")
        start_time = time.time()
        loader = self.setup_dataloader("train")

        for it, (x, y) in enumerate(
            tqdm(loader, desc="One Epoch", dynamic_ncols=True), start=self.start_iter
        ):
            x, y = x.to(self.device), y.to(self.device)

            lr = self.get_lr(it)
            for pg in self.optimizer.param_groups:
                pg["lr"] = lr

            if self.scaler is not None:
                with torch.cuda.amp.autocast():
                    logits = self.model(x)
                    loss = F.cross_entropy(logits.view(-1, logits.size(-1)), y.view(-1))
                self.scaler.scale(loss).backward()
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                logits = self.model(x)
                loss = F.cross_entropy(logits.view(-1, logits.size(-1)), y.view(-1))
                loss.backward()
                self.optimizer.step()

            self.optimizer.zero_grad(set_to_none=True)

            if it % self.eval_interval == 0:
                losses = self.estimate_loss()
                elapsed = time.time() - start_time
                print(
                    f"Iter {it}: train {losses['train']:.4f}, val {losses['val']:.4f}, "
                    f"lr {lr:.2e}, time {elapsed:.1f}s"
                )

                # --- W&B Logging ---
                wandb.log(
                    {
                        "iter": it,
                        "train_loss": losses["train"],
                        "val_loss": losses["val"],
                        "lr": lr,
                        # "elapsed_time": elapsed,
                    }
                )

                self.save_checkpoint(it, losses["val"])

            if it % 100 == 0:
                gc.collect()
                if self.device == "cuda":
                    torch.cuda.empty_cache()

        print(f"One epoch finished in {time.time() - start_time:.1f}s")

    def load_checkpoint(self, path):
        checkpoint = torch.load(path, map_location=self.device)
        self.model.load_state_dict(checkpoint["model"])
        self.optimizer.load_state_dict(checkpoint["optimizer"])
        self.start_iter = checkpoint["iter_num"] + 1
        self.best_val_loss = checkpoint.get("val_loss", float("inf"))
