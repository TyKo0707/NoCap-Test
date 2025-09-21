import sys
import uuid
import math
import glob
import time
import argparse
import os
from dataclasses import dataclass
from contextlib import nullcontext

import numpy as np
import torch
from torch import nn
import torch.distributed as dist
import torch.nn.functional as F
import torch._inductor.config as config
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed import init_process_group, destroy_process_group

from utils import *

import torch._dynamo
torch._dynamo.config.suppress_errors = True

print0(f"Running pytorch {torch.version.__version__}")

parser = argparse.ArgumentParser()

# file system input / output
parser.add_argument("--input_bin", type=str, default="/fineweb10B_balanced_buckets/bucket0*.bin",
                    help="input .bin to train on", )
parser.add_argument("--input_val_bin", type=str, default="/fineweb10B_balanced_buckets/bucket_val_*.bin",
                    help="input .bin to eval validation loss on")
parser.add_argument("--output_dir", type=str, default="",
                    help="output directory to which to write logs and checkpoints")
parser.add_argument("--model", type=str, default="d12", help="d12|d24|d36|d48")

# token layout for each step of the optimization
parser.add_argument("--batch_size", type=int, default=4, help="batch size, in units of #batch dimensions")
parser.add_argument("--grad_accumulation_steps", type=int, default=1, help="number of gradient accumulation steps")
parser.add_argument("--sequence_length", type=int, default=64, help="sequence length")

# workload (number of steps)
parser.add_argument("--num_iterations", type=int, default=100, help="number of iterations to run")

# optimization
parser.add_argument("--learning_rate", type=float, default=1e-4, help="learning rate warmup iterations")
parser.add_argument("--warmup_iters", type=int, default=0, help="learning rate warmup iterations")
parser.add_argument("--warmdown_iters", type=int, default=0, help="learning rate warmdown iterations")
parser.add_argument("--weight_decay", type=float, default=0.0, help="weight decay")

# evaluation
parser.add_argument("--val_loss_every", type=int, default=0, help="every how many steps to evaluate val loss?")
parser.add_argument("--val_batch_size", type=int, default=16, help="how many batches of val to average?")
parser.add_argument("--save_every", type=int, default=5000, help="every how many steps to save the checkpoint")
parser.add_argument("--log_wandb", action="store_true", help="log to wandb")
args = parser.parse_args()

with open(sys.argv[0]) as f:
    code = f.read()


# -----------------------------------------------------------------------------
# PyTorch nn.Module definitions for the GPT-2 model

class Rotary(torch.nn.Module):
    def __init__(self, dim, base=10000):
        super().__init__()
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv_freq)
        self.seq_len_cached = None
        self.cos_cached = None
        self.sin_cached = None

    def forward(self, x):
        seq_len = x.shape[1]
        if seq_len != self.seq_len_cached:
            self.seq_len_cached = seq_len
            t = torch.arange(seq_len, device=x.device).type_as(self.inv_freq)
            freqs = torch.outer(t, self.inv_freq).to(x.device)
            self.cos_cached = freqs.cos()
            self.sin_cached = freqs.sin()
        return self.cos_cached[None, :, None, :], self.sin_cached[None, :, None, :]


def apply_rotary_emb(x, cos, sin):
    assert x.ndim == 4  # multihead attention
    d = x.shape[3] // 2
    x1 = x[..., :d]
    x2 = x[..., d:]
    y1 = x1 * cos + x2 * sin
    y2 = x1 * (-sin) + x2 * cos
    return torch.cat([y1, y2], 3)


def rmsnorm(x0, eps=1e-6):
    x = x0.float()
    x = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + eps)
    return x.type_as(x0)


class CausalSelfAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.n_head = config.n_head
        self.n_embd = config.n_embd
        self.head_dim = self.n_embd // self.n_head
        assert self.n_embd % self.n_head == 0
        self.c_attn = nn.Linear(self.n_embd, 3 * self.n_embd, bias=False)
        self.c_proj = nn.Linear(self.n_embd, self.n_embd, bias=False)
        self.rotary = Rotary(self.head_dim)

    def forward(self, x):
        B, T, C = x.size()
        qkv = self.c_attn(x)
        q, k, v = qkv.split(self.n_embd, dim=2)
        k = k.view(B, T, self.n_head, self.head_dim)
        q = q.view(B, T, self.n_head, self.head_dim)
        v = v.view(B, T, self.n_head, self.head_dim)
        cos, sin = self.rotary(q)
        q = apply_rotary_emb(q, cos, sin)
        k = apply_rotary_emb(k, cos, sin)
        y = F.scaled_dot_product_attention(
            q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2), is_causal=True
        )
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        y = self.c_proj(y)
        return y


class MLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.c_fc = nn.Linear(config.n_embd, 4 * config.n_embd, bias=False)
        self.c_proj = nn.Linear(4 * config.n_embd, config.n_embd, bias=False)

    def forward(self, x):
        x = self.c_fc(x)
        x = F.gelu(x)
        x = self.c_proj(x)
        return x


class Block(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.attn = CausalSelfAttention(config)
        self.mlp = MLP(config)
        self.attn_scale = 1 / math.sqrt(2 * config.n_layer)

    def forward(self, x):
        x = x + self.attn_scale * self.attn(rmsnorm(x))
        x = x + self.mlp(rmsnorm(x))
        return x


# -----------------------------------------------------------------------------
# The main GPT-2 model

@dataclass
class GPTConfig:
    vocab_size: int = 50257
    n_layer: int = 12
    n_head: int = 12
    n_embd: int = 768


class GPT(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.transformer = nn.ModuleDict(
            dict(
                wte=nn.Embedding(config.vocab_size, config.n_embd),
                h=nn.ModuleList([Block(config) for _ in range(config.n_layer)]),
            )
        )
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        self.transformer.wte.weight = self.lm_head.weight  # weight tying

    def forward(self, idx, targets=None, return_logits=True):
        b, t = idx.size()
        x = self.transformer.wte(idx)
        for block in self.transformer.h:
            x = block(x)
        x = rmsnorm(x)

        if targets is not None:
            logits = self.lm_head(x)
            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=-1
            )
        else:
            logits = self.lm_head(x[:, [-1], :])
            loss = None

        if not return_logits:
            logits = None
        return logits, loss

    def configure_optimizers(self, weight_decay, learning_rate, betas, device_type):
        optimizer = torch.optim.AdamW(
            self.parameters(), lr=learning_rate, weight_decay=weight_decay, betas=betas
        )
        return optimizer


# -----------------------------------------------------------------------------
# Our own simple Distributed Data Loader

def _peek_data_shard(filename):
    with open(filename, "rb") as f:
        header = np.frombuffer(f.read(256 * 4), dtype=np.int32)
    if header[0] != 20240520:
        print("ERROR: magic number mismatch in the data .bin file!")
        print("---> HINT: Are you passing in a correct file with --input_bin?")
        print("---> HINT: Dataset encoding changed recently, re-run data prepro or refer again to README")
        print("---> HINT: For example re-run: `python dev/data/tinyshakespeare.py`, then re-try")
        exit(1)
    assert header[1] == 1, "unsupported version"
    ntok = header[2]
    return ntok


def _load_data_shard(filename):
    with open(filename, "rb") as f:
        header = np.frombuffer(f.read(256 * 4), dtype=np.int32)
        assert header[0] == 20240520, "magic number mismatch in the data .bin file"
        assert header[1] == 1, "unsupported version"
        ntok = header[2]
        tokens = np.frombuffer(f.read(), dtype=np.uint16)
    assert len(tokens) == ntok, "number of tokens read does not match header?"
    return tokens


class DistributedDataLoader:
    def __init__(self, filename_pattern, B, T, process_rank, num_processes, device, non_blocking):
        self.process_rank = process_rank
        self.num_processes = num_processes
        self.B = B
        self.T = T
        self.device = device
        self.non_blocking = non_blocking

        self.files = sorted(glob.glob(filename_pattern))
        assert len(self.files) > 0, f"did not find any files that match the pattern {filename_pattern}"

        ntok_total = np.int64(0)
        for fname in self.files:
            shard_ntok = _peek_data_shard(fname)
            assert shard_ntok >= num_processes * B * T + 1
            ntok_total += shard_ntok
        self.ntok_total = ntok_total
        print0(f"DataLoader: total number of tokens: {ntok_total:,} across {len(self.files)} files")

        self.reset()

    def reset(self):
        self.current_shard = 0
        self.current_position = self.process_rank * self.B * self.T
        self.tokens = _load_data_shard(self.files[self.current_shard])

    def advance(self):
        self.current_shard = (self.current_shard + 1) % len(self.files)
        self.current_position = self.process_rank * self.B * self.T
        self.tokens = _load_data_shard(self.files[self.current_shard])

    def next_batch(self):
        B = self.B
        T = self.T
        buf = self.tokens[self.current_position: self.current_position + B * T + 1]
        buf = torch.tensor(buf.astype(np.int32), dtype=torch.long)
        x = (buf[:-1]).view(B, T)
        y = (buf[1:]).view(B, T)
        self.current_position += B * T * self.num_processes
        if self.current_position + (B * T * self.num_processes + 1) > len(self.tokens):
            self.advance()
        # device-aware transfer
        x = x.to(self.device, non_blocking=self.non_blocking)
        y = y.to(self.device, non_blocking=self.non_blocking)
        return x, y


# -----------------------------------------------------------------------------
# int main

VAL_TOKENS = 1_048_576  # fixed for consistent comparisons

if __name__ == "__main__":
    B, T = args.batch_size, args.sequence_length
    assert args.model in {"d12", "d24", "d36", "d48"}

    use_cuda = torch.cuda.is_available()
    # Detect distributed launch (torchrun sets these)
    env_world_size = int(os.environ.get("WORLD_SIZE", "1"))
    distributed = env_world_size > 1

    # Backend selection
    backend = "nccl" if use_cuda else "gloo"

    # Ranks
    ddp_rank = int(os.environ.get("RANK", "0"))
    ddp_local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    ddp_world_size = env_world_size

    # Device
    if use_cuda:
        device = torch.device(f"cuda:{ddp_local_rank}")
        torch.cuda.set_device(device)
        non_blocking = True
    else:
        device = torch.device("cpu")
        non_blocking = False

    master_process = ddp_rank == 0
    print(f"using device: {device}, distributed={distributed}, world_size={ddp_world_size}")

    # Initialize process group only if actually distributed
    if distributed and not dist.is_initialized():
        init_process_group(backend=backend)

    # grad accumulation scaling per-GPU only when distributed
    assert args.grad_accumulation_steps % max(ddp_world_size, 1) == 0, \
        "grad_accumulation_steps must be divisible by world size"
    args.grad_accumulation_steps //= max(ddp_world_size, 1)

    if args.log_wandb and master_process:
        import wandb
        import datetime
        start_time = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        wandb.init(project="benchmark_gpt2", name=f"gpt2-{args.model} {start_time}")
        wandb.config.update(args)
        wandb.save("train_gpt2.py")
        wandb.save("run.sh")

    tokens_per_iter = B * T * max(ddp_world_size, 1) * args.grad_accumulation_steps
    print0(f"tokens per iteration: {tokens_per_iter:,}")

    # AMP/autocast context
    if use_cuda:
        ctx = torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16)
    else:
        ctx = nullcontext()  # disable autocast on CPU for portability

    # Data loaders
    train_loader = DistributedDataLoader(args.input_bin, B, T, ddp_rank, max(ddp_world_size, 1),
                                         device=device, non_blocking=non_blocking)
    tokens_per_iter_val = args.val_batch_size * T * max(ddp_world_size, 1)
    assert VAL_TOKENS % tokens_per_iter_val == 0
    val_steps = VAL_TOKENS // tokens_per_iter_val
    val_loader = DistributedDataLoader(args.input_val_bin, args.val_batch_size, T, ddp_rank, max(ddp_world_size, 1),
                                       device=device, non_blocking=non_blocking)
    x, y = train_loader.next_batch()

    # Model
    num_vocab = 50257
    model_config = {
        "d12": GPTConfig(vocab_size=num_vocab, n_layer=12, n_head=12, n_embd=768),
        "d24": GPTConfig(vocab_size=num_vocab, n_layer=24, n_head=16, n_embd=1024),
        "d36": GPTConfig(vocab_size=num_vocab, n_layer=36, n_head=20, n_embd=1280),
        "d48": GPTConfig(vocab_size=num_vocab, n_layer=48, n_head=25, n_embd=1600),
    }[args.model]
    model = GPT(model_config).train().to(device)

    if hasattr(config, "coordinate_descent_tuning"):
        config.coordinate_descent_tuning = True  # suggested by @Chillee
    print0("compiling the model...")
    try:
        model = torch.compile(model)  # works on CPU & CUDA; disable if it causes issues
    except Exception as e:
        print0(f"torch.compile disabled due to: {e}")

    # Wrap with DDP only if world_size > 1
    if distributed and ddp_world_size > 1:
        if use_cuda:
            model = DDP(model, device_ids=[ddp_local_rank])
        else:
            model = DDP(model)  # CPU DDP with Gloo (for multi-proc CPU testing)
    raw_model = model.module if isinstance(model, DDP) else model

    # Optimizer
    optimizer = raw_model.configure_optimizers(
        weight_decay=args.weight_decay,
        learning_rate=args.learning_rate,
        betas=(0.9, 0.95),
        device_type=("cuda" if use_cuda else "cpu"),
    )

    # LR schedule
    def get_lr(it):
        assert it <= args.num_iterations
        if it < args.warmup_iters and args.warmup_iters > 0:
            return args.learning_rate * (it + 1) / args.warmup_iters
        elif it < args.num_iterations - args.warmdown_iters:
            return args.learning_rate
        else:
            decay_ratio = (args.num_iterations - it) / max(args.warmdown_iters, 1)
            return args.learning_rate * decay_ratio

    run_id = str(uuid.uuid4())

    # logging
    logfile = None
    if master_process and args.output_dir:
        os.makedirs(args.output_dir, exist_ok=True)
        logfile = os.path.join(args.output_dir, f"{run_id}.log")
        with open(logfile, "w"):
            pass

    def device_sync():
        if use_cuda:
            torch.cuda.synchronize()

    training_time_ms = 0.0
    device_sync()
    t0 = time.perf_counter()

    # training
    for step in range(args.num_iterations + 1):
        last_step = step == args.num_iterations

        if args.val_loss_every > 0 and (step % args.val_loss_every == 0 or last_step):
            device_sync()
            training_time_ms += 1000 * (time.perf_counter() - t0)
            model.eval()
            val_loader.reset()
            with torch.no_grad():
                val_loss = torch.zeros(1, device=device)
                for _ in range(val_steps):
                    x_val, y_val = val_loader.next_batch()
                    _, loss = model(x_val, y_val, return_logits=False)
                    val_loss += loss
                if dist.is_initialized() and ddp_world_size > 1:
                    dist.all_reduce(val_loss, op=dist.ReduceOp.AVG)
                val_loss /= val_steps

            print0(f"step:{step}/{args.num_iterations} | val loss {val_loss.item():.6f}")
            if master_process:
                if args.log_wandb:
                    import wandb
                    wandb.log({"val_loss": val_loss.item()}, step=step * tokens_per_iter)
                    wandb.log({"time": training_time_ms}, step=step * tokens_per_iter)
                if logfile is not None:
                    with open(logfile, "a") as f:
                        f.write(f"s:{step} val:{val_loss.item():.6f}\n")

            device_sync()
            t0 = time.perf_counter()

        if last_step:
            break

        model.train()
        train_loss = torch.zeros(1, device=device)
        for micro_step in range(args.grad_accumulation_steps):
            if isinstance(model, DDP):
                model.require_backward_grad_sync = (micro_step == args.grad_accumulation_steps - 1)
            with ctx:
                _, loss = model(x, y, return_logits=False)
                loss = loss / args.grad_accumulation_steps
                train_loss += loss.detach()
            x, y = train_loader.next_batch()
            loss.backward()

        train_loss /= args.grad_accumulation_steps

        lr = get_lr(step)
        for pg in optimizer.param_groups:
            pg["lr"] = lr
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)

        device_sync()
        approx_training_time_ms = training_time_ms + 1000 * (time.perf_counter() - t0)
        if dist.is_initialized() and ddp_world_size > 1:
            dist.all_reduce(train_loss, op=dist.ReduceOp.AVG)
        lossf = train_loss.item()
        print0(
            f"step:{step}/{args.num_iterations} | loss {lossf:.6f} | "
            f"train_time:{approx_training_time_ms / 1000:.2f}s | "
            f"step_avg:{approx_training_time_ms / (step + 1):.2f}ms"
        )
        if master_process and logfile is not None:
            with open(logfile, "a") as f:
                f.write(f"s:{step} trn:{lossf:.6f}\n")

        if master_process and (step + 1) % args.save_every == 0:
            log = dict(model=raw_model.state_dict(), code=code, args=args.__dict__)
            os.makedirs(f"logs/{run_id}", exist_ok=True)
            torch.save(log, f"logs/{run_id}/model_step{step:06d}.pt")

    if use_cuda:
        print0(f"peak memory consumption: {torch.cuda.max_memory_allocated() // 1024 // 1024} MiB")
    else:
        print0("peak memory consumption: N/A (CPU)")

    if master_process:
        log = dict(model=raw_model.state_dict(), code=code, args=args.__dict__)
        os.makedirs(f"logs/{run_id}", exist_ok=True)
        torch.save(log, f"logs/{run_id}/final.pt")

    if dist.is_initialized():
        destroy_process_group()
