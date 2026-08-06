# deep_rbsde_skeleton.py
from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Callable, Dict, Any, Optional, Tuple, List

import os
import csv
import copy
import time 

import math
import torch
import torch.nn as nn

# ==========================================================
# GLOBAL MODE


ETAT = "RUN"   # "TEST" or "RUN"
# ==========================================================

def forward_in_chunks(module, x: torch.Tensor, chunk_size: int) -> torch.Tensor:
    """
    Apply `module` to `x` by chunks along dimension 0, then concatenate.
    Useful to avoid CUDA OOM on very large batches.
    """
    if chunk_size is None or chunk_size <= 0 or x.shape[0] <= chunk_size:
        return module(x)

    out_chunks = []
    for start in range(0, x.shape[0], chunk_size):
        end = min(start + chunk_size, x.shape[0])
        out_chunks.append(module(x[start:end]))
    return torch.cat(out_chunks, dim=0)

def _relative_metric_with_ci(value: float, ci_low: float, ci_high: float, price: float) -> Dict[str, float]:
    if (not math.isfinite(price)) or abs(price) < 1e-14:
        return {"value": float("nan"), "ci_low": float("nan"), "ci_high": float("nan")}
    rel_value = value / price
    rel_low_raw = ci_low / price
    rel_high_raw = ci_high / price
    return {
        "value": rel_value,
        "ci_low": min(rel_low_raw, rel_high_raw),
        "ci_high": max(rel_low_raw, rel_high_raw),
    }



# ============================
# Incremental CSV helpers
# ============================
def result_row_from_res(res: Dict[str, Any]) -> Dict[str, Any]:
    params = res.get("params", {})
    sensi = res.get("sensitivities") or {}
    best = res.get("best", {})

    return {
        "K": float(params.get("K", float("nan"))),
        "r": float(params.get("r", float("nan"))),
        "sigma": float(params.get("sigma", float("nan"))),
        "S0": float(params.get("S0", float("nan"))),
        "T": float(params.get("T", float("nan"))),

        "price": float(sensi.get("price", float("nan"))),
        "price_std": float(sensi.get("price_std", float("nan"))),
        "price_se": float(sensi.get("price_se", float("nan"))),
        "price_ci_low": float(sensi.get("price_ci_low", float("nan"))),
        "price_ci_high": float(sensi.get("price_ci_high", float("nan"))),

        "E_disc_int_absZ": float(sensi.get("E_disc_int_absZ", float("nan"))),
        "sqrt_E_disc_int_Z2": float(sensi.get("sqrt_E_disc_int_Z2", float("nan"))),

        "Vinf_prime0": float(sensi.get("Vinf_prime0", float("nan"))),
        "Vinf_prime0_ci_low": float(sensi.get("Vinf_prime0_ci_low", float("nan"))),
        "Vinf_prime0_ci_high": float(sensi.get("Vinf_prime0_ci_high", float("nan"))),

        "V2_prime0": float(sensi.get("V2_prime0", float("nan"))),
        "V2_prime0_ci_low": float(sensi.get("V2_prime0_ci_low", float("nan"))),
        "V2_prime0_ci_high": float(sensi.get("V2_prime0_ci_high", float("nan"))),

        "relative_Vinf_prime0": float(sensi.get("relative_Vinf_prime0", float("nan"))),
        "relative_Vinf_prime0_ci_low": float(sensi.get("relative_Vinf_prime0_ci_low", float("nan"))),
        "relative_Vinf_prime0_ci_high": float(sensi.get("relative_Vinf_prime0_ci_high", float("nan"))),

        "relative_V2_prime0": float(sensi.get("relative_V2_prime0", float("nan"))),
        "relative_V2_prime0_ci_low": float(sensi.get("relative_V2_prime0_ci_low", float("nan"))),
        "relative_V2_prime0_ci_high": float(sensi.get("relative_V2_prime0_ci_high", float("nan"))),

        "compute_device": res.get("compute_device", ""),
        "best_loss": float(best.get("loss", float("nan"))),
        "best_step": int(best.get("step", -1)),
        "final_step": int(res.get("final_step", -1)),
    }

def append_result_csv(res: Dict[str, Any], csv_path: str) -> None:
    row = result_row_from_res(res)
    os.makedirs(os.path.dirname(csv_path) or ".", exist_ok=True)
    file_exists = os.path.exists(csv_path)

    if file_exists:
        # avoid duplicates on K if rerun / resume
        try:
            with open(csv_path, "r", newline="") as f:
                reader = csv.DictReader(f)
                for existing in reader:
                    try:
                        if float(existing.get("K", "nan")) == row["K"]:
                            return
                    except Exception:
                        pass
        except Exception:
            pass

    with open(csv_path, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(row.keys()))
        if not file_exists:
            writer.writeheader()
        writer.writerow(row)


# ----------------------------
# Config
# ----------------------------
@dataclass

class RBSDEConfigTest:

    DEVICE: str = "cuda"
    DTYPE: torch.dtype = torch.float32
    SEED: int = 1234

    N_TIME_STEPS: int = 50
    N_BATCH: int = 2**12

    LOSS_EPSILON_STOPPING: float = 1e-2
    MIN_DELTA: float = 1e-6
    PATIENCE: int = 15
    CHECK_EVERY: int = 50

    LR: float = 1e-3
    MAX_STEPS: int = 2000
    GRAD_CLIP: float = 1.0

    HIDDEN_DIM: int = 128
    N_HIDDEN_LAYERS: int = 3
    ACTIVATION: str = "silu"
    USE_LAYER_NORM: bool = False

    LEARNABLE_Y0: bool = True
    USE_BF16_WARMUP: bool = True
    BF16_TO_FP32_LOSS_SWITCH: float = 1e-1
    BF16_WARMUP_MIN_STEPS: int = 0

    ZNET_CHUNK_SIZE: int = 2**16

@dataclass
class RBSDEConfigRun:

    DEVICE: str = "cuda"
    DTYPE: torch.dtype = torch.float32
    SEED: int = 1234

    N_TIME_STEPS: int = 100
    N_BATCH: int = 2**17

    LOSS_EPSILON_STOPPING: float = 5e-3
    MIN_DELTA: float = 1e-6
    PATIENCE: int = 10
    CHECK_EVERY: int = 50

    LR: float = 1e-3
    MAX_STEPS: int = 20000
    GRAD_CLIP: float = 1.0

    HIDDEN_DIM: int = 192
    N_HIDDEN_LAYERS: int = 4
    ACTIVATION: str = "silu"
    USE_LAYER_NORM: bool = False

    LEARNABLE_Y0: bool = True

    ZNET_CHUNK_SIZE: int = 2**15


def get_rbsde_config():

    if ETAT == "TEST":
        cfg = RBSDEConfigTest()

    elif ETAT == "RUN":
        cfg = RBSDEConfigRun()

    else:
        raise ValueError(f"Unknown ETAT = {ETAT}")

    print("======================================")
    print(f"RBSDE CONFIG MODE : {ETAT}")
    print("======================================")

    return cfg      

RBSDEConfig = RBSDEConfigTest | RBSDEConfigRun

def _cpu_clone_state_dict(state_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    return {k: v.detach().cpu().clone() for k, v in state_dict.items()}


def _save_checkpoint_atomic(checkpoint_path: str, payload: Dict[str, Any]) -> None:
    directory = os.path.dirname(checkpoint_path) or "."
    os.makedirs(directory, exist_ok=True)
    tmp_path = checkpoint_path + ".tmp"
    torch.save(payload, tmp_path)
    os.replace(tmp_path, checkpoint_path)


def _load_checkpoint_if_exists(checkpoint_path: Optional[str]) -> Optional[Dict[str, Any]]:
    if checkpoint_path is None or not os.path.exists(checkpoint_path):
        return None
    return torch.load(checkpoint_path, map_location="cpu")


def _checkpoint_path_for_index(status_file: str, idx: int, params: Dict[str, Any]) -> str:
    checkpoint_dir = os.path.join(os.path.dirname(status_file) or ".", "checkpoints")
    k_value = params.get("K", "NA")
    return os.path.join(checkpoint_dir, f"rbsde_idx_{idx:03d}_K_{k_value}.pt")


# ----------------------------
# Generic MLP
# ----------------------------


def set_seed(seed: int) -> None:
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def get_activation(name: str) -> nn.Module:
    name = name.lower()
    if name in ("relu",):
        return nn.ReLU()
    if name in ("tanh",):
        return nn.Tanh()
    if name in ("gelu",):
        return nn.GELU()
    if name in ("silu", "swish"):
        return nn.SiLU()
    raise ValueError(f"Unknown activation: {name}")


class MLP(nn.Module):
    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        hidden_dim: int,
        n_hidden_layers: int,
        activation: nn.Module,
        use_layer_norm: bool = False,
    ):
        super().__init__()
        layers: List[nn.Module] = []
        d = in_dim
        for _ in range(n_hidden_layers):
            layers.append(nn.Linear(d, hidden_dim))
            if use_layer_norm:
                layers.append(nn.LayerNorm(hidden_dim))
            layers.append(activation)
            d = hidden_dim
        layers.append(nn.Linear(d, out_dim))
        self.net = nn.Sequential(*layers)

        # Reasonable init
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


# ----------------------------
# State / path simulation (plug-and-play)
# ----------------------------

@dataclass
class PathBundle:
    # Shapes:
    # t_grid: (N+1,)
    # W: (B, N+1, dW_dim)
    # dW: (B, N, dW_dim)
    # S: (B, N+1, 1)
    # M: (B, N+1, 1)
    t_grid: torch.Tensor
    W: torch.Tensor
    dW: torch.Tensor
    S: torch.Tensor
    M: torch.Tensor


class GBMRunningMaxSimulator:
    """
    Example simulator for the user case:
    S_t = S0 exp((r - 0.5 sigma^2)t + sigma W_t)
    M_t = max_{s<=t} S_s
    Brownian dimension dW_dim = 1 here, but structure is easy to generalize.
    """
    def __init__(self, cfg: RBSDEConfig):
        self.cfg = cfg

    @torch.no_grad()
    def sample(self, params: Dict[str, Any]) -> PathBundle:
        cfg = self.cfg
        device, dtype = cfg.DEVICE, cfg.DTYPE
        B = cfg.N_BATCH
        N = cfg.N_TIME_STEPS
        T = float(params["T"])
        dt = T / N

        r = float(params["r"])
        sigma = float(params["sigma"])
        S0 = float(params["S0"])

        t_grid = torch.linspace(0.0, T, N + 1, device=device, dtype=dtype)  # (N+1,)
        dW = torch.randn(B, N, 1, device=device, dtype=dtype) * math.sqrt(dt)  # (B,N,1)
        W = torch.zeros(B, N + 1, 1, device=device, dtype=dtype)
        W[:, 1:, :] = torch.cumsum(dW, dim=1)

        # log S
        t = t_grid.view(1, N + 1, 1)
        logS = math.log(S0) + (r - 0.5 * sigma * sigma) * t + sigma * W
        S = torch.exp(logS)  # (B,N+1,1)
        M, _ = torch.cummax(S, dim=1)  # running max along time

        return PathBundle(t_grid=t_grid, W=W, dW=dW, S=S, M=M)


# ----------------------------
# Xi / obstacle interface
# ----------------------------
XiFn = Callable[[torch.Tensor, Dict[str, torch.Tensor], Dict[str, Any]], torch.Tensor]
# signature:
#   xi(t, path_dict, params) -> (B,1) obstacle value at that time
#
# path_dict provides tensors with time-dimension already indexed, e.g.:
#   {"S": S_t (B,1), "M": M_t (B,1), "t": t_scalar (B,1) or (1,1)}
# params is python dict for scalar params.


def default_phi(x: torch.Tensor, smooth_eps: float = 1e-3) -> torch.Tensor:
    """
    Smooth approx of x^+ : softplus
    """
    return torch.nn.functional.softplus(x / smooth_eps) * smooth_eps


def xi_american_lookback(
    t: torch.Tensor,
    path_t: Dict[str, torch.Tensor],
    params: Dict[str, Any],
    ) -> torch.Tensor:
    """
    Obstacle for Proposition: xi_t = -phi(M_t - K).
    """
    K = float(params["K"])
    M_t = path_t["M"]  # (B,1)
    return -default_phi(M_t - K)

# ----------------------------
# Networks: Z(t, state) and optionally Y0
# ----------------------------
class ZNet(nn.Module):
    """
    Approximates Z_t = z_theta(t, X_t) where X_t is chosen state.
    In your example, X_t could be (logS_t, M_t) or (S_t, M_t).
    """
    def __init__(self, in_dim: int, cfg: RBSDEConfig):
        super().__init__()
        act = get_activation(cfg.ACTIVATION)
        self.mlp = MLP(
            in_dim=in_dim,
            out_dim=1,  # Brownian dimension = 1 here; generalize to d
            hidden_dim=cfg.HIDDEN_DIM,
            n_hidden_layers=cfg.N_HIDDEN_LAYERS,
            activation=act,
            use_layer_norm=cfg.USE_LAYER_NORM,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.mlp(x)


class LearnableY0(nn.Module):
    """
    In Deep BSDE variants, Y0 is often a direct parameter.
    """
    def __init__(self, init_value: float = 0.0):
        super().__init__()
        self.y0 = nn.Parameter(torch.tensor([init_value], dtype=torch.float32))

    def forward(self, batch_size: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        return self.y0.to(device=device, dtype=dtype).view(1, 1).repeat(batch_size, 1)


# ----------------------------
# Model wrapper
# ----------------------------
class DeepRBSDEModel(nn.Module):
    def __init__(self, cfg: RBSDEConfig, state_dim: int):
        super().__init__()
        self.cfg = cfg
        self.z_net = ZNet(in_dim=state_dim, cfg=cfg)
        self.y0_net: Optional[nn.Module] = LearnableY0(0.0) if cfg.LEARNABLE_Y0 else None

    def init_y0(self, batch_size: int) -> torch.Tensor:
        cfg = self.cfg
        device = torch.device(cfg.DEVICE)
        dtype = cfg.DTYPE
        if self.y0_net is None:
            # fallback: start from zeros
            return torch.zeros(batch_size, 1, device=device, dtype=dtype)
        return self.y0_net(batch_size, device, dtype)


# ----------------------------
# Utilities for early stopping
# ----------------------------
@torch.no_grad()
def grad_norm(model: nn.Module) -> float:
    total = 0.0
    for p in model.parameters():
        if p.grad is None:
            continue
        g = p.grad.detach()
        total += float(torch.sum(g * g).item())
    return math.sqrt(total)


# ----------------------------
# Parameter sweep helper
# ----------------------------
def iter_param_grid(param_grid: Dict[str, List[Any]]) -> List[Dict[str, Any]]:
    """
    Example:
    param_grid = {"K":[0.9,1.0,1.1], "sigma":[0.2,0.3]}
    returns list of dicts with cartesian product.
    """
    keys = list(param_grid.keys())
    grids = [param_grid[k] for k in keys]
    out: List[Dict[str, Any]] = []

    def rec(i: int, cur: Dict[str, Any]) -> None:
        if i == len(keys):
            out.append(dict(cur))
            return
        k = keys[i]
        for v in grids[i]:
            cur[k] = v
            rec(i + 1, cur)

    rec(0, {})
    return out



def _compute_rbsde_losses(
    cfg: RBSDEConfig,
    params: Dict[str, Any],
    xi_fn: XiFn,
    paths: PathBundle,
    model: DeepRBSDEModel,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    batch_size = cfg.N_BATCH
    n_time_steps = cfg.N_TIME_STEPS
    dt = float(params["T"]) / n_time_steps
    r = float(params["r"])
    tau_tol = 1e-6

    S = paths.S
    M = paths.M
    dW = paths.dW
    t_grid = paths.t_grid

    Y = model.init_y0(batch_size)
    K_acc = torch.zeros_like(Y)

    use_fast_vectorized_run = (ETAT == "RUN")

    if use_fast_vectorized_run:
        t_feat = t_grid[:-1].view(1, n_time_steps, 1).expand(batch_size, n_time_steps, 1)
        logS_all = torch.log(S[:, :-1, :] + 1e-12)
        M_all = M[:, :-1, :]

        state_all = torch.cat([t_feat, logS_all, M_all], dim=2)
        flat_state = state_all.reshape(batch_size * n_time_steps, 3)

        Z_all = forward_in_chunks(
            model.z_net,
            flat_state,
            chunk_size=cfg.ZNET_CHUNK_SIZE,
        )
        Z_all = Z_all.reshape(batch_size, n_time_steps, 1)

        if xi_fn is xi_american_lookback:
            K_value = float(params["K"])
            xi_all = -default_phi(M_all - K_value)
            xi_T = -default_phi(M[:, -1, :] - K_value)

            for i in range(n_time_steps):
                Z = Z_all[:, i, :]
                xi_t = xi_all[:, i, :]

                Y = Y + r * Y * dt + Z * dW[:, i, :]

                violation = Y - (xi_t + tau_tol)
                dK = torch.clamp(violation, min=0.0)
                Y = Y - dK
                K_acc = K_acc + dK
        else:
            for i in range(n_time_steps):
                t_i = t_grid[i].view(1, 1).expand(batch_size, 1)
                Z = Z_all[:, i, :]

                Y = Y + r * Y * dt + Z * dW[:, i, :]

                path_t = {"S": S[:, i, :], "M": M[:, i, :], "t": t_i}
                xi_t = xi_fn(t_i, path_t, params)

                violation = Y - (xi_t + tau_tol)
                dK = torch.clamp(violation, min=0.0)
                Y = Y - dK
                K_acc = K_acc + dK

            t_T = t_grid[-1].view(1, 1).expand(batch_size, 1)
            path_T = {"S": S[:, -1, :], "M": M[:, -1, :], "t": t_T}
            xi_T = xi_fn(t_T, path_T, params)
    else:
        for i in range(n_time_steps):
            t_i = t_grid[i].view(1, 1).expand(batch_size, 1)

            logS_t = torch.log(S[:, i, :] + 1e-12)
            M_t = M[:, i, :]
            state = torch.cat([t_i, logS_t, M_t], dim=1)

            Z = model.z_net(state)

            Y = Y + r * Y * dt + Z * dW[:, i, :]

            path_t = {"S": S[:, i, :], "M": M[:, i, :], "t": t_i}
            xi_t = xi_fn(t_i, path_t, params)

            violation = Y - (xi_t + tau_tol)
            dK = torch.clamp(violation, min=0.0)
            Y = Y - dK
            K_acc = K_acc + dK

        t_T = t_grid[-1].view(1, 1).expand(batch_size, 1)
        path_T = {"S": S[:, -1, :], "M": M[:, -1, :], "t": t_T}
        xi_T = xi_fn(t_T, path_T, params)

    loss_terminal = torch.mean((Y - xi_T) ** 2)
    loss_reflection = torch.mean(K_acc ** 2)
    loss = loss_terminal + 0.1 * loss_reflection
    return loss_terminal, loss_reflection, loss


# ----------------------------
# Skeleton of training loop (placeholder)
# ----------------------------
def train_one_setting(
    cfg: RBSDEConfig,
    params: Dict[str, Any],
    xi_fn: XiFn,
    init_state_dict: Optional[Dict[str, Any]] = None,
    lr_override: Optional[float] = None,
    checkpoint_path: Optional[str] = None,
    ) -> Dict[str, Any]:
    import time

    PROFILE_MODE = (ETAT == "TEST")

    def cuda_sync_if_needed() -> None:
        if PROFILE_MODE and cfg.DEVICE.startswith("cuda") and torch.cuda.is_available():
            torch.cuda.synchronize()

    set_seed(cfg.SEED)
    device = torch.device(cfg.DEVICE)

    simulator = GBMRunningMaxSimulator(cfg)
    model = DeepRBSDEModel(cfg, state_dim=3).to(device)

    if init_state_dict is not None:
        model.load_state_dict(init_state_dict)

    lr = lr_override if lr_override is not None else cfg.LR
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=0.5,
        patience=3,
        threshold=cfg.MIN_DELTA,
        min_lr=1e-6,
    )

    use_bf16_warmup = (
        cfg.DEVICE.startswith("cuda")
        and getattr(cfg, "USE_BF16_WARMUP", True)
    )
    mixed_precision_enabled = use_bf16_warmup
    bf16_switch_loss = getattr(cfg, "BF16_TO_FP32_LOSS_SWITCH", 1e-1)
    bf16_warmup_min_steps = getattr(cfg, "BF16_WARMUP_MIN_STEPS", 0)
    scaler = torch.amp.GradScaler("cuda", enabled=mixed_precision_enabled)

    best = {"loss": float("inf"), "step": -1}
    no_improve_count = 0
    switched_to_fp32 = False
    start_step = 0

    checkpoint = _load_checkpoint_if_exists(checkpoint_path)
    if checkpoint is not None:
        if checkpoint.get("completed", False):
            print(
                f"[train_one_setting] CHECKPOINT COMPLETE | K={params['K']} | loading saved result",
                flush=True,
            )
            return checkpoint["result"]

        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        scheduler.load_state_dict(checkpoint["scheduler_state_dict"])

        mixed_precision_enabled = bool(checkpoint.get("mixed_precision_enabled", mixed_precision_enabled))
        switched_to_fp32 = bool(checkpoint.get("switched_to_fp32", switched_to_fp32))
        best = checkpoint.get("best", best)
        no_improve_count = int(checkpoint.get("no_improve_count", no_improve_count))
        start_step = int(checkpoint.get("step", -1)) + 1

        scaler = torch.amp.GradScaler("cuda", enabled=mixed_precision_enabled)
        scaler_state = checkpoint.get("scaler_state_dict")
        if scaler_state is not None and mixed_precision_enabled:
            scaler.load_state_dict(scaler_state)

        cpu_rng_state = checkpoint.get("cpu_rng_state")
        if cpu_rng_state is not None:
            torch.random.set_rng_state(cpu_rng_state)
        cuda_rng_state_all = checkpoint.get("cuda_rng_state_all")
        if cuda_rng_state_all is not None and torch.cuda.is_available():
            torch.cuda.set_rng_state_all(cuda_rng_state_all)

        print(
            f"[train_one_setting] RESUME | MODE={ETAT} | K={params['K']} | from_step={start_step}",
            flush=True,
        )

    precision_mode = "bf16-warmup" if mixed_precision_enabled else "fp32"
    print(
        f"[train_one_setting] START | "
        f"MODE={ETAT} | "
        f"K={params['K']} | "
        f"N_BATCH={cfg.N_BATCH} | "
        f"N_TIME_STEPS={cfg.N_TIME_STEPS} | "
        f"MAX_STEPS={cfg.MAX_STEPS} | "
        f"precision={precision_mode} | "
        f"switch_loss={bf16_switch_loss:.2e}",
        flush=True,
    )

    t_train_start = time.time()
    block_path_time = 0.0
    block_train_time = 0.0
    block_steps = 0
    save_every = getattr(cfg, "SAVE_EVERY", cfg.CHECK_EVERY)

    for step in range(start_step, cfg.MAX_STEPS):
        cuda_sync_if_needed()
        t_paths_start = time.time()
        paths = simulator.sample(params)
        cuda_sync_if_needed()
        t_paths_end = time.time()

        path_gen_time = t_paths_end - t_paths_start
        block_path_time += path_gen_time

        cuda_sync_if_needed()
        t_step_start = time.time()

        optimizer.zero_grad(set_to_none=True)

        if mixed_precision_enabled:
            with torch.amp.autocast("cuda", enabled=True, dtype=torch.bfloat16):
                loss_terminal, loss_reflection, loss = _compute_rbsde_losses(
                    cfg=cfg,
                    params=params,
                    xi_fn=xi_fn,
                    paths=paths,
                    model=model,
                )

            scaler.scale(loss).backward()

            if cfg.GRAD_CLIP is not None and cfg.GRAD_CLIP > 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.GRAD_CLIP)

            scaler.step(optimizer)
            scaler.update()
        else:
            loss_terminal, loss_reflection, loss = _compute_rbsde_losses(
                cfg=cfg,
                params=params,
                xi_fn=xi_fn,
                paths=paths,
                model=model,
            )

            loss.backward()

            if cfg.GRAD_CLIP is not None and cfg.GRAD_CLIP > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.GRAD_CLIP)

            optimizer.step()

        cuda_sync_if_needed()
        t_step_end = time.time()

        train_step_time = t_step_end - t_step_start
        block_train_time += train_step_time
        block_steps += 1

        if checkpoint_path is not None and (step + 1) % save_every == 0:
            payload = {
                "completed": False,
                "step": step,
                "params": dict(params),
                "best": best,
                "no_improve_count": no_improve_count,
                "mixed_precision_enabled": mixed_precision_enabled,
                "switched_to_fp32": switched_to_fp32,
                "model_state_dict": _cpu_clone_state_dict(model.state_dict()),
                "optimizer_state_dict": optimizer.state_dict(),
                "scheduler_state_dict": scheduler.state_dict(),
                "scaler_state_dict": scaler.state_dict() if mixed_precision_enabled else None,
                "cpu_rng_state": torch.random.get_rng_state(),
                "cuda_rng_state_all": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
            }
            _save_checkpoint_atomic(checkpoint_path, payload)

        if step % cfg.CHECK_EVERY == 0 and step > 0:
            current_loss = float(loss.item())
            current_loss_terminal = float(loss_terminal.item())
            current_loss_reflection = float(loss_reflection.item())
            scheduler.step(current_loss)

            just_switched_to_fp32 = False

            if (
                mixed_precision_enabled
                and (current_loss < bf16_switch_loss)
                and (step >= bf16_warmup_min_steps)
            ):
                mixed_precision_enabled = False
                just_switched_to_fp32 = True
                switched_to_fp32 = True
                scaler = torch.amp.GradScaler("cuda", enabled=False)

                print(
                    f"[train_one_setting] SWITCH PRECISION | "
                    f"K={params['K']} | step={step} | "
                    f"loss={current_loss:.6e} | new_precision=fp32",
                    flush=True,
                )

            if current_loss < best["loss"] - cfg.MIN_DELTA:
                best = {
                    "loss": current_loss,
                    "loss_terminal": current_loss_terminal,
                    "loss_reflection": current_loss_reflection,
                    "step": step,
                }
                no_improve_count = 0
            else:
                no_improve_count += 1

            elapsed_train = time.time() - t_train_start
            current_lr = optimizer.param_groups[0]["lr"]
            precision_label = "bf16" if mixed_precision_enabled else "fp32"

            if PROFILE_MODE:
                avg_path_time = block_path_time / max(block_steps, 1)
                avg_train_time = block_train_time / max(block_steps, 1)
                avg_total_step_time = (block_path_time + block_train_time) / max(block_steps, 1)

                print(
                    f"[train_one_setting] MODE=TEST | "
                    f"K={params['K']} | "
                    f"step={step}/{cfg.MAX_STEPS} | "
                    f"loss={current_loss:.6e} | "
                    f"loss_terminal={current_loss_terminal:.6e} | "
                    f"loss_reflection={current_loss_reflection:.6e} | "
                    f"best_loss={best['loss']:.6e} | "
                    f"no_improve={no_improve_count}/{cfg.PATIENCE} | "
                    f"lr={current_lr:.2e} | "
                    f"precision={precision_label} | "
                    f"avg_path_gen={avg_path_time:.4f}s | "
                    f"avg_train_step={avg_train_time:.4f}s | "
                    f"avg_total_step={avg_total_step_time:.4f}s | "
                    f"elapsed={elapsed_train:.2f}s",
                    flush=True,
                )
            else:
                print(
                    f"[train_one_setting] MODE=RUN | "
                    f"K={params['K']} | "
                    f"step={step}/{cfg.MAX_STEPS} | "
                    f"loss={current_loss:.6e} | "
                    f"best_loss={best['loss']:.6e} | "
                    f"no_improve={no_improve_count}/{cfg.PATIENCE} | "
                    f"lr={current_lr:.2e} | "
                    f"precision={precision_label} | "
                    f"elapsed={elapsed_train:.2f}s",
                    flush=True,
                )

            block_path_time = 0.0
            block_train_time = 0.0
            block_steps = 0

            can_early_stop = (not mixed_precision_enabled) and (not just_switched_to_fp32)

            if can_early_stop and current_loss < cfg.LOSS_EPSILON_STOPPING:
                print(
                    f"[train_one_setting] EARLY STOP (loss_tol reached) | "
                    f"K={params['K']} | step={step} | loss={current_loss:.6e}",
                    flush=True,
                )
                break

            if can_early_stop and no_improve_count >= cfg.PATIENCE:
                print(
                    f"[train_one_setting] EARLY STOP (plateau) | "
                    f"K={params['K']} | step={step} | best_loss={best['loss']:.6e}",
                    flush=True,
                )
                break

    cuda_sync_if_needed()
    t_train_end = time.time()

    final_precision = "fp32" if (switched_to_fp32 or not use_bf16_warmup) else "bf16"
    print(
        f"[train_one_setting] END TRAIN | "
        f"MODE={ETAT} | "
        f"K={params['K']} | "
        f"final_precision={final_precision} | "
        f"total_train_time={t_train_end - t_train_start:.2f}s",
        flush=True,
    )

    sensi = evaluate_sensitivities(
        cfg=cfg,
        model=model,
        simulator=simulator,
        params=params,
        xi_fn=xi_fn,
        n_eval_batches=4,
        tau_tol=1e-6,
    )

    result = {
        "params": params,
        "best": best,
        "final_step": step,
        "config": asdict(cfg),
        "sensitivities": sensi,
        "compute_device": get_compute_device_name(cfg.DEVICE),
        "model_state_dict": _cpu_clone_state_dict(model.state_dict()),
    }

    if checkpoint_path is not None:
        payload = {
            "completed": True,
            "step": step,
            "params": dict(params),
            "best": best,
            "no_improve_count": no_improve_count,
            "mixed_precision_enabled": mixed_precision_enabled,
            "switched_to_fp32": switched_to_fp32,
            "model_state_dict": result["model_state_dict"],
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "scaler_state_dict": scaler.state_dict() if mixed_precision_enabled else None,
            "cpu_rng_state": torch.random.get_rng_state(),
            "cuda_rng_state_all": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
            "result": result,
        }
        _save_checkpoint_atomic(checkpoint_path, payload)

    return result

def evaluate_sensitivities(
    cfg: RBSDEConfig,
    model: DeepRBSDEModel,
    simulator: GBMRunningMaxSimulator,
    params: Dict[str, Any],
    xi_fn: XiFn,
    n_eval_batches: int = 4,
    tau_tol: float = 1e-6) -> Dict[str, float]:
    r"""
    Estimates the Proposition's sensitivities (discrete-time approximation):
    """
    device = torch.device(cfg.DEVICE)
    model.eval()

    n = cfg.N_TIME_STEPS
    r = float(params["r"])

    samples_disc_int_absZ: List[torch.Tensor] = []
    samples_disc_int_Z2: List[torch.Tensor] = []

    with torch.no_grad():
        # Price = learned initial value Y0 in the Deep RBSDE scheme
        price = float(model.init_y0(1).detach().cpu().view(-1)[0].item())
        price_std = 0.0
        price_se = 0.0
        price_ci_low = price
        price_ci_high = price

        for _ in range(n_eval_batches):
            paths = simulator.sample(params)
            batch_size = paths.S.shape[0]

            S = paths.S
            M = paths.M
            dW = paths.dW
            t_grid = paths.t_grid

            Y = model.init_y0(batch_size)

            alive = torch.ones(batch_size, 1, device=device, dtype=cfg.DTYPE)

            tau_hat = torch.full(
                (batch_size, 1),
                float(params["T"]),
                device=device,
                dtype=cfg.DTYPE,
            )

            int_absZ = torch.zeros(batch_size, 1, device=device, dtype=cfg.DTYPE)
            int_Z2 = torch.zeros(batch_size, 1, device=device, dtype=cfg.DTYPE)

            for i in range(n):
                t_i = t_grid[i].view(1, 1).expand(batch_size, 1)

                if i < n - 1:
                    dt_i = (t_grid[i + 1] - t_grid[i]).item()
                else:
                    dt_i = (t_grid[i] - t_grid[i - 1]).item()

                logS_t = torch.log(S[:, i, :] + 1e-12)
                M_t = M[:, i, :]
                state = torch.cat([t_i, logS_t, M_t], dim=1)

                Z = model.z_net(state)

                int_absZ = int_absZ + alive * torch.abs(Z) * dt_i
                int_Z2 = int_Z2 + alive * (Z * Z) * dt_i

                Y = Y + r * Y * dt_i + Z * dW[:, i, :]

                path_t = {"S": S[:, i, :], "M": M[:, i, :], "t": t_i}
                xi_t = xi_fn(t_i, path_t, params)

                violation = Y - (xi_t + tau_tol)
                dK = torch.clamp(violation, min=0.0)
                Y = Y - dK

                contact = (dK > 0.0) | (Y <= xi_t + tau_tol)
                newly_hit = (alive > 0.0) & contact
                tau_hat = torch.where(newly_hit, t_i, tau_hat)
                alive = alive * (~contact).to(alive.dtype)

            disc = torch.exp(-r * tau_hat)
            disc_int_absZ = disc * int_absZ
            disc_int_Z2 = disc * int_Z2

            samples_disc_int_absZ.append(disc_int_absZ.squeeze(-1).detach().cpu())
            samples_disc_int_Z2.append(disc_int_Z2.squeeze(-1).detach().cpu())

    x_abs = torch.cat(samples_disc_int_absZ, dim=0).to(torch.float64)
    x_z2 = torch.cat(samples_disc_int_Z2, dim=0).to(torch.float64)

    n_abs = int(x_abs.numel())
    n_z2 = int(x_z2.numel())

    mean_abs = float(x_abs.mean().item()) if n_abs > 0 else float("nan")
    mean_z2 = float(x_z2.mean().item()) if n_z2 > 0 else float("nan")

    std_abs = float(x_abs.std(unbiased=True).item()) if n_abs > 1 else 0.0
    std_z2 = float(x_z2.std(unbiased=True).item()) if n_z2 > 1 else 0.0

    se_abs = std_abs / math.sqrt(n_abs) if n_abs > 0 else float("nan")
    se_z2 = std_z2 / math.sqrt(n_z2) if n_z2 > 0 else float("nan")

    ci_abs_low = mean_abs - 1.96 * se_abs if math.isfinite(se_abs) else float("nan")
    ci_abs_high = mean_abs + 1.96 * se_abs if math.isfinite(se_abs) else float("nan")

    ci_z2_low = mean_z2 - 1.96 * se_z2 if math.isfinite(se_z2) else float("nan")
    ci_z2_high = mean_z2 + 1.96 * se_z2 if math.isfinite(se_z2) else float("nan")

    sqrt_mean_z2 = math.sqrt(max(mean_z2, 0.0)) if math.isfinite(mean_z2) else float("nan")
    sqrt_ci_low = math.sqrt(max(ci_z2_low, 0.0)) if math.isfinite(ci_z2_low) else float("nan")
    sqrt_ci_high = math.sqrt(max(ci_z2_high, 0.0)) if math.isfinite(ci_z2_high) else float("nan")

    rel_vinf = _relative_metric_with_ci(mean_abs, ci_abs_low, ci_abs_high, price)
    rel_v2 = _relative_metric_with_ci(sqrt_mean_z2, sqrt_ci_low, sqrt_ci_high, price)

    return {
        "price": price,
        "price_std": price_std,
        "price_se": price_se,
        "price_ci_low": price_ci_low,
        "price_ci_high": price_ci_high,

        "E_disc_int_absZ": mean_abs,
        "E_disc_int_absZ_std": std_abs,
        "E_disc_int_absZ_se": se_abs,
        "E_disc_int_absZ_ci_low": ci_abs_low,
        "E_disc_int_absZ_ci_high": ci_abs_high,

        "sqrt_E_disc_int_Z2": sqrt_mean_z2,
        "E_disc_int_Z2_std": std_z2,
        "E_disc_int_Z2_se": se_z2,
        "sqrt_E_disc_int_Z2_ci_low": sqrt_ci_low,
        "sqrt_E_disc_int_Z2_ci_high": sqrt_ci_high,

        "Vinf_prime0": mean_abs,
        "Vinf_prime0_ci_low": ci_abs_low,
        "Vinf_prime0_ci_high": ci_abs_high,

        "V2_prime0": sqrt_mean_z2,
        "V2_prime0_ci_low": sqrt_ci_low,
        "V2_prime0_ci_high": sqrt_ci_high,

        "relative_Vinf_prime0": rel_vinf["value"],
        "relative_Vinf_prime0_ci_low": rel_vinf["ci_low"],
        "relative_Vinf_prime0_ci_high": rel_vinf["ci_high"],

        "relative_V2_prime0": rel_v2["value"],
        "relative_V2_prime0_ci_low": rel_v2["ci_low"],
        "relative_V2_prime0_ci_high": rel_v2["ci_high"],
    }

def run_sweep(
    cfg: RBSDEConfig,
    param_grid: Dict[str, List[Any]],
    xi_fn: XiFn,
    warm_start: bool = True,
    status_file: str = "OFFICIAL/progress.txt",
    incremental_csv_path: str = "OFFICIAL/sensitivities.csv",
    ) -> List[Dict[str, Any]]:

    results = []
    previous_state_dict = None

    grid_list = iter_param_grid(param_grid)
    total_params = len(grid_list)
    t0_global = time.time()

    def format_seconds(seconds: float) -> str:
        seconds = max(float(seconds), 0.0)
        h = int(seconds // 3600)
        m = int((seconds % 3600) // 60)
        s = int(seconds % 60)
        return f"{h:02d}:{m:02d}:{s:02d}"

    os.makedirs(os.path.dirname(status_file) or ".", exist_ok=True)

    print("======================================", flush=True)
    print(f"Starting parameter sweep: {total_params} parameter values", flush=True)
    print(f"Warm start: {warm_start}", flush=True)
    print(f"Live progress written to: {status_file}", flush=True)
    print("======================================", flush=True)

    for idx, p in enumerate(grid_list, start=1):
        params = dict(p)
        t0_param = time.time()
        checkpoint_path = _checkpoint_path_for_index(status_file, idx, params)

        if warm_start and previous_state_dict is not None:
            current_lr = 0.3 * cfg.LR
        else:
            current_lr = cfg.LR

        res = train_one_setting(
            cfg,
            params=params,
            xi_fn=xi_fn,
            init_state_dict=previous_state_dict if warm_start else None,
            lr_override=current_lr,
            checkpoint_path=checkpoint_path,
        )

        if warm_start and "model_state_dict" in res:
            previous_state_dict = copy.deepcopy(res["model_state_dict"])

        res.pop("model_state_dict", None)
        results.append(res)
        append_result_csv(res, incremental_csv_path)

        elapsed_global = time.time() - t0_global
        elapsed_param = time.time() - t0_param
        avg_time_per_param = elapsed_global / idx
        remaining_params = total_params - idx
        eta_seconds = avg_time_per_param * remaining_params

        k_value = params["K"]

        status_line = (
            f"[{idx:03d}/{total_params:03d}] "
            f"K={k_value} | "
            f"last={format_seconds(elapsed_param)} | "
            f"avg={format_seconds(avg_time_per_param)} | "
            f"elapsed={format_seconds(elapsed_global)} | "
            f"ETA={format_seconds(eta_seconds)}"
        )
        with open(status_file, "w") as f:
            f.write(status_line + "\n")

    total_elapsed = time.time() - t0_global
    with open(status_file, "w") as f:
        f.write(f"Sweep finished in {format_seconds(total_elapsed)} \n")

    print("======================================", flush=True)
    print(
        f"Sweep finished: {total_params} parameter values processed "
        f"in {format_seconds(total_elapsed)}",
        flush=True,
    )
    print("======================================", flush=True)

    return results

# ----------------------------
# CSV export helper
# ----------------------------
def write_sensi_csv(results: List[Dict[str, Any]], csv_path: str) -> None:
    """Write prices, sensitivities, and relative sensitivities to a CSV."""
    rows = []
    for res in results:
        rows.append(result_row_from_res(res))

    rows.sort(key=lambda d: (d.get("r", 0.0), d.get("sigma", 0.0), d.get("K", 0.0)))

    os.makedirs(os.path.dirname(csv_path) or ".", exist_ok=True)
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()) if rows else [])
        writer.writeheader()
        writer.writerows(rows)

def get_compute_device_name(device: str) -> str:
    if device.startswith("cuda") and torch.cuda.is_available():
        if ":" in device:
            device_index = int(device.split(":", 1)[1])
        else:
            device_index = torch.cuda.current_device()
        return torch.cuda.get_device_name(device_index)
    return "CPU"


def main() -> None:
    """
    Minimal runnable entry point for batch jobs.
    Adjust the parameter grid here if needed.
    """
    cfg = get_rbsde_config()
    compute_device = get_compute_device_name(cfg.DEVICE)

    if ETAT == "TEST":
        strike_grid = [round(9.5 + 0.1 * i, 10) for i in range(10)]
    elif ETAT == "RUN":
        strike_grid = [round(9.5 + 0.02 * i, 10) for i in range(500)]
    else:
        raise ValueError(f"Unknown ETAT = {ETAT}")

    param_grid = {
        "r": [0.02],
        "sigma": [0.2],
        "S0": [10.0],
        "T": [0.5],
        "K": strike_grid,
    }

    print(f"Compute device used: {compute_device}", flush=True)
    print(f"Number of parameter combinations: {len(iter_param_grid(param_grid))}", flush=True)

    results = run_sweep(
        cfg=cfg,
        param_grid=param_grid,
        xi_fn=xi_american_lookback,
        warm_start=True,
    )

    csv_path = "OFFICIAL/sensitivities.csv"
    print("Writing CSV...", flush=True)
    write_sensi_csv(results, csv_path)
    print(f"Results written to {csv_path}", flush=True)


if __name__ == "__main__":
    main()
