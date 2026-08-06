from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Callable, Dict, Any, Optional, List
import copy
import csv
import math
import os
import time

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
# Checkpoint helpers
# ============================
def save_checkpoint_atomic(checkpoint_path: str, payload: Dict[str, Any]) -> None:
    import tempfile
    checkpoint_dir = os.path.dirname(checkpoint_path) or "."
    os.makedirs(checkpoint_dir, exist_ok=True)

    fd, tmp_path = tempfile.mkstemp(
        prefix="tmp_ckpt_",
        suffix=".pt",
        dir=checkpoint_dir,
    )
    os.close(fd)

    try:
        torch.save(payload, tmp_path)
        os.replace(tmp_path, checkpoint_path)
    finally:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)


def load_checkpoint_if_exists(checkpoint_path: str) -> Optional[Dict[str, Any]]:
    if os.path.exists(checkpoint_path):
        return torch.load(checkpoint_path, map_location="cpu")
    return None


# ============================
# Incremental CSV helpers
# ============================
def result_row_from_res(res: Dict[str, Any]) -> Dict[str, Any]:
    params = res.get("params", {})
    sensi = res.get("sensitivities", {})
    best = res.get("best", {})
    return {
        "K": float(params.get("K", float("nan"))),
        "r0": float(params.get("r0", float("nan"))),
        "sigma": float(params.get("sigma", float("nan"))),
        "S0": float(params.get("S0", float("nan"))),
        "T": float(params.get("T", float("nan"))),
        "a": float(params.get("a", float("nan"))),
        "r_min": float(params.get("r_min", float("nan"))),
        "r_max": float(params.get("r_max", float("nan"))),
        "r_bar": float(params.get("r_bar", float("nan"))),
        "phi_cap": float(params.get("phi_cap", float("nan"))),
        "phi_smooth_eps": float(params.get("phi_smooth_eps", float("nan"))),

        "price": float(sensi.get("price", float("nan"))),
        "price_std": float(sensi.get("price_std", float("nan"))),
        "price_se": float(sensi.get("price_se", float("nan"))),
        "price_ci_low": float(sensi.get("price_ci_low", float("nan"))),
        "price_ci_high": float(sensi.get("price_ci_high", float("nan"))),

        "n_eval_paths": int(sensi.get("n_eval_paths", -1)),

        "E_int_absZ": float(sensi.get("E_int_absZ", float("nan"))),
        "E_int_absZ_std": float(sensi.get("E_int_absZ_std", float("nan"))),
        "E_int_absZ_se": float(sensi.get("E_int_absZ_se", float("nan"))),
        "E_int_absZ_ci_low": float(sensi.get("E_int_absZ_ci_low", float("nan"))),
        "E_int_absZ_ci_high": float(sensi.get("E_int_absZ_ci_high", float("nan"))),

        "E_int_Z2": float(sensi.get("E_int_Z2", float("nan"))),
        "E_int_Z2_std": float(sensi.get("E_int_Z2_std", float("nan"))),
        "E_int_Z2_se": float(sensi.get("E_int_Z2_se", float("nan"))),
        "E_int_Z2_ci_low": float(sensi.get("E_int_Z2_ci_low", float("nan"))),
        "E_int_Z2_ci_high": float(sensi.get("E_int_Z2_ci_high", float("nan"))),

        "sqrt_E_int_Z2": float(sensi.get("sqrt_E_int_Z2", float("nan"))),

        "Vprime_infty_0": float(sensi.get("Vprime_infty_0", float("nan"))),
        "Vprime_infty_0_ci_low": float(sensi.get("Vprime_infty_0_ci_low", float("nan"))),
        "Vprime_infty_0_ci_high": float(sensi.get("Vprime_infty_0_ci_high", float("nan"))),

        "Vprime_2_0": float(sensi.get("Vprime_2_0", float("nan"))),
        "Vprime_2_0_ci_low": float(sensi.get("Vprime_2_0_ci_low", float("nan"))),
        "Vprime_2_0_ci_high": float(sensi.get("Vprime_2_0_ci_high", float("nan"))),

        "relative_Vprime_infty_0": float(sensi.get("relative_Vprime_infty_0", float("nan"))),
        "relative_Vprime_infty_0_ci_low": float(sensi.get("relative_Vprime_infty_0_ci_low", float("nan"))),
        "relative_Vprime_infty_0_ci_high": float(sensi.get("relative_Vprime_infty_0_ci_high", float("nan"))),

        "relative_Vprime_2_0": float(sensi.get("relative_Vprime_2_0", float("nan"))),
        "relative_Vprime_2_0_ci_low": float(sensi.get("relative_Vprime_2_0_ci_low", float("nan"))),
        "relative_Vprime_2_0_ci_high": float(sensi.get("relative_Vprime_2_0_ci_high", float("nan"))),

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



# ============================
# Config (training / solver only)
# ============================
@dataclass
class BSDEControlConfigTest:
    DEVICE: str = "cuda"
    DTYPE: torch.dtype = torch.float32
    SEED: int = 1234

    N_TIME_STEPS: int = 50
    N_BATCH: int = 2**10

    LR: float = 1e-3
    MAX_STEPS: int = 2_000
    GRAD_CLIP: float = 1.0
    ZNET_CHUNK_SIZE: int = 2**16
    EPSILON_STOPPING: float = 1e-2
    PATIENCE: int = 10
    CHECK_EVERY: int = 50
    MIN_DELTA: float = 1e-6

    HIDDEN_DIM: int = 128
    N_HIDDEN_LAYERS: int = 3
    ACTIVATION: str = "silu"
    USE_LAYER_NORM: bool = False
    LEARNABLE_Y0: bool = True
    USE_BF16_WARMUP: bool = True
    BF16_TO_FP32_LOSS_SWITCH: float = 1e-1
    BF16_WARMUP_MIN_STEPS: int = 0


@dataclass
class BSDEControlConfigRun:
    DEVICE: str = "cuda"
    DTYPE: torch.dtype = torch.float32
    SEED: int = 1234
    ZNET_CHUNK_SIZE: int = 2**16
    N_TIME_STEPS: int = 100
    N_BATCH: int = 2**17

    LR: float = 5e-3
    MAX_STEPS: int = 20_000
    GRAD_CLIP: float = 1.0
    MIN_DELTA: float = 1e-6

    EPSILON_STOPPING: float = 5e-3
    PATIENCE: int = 20
    CHECK_EVERY: int = 100

    HIDDEN_DIM: int = 192
    N_HIDDEN_LAYERS: int = 4
    ACTIVATION: str = "silu"
    USE_LAYER_NORM: bool = False
    LEARNABLE_Y0: bool = True
    USE_BF16_WARMUP: bool = True
    BF16_TO_FP32_LOSS_SWITCH: float = 1e-1
    BF16_WARMUP_MIN_STEPS: int = 0


def get_bsde_control_config() -> BSDEControlConfigTest | BSDEControlConfigRun:
    if ETAT == "TEST":
        cfg = BSDEControlConfigTest()
    elif ETAT == "RUN":
        cfg = BSDEControlConfigRun()
    else:
        raise ValueError(f"Unknown ETAT = {ETAT}")

    print("======================================", flush=True)
    print(f"BSDE CONTROL CONFIG MODE : {ETAT}", flush=True)
    print("======================================", flush=True)
    print(asdict(cfg), flush=True)
    return cfg


def set_seed(seed: int) -> None:
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def get_compute_device_name(device: str) -> str:
    if device.startswith("cuda") and torch.cuda.is_available():
        if ":" in device:
            device_index = int(device.split(":", 1)[1])
        else:
            device_index = torch.cuda.current_device()
        return torch.cuda.get_device_name(device_index)
    return "CPU"


def get_activation(name: str) -> nn.Module:
    name = name.lower()
    if name == "relu":
        return nn.ReLU()
    if name == "tanh":
        return nn.Tanh()
    if name == "gelu":
        return nn.GELU()
    if name in ("silu", "swish"):
        return nn.SiLU()
    raise ValueError(f"Unknown activation: {name}")


# ============================
# MLP
# ============================
class MLP(nn.Module):
    def __init__(self, in_dim: int, out_dim: int, cfg: BSDEControlConfigTest | BSDEControlConfigRun):
        super().__init__()
        act = get_activation(cfg.ACTIVATION)
        layers: List[nn.Module] = []
        d = in_dim
        for _ in range(cfg.N_HIDDEN_LAYERS):
            layers.append(nn.Linear(d, cfg.HIDDEN_DIM))
            if cfg.USE_LAYER_NORM:
                layers.append(nn.LayerNorm(cfg.HIDDEN_DIM))
            layers.append(act)
            d = cfg.HIDDEN_DIM
        layers.append(nn.Linear(d, out_dim))
        self.net = nn.Sequential(*layers)
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class LearnableY0(nn.Module):
    def __init__(self, init_value: float = 0.0):
        super().__init__()
        self.y0 = nn.Parameter(torch.tensor([init_value], dtype=torch.float32))

    def forward(self, batch_size: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        return self.y0.to(device=device, dtype=dtype).view(1, 1).repeat(batch_size, 1)


# ============================
# Paths: GBM + Running Max
# ============================
@dataclass
class PathBundle:
    t_grid: torch.Tensor
    dW: torch.Tensor
    W: torch.Tensor
    S: torch.Tensor
    M: torch.Tensor


class GBMRunningMaxSimulator:
    def __init__(self, cfg: BSDEControlConfigTest | BSDEControlConfigRun):
        self.cfg = cfg

    @torch.no_grad()
    def sample(self, params: Dict[str, Any]) -> PathBundle:
        cfg = self.cfg
        device, dtype = cfg.DEVICE, cfg.DTYPE

        batch_size = cfg.N_BATCH
        n_time_steps = cfg.N_TIME_STEPS
        maturity = float(params["T"])
        dt = maturity / n_time_steps

        r0 = float(params["r0"])
        sigma = float(params["sigma"])
        S0 = float(params["S0"])

        t_grid = torch.linspace(0.0, maturity, n_time_steps + 1, device=device, dtype=dtype)
        dW = torch.randn(batch_size, n_time_steps, 1, device=device, dtype=dtype) * math.sqrt(dt)
        W = torch.zeros(batch_size, n_time_steps + 1, 1, device=device, dtype=dtype)
        W[:, 1:, :] = torch.cumsum(dW, dim=1)

        t = t_grid.view(1, n_time_steps + 1, 1)
        logS = math.log(S0) + (r0 - 0.5 * sigma * sigma) * t + sigma * W
        S = torch.exp(logS)
        M, _ = torch.cummax(S, dim=1)
        return PathBundle(t_grid=t_grid, dW=dW, W=W, S=S, M=M)


# ============================
# Payoff: bounded smooth approx of (M_T - K)^+
# ============================
def bounded_positive_part(x: torch.Tensor, cap: float, eps: float) -> torch.Tensor:
    x_pos = torch.nn.functional.softplus(x / eps) * eps
    return cap * torch.tanh(x_pos / cap)


XiTerminalFn = Callable[[Dict[str, torch.Tensor], Dict[str, Any]], torch.Tensor]


def xi_terminal_lookback(path_T: Dict[str, torch.Tensor], params: Dict[str, Any]) -> torch.Tensor:
    strike = float(params["K"])
    cap = float(params["phi_cap"])
    eps = float(params["phi_smooth_eps"])
    M_T = path_T["M"]
    return bounded_positive_part(M_T - strike, cap=cap, eps=eps)


# ============================
# Hamiltonian
# ============================
def alpha_star(y: torch.Tensor, z: torch.Tensor, params: Dict[str, Any]) -> torch.Tensor:
    a = float(params["a"])
    r_min = float(params["r_min"])
    r_max = float(params["r_max"])
    r_bar = float(params["r_bar"])

    theta = y + z
    r_unconstr = r_bar + theta / (2.0 * a)
    return torch.clamp(r_unconstr, min=r_min, max=r_max)


def generator_f(y: torch.Tensor, z: torch.Tensor, params: Dict[str, Any]) -> torch.Tensor:
    a = float(params["a"])
    r_bar = float(params["r_bar"])
    r = alpha_star(y, z, params)
    return a * (r - r_bar) ** 2 - r * (y + z)


def k_star(y: torch.Tensor, z: torch.Tensor, params: Dict[str, Any]) -> torch.Tensor:
    return alpha_star(y, z, params)


# ============================
# Deep BSDE model
# ============================
class DeepBSDEControl(nn.Module):
    def __init__(self, cfg: BSDEControlConfigTest | BSDEControlConfigRun, state_dim: int):
        super().__init__()
        self.cfg = cfg
        self.z_net = MLP(in_dim=state_dim, out_dim=1, cfg=cfg)
        self.y0 = LearnableY0(0.0) if cfg.LEARNABLE_Y0 else None

    def init_y0(self, batch_size: int) -> torch.Tensor:
        device = torch.device(self.cfg.DEVICE)
        dtype = self.cfg.DTYPE
        if self.y0 is None:
            return torch.zeros(batch_size, 1, device=device, dtype=dtype)
        return self.y0(batch_size, device, dtype)


@torch.no_grad()
def grad_norm(model: nn.Module) -> float:
    total = 0.0
    for p in model.parameters():
        if p.grad is None:
            continue
        g = p.grad.detach()
        total += float(torch.sum(g * g).item())
    return math.sqrt(total)


# ============================
# Parameter grid sweep
# ============================
def iter_param_grid(param_grid: Dict[str, List[Any]]) -> List[Dict[str, Any]]:
    keys = list(param_grid.keys())
    grids = [param_grid[k] for k in keys]
    out: List[Dict[str, Any]] = []

    def rec(i: int, cur: Dict[str, Any]) -> None:
        if i == len(keys):
            out.append(dict(cur))
            return
        key = keys[i]
        for value in grids[i]:
            cur[key] = value
            rec(i + 1, cur)

    rec(0, {})
    return out


# ============================
# Training one setting
# ============================


def train_one_setting(
    cfg,
    params: Dict[str, Any],
    xi_terminal_fn,
    init_state_dict: Optional[Dict[str, torch.Tensor]] = None,
    lr_override: Optional[float] = None,
    checkpoint_path: Optional[str] = None,
    ):
    import time

    PROFILE = (ETAT == "TEST")
    FAST = (ETAT == "RUN")

    def cuda_sync():
        if PROFILE and cfg.DEVICE.startswith("cuda") and torch.cuda.is_available():
            torch.cuda.synchronize()

    set_seed(cfg.SEED)
    device = torch.device(cfg.DEVICE)

    simulator = GBMRunningMaxSimulator(cfg)
    state_dim = 3
    model = DeepBSDEControl(cfg, state_dim=state_dim).to(device)

    if init_state_dict is not None:
        model.load_state_dict(init_state_dict)

    lr = lr_override if lr_override is not None else cfg.LR
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    best = {"loss": float("inf"), "step": -1}
    no_improve = 0
    start_step = 0

    use_bf16_phase = bool(
        getattr(cfg, "USE_BF16_WARMUP", True)
        and cfg.DEVICE.startswith("cuda")
        and torch.cuda.is_available()
    )
    bf16_switch_loss = float(getattr(cfg, "BF16_TO_FP32_LOSS_SWITCH", 1e-1))
    bf16_min_steps = int(getattr(cfg, "BF16_WARMUP_MIN_STEPS", 0))
    switched_to_fp32 = False

    # -------- resume from checkpoint if present --------
    if checkpoint_path is not None:
        ckpt = load_checkpoint_if_exists(checkpoint_path)
        if ckpt is not None:
            if ckpt.get("finished", False):
                print(f"[train_one_setting] ALREADY FINISHED | K={params['K']}", flush=True)
                model.load_state_dict(ckpt["model_state_dict"])
                sensitivities = evaluate_sensitivities(cfg, model, simulator, params)
                return {
                    "params": params,
                    "best": ckpt.get("best", {"loss": float("inf"), "step": -1}),
                    "final_step": ckpt.get("step", -1),
                    "config": asdict(cfg),
                    "sensitivities": sensitivities,
                    "compute_device": get_compute_device_name(cfg.DEVICE),
                    "model_state_dict": {
                        k: v.detach().cpu().clone() for k, v in model.state_dict().items()
                    },
                }
            model.load_state_dict(ckpt["model_state_dict"])
            optimizer.load_state_dict(ckpt["optimizer_state_dict"])
            best = ckpt.get("best", best)
            no_improve = ckpt.get("no_improve", 0)
            use_bf16_phase = ckpt.get("use_bf16_phase", use_bf16_phase)
            switched_to_fp32 = ckpt.get("switched_to_fp32", False)
            start_step = int(ckpt.get("step", -1)) + 1

            if "cpu_rng_state" in ckpt:
                torch.random.set_rng_state(ckpt["cpu_rng_state"])
            if torch.cuda.is_available() and "cuda_rng_state_all" in ckpt and ckpt["cuda_rng_state_all"] is not None:
                torch.cuda.set_rng_state_all(ckpt["cuda_rng_state_all"])

            print(
                f"[train_one_setting] RESUME | K={params['K']} | from_step={start_step} | "
                f"precision={'bf16' if use_bf16_phase else 'fp32'}",
                flush=True,
            )

    print(
        f"[train] MODE={ETAT} | K={params['K']} | LR={lr:.2e} | "
        f"N_BATCH={cfg.N_BATCH} | N_TIME_STEPS={cfg.N_TIME_STEPS} | MAX_STEPS={cfg.MAX_STEPS} | "
        f"precision={'bf16' if use_bf16_phase else 'fp32'}",
        flush=True,
    )

    B = cfg.N_BATCH
    N = cfg.N_TIME_STEPS
    T = float(params["T"])
    dt = T / N

    t0 = time.time()
    block_path_time = 0.0
    block_train_time = 0.0
    block_steps = 0
    save_every = int(getattr(cfg, "SAVE_EVERY", cfg.CHECK_EVERY))

    for step in range(start_step, cfg.MAX_STEPS):
        # ======================
        # PATHS
        # ======================
        cuda_sync()
        t_path = time.time()
        paths = simulator.sample(params)
        cuda_sync()
        block_path_time += time.time() - t_path

        # ======================
        # TRAIN STEP
        # ======================
        cuda_sync()
        t_step = time.time()
        optimizer.zero_grad(set_to_none=True)

        S = paths.S
        M = paths.M
        dW = paths.dW
        t_grid = paths.t_grid

        Y = model.init_y0(B)

        if FAST:
            t_feat = t_grid[:-1].view(1, N, 1).expand(B, N, 1)
            logS = torch.log(S[:, :-1, :] + 1e-12)
            M_path = M[:, :-1, :]
            state_all = torch.cat([t_feat, logS, M_path], dim=2)

            if use_bf16_phase:
                with torch.amp.autocast("cuda", enabled=True, dtype=torch.bfloat16):
                    flat_state = state_all.reshape(B * N, 3)

                    Z_all = forward_in_chunks(
                        model.z_net,
                        flat_state,
                        chunk_size=cfg.ZNET_CHUNK_SIZE,
                    ).reshape(B, N, 1)
                Z_all = Z_all.float()
            else:
                flat_state = state_all.reshape(B * N, 3)

                Z_all = forward_in_chunks(
                    model.z_net,
                    flat_state,
                    chunk_size=cfg.ZNET_CHUNK_SIZE,
                ).reshape(B, N, 1)

            for i in range(N):
                Z = Z_all[:, i, :]
                Y = Y - generator_f(Y, Z, params) * dt + Z * dW[:, i, :]
        else:
            for i in range(N):
                t_i = t_grid[i].view(1, 1).expand(B, 1)
                logS = torch.log(S[:, i, :] + 1e-12)
                M_t = M[:, i, :]
                state = torch.cat([t_i, logS, M_t], dim=1)

                if use_bf16_phase:
                    with torch.amp.autocast("cuda", enabled=True, dtype=torch.bfloat16):
                        Z = model.z_net(state)
                    Z = Z.float()
                else:
                    Z = model.z_net(state)

                Y = Y - generator_f(Y, Z, params) * dt + Z * dW[:, i, :]

        xi = xi_terminal_fn(
            {"S": S[:, -1, :], "M": M[:, -1, :], "t": t_grid[-1]},
            params,
        )

        loss = torch.mean((Y - xi) ** 2)
        loss.backward()

        if cfg.GRAD_CLIP and cfg.GRAD_CLIP > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.GRAD_CLIP)

        optimizer.step()

        cuda_sync()
        block_train_time += time.time() - t_step
        block_steps += 1

        # periodic checkpoint
        if checkpoint_path is not None and ((step + 1) % save_every == 0):
            payload = {
                "step": step,
                "params": dict(params),
                "best": best,
                "no_improve": no_improve,
                "use_bf16_phase": use_bf16_phase,
                "switched_to_fp32": switched_to_fp32,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "cpu_rng_state": torch.random.get_rng_state(),
                "cuda_rng_state_all": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
            }
            save_checkpoint_atomic(checkpoint_path, payload)

        # ======================
        # EARLY STOPPING
        # ======================
        if step % cfg.CHECK_EVERY == 0 and step > 0:
            current_loss = float(loss.item())

            just_switched_to_fp32 = False
            if use_bf16_phase and step >= bf16_min_steps and current_loss < bf16_switch_loss:
                use_bf16_phase = False
                switched_to_fp32 = True
                just_switched_to_fp32 = True
                print(
                    f"[train_one_setting] SWITCH PRECISION | "
                    f"K={params['K']} | step={step} | loss={current_loss:.6e} | "
                    f"new_precision=fp32",
                    flush=True,
                )

            if current_loss < best["loss"] - cfg.MIN_DELTA:
                best = {"loss": current_loss, "step": step}
                no_improve = 0
            else:
                no_improve += 1

            elapsed = time.time() - t0
            precision_label = "bf16" if use_bf16_phase else "fp32"

            if PROFILE:
                avg_path_time = block_path_time / max(block_steps, 1)
                avg_train_time = block_train_time / max(block_steps, 1)
                avg_total_step_time = (block_path_time + block_train_time) / max(block_steps, 1)
                print(
                    f"[train_one_setting] MODE=TEST | "
                    f"K={params['K']} | "
                    f"step={step}/{cfg.MAX_STEPS} | "
                    f"loss={current_loss:.6e} | "
                    f"best_loss={best['loss']:.6e} | "
                    f"no_improve={no_improve}/{cfg.PATIENCE} | "
                    f"lr={optimizer.param_groups[0]['lr']:.2e} | "
                    f"precision={precision_label} | "
                    f"avg_path_gen={avg_path_time:.4f}s | "
                    f"avg_train_step={avg_train_time:.4f}s | "
                    f"avg_total_step={avg_total_step_time:.4f}s | "
                    f"elapsed={elapsed:.2f}s",
                    flush=True,
                )
            else:
                print(
                    f"[train_one_setting] MODE=RUN | "
                    f"K={params['K']} | "
                    f"step={step}/{cfg.MAX_STEPS} | "
                    f"loss={current_loss:.6e} | "
                    f"best_loss={best['loss']:.6e} | "
                    f"no_improve={no_improve}/{cfg.PATIENCE} | "
                    f"lr={optimizer.param_groups[0]['lr']:.2e} | "
                    f"precision={precision_label} | "
                    f"elapsed={elapsed:.2f}s",
                    flush=True,
                )

            block_path_time = 0.0
            block_train_time = 0.0
            block_steps = 0

            can_early_stop = (not use_bf16_phase) and (not just_switched_to_fp32)

            if can_early_stop and current_loss < cfg.EPSILON_STOPPING:
                print(f"[STOP loss_tol] K={params['K']} step={step}", flush=True)
                break

            if can_early_stop and no_improve >= cfg.PATIENCE:
                print(f"[STOP plateau] K={params['K']} step={step}", flush=True)
                break

    # final checkpoint
    if checkpoint_path is not None:
        payload = {
            "step": step,
            "params": dict(params),
            "best": best,
            "no_improve": no_improve,
            "use_bf16_phase": use_bf16_phase,
            "switched_to_fp32": switched_to_fp32,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "cpu_rng_state": torch.random.get_rng_state(),
            "cuda_rng_state_all": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
            "finished": True,
        }
        save_checkpoint_atomic(checkpoint_path, payload)

    print(
        f"[train END] K={params['K']} | total={time.time()-t0:.1f}s | "
        f"final_precision={'bf16' if use_bf16_phase else 'fp32'}",
        flush=True,
    )

    sensitivities = evaluate_sensitivities(cfg, model, simulator, params)

    return {
        "params": params,
        "best": best,
        "final_step": step,
        "config": asdict(cfg),
        "sensitivities": sensitivities,
        "compute_device": get_compute_device_name(cfg.DEVICE),
        "model_state_dict": {
            k: v.detach().cpu().clone() for k, v in model.state_dict().items()
        },
    }
# ============================
# Sensitivity evaluation under P^{alpha*}
# ============================
@torch.no_grad()
def evaluate_sensitivities(
    cfg: BSDEControlConfigTest | BSDEControlConfigRun,
    model: DeepBSDEControl,
    simulator: GBMRunningMaxSimulator,
    params: Dict[str, Any],
    n_eval_batches: int = 4,
    ) -> Dict[str, float]:
    import math

    device = torch.device(cfg.DEVICE)
    model.eval()

    n_time_steps = cfg.N_TIME_STEPS
    samples_int_absZ: List[torch.Tensor] = []
    samples_int_Z2: List[torch.Tensor] = []

    for _ in range(n_eval_batches):
        paths = simulator.sample(params)
        batch_size = paths.S.shape[0]
        maturity = float(params["T"])
        dt = maturity / n_time_steps

        S, M, dW, t_grid = paths.S, paths.M, paths.dW, paths.t_grid
        Y = model.init_y0(batch_size)

        A = torch.zeros(batch_size, 1, device=device, dtype=cfg.DTYPE)
        int_absZ = torch.zeros(batch_size, 1, device=device, dtype=cfg.DTYPE)
        int_Z2 = torch.zeros(batch_size, 1, device=device, dtype=cfg.DTYPE)

        for i in range(n_time_steps):
            t_i = t_grid[i].view(1, 1).expand(batch_size, 1)
            logS_t = torch.log(S[:, i, :] + 1e-12)
            M_t = M[:, i, :]
            state = torch.cat([t_i, logS_t, M_t], dim=1)

            Z = model.z_net(state)
            r_star = alpha_star(Y, Z, params)

            A = A + r_star * dt
            Kdisc = torch.exp(-A)

            int_absZ = int_absZ + Kdisc * torch.abs(Z) * dt
            int_Z2 = int_Z2 + Kdisc * (Z * Z) * dt

            f_val = generator_f(Y, Z, params)
            Y = Y - f_val * dt + Z * dW[:, i, :]

        samples_int_absZ.append(int_absZ.squeeze(-1).detach().cpu())
        samples_int_Z2.append(int_Z2.squeeze(-1).detach().cpu())

    x_absZ = torch.cat(samples_int_absZ, dim=0)
    x_Z2 = torch.cat(samples_int_Z2, dim=0)
    n_samples = x_absZ.numel()

    E_int_absZ = float(x_absZ.mean().item())
    E_int_Z2 = float(x_Z2.mean().item())

    if n_samples > 1:
        std_int_absZ = float(x_absZ.std(unbiased=True).item())
        std_int_Z2 = float(x_Z2.std(unbiased=True).item())
    else:
        std_int_absZ = 0.0
        std_int_Z2 = 0.0

    se_int_absZ = std_int_absZ / math.sqrt(max(n_samples, 1))
    se_int_Z2 = std_int_Z2 / math.sqrt(max(n_samples, 1))

    z_975 = 1.96
    E_int_absZ_ci_low = E_int_absZ - z_975 * se_int_absZ
    E_int_absZ_ci_high = E_int_absZ + z_975 * se_int_absZ
    E_int_Z2_ci_low = E_int_Z2 - z_975 * se_int_Z2
    E_int_Z2_ci_high = E_int_Z2 + z_975 * se_int_Z2

    price = float(model.init_y0(1).squeeze().detach().cpu().item())
    price_std = 0.0
    price_se = 0.0
    price_ci_low = price
    price_ci_high = price

    Vprime_infty_0 = E_int_absZ
    Vprime_infty_0_ci_low = E_int_absZ_ci_low
    Vprime_infty_0_ci_high = E_int_absZ_ci_high

    E_int_Z2_ci_low_clipped = max(E_int_Z2_ci_low, 0.0)
    E_int_Z2_ci_high_clipped = max(E_int_Z2_ci_high, 0.0)
    Vprime_2_0 = float(math.sqrt(max(E_int_Z2, 0.0)))
    Vprime_2_0_ci_low = float(math.sqrt(E_int_Z2_ci_low_clipped))
    Vprime_2_0_ci_high = float(math.sqrt(E_int_Z2_ci_high_clipped))

    rel_infty = _relative_metric_with_ci(
        Vprime_infty_0,
        Vprime_infty_0_ci_low,
        Vprime_infty_0_ci_high,
        price,
    )
    rel_l2 = _relative_metric_with_ci(
        Vprime_2_0,
        Vprime_2_0_ci_low,
        Vprime_2_0_ci_high,
        price,
    )

    return {
        "price": price,
        "price_std": price_std,
        "price_se": price_se,
        "price_ci_low": price_ci_low,
        "price_ci_high": price_ci_high,
        "n_eval_paths": int(n_samples),
        "E_int_absZ": E_int_absZ,
        "E_int_absZ_std": std_int_absZ,
        "E_int_absZ_se": se_int_absZ,
        "E_int_absZ_ci_low": E_int_absZ_ci_low,
        "E_int_absZ_ci_high": E_int_absZ_ci_high,
        "E_int_Z2": E_int_Z2,
        "E_int_Z2_std": std_int_Z2,
        "E_int_Z2_se": se_int_Z2,
        "E_int_Z2_ci_low": E_int_Z2_ci_low,
        "E_int_Z2_ci_high": E_int_Z2_ci_high,
        "sqrt_E_int_Z2": float(math.sqrt(max(E_int_Z2, 0.0))),
        "Vprime_infty_0": Vprime_infty_0,
        "Vprime_infty_0_ci_low": Vprime_infty_0_ci_low,
        "Vprime_infty_0_ci_high": Vprime_infty_0_ci_high,
        "Vprime_2_0": Vprime_2_0,
        "Vprime_2_0_ci_low": Vprime_2_0_ci_low,
        "Vprime_2_0_ci_high": Vprime_2_0_ci_high,
        "relative_Vprime_infty_0": rel_infty["value"],
        "relative_Vprime_infty_0_ci_low": rel_infty["ci_low"],
        "relative_Vprime_infty_0_ci_high": rel_infty["ci_high"],
        "relative_Vprime_2_0": rel_l2["value"],
        "relative_Vprime_2_0_ci_low": rel_l2["ci_low"],
        "relative_Vprime_2_0_ci_high": rel_l2["ci_high"],
    }


# ============================
# Sweep runner
# ============================
def run_sweep(
    cfg: BSDEControlConfigTest | BSDEControlConfigRun,
    param_grid: Dict[str, List[Any]],
    xi_terminal_fn: XiTerminalFn = xi_terminal_lookback,
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

    print(f"Starting sweep over {total_params} parameter values.", flush=True)
    print(f"Live progress written to: {status_file}", flush=True)

    for idx, p in enumerate(grid_list, start=1):
        params = dict(p)
        t0_param = time.time()

        checkpoint_dir = "OFFICIAL/checkpoints"
        os.makedirs(checkpoint_dir, exist_ok=True)
        checkpoint_path = os.path.join(checkpoint_dir, f"control_K_{float(params['K']):.10f}.pt")

        current_lr = cfg.LR
        if warm_start and previous_state_dict is not None:
            current_lr = 0.3 * cfg.LR

        existing_ckpt = load_checkpoint_if_exists(checkpoint_path)
        if existing_ckpt is not None and existing_ckpt.get("finished", False):
            print(f"[run_sweep] Reusing finished checkpoint for K={params['K']}", flush=True)

        res = train_one_setting(
            cfg,
            params=params,
            xi_terminal_fn=xi_terminal_fn,
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

        strike_value = params.get("K", "NA")
        status_line = (
            f"[{idx:03d}/{total_params:03d}] "
            f"K={strike_value} | "
            f"last={format_seconds(elapsed_param)} | "
            f"avg={format_seconds(avg_time_per_param)} | "
            f"elapsed={format_seconds(elapsed_global)} | "
            f"ETA={format_seconds(eta_seconds)}"
        )

        with open(status_file, "w") as f:
            f.write(status_line + "\n")

    total_elapsed = time.time() - t0_global
    with open(status_file, "w") as f:
        f.write(f"Finished in {format_seconds(total_elapsed)}\n")

    print("Sweep finished.", flush=True)
    return results


# ============================
# CSV export helper
# ============================
def write_sensi_csv(results: List[Dict[str, Any]], csv_path: str) -> None:
    rows = [result_row_from_res(res) for res in results]
    rows.sort(key=lambda d: (d.get("r0", 0.0), d.get("sigma", 0.0), d.get("K", 0.0)))
    os.makedirs(os.path.dirname(csv_path) or ".", exist_ok=True)
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()) if rows else [])
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    cfg = get_bsde_control_config()
    compute_device = get_compute_device_name(cfg.DEVICE)


    if ETAT == "TEST":
        strike_grid = [round(9.5 + 0.1 * i, 10) for i in range(10)]
    elif ETAT == "RUN":
        strike_grid = [round(9.5 + 0.02 * i, 10) for i in range(500)]
    else:
        raise ValueError(f"Unknown ETAT = {ETAT}")

    param_grid = {
        "T": [0.5],
        "r0": [0.02],
        "sigma": [0.2],
        "S0": [10.0],
        "K": strike_grid,
        "a": [1.0],
        "r_min": [0.02],
        "r_max": [0.08],
        "r_bar": [0.04],
        "phi_cap": [1.0],
        "phi_smooth_eps": [1e-3],
    }

    print(f"Compute device used: {compute_device}", flush=True)
    print(f"Parameter grid keys: {list(param_grid.keys())}", flush=True)

    csv_path = "OFFICIAL/control_sensitivities.csv"
    results = run_sweep(
        cfg=cfg,
        param_grid=param_grid,
        xi_terminal_fn=xi_terminal_lookback,
        warm_start=True,
        status_file="OFFICIAL/progress.txt",
        incremental_csv_path=csv_path,
    )

    print(f"Incremental results available in {csv_path}", flush=True)


if __name__ == "__main__":
    main()
