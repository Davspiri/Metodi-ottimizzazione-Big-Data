#!/usr/bin/env python3
"""
Mars Operations: Base Deployment (MOBD) - Optimizer
Randomized Block Coordinate Descent (RBCD), Gauss-Seidel variant.

Problem
-------
Minimise f(x) = f1(x) + f2(x) + f3(x) + f4(x)
  x ∈ R^{2000}: concatenation of 2-D positions of m=1000 modules.

  f1 - network dispersion: sum_{i<j} 0.1*(1-exp(-||x^i-x^j||^2))
  f2 - distance from fixed stations: (1/m)*sum_i sum_q ||x^i-z^q||^2
  f3 - deviation from ideal operating distance:
       1000*sum_i (log(1+||x^i-s^i||^2)/log(1+r^2) - 1)^2,  r=100
  f4 - black-box environmental cost, evaluated via external simulator.

Algorithm (RBCD Gauss-Seidel with diminishing step size)
---------------------------------------------------------
  Initialisation : warm start (f3=0 by construction)
  Each epoch     : random permutation of module indices, then for each i
                     g_i = ∂f1/∂x^i + ∂f2/∂x^i + ∂f3/∂x^i + ∂f4/∂x^i
                     x^i ← x^i − α_k · g_i
                   α_k = α_0 / (1 + β·k)^γ
  f4 gradient    : forward finite differences, step H=1e-4.
                   All 2m=2000 perturbation evaluations per epoch are
                   dispatched to a thread pool (N_WORKERS threads), so the
                   total wall time per epoch is ≈ ceil(2m/N_WORKERS)*5 s
                   instead of 2m*5 s sequential.
                   "Stale-base" strategy: perturbations all use the
                   epoch-start snapshot X_snap and f4_base=f4(X_snap) from
                   the previous epoch-end evaluation, so the base is always
                   consistent with the perturbed vectors.
  Checkpointing  : after every epoch, total cost = f_known + f4 is computed;
                   if it is a new minimum, x.txt is overwritten immediately.
"""

import numpy as np
import subprocess
import os
import sys
import time
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed

# ── Problem constants ──────────────────────────────────────────────────────────
M       = 1000
R       = 100.0
H       = 1e-4          # finite-difference step for ∂f4/∂x
LOG1R2  = np.log(1.0 + R * R)

# Fixed stations z^0 … z^4.  Their sum is (0,0) → grad_f2 = (10/m)*x^i.
STATIONS = np.array([[0., 0.], [100., 100.], [-100., 100.],
                     [-100., -100.], [100., -100.]], dtype=np.float64)

# ── File paths ─────────────────────────────────────────────────────────────────
_DIR      = os.path.dirname(os.path.abspath(__file__))
SIMULATOR = os.path.join(_DIR, "linux", "mobd")
OUTPUT    = os.path.join(_DIR, "x.txt")
_SHM      = "/dev/shm"          # RAM disk for simulator I/O

N_WORKERS = max(1, os.cpu_count() or 4)   # parallel threads for f4 perturbations


# ══════════════════════════════════════════════════════════════════════════════
# Simulator interface
# ══════════════════════════════════════════════════════════════════════════════

def _write_x(x_flat: np.ndarray, path: str) -> None:
    """Write 2000 floats (one per line) to a RAM-disk path."""
    np.savetxt(path, x_flat, fmt="%.15g")


def eval_f4(x_flat: np.ndarray, path: str) -> float:
    """Invoke  mobd <path> -b  and return the float value of f4."""
    _write_x(x_flat, path)
    proc = subprocess.run([SIMULATOR, path, "-b"],
                          capture_output=True, text=True, check=True)
    return float(proc.stdout.strip())


def _perturb_eval(args: tuple) -> float:
    """Worker function: perturb x_flat at index idx by +H and eval f4."""
    x_snap, idx, worker_path = args
    p = x_snap.copy()
    p[idx] += H
    return eval_f4(p, worker_path)


def eval_f4_all_partials(
    x_snap: np.ndarray,
    f4_base: float,
) -> np.ndarray:
    """
    Compute the f4 partial-gradient estimate for every block simultaneously.

    Uses forward finite differences:
        ∂f4/∂x^i_j ≈ (f4(x + H·e_{2i+j}) - f4_base) / H,   j∈{0,1}

    All 2*M perturbation evaluations are dispatched to N_WORKERS threads so
    that wall time ≈ ceil(2M / N_WORKERS) × 5 s  instead of  2M × 5 s.

    Returns g_f4 : (m, 2) array of partial gradient estimates.
    """
    # Build task list: (x_snap, coord_index, tmp_path)
    worker_paths = [os.path.join(_SHM, f"mobd_w{k}.txt")
                    for k in range(N_WORKERS)]
    tasks = []
    for i in range(M):
        tasks.append((x_snap, 2 * i,     worker_paths[(2 * i)     % N_WORKERS]))
        tasks.append((x_snap, 2 * i + 1, worker_paths[(2 * i + 1) % N_WORKERS]))

    results = [None] * (2 * M)
    with ThreadPoolExecutor(max_workers=N_WORKERS) as pool:
        future_to_idx = {
            pool.submit(_perturb_eval, t): k
            for k, t in enumerate(tasks)
        }
        for fut in as_completed(future_to_idx):
            results[future_to_idx[fut]] = fut.result()

    g_f4 = np.empty((M, 2), dtype=np.float64)
    for i in range(M):
        g_f4[i, 0] = (results[2 * i]     - f4_base) / H
        g_f4[i, 1] = (results[2 * i + 1] - f4_base) / H
    return g_f4


# ══════════════════════════════════════════════════════════════════════════════
# Analytical cost functions
# ══════════════════════════════════════════════════════════════════════════════

def compute_f1(X: np.ndarray) -> float:
    """f1 = 0.1 * sum_{i<j} (1 - exp(-||x^i - x^j||^2))"""
    diff = X[:, None, :] - X[None, :, :]          # (m, m, 2)
    sq   = np.einsum("ijk,ijk->ij", diff, diff)    # (m, m)
    return float(0.1 * (M * (M - 1) / 2 - np.triu(np.exp(-sq), k=1).sum()))


def compute_f2(X: np.ndarray) -> float:
    """f2 = (1/m) * sum_i sum_q ||x^i - z^q||^2"""
    diff = X[:, None, :] - STATIONS[None, :, :]   # (m, 5, 2)
    return float(np.einsum("ijk,ijk->", diff, diff) / M)


def compute_f3(X: np.ndarray, S: np.ndarray) -> float:
    """f3 = 1000 * sum_i (log(1+||x^i-s^i||^2)/log(1+r^2) - 1)^2"""
    d2  = np.einsum("ij,ij->i", X - S, X - S)
    phi = np.log1p(d2) / LOG1R2 - 1.0
    return float(1000.0 * phi @ phi)


def compute_f_known(X: np.ndarray, S: np.ndarray) -> float:
    return compute_f1(X) + compute_f2(X) + compute_f3(X, S)


# ══════════════════════════════════════════════════════════════════════════════
# Analytical partial gradients  (per block i)
# ══════════════════════════════════════════════════════════════════════════════

def grad_f1_i(i: int, X: np.ndarray) -> np.ndarray:
    """∂f1/∂x^i = 0.2 * sum_{j≠i} (x^i-x^j) * exp(-||x^i-x^j||^2)"""
    d  = X[i] - X
    sq = np.einsum("ij,ij->i", d, d)
    w  = np.exp(-sq)
    w[i] = 0.0
    return 0.2 * (w[:, None] * d).sum(axis=0)


def grad_f2_i(i: int, X: np.ndarray) -> np.ndarray:
    """∂f2/∂x^i = (10/m)*x^i   (uses sum_q z^q = 0)"""
    return (10.0 / M) * X[i]


def grad_f3_i(i: int, X: np.ndarray, S: np.ndarray) -> np.ndarray:
    """∂f3/∂x^i = 4000*phi_i*(x^i-s^i) / ((1+D_i)*log(1+r^2))"""
    d   = X[i] - S[i]
    D   = float(d @ d)
    phi = np.log1p(D) / LOG1R2 - 1.0
    return (4000.0 * phi / ((1.0 + D) * LOG1R2)) * d


# ══════════════════════════════════════════════════════════════════════════════
# Warm start  (f3 = 0 exactly at initialisation)
# ══════════════════════════════════════════════════════════════════════════════

def warm_start() -> tuple:
    """
    Reference points: s^i = ((-1)^i * i * 0.2,  (-1)^i * i * 0.2), i=1…m.

    Place each module at distance r from s^i oriented toward the origin:
        x^i = s^i + r * (−s^i / ‖s^i‖)
    This satisfies ‖x^i − s^i‖ = r  ⟹  phi_i = 0  ⟹  f3 = 0.
    """
    idx  = np.arange(1, M + 1, dtype=np.float64)
    sign = np.where(idx % 2 == 0, 1.0, -1.0)
    S    = np.column_stack([sign * idx * 0.2, sign * idx * 0.2])  # (m, 2)
    nrm  = np.linalg.norm(S, axis=1, keepdims=True)               # (m, 1)
    safe = np.where(nrm < 1e-12, 1.0, nrm)
    X    = S + R * (-S / safe)
    X[nrm[:, 0] < 1e-12] = [R, 0.0]
    return X, S


# ══════════════════════════════════════════════════════════════════════════════
# RBCD optimiser
# ══════════════════════════════════════════════════════════════════════════════

def rbcd(
    n_epochs:    int   = 20,
    alpha_0:     float = 0.02,
    beta:        float = 0.05,
    gamma:       float = 0.60,
    seed:        int   = 42,
    use_f4_grad: bool  = True,
    resume:      bool  = True,
) -> np.ndarray:
    """
    Randomized Block Coordinate Descent.

    Parameters
    ----------
    n_epochs    : number of full sweeps over all m=1000 modules
    alpha_0     : initial step size
    beta, gamma : diminishing schedule  α_k = α_0 / (1 + β·k)^γ
    use_f4_grad : True  → spec-compliant; forward-FD for f4, parallelised
                  False → analytical gradient only (fast diagnostic mode,
                          beware: may worsen f4 at post-warm-start positions)
    resume      : load existing x.txt as starting point if valid
    """
    rng = np.random.default_rng(seed)

    # ── Initialise ─────────────────────────────────────────────────────────
    X, S = warm_start()
    print("Warm start computed  (f3 = 0 by construction)")

    if resume and os.path.exists(OUTPUT):
        try:
            xl = np.loadtxt(OUTPUT)
            if xl.size == 2 * M:
                X = xl.reshape(M, 2)
                print(f"Resumed from existing  {OUTPUT}")
        except Exception as exc:
            print(f"Could not load {OUTPUT}: {exc}")

    # Always save a valid warm-start checkpoint immediately
    np.savetxt(OUTPUT, X.flatten(), fmt="%.15g")

    # ── Initial evaluation ─────────────────────────────────────────────────
    t0     = time.time()
    tmp0   = os.path.join(_SHM, "mobd_init.txt")
    f4_now = eval_f4(X.flatten(), tmp0)
    fk     = compute_f_known(X, S)
    total  = fk + f4_now
    best   = total
    best_X = X.copy()
    np.savetxt(OUTPUT, best_X.flatten(), fmt="%.15g")

    mode_str = (f"FULL — forward FD for f4  [{N_WORKERS} parallel workers, "
                f"~{int(np.ceil(2*M/N_WORKERS)*5/60)} min/epoch]"
                if use_f4_grad
                else "FAST — analytical gradient only  [~0 s/epoch]")
    print(f"\n{'═'*64}")
    print(f"Mode  : {mode_str}")
    print(f"Init  : f1={compute_f1(X):.2f}  f2={compute_f2(X):.2f}"
          f"  f3={compute_f3(X,S):.2f}  f4={f4_now:.2f}")
    print(f"Total : {total:.2f}   ({time.time()-t0:.1f} s)")
    print(f"{'═'*64}\n")

    for ep in range(n_epochs):
        t_ep  = time.time()
        alpha = alpha_0 / (1.0 + beta * ep) ** gamma
        order = rng.permutation(M)

        # ── Compute f4 gradient cache for this epoch (parallelised) ────────
        if use_f4_grad:
            snap    = X.flatten()   # epoch-start snapshot (consistent base)
            f4_base = f4_now        # f4(snap) from previous epoch-end eval
            t_fd    = time.time()
            g_f4    = eval_f4_all_partials(snap, f4_base)
            print(f"  [f4 FD done in {time.time()-t_fd:.1f}s]", flush=True)
        else:
            g_f4 = np.zeros((M, 2), dtype=np.float64)

        # ── Gauss-Seidel block updates ──────────────────────────────────────
        for i in order:
            g = (grad_f1_i(i, X)
                 + grad_f2_i(i, X)
                 + grad_f3_i(i, X, S)
                 + g_f4[i])
            X[i] -= alpha * g

        # ── Epoch-end: evaluate total cost and checkpoint ───────────────────
        tmp_ep = os.path.join(_SHM, "mobd_epoch.txt")
        f4_now = eval_f4(X.flatten(), tmp_ep)
        fk     = compute_f_known(X, S)
        total  = fk + f4_now
        dt     = time.time() - t_ep

        print(f"Epoch {ep+1:3d}/{n_epochs}  total={total:.2f}"
              f"  (f1={compute_f1(X):.2f}  f2={compute_f2(X):.2f}"
              f"  f3={compute_f3(X,S):.2f}  f4={f4_now:.2f})"
              f"  α={alpha:.5f}  t={dt:.1f}s",
              flush=True)

        if total < best:
            best   = total
            best_X = X.copy()
            np.savetxt(OUTPUT, best_X.flatten(), fmt="%.15g")
            print(f"  ↳ New best! Saved to {OUTPUT}  (Δ={best-total:.2f})",
                  flush=True)

    return best_X


# ══════════════════════════════════════════════════════════════════════════════
# Entry point
# ══════════════════════════════════════════════════════════════════════════════

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="MOBD RBCD optimizer — Mars Operations Base Deployment",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument(
        "--mode",
        choices=["fast", "full", "both"],
        default="full",
        help=(
            "full : spec-compliant RBCD with f4 FD (parallelised) — RECOMMENDED | "
            "fast : analytical gradient only (0 sim calls mid-epoch, "
            "NOTE: may worsen f4 at warm-start positions) | "
            "both : fast phase first, then full phase"
        ),
    )
    p.add_argument("--fast-epochs", type=int, default=50,
                   help="Analytical-only epochs (used in mode=fast or both)")
    p.add_argument("--full-epochs", type=int, default=20,
                   help="Full RBCD epochs with f4 FD (used in mode=full or both)")
    p.add_argument("--alpha0",  type=float, default=0.01,
                   help="Initial step size α_0")
    p.add_argument("--beta",    type=float, default=0.05,
                   help="Diminishing schedule β")
    p.add_argument("--gamma",   type=float, default=0.60,
                   help="Diminishing schedule γ")
    p.add_argument("--seed",    type=int,   default=42,
                   help="Random seed")
    p.add_argument("--no-resume", action="store_true",
                   help="Ignore existing x.txt and restart from warm start")
    p.add_argument("--workers", type=int, default=N_WORKERS,
                   help="Number of parallel threads for f4 perturbations")
    return p.parse_args()


if __name__ == "__main__":
    args   = parse_args()
    resume = not args.no_resume

    # Override global worker count if specified
    import mobd_optimizer as _self
    _self.N_WORKERS = args.workers
    N_WORKERS = args.workers

    if args.mode in ("fast", "both"):
        print("╔══════════════════════════════════════════╗")
        print("║  Phase 1 — Analytical RBCD  (FAST)      ║")
        print("╚══════════════════════════════════════════╝")
        rbcd(
            n_epochs    = args.fast_epochs,
            alpha_0     = args.alpha0,
            beta        = args.beta,
            gamma       = args.gamma,
            seed        = args.seed,
            use_f4_grad = False,
            resume      = resume,
        )
        resume = True   # phase 2 always resumes from phase 1 output

    if args.mode in ("full", "both"):
        print("\n╔══════════════════════════════════════════╗")
        print("║  Phase 2 — Full RBCD with f4 FD         ║")
        print("╚══════════════════════════════════════════╝")
        rbcd(
            n_epochs    = args.full_epochs,
            alpha_0     = args.alpha0 * 0.5,   # smaller α after warm-up
            beta        = args.beta,
            gamma       = args.gamma,
            seed        = args.seed + 1,
            use_f4_grad = True,
            resume      = True,
        )

    print("\nDone.  Best solution in:", OUTPUT)
