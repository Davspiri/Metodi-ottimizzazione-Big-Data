#!/usr/bin/env python3
"""
Ottimizzatore per il progetto MOBD (Mars Operations: Base Deployment).

Vogliamo minimizzare  f(x) = f1(x) + f2(x) + f3(x) + f4(x),  con x in R^2000
(le posizioni 2D dei m=1000 moduli messe in fila una dopo l'altra).

  f1 - dispersione della rete:  sum_{i<j} 0.1*(1 - exp(-||x^i - x^j||^2))
  f2 - distanza dalle stazioni fisse:  (1/m)*sum_i sum_q ||x^i - z^q||^2
  f3 - deviazione dalla distanza operativa ideale:
       1000*sum_i (log(1+||x^i - s^i||^2)/log(1+r^2) - 1)^2,  r=100
  f4 - costo ambientale "black-box": non ha formula, lo valutiamo col
       simulatore esterno fornito col progetto.

Per l'ottimizzazione usiamo il metodo RBCD (Randomized Block Coordinate
Descent) nella variante Gauss-Seidel con passo decrescente:
  - all'inizio partiamo da un warm start scelto in modo che f3 sia gia' 0;
  - ad ogni epoca permutiamo a caso i moduli e, per ogni modulo i,
    aggiorniamo solo il suo blocco:
        g_i = grad f1 + grad f2 + grad f3 + grad f4   (rispetto a x^i)
        x^i = x^i - alpha_k * g_i
    con passo alpha_k = alpha_0 / (1 + beta*k)^gamma;
  - il gradiente di f4 non lo conosciamo, quindi lo stimiamo con le
    differenze finite in avanti (passo H=1e-4). Siccome ogni chiamata al
    simulatore costa diversi secondi, lanciamo le 2000 valutazioni di
    un'epoca in parallelo su piu' thread. Come punto base usiamo sempre la
    x di inizio epoca, cosi' resta coerente con tutte le perturbazioni;
  - alla fine di ogni epoca calcoliamo il costo totale e, se e' il migliore
    trovato finora, riscriviamo subito x.txt.
"""

import numpy as np
import subprocess
import os
import sys
import time
import argparse
import tempfile
import platform
from concurrent.futures import ThreadPoolExecutor, as_completed

# Costanti del problema
M       = 1000
R       = 100.0
H       = 1e-4          # passo delle differenze finite per il gradiente di f4
LOG1R2  = np.log(1.0 + R * R)

# Stazioni fisse z^0..z^4. La loro somma fa (0,0), quindi grad f2 = (10/m)*x^i.
STATIONS = np.array([[0., 0.], [100., 100.], [-100., 100.],
                     [-100., -100.], [100., -100.]], dtype=np.float64)

# Percorsi dei file
_DIR = os.path.dirname(os.path.abspath(__file__))

_PLATFORM = platform.system()
if _PLATFORM == "Windows":
    SIMULATOR = os.path.join(_DIR, "win", "mobd.exe")
elif _PLATFORM == "Darwin":
    SIMULATOR = os.path.join(_DIR, "macos", "mobd")
else:
    SIMULATOR = os.path.join(_DIR, "linux", "mobd")

OUTPUT = os.path.join(_DIR, "x.txt")
_SHM   = "/dev/shm" if _PLATFORM == "Linux" else tempfile.gettempdir()

N_WORKERS = max(1, os.cpu_count() or 4)   # thread paralleli per le perturbazioni di f4


# ---------------------------------------------------------------------------
# Interfaccia con il simulatore
# ---------------------------------------------------------------------------

def _write_x(x_flat: np.ndarray, path: str) -> None:
    """Scrive i 2000 valori (uno per riga) nel file che legge il simulatore."""
    np.savetxt(path, x_flat, fmt="%.15g")


def eval_f4(x_flat: np.ndarray, path: str) -> float:
    """Lancia  mobd <path> -b  e restituisce il valore di f4."""
    _write_x(x_flat, path)
    proc = subprocess.run([SIMULATOR, path, "-b"],
                          capture_output=True, text=True, check=True)
    return float(proc.stdout.strip())


def _perturb_eval(args: tuple) -> float:
    """Perturba x_flat nella coordinata idx di +H e valuta f4 (usata dai thread)."""
    x_snap, idx, worker_path = args
    p = x_snap.copy()
    p[idx] += H
    return eval_f4(p, worker_path)


def eval_f4_all_partials(
    x_snap: np.ndarray,
    f4_base: float,
) -> np.ndarray:
    """
    Stima il gradiente di f4 per tutti i moduli, con le differenze finite
    in avanti:
        df4/dx^i_j ~ (f4(x + H*e_{2i+j}) - f4_base) / H,   j in {0,1}

    Le 2*M perturbazioni vengono lanciate su N_WORKERS thread in parallelo,
    cosi' non aspettiamo una chiamata al simulatore alla volta.

    Ritorna g_f4: array (m, 2) con le stime dei gradienti.
    """
    # lista dei task: (x_snap, indice della coordinata, file temporaneo)
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


# ---------------------------------------------------------------------------
# Funzioni di costo (calcolate analiticamente)
# ---------------------------------------------------------------------------

def compute_f1(X: np.ndarray) -> float:
    """f1 = 0.1 * sum_{i<j} (1 - exp(-||x^i - x^j||^2))."""
    diff = X[:, None, :] - X[None, :, :]          # (m, m, 2)
    sq   = np.einsum("ijk,ijk->ij", diff, diff)    # (m, m)
    return float(0.1 * (M * (M - 1) / 2 - np.triu(np.exp(-sq), k=1).sum()))


def compute_f2(X: np.ndarray) -> float:
    """f2 = (1/m) * sum_i sum_q ||x^i - z^q||^2."""
    diff = X[:, None, :] - STATIONS[None, :, :]   # (m, 5, 2)
    return float(np.einsum("ijk,ijk->", diff, diff) / M)


def compute_f3(X: np.ndarray, S: np.ndarray) -> float:
    """f3 = 1000 * sum_i (log(1+||x^i-s^i||^2)/log(1+r^2) - 1)^2."""
    d2  = np.einsum("ij,ij->i", X - S, X - S)
    phi = np.log1p(d2) / LOG1R2 - 1.0
    return float(1000.0 * phi @ phi)


def compute_f_known(X: np.ndarray, S: np.ndarray) -> float:
    return compute_f1(X) + compute_f2(X) + compute_f3(X, S)


# ---------------------------------------------------------------------------
# Gradienti parziali analitici (rispetto al blocco i)
# ---------------------------------------------------------------------------

def grad_f1_i(i: int, X: np.ndarray) -> np.ndarray:
    """Gradiente di f1 su x^i = 0.2 * sum_{j!=i} (x^i-x^j) * exp(-||x^i-x^j||^2)."""
    d  = X[i] - X
    sq = np.einsum("ij,ij->i", d, d)
    w  = np.exp(-sq)
    w[i] = 0.0
    return 0.2 * (w[:, None] * d).sum(axis=0)


def grad_f2_i(i: int, X: np.ndarray) -> np.ndarray:
    """Gradiente di f2 su x^i = (10/m)*x^i  (sfrutta il fatto che sum_q z^q = 0)."""
    return (10.0 / M) * X[i]


def grad_f3_i(i: int, X: np.ndarray, S: np.ndarray) -> np.ndarray:
    """Gradiente di f3 su x^i = 4000*phi_i*(x^i-s^i) / ((1+D_i)*log(1+r^2))."""
    d   = X[i] - S[i]
    D   = float(d @ d)
    phi = np.log1p(D) / LOG1R2 - 1.0
    return (4000.0 * phi / ((1.0 + D) * LOG1R2)) * d


# ---------------------------------------------------------------------------
# Warm start (cosi' all'inizio f3 vale esattamente 0)
# ---------------------------------------------------------------------------

def warm_start() -> tuple:
    """
    Punti di riferimento: s^i = ((-1)^i * i * 0.2, (-1)^i * i * 0.2), i=1..m.

    Mettiamo ogni modulo a distanza r da s^i, in direzione dell'origine:
        x^i = s^i + r * (-s^i / ||s^i||)
    In questo modo ||x^i - s^i|| = r, quindi phi_i = 0 e di conseguenza f3 = 0.
    """
    idx  = np.arange(1, M + 1, dtype=np.float64)
    sign = np.where(idx % 2 == 0, 1.0, -1.0)
    S    = np.column_stack([sign * idx * 0.2, sign * idx * 0.2])  # (m, 2)
    nrm  = np.linalg.norm(S, axis=1, keepdims=True)               # (m, 1)
    safe = np.where(nrm < 1e-12, 1.0, nrm)
    X    = S + R * (-S / safe)
    X[nrm[:, 0] < 1e-12] = [R, 0.0]
    return X, S


# ---------------------------------------------------------------------------
# Ottimizzatore RBCD
# ---------------------------------------------------------------------------

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

    Parametri
    ---------
    n_epochs    : quante passate complete fare su tutti gli m=1000 moduli
    alpha_0     : passo iniziale
    beta, gamma : passo decrescente  alpha_k = alpha_0 / (1 + beta*k)^gamma
    use_f4_grad : True  -> include il gradiente di f4 con le differenze finite
                           (parallelizzate); e' la modalita' completa
                  False -> usa solo i gradienti analitici (modalita' veloce
                           per fare prove; attenzione: puo' peggiorare f4)
    resume      : se trova un x.txt valido riparte da quello
    """
    rng = np.random.default_rng(seed)

    # inizializzazione
    X, S = warm_start()
    print("Warm start fatto (f3 = 0 all'inizio)")

    if resume and os.path.exists(OUTPUT):
        try:
            xl = np.loadtxt(OUTPUT)
            if xl.size == 2 * M:
                X = xl.reshape(M, 2)
                print(f"Riparto da {OUTPUT}")
        except Exception as exc:
            print(f"Non riesco a leggere {OUTPUT}: {exc}")

    # salviamo subito un x.txt valido (almeno il warm start)
    np.savetxt(OUTPUT, X.flatten(), fmt="%.15g")

    # valutazione iniziale
    t0     = time.time()
    tmp0   = os.path.join(_SHM, "mobd_init.txt")
    f4_now = eval_f4(X.flatten(), tmp0)
    fk     = compute_f_known(X, S)
    total  = fk + f4_now
    best   = total
    best_X = X.copy()
    np.savetxt(OUTPUT, best_X.flatten(), fmt="%.15g")

    mode_str = (f"completa - f4 con differenze finite ({N_WORKERS} thread, "
                f"circa {int(np.ceil(2*M/N_WORKERS)*5/60)} min a epoca)"
                if use_f4_grad
                else "veloce - solo gradienti analitici (~0 s a epoca)")
    print(f"\nModalita': {mode_str}")
    print(f"Inizio: f1={compute_f1(X):.2f}  f2={compute_f2(X):.2f}"
          f"  f3={compute_f3(X,S):.2f}  f4={f4_now:.2f}")
    print(f"Totale: {total:.2f}  ({time.time()-t0:.1f} s)\n")

    for ep in range(n_epochs):
        t_ep  = time.time()
        alpha = alpha_0 / (1.0 + beta * ep) ** gamma
        order = rng.permutation(M)

        # gradiente di f4 per questa epoca (calcolato una volta, in parallelo)
        if use_f4_grad:
            snap    = X.flatten()   # x di inizio epoca, usata come base comune
            f4_base = f4_now        # f4(snap), gia' calcolata a fine epoca precedente
            t_fd    = time.time()
            g_f4    = eval_f4_all_partials(snap, f4_base)
            print(f"  [gradiente di f4 calcolato in {time.time()-t_fd:.1f}s]", flush=True)
        else:
            g_f4 = np.zeros((M, 2), dtype=np.float64)

        # aggiornamento dei blocchi (Gauss-Seidel, un modulo alla volta)
        for i in order:
            g = (grad_f1_i(i, X)
                 + grad_f2_i(i, X)
                 + grad_f3_i(i, X, S)
                 + g_f4[i])
            X[i] -= alpha * g

        # fine epoca: calcoliamo il costo totale e salviamo se migliora
        tmp_ep = os.path.join(_SHM, "mobd_epoch.txt")
        f4_now = eval_f4(X.flatten(), tmp_ep)
        fk     = compute_f_known(X, S)
        total  = fk + f4_now
        dt     = time.time() - t_ep

        print(f"Epoca {ep+1:3d}/{n_epochs}  tot={total:.2f}"
              f"  (f1={compute_f1(X):.2f}  f2={compute_f2(X):.2f}"
              f"  f3={compute_f3(X,S):.2f}  f4={f4_now:.2f})"
              f"  alpha={alpha:.5f}  t={dt:.1f}s",
              flush=True)

        if total < best:
            best   = total
            best_X = X.copy()
            np.savetxt(OUTPUT, best_X.flatten(), fmt="%.15g")
            print(f"  nuovo minimo, salvato in {OUTPUT}",
                  flush=True)

    return best_X


# ---------------------------------------------------------------------------
# Avvio del programma
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Ottimizzatore RBCD per il progetto MOBD",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument(
        "--mode",
        choices=["fast", "full", "both"],
        default="full",
        help=(
            "full : RBCD completo con f4 alle differenze finite | "
            "fast : solo gradienti analitici (non usa il simulatore durante l'epoca) | "
            "both : prima la fase veloce, poi quella completa"
        ),
    )
    p.add_argument("--fast-epochs", type=int, default=50,
                   help="Numero di epoche solo-analitiche (per mode=fast o both)")
    p.add_argument("--full-epochs", type=int, default=20,
                   help="Numero di epoche complete con f4 (per mode=full o both)")
    p.add_argument("--alpha0",  type=float, default=0.01,
                   help="Passo iniziale alpha_0")
    p.add_argument("--beta",    type=float, default=0.05,
                   help="Parametro beta del passo decrescente")
    p.add_argument("--gamma",   type=float, default=0.60,
                   help="Parametro gamma del passo decrescente")
    p.add_argument("--seed",    type=int,   default=42,
                   help="Seme per i numeri casuali")
    p.add_argument("--no-resume", action="store_true",
                   help="Ignora l'x.txt esistente e riparte dal warm start")
    p.add_argument("--workers", type=int, default=N_WORKERS,
                   help="Numero di thread paralleli per le perturbazioni di f4")
    return p.parse_args()


if __name__ == "__main__":
    args   = parse_args()
    resume = not args.no_resume

    # aggiorna il numero di thread se passato da riga di comando
    import mobd_optimizer as _self
    _self.N_WORKERS = args.workers
    N_WORKERS = args.workers

    if args.mode in ("fast", "both"):
        print("Fase 1: RBCD con soli gradienti analitici (veloce)")
        rbcd(
            n_epochs    = args.fast_epochs,
            alpha_0     = args.alpha0,
            beta        = args.beta,
            gamma       = args.gamma,
            seed        = args.seed,
            use_f4_grad = False,
            resume      = resume,
        )
        resume = True   # la fase 2 riparte sempre dal risultato della fase 1

    if args.mode in ("full", "both"):
        print("\nFase 2: RBCD completo con il gradiente di f4")
        rbcd(
            n_epochs    = args.full_epochs,
            alpha_0     = args.alpha0 * 0.5,   # passo piu' piccolo dopo la fase iniziale
            beta        = args.beta,
            gamma       = args.gamma,
            seed        = args.seed + 1,
            use_f4_grad = True,
            resume      = True,
        )

    print("\nFinito. La soluzione migliore e' in:", OUTPUT)
