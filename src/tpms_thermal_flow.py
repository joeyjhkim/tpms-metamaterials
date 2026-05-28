#!/usr/bin/env python3
"""
TPMS Thermal & Flow Simulation
===============================
Simulates heat transfer, thermal dissipation, and pressure-driven fluid flow
through novel TPMS metamaterials in static (undeformed) configuration.

Physics:
  - Thermal: transient heat conduction (implicit FDM, heterogeneous k)
  - Flow: pressure-driven creeping flow through void channels (Darcy)

Outputs per geometry:
  - geo_XXX_thermal.mp4   — transient heat conduction animation
  - geo_XXX_flow.mp4      — fluid flow streamline animation
  - geo_XXX_heat_summary.png — steady-state dashboard

Overall:
  - thermal_flow_summary.csv/.png — comparison metrics
"""

import sys as _sys
if hasattr(_sys.stdout, 'reconfigure'):
    _sys.stdout.reconfigure(line_buffering=True)

import warnings
warnings.filterwarnings("ignore", category=RuntimeWarning)

import os, csv, time, sys
import numpy as np
from scipy import sparse
from scipy.sparse.linalg import spsolve, splu, cg
from scipy.ndimage import binary_erosion
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.colors import Normalize
from datetime import date

# =============================================================================
# === PARAMETERS ===
# =============================================================================
N_CELLS    = 3
DOMAIN_M   = 0.1524          # 6 inches
SIM_RES    = 60              # 60^3 voxels
L_TPMS     = 2.0 * np.pi * N_CELLS

# Thermal
K_SOLID    = 200.0           # W/(m·K), aluminum
K_VOID     = 0.026           # W/(m·K), air
RHO_SOLID  = 2700.0          # kg/m³
CP_SOLID   = 900.0           # J/(kg·K)
RHO_VOID   = 1.2             # kg/m³
CP_VOID    = 1005.0          # J/(kg·K)
T_HOT      = 100.0           # °C
T_COLD     = 20.0            # °C

# Flow
MU_FLUID   = 1.81e-5         # Pa·s
P_IN       = 100.0           # Pa
P_OUT      = 0.0             # Pa

# Sim control
N_THERMAL_STEPS = 50
N_STREAMLINES   = 80
STREAM_MAX_STEPS = 2000

DATE_STR = date.today().strftime("%Y%m%d")
OUT      = os.getcwd()

# =============================================================================
# === FOURIER BASIS (identical to tpms_compression.py) ===
# =============================================================================
def _b_P(X, Y, Z):    return np.cos(X) + np.cos(Y) + np.cos(Z)
def _b_G(X, Y, Z):    return np.sin(X)*np.cos(Y) + np.sin(Y)*np.cos(Z) + np.sin(Z)*np.cos(X)
def _b_D(X, Y, Z):    return (np.sin(X)*np.sin(Y)*np.sin(Z) + np.sin(X)*np.cos(Y)*np.cos(Z)
                               + np.cos(X)*np.sin(Y)*np.cos(Z) + np.cos(X)*np.cos(Y)*np.sin(Z))
def _b_IWP(X, Y, Z):  return np.cos(X)*np.cos(Y) + np.cos(Y)*np.cos(Z) + np.cos(Z)*np.cos(X)
def _b_N(X, Y, Z):    return np.cos(X)*np.cos(Y)*np.cos(Z)
def _b_SS(X, Y, Z):   return np.sin(X)*np.sin(Y) + np.sin(Y)*np.sin(Z) + np.sin(Z)*np.sin(X)
def _b_P2(X, Y, Z):   return np.cos(2*X) + np.cos(2*Y) + np.cos(2*Z)
def _b_G2(X, Y, Z):   return np.sin(2*X)*np.cos(Y) + np.sin(2*Y)*np.cos(Z) + np.sin(2*Z)*np.cos(X)
def _b_CC2(X, Y, Z):  return np.cos(2*X)*np.cos(Y) + np.cos(2*Y)*np.cos(Z) + np.cos(2*Z)*np.cos(X)
def _b_SS2(X, Y, Z):  return np.sin(2*X)*np.sin(Y) + np.sin(2*Y)*np.sin(Z) + np.sin(2*Z)*np.sin(X)
def _b_CSC(X, Y, Z):  return (np.cos(X)*np.sin(Y)*np.cos(Z) + np.sin(X)*np.cos(Y)*np.sin(Z)
                               + np.cos(Y)*np.sin(Z)*np.cos(X))
def _b_L(X, Y, Z):    return (np.sin(2*X)*np.cos(Y)*np.sin(Z) + np.sin(2*Y)*np.cos(Z)*np.sin(X)
                               + np.sin(2*Z)*np.cos(X)*np.sin(Y))

BASIS = [
    ("P", _b_P), ("G", _b_G), ("D", _b_D), ("IWP", _b_IWP),
    ("N", _b_N), ("SS", _b_SS), ("P2", _b_P2), ("G2", _b_G2),
    ("CC2", _b_CC2), ("SS2", _b_SS2), ("CSC", _b_CSC), ("L", _b_L),
]
NAME_TO_IDX = {name: i for i, (name, _) in enumerate(BASIS)}
N_BASIS = len(BASIS)


# =============================================================================
# === PARSE GEOMETRY PARAMETERS ===
# =============================================================================
def parse_geometries(eq_path):
    """Parse geometry parameters from the equations log file."""
    geometries = []
    with open(eq_path, "r") as f:
        text = f.read()
    blocks = text.strip().split("\n\n")
    for block in blocks:
        if not block.strip():
            continue
        lines = block.strip().split("\n")
        geo = {}
        for line in lines:
            if line.startswith("Index:"):
                geo["index"] = int(line.split(":")[1].strip())
            elif line.startswith("Name:"):
                geo["name"] = line.split(":", 1)[1].strip()
            elif line.startswith("Coefficients:"):
                coeff_str = line.split(":", 1)[1].strip()
                coeffs = np.zeros(N_BASIS)
                active_idx = []
                for pair in coeff_str.split(","):
                    pair = pair.strip()
                    name, val = pair.split("=")
                    idx = NAME_TO_IDX[name.strip()]
                    coeffs[idx] = float(val)
                    active_idx.append(idx)
                geo["coeffs"] = coeffs
                geo["active_idx"] = sorted(active_idx)
            elif line.startswith("Isovalue t:"):
                geo["isovalue_t"] = float(line.split(":")[1].strip())
            elif line.startswith("Freq:"):
                geo["freq"] = float(line.split(":")[1].strip())
            elif line.startswith("Wall half-iso:"):
                geo["wall_half_iso"] = float(line.split(":")[1].strip())
            elif line.startswith("Equation:"):
                geo["equation"] = line.split(":", 1)[1].strip()
        if "coeffs" in geo:
            geometries.append(geo)
    return geometries


# =============================================================================
# === VOXELIZATION ===
# =============================================================================
def voxelize(params, res):
    """Create solid/void mask on a 3D grid. Returns (solid, dx)."""
    dx = DOMAIN_M / res
    half = dx / 2.0
    x = np.linspace(half, DOMAIN_M - half, res)
    X, Y, Z = np.meshgrid(x, x, x, indexing="ij")
    s = L_TPMS / DOMAIN_M
    f = params["freq"]
    Xf, Yf, Zf = f * s * X, f * s * Y, f * s * Z
    F = np.zeros_like(X)
    for i in params["active_idx"]:
        F += params["coeffs"][i] * BASIS[i][1](Xf, Yf, Zf)
    F -= params["isovalue_t"]
    solid = np.abs(F) < params["wall_half_iso"]
    return solid, dx


# =============================================================================
# === THERMAL — Vectorized implicit FDM ===
# =============================================================================
def _build_diffusion_matrix(solid, dx):
    """Build sparse diffusion operator A for heterogeneous medium.

    Heat equation: rho*cp * dT/dt = div(k * grad(T))
    Discretised: dT_n/dt = sum_m [k_nm / (rho_cp_n * dx^2)] * (T_m - T_n)
    Matrix A: A_nm = k_nm / (rho_cp_n * dx^2) for neighbours
              A_nn = -sum_m A_nm

    Returns A (sparse CSR), k_field, rho_cp_field.
    """
    nx, ny, nz = solid.shape
    N = nx * ny * nz

    k_field = np.where(solid, K_SOLID, K_VOID)
    rho_cp = np.where(solid, RHO_SOLID * CP_SOLID, RHO_VOID * CP_VOID)
    k_flat = k_field.ravel()
    rho_cp_flat = rho_cp.ravel()

    idx_arr = np.arange(N, dtype=np.int64).reshape(nx, ny, nz)

    all_rows = []
    all_cols = []
    all_vals = []
    diag_contrib = np.zeros(N)

    # 3 axis directions: (from_slice, to_slice)
    directions = [
        ((slice(None, -1), slice(None), slice(None)),
         (slice(1, None),  slice(None), slice(None))),   # x
        ((slice(None), slice(None, -1), slice(None)),
         (slice(None), slice(1, None),  slice(None))),   # y
        ((slice(None), slice(None), slice(None, -1)),
         (slice(None), slice(None), slice(1, None))),    # z
    ]

    for sf, st in directions:
        from_idx = idx_arr[sf].ravel()
        to_idx = idx_arr[st].ravel()

        k_f = k_flat[from_idx]
        k_t = k_flat[to_idx]
        k_harm = 2.0 * k_f * k_t / np.maximum(k_f + k_t, 1e-30)

        # Forward: entry A[from, to]
        coeff_ft = k_harm / (rho_cp_flat[from_idx] * dx * dx)
        all_rows.append(from_idx)
        all_cols.append(to_idx)
        all_vals.append(coeff_ft)
        np.add.at(diag_contrib, from_idx, -coeff_ft)

        # Backward: entry A[to, from]
        coeff_tf = k_harm / (rho_cp_flat[to_idx] * dx * dx)
        all_rows.append(to_idx)
        all_cols.append(from_idx)
        all_vals.append(coeff_tf)
        np.add.at(diag_contrib, to_idx, -coeff_tf)

    rows = np.concatenate(all_rows + [np.arange(N)])
    cols = np.concatenate(all_cols + [np.arange(N)])
    vals = np.concatenate(all_vals + [diag_contrib])

    A = sparse.coo_matrix((vals, (rows, cols)), shape=(N, N)).tocsr()
    return A, k_field, rho_cp


def simulate_thermal(solid, dx, n_steps):
    """Implicit transient heat conduction.

    BCs: T_HOT at z=0, T_COLD at z=L, insulated sides.
    Returns: snapshots (list of T arrays), dt, k_field.
    """
    nx, ny, nz = solid.shape
    N = nx * ny * nz

    print("    Building diffusion matrix (vectorized)...", flush=True)
    A, k_field, rho_cp = _build_diffusion_matrix(solid, dx)

    alpha_max = K_SOLID / (RHO_SOLID * CP_SOLID)
    t_total = (DOMAIN_M ** 2 / alpha_max) * 1.5
    dt = t_total / n_steps

    # Initial condition
    T = np.full(N, T_COLD)

    # BC masks
    idx_arr = np.arange(N, dtype=np.int64).reshape(nx, ny, nz)
    bc_hot = np.zeros(N, dtype=bool)
    bc_cold = np.zeros(N, dtype=bool)
    bc_hot[idx_arr[:, :, 0].ravel()] = True
    bc_cold[idx_arr[:, :, nz - 1].ravel()] = True
    T[bc_hot] = T_HOT
    T[bc_cold] = T_COLD

    bc_all = bc_hot | bc_cold
    free_idx = np.where(~bc_all)[0]
    bc_idx = np.where(bc_all)[0]

    # Implicit system: (I - dt*A) T^{n+1} = T^n
    M = sparse.eye(N, format="csr") - dt * A
    M_ff = M[np.ix_(free_idx, free_idx)]
    M_fb = M[np.ix_(free_idx, bc_idx)]

    T_bc_vals = np.zeros(len(bc_idx))
    T_bc_vals[bc_hot[bc_all]] = T_HOT
    T_bc_vals[bc_cold[bc_all]] = T_COLD
    rhs_bc = -M_fb @ T_bc_vals

    print(f"    Factoring ({len(free_idx)} free DOFs)...", flush=True)
    try:
        factor = splu(M_ff.tocsc())
        use_direct = True
    except Exception:
        use_direct = False
        print("    (using iterative solver)", flush=True)

    snapshots = [T.copy()]
    print(f"    Stepping {n_steps}x (dt={dt:.2f}s, total={t_total:.0f}s)...",
          flush=True)

    for step in range(1, n_steps + 1):
        rhs = T[free_idx] + rhs_bc
        if use_direct:
            T_free = factor.solve(rhs)
        else:
            T_free, _ = cg(M_ff, rhs, x0=T[free_idx], maxiter=500, atol=1e-8)
        T_new = T.copy()
        T_new[free_idx] = T_free
        T = T_new
        snapshots.append(T.copy())

        if step % 10 == 0 or step == n_steps:
            T3 = T.reshape(nx, ny, nz)
            print(f"      step {step:3d}/{n_steps}  "
                  f"T=[{T.min():.1f}, {T.max():.1f}]  "
                  f"mid={T3[nx//2, ny//2, nz//2]:.1f}°C", flush=True)

    return snapshots, dt, k_field


# =============================================================================
# === FLOW — Vectorized pressure Laplace solve ===
# =============================================================================
def _build_void_laplacian(void, dx):
    """Build Laplacian on void-only DOFs. Returns L, void_indices, g2l."""
    nx, ny, nz = void.shape
    N = nx * ny * nz
    void_flat = void.ravel()
    void_indices = np.where(void_flat)[0]
    n_void = len(void_indices)

    g2l = np.full(N, -1, dtype=np.int64)
    g2l[void_indices] = np.arange(n_void)

    idx_arr = np.arange(N, dtype=np.int64).reshape(nx, ny, nz)
    coeff = 1.0 / (dx * dx)

    all_rows = []
    all_cols = []
    all_vals = []
    diag_contrib = np.zeros(n_void)

    directions = [
        ((slice(None, -1), slice(None), slice(None)),
         (slice(1, None),  slice(None), slice(None))),
        ((slice(None), slice(None, -1), slice(None)),
         (slice(None), slice(1, None),  slice(None))),
        ((slice(None), slice(None), slice(None, -1)),
         (slice(None), slice(None), slice(1, None))),
    ]

    for sf, st in directions:
        from_g = idx_arr[sf].ravel()
        to_g = idx_arr[st].ravel()
        both = void_flat[from_g] & void_flat[to_g]
        fg = from_g[both]
        tg = to_g[both]
        fl = g2l[fg]
        tl = g2l[tg]
        n_p = len(fl)
        c = np.full(n_p, coeff)

        all_rows.append(fl);  all_cols.append(tl);  all_vals.append(c)
        np.add.at(diag_contrib, fl, -coeff)
        all_rows.append(tl);  all_cols.append(fl);  all_vals.append(c)
        np.add.at(diag_contrib, tl, -coeff)

    rows = np.concatenate(all_rows + [np.arange(n_void)])
    cols = np.concatenate(all_cols + [np.arange(n_void)])
    vals = np.concatenate(all_vals + [diag_contrib])

    L = sparse.coo_matrix((vals, (rows, cols)), shape=(n_void, n_void)).tocsr()
    return L, void_indices, g2l


def simulate_flow(solid, dx):
    """Pressure-driven flow through void channels.

    Returns: pressure (3D), vx, vy, vz (3D), streamlines (list of paths).
    """
    nx, ny, nz = solid.shape
    N = nx * ny * nz
    void = ~solid
    void_flat = void.ravel()

    inlet_count = int(void[:, :, 0].sum())
    outlet_count = int(void[:, :, -1].sum())
    total_void = int(void.sum())
    print(f"    Void: {total_void} ({100*total_void/N:.1f}%)  "
          f"inlet={inlet_count} outlet={outlet_count}", flush=True)

    if inlet_count == 0 or outlet_count == 0:
        print("    No flow path — skipping", flush=True)
        return None, None, None, None, []

    print("    Building void Laplacian (vectorized)...", flush=True)
    L_mat, void_indices, g2l = _build_void_laplacian(void, dx)
    n_void = len(void_indices)

    # BC masks (in local void index space)
    idx_arr = np.arange(N, dtype=np.int64).reshape(nx, ny, nz)
    bc_in_g = idx_arr[:, :, 0].ravel()
    bc_out_g = idx_arr[:, :, nz - 1].ravel()

    bc_in_mask = np.zeros(n_void, dtype=bool)
    bc_out_mask = np.zeros(n_void, dtype=bool)
    for g in bc_in_g:
        if void_flat[g]:
            bc_in_mask[g2l[g]] = True
    for g in bc_out_g:
        if void_flat[g]:
            bc_out_mask[g2l[g]] = True

    bc_mask = bc_in_mask | bc_out_mask
    free_local = np.where(~bc_mask)[0]
    bc_local = np.where(bc_mask)[0]

    p_bc = np.zeros(n_void)
    p_bc[bc_in_mask] = P_IN
    p_bc[bc_out_mask] = P_OUT

    L_ff = L_mat[np.ix_(free_local, free_local)]
    L_fb = L_mat[np.ix_(free_local, bc_local)]
    rhs = -L_fb @ p_bc[bc_local]

    print(f"    Solving pressure ({len(free_local)} free DOFs)...", flush=True)
    try:
        p_free = spsolve(L_ff.tocsc(), rhs)
    except Exception:
        p_free, _ = cg(L_ff, rhs, maxiter=2000, atol=1e-8)

    p_void_vals = p_bc.copy()
    p_void_vals[~bc_mask] = p_free

    # Map to full 3D
    pressure = np.full((nx, ny, nz), np.nan)
    pressure_flat = pressure.ravel()
    pressure_flat[void_indices] = p_void_vals
    pressure = pressure_flat.reshape(nx, ny, nz)

    # Velocity: v = -(1/mu) * grad(p)  (vectorized with np.gradient)
    p_clean = np.nan_to_num(pressure, nan=0.0)
    dp = np.gradient(p_clean, dx)  # [dp_dx, dp_dy, dp_dz]
    vx = np.where(void, -(1.0 / MU_FLUID) * dp[0], 0.0)
    vy = np.where(void, -(1.0 / MU_FLUID) * dp[1], 0.0)
    vz = np.where(void, -(1.0 / MU_FLUID) * dp[2], 0.0)

    # Trace streamlines
    print("    Tracing streamlines...", flush=True)
    streamlines = _trace_streamlines(vx, vy, vz, void, dx)
    print(f"    {len(streamlines)} streamlines traced", flush=True)

    return pressure, vx, vy, vz, streamlines


def _trace_streamlines(vx, vy, vz, void, dx):
    """Trace streamlines from z=0 inlet through void channels."""
    nx, ny, nz = void.shape
    rng = np.random.default_rng(42)

    inlet_ij = np.argwhere(void[:, :, 0])
    if len(inlet_ij) == 0:
        return []

    n = min(N_STREAMLINES, len(inlet_ij))
    chosen = rng.choice(len(inlet_ij), size=n, replace=False)

    v_mag = np.sqrt(vx**2 + vy**2 + vz**2)
    max_v = max(float(v_mag.max()), 1e-10)
    dt_s = 0.3 * dx / max_v

    lines = []
    for ci in chosen:
        si, sj = inlet_ij[ci]
        pos = np.array([(si + 0.5) * dx, (sj + 0.5) * dx, 0.5 * dx])
        path = [pos.copy()]

        for _ in range(STREAM_MAX_STEPS):
            ii = min(int(pos[0] / dx), nx - 1)
            jj = min(int(pos[1] / dx), ny - 1)
            kk = min(int(pos[2] / dx), nz - 1)
            if ii < 0 or jj < 0 or kk < 0:
                break
            if not void[ii, jj, kk]:
                break
            v = np.array([vx[ii, jj, kk], vy[ii, jj, kk], vz[ii, jj, kk]])
            if np.linalg.norm(v) < 1e-12:
                break
            pos = np.clip(pos + v * dt_s, 0, DOMAIN_M - 1e-10)
            path.append(pos.copy())
            if pos[2] >= DOMAIN_M - dx:
                break

        if len(path) > 5:
            lines.append(np.array(path))
    return lines


# =============================================================================
# === METRICS ===
# =============================================================================
def compute_thermal_metrics(T_steady, solid, dx, k_field):
    nx, ny, nz = solid.shape
    T_3d = T_steady.reshape(nx, ny, nz)

    # Heat flux through outlet face (z = nz-1)
    dTdz_outlet = (T_3d[:, :, -1] - T_3d[:, :, -2]) / dx
    k_outlet = k_field[:, :, -1]
    q_per_voxel = -k_outlet * dTdz_outlet * dx * dx
    q_total = float(np.sum(q_per_voxel))

    A_face = DOMAIN_M ** 2
    dT = T_HOT - T_COLD
    k_eff = abs(q_total) * DOMAIN_M / (A_face * dT) if dT != 0 else 0
    R_th = DOMAIN_M / (max(k_eff, 1e-30) * A_face)
    T_z = np.mean(T_3d, axis=(0, 1))

    return {"k_eff": k_eff, "R_thermal": R_th,
            "q_total": abs(q_total), "T_z_profile": T_z}


def compute_flow_metrics(pressure, vx, vy, vz, void, dx):
    if pressure is None:
        return {"permeability": 0, "avg_velocity": 0, "flow_rate": 0,
                "porosity": 0, "v_z_profile": np.zeros(SIM_RES)}
    nx, ny, nz = void.shape
    porosity = float(void.sum()) / (nx * ny * nz)
    v_mag = np.sqrt(vx**2 + vy**2 + vz**2)
    avg_vel = float(v_mag[void].mean()) if void.any() else 0

    Q_flow = float(np.sum(vz[:, :, -1][void[:, :, -1]] * dx * dx))
    A = DOMAIN_M ** 2
    dP = P_IN - P_OUT
    K_perm = abs(Q_flow) * MU_FLUID * DOMAIN_M / (A * max(dP, 1e-30))

    v_z = np.array([float(v_mag[:, :, k][void[:, :, k]].mean())
                     if void[:, :, k].any() else 0.0 for k in range(nz)])

    return {"permeability": K_perm, "avg_velocity": avg_vel,
            "flow_rate": abs(Q_flow), "porosity": porosity,
            "v_z_profile": v_z}


# =============================================================================
# === SURFACE VOXEL DETECTION (vectorized) ===
# =============================================================================
def find_surface_points(solid, dx, max_pts=4000):
    """Find solid voxels adjacent to void (surface). Subsample for plotting."""
    interior = binary_erosion(solid, structure=np.ones((3, 3, 3)))
    surface = solid & ~interior
    pts = np.argwhere(surface)
    if len(pts) > max_pts:
        step = max(1, len(pts) // max_pts)
        pts = pts[::step]
    coords_mm = (pts + 0.5) * dx * 1000.0
    flat_idx = pts[:, 0] * solid.shape[1] * solid.shape[2] + \
               pts[:, 1] * solid.shape[2] + pts[:, 2]
    return coords_mm, flat_idx


# =============================================================================
# === VIDEO — THERMAL ===
# =============================================================================
def make_thermal_video(snapshots, solid, dx, geo_name, mp4_path):
    nx, ny, nz = solid.shape
    n_frames = len(snapshots)

    surf_mm, surf_flat = find_surface_points(solid, dx)
    z_mm = np.linspace(dx / 2, DOMAIN_M - dx / 2, nz) * 1000

    fig = plt.figure(figsize=(18, 8))
    ax3d = fig.add_subplot(131, projection="3d")
    ax_xz = fig.add_subplot(132)
    ax_prof = fig.add_subplot(133)
    fig.suptitle(f"{geo_name} — Transient Heat Conduction",
                 fontsize=13, fontweight="bold")

    norm = Normalize(vmin=T_COLD, vmax=T_HOT)

    def update(frame):
        T = snapshots[frame]
        T_3d = T.reshape(nx, ny, nz)

        ax3d.clear()
        ax3d.scatter(surf_mm[:, 0], surf_mm[:, 1], surf_mm[:, 2],
                     c=T[surf_flat], cmap="coolwarm", norm=norm,
                     s=4, alpha=0.8, edgecolors="none")
        ax3d.set_xlim(0, DOMAIN_M * 1e3)
        ax3d.set_ylim(0, DOMAIN_M * 1e3)
        ax3d.set_zlim(0, DOMAIN_M * 1e3)
        ax3d.set_xlabel("X mm", fontsize=8)
        ax3d.set_ylabel("Y mm", fontsize=8)
        ax3d.set_zlabel("Z mm", fontsize=8)
        ax3d.view_init(elev=25, azim=-60 + frame * 1.5)
        ax3d.set_title(f"Step {frame}/{n_frames - 1}", fontsize=9)

        ax_xz.clear()
        mid_j = ny // 2
        T_xz = T_3d[:, mid_j, :].T
        ax_xz.imshow(T_xz, origin="lower", cmap="coolwarm", norm=norm,
                      aspect="equal",
                      extent=[0, DOMAIN_M * 1e3, 0, DOMAIN_M * 1e3])
        solid_xz = solid[:, mid_j, :].T.astype(float)
        ax_xz.contour(solid_xz, levels=[0.5],
                       extent=[0, DOMAIN_M * 1e3, 0, DOMAIN_M * 1e3],
                       colors="gray", linewidths=0.5, alpha=0.5)
        ax_xz.set_xlabel("X mm"); ax_xz.set_ylabel("Z mm")
        ax_xz.set_title("XZ slice (y=mid)", fontsize=9)

        ax_prof.clear()
        T_z_avg = np.mean(T_3d, axis=(0, 1))
        T_z_s = np.array([T_3d[:, :, k][solid[:, :, k]].mean()
                          if solid[:, :, k].any() else np.nan
                          for k in range(nz)])
        T_z_v = np.array([T_3d[:, :, k][~solid[:, :, k]].mean()
                          if (~solid[:, :, k]).any() else np.nan
                          for k in range(nz)])
        ax_prof.plot(z_mm, T_z_avg, "k-", lw=2, label="Average")
        ax_prof.plot(z_mm, T_z_s, "r--", lw=1.5, label="Solid")
        ax_prof.plot(z_mm, T_z_v, "b:", lw=1.5, label="Void (air)")
        ax_prof.set_xlabel("Z mm"); ax_prof.set_ylabel("T (°C)")
        ax_prof.set_ylim(T_COLD - 5, T_HOT + 5)
        ax_prof.set_title("T(z) profile", fontsize=9)
        ax_prof.legend(fontsize=7); ax_prof.grid(True, alpha=0.3)
        return []

    anim = FuncAnimation(fig, update, frames=n_frames, interval=300, blit=False)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    try:
        from matplotlib.animation import FFMpegWriter
        anim.save(mp4_path, writer=FFMpegWriter(fps=5), dpi=120)
        print(f"    Saved {os.path.basename(mp4_path)}", flush=True)
    except Exception as e:
        print(f"    WARN: mp4 failed ({e}), trying gif...", flush=True)
        try:
            from matplotlib.animation import PillowWriter
            anim.save(mp4_path.replace(".mp4", ".gif"),
                      writer=PillowWriter(fps=5), dpi=100)
        except Exception:
            pass
    plt.close(fig)


# =============================================================================
# === VIDEO — FLOW ===
# =============================================================================
def make_flow_video(pressure, vx, vy, vz, streamlines, solid, dx,
                    geo_name, mp4_path):
    if pressure is None or len(streamlines) == 0:
        print("    Skipping flow video (no data)", flush=True)
        return

    nx, ny, nz = solid.shape
    void = ~solid
    max_path_len = max(len(s) for s in streamlines)
    n_frames = min(80, max_path_len)

    v_mag = np.sqrt(vx**2 + vy**2 + vz**2)
    v_max = max(float(v_mag.max()), 1e-10)

    # Precompute per-streamline velocity colors
    stream_v = []
    for sl in streamlines:
        c = np.zeros(len(sl))
        for pi, pt in enumerate(sl):
            ii = min(int(pt[0] / dx), nx - 1)
            jj = min(int(pt[1] / dx), ny - 1)
            kk = min(int(pt[2] / dx), nz - 1)
            c[pi] = v_mag[max(ii, 0), max(jj, 0), max(kk, 0)]
        stream_v.append(c)

    fig = plt.figure(figsize=(16, 8))
    ax3d = fig.add_subplot(121, projection="3d")
    ax_pz = fig.add_subplot(122)
    fig.suptitle(f"{geo_name} — Pressure-Driven Flow",
                 fontsize=13, fontweight="bold")
    norm_p = Normalize(vmin=P_OUT, vmax=P_IN)

    mid_j = ny // 2
    solid_xz = solid[:, mid_j, :].T.astype(float)
    void_xz = void[:, mid_j, :].T

    def update(frame):
        frac = frame / max(n_frames - 1, 1)

        ax3d.clear()
        for si, sl in enumerate(streamlines):
            if len(sl) < 2:
                continue
            pts = sl * 1000
            c_mean = stream_v[si].mean() / v_max

            ax3d.plot(pts[:, 0], pts[:, 1], pts[:, 2],
                      color=plt.cm.viridis(c_mean), alpha=0.25, lw=0.5)

            p_idx = min(int(frac * len(sl)), len(sl) - 1)
            ax3d.scatter([pts[p_idx, 0]], [pts[p_idx, 1]], [pts[p_idx, 2]],
                         c=[stream_v[si][p_idx] / v_max], cmap="viridis",
                         vmin=0, vmax=1, s=18, alpha=0.9, edgecolors="none")

        L_mm = DOMAIN_M * 1000
        ax3d.set_xlim(0, L_mm); ax3d.set_ylim(0, L_mm); ax3d.set_zlim(0, L_mm)
        ax3d.set_xlabel("X mm", fontsize=8)
        ax3d.set_ylabel("Y mm", fontsize=8)
        ax3d.set_zlabel("Z mm", fontsize=8)
        ax3d.view_init(elev=20, azim=-60 + frame * 2)
        ax3d.set_title(f"Streamlines (t={frac * 100:.0f}%)", fontsize=9)

        ax_pz.clear()
        p_slice = pressure[:, mid_j, :].T
        p_disp = np.where(void_xz, p_slice, np.nan)
        ext = [0, L_mm, 0, L_mm]
        ax_pz.imshow(p_disp, origin="lower", cmap="RdYlBu_r",
                      norm=norm_p, aspect="equal", extent=ext)
        ax_pz.contourf(solid_xz, extent=ext, levels=[0.5, 1.5],
                        colors=["gray"], alpha=0.4)
        ax_pz.contour(solid_xz, levels=[0.5], extent=ext,
                       colors="black", linewidths=0.5)

        # Velocity quiver (subsampled)
        step_v = max(1, nx // 12)
        xi = np.arange(0, nx, step_v)
        zi = np.arange(0, nz, step_v)
        Xi, Zi = np.meshgrid(xi, zi)
        Vx_s = vx[Xi, mid_j, Zi]
        Vz_s = vz[Xi, mid_j, Zi]
        void_s = void[Xi, mid_j, Zi]
        X_mm = (Xi + 0.5) * dx * 1000
        Z_mm = (Zi + 0.5) * dx * 1000
        Vx_s = np.where(void_s, Vx_s, 0)
        Vz_s = np.where(void_s, Vz_s, 0)
        ax_pz.quiver(X_mm, Z_mm, Vx_s, Vz_s, color="black", alpha=0.6,
                      scale_units="inches", scale=v_max * 0.8,
                      width=0.003, headwidth=3)
        ax_pz.set_xlabel("X mm"); ax_pz.set_ylabel("Z mm")
        ax_pz.set_title("Pressure + velocity (y=mid)", fontsize=9)
        return []

    anim = FuncAnimation(fig, update, frames=n_frames, interval=200, blit=False)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    try:
        from matplotlib.animation import FFMpegWriter
        anim.save(mp4_path, writer=FFMpegWriter(fps=8), dpi=120)
        print(f"    Saved {os.path.basename(mp4_path)}", flush=True)
    except Exception as e:
        print(f"    WARN: mp4 failed ({e}), trying gif...", flush=True)
        try:
            from matplotlib.animation import PillowWriter
            anim.save(mp4_path.replace(".mp4", ".gif"),
                      writer=PillowWriter(fps=5), dpi=100)
        except Exception:
            pass
    plt.close(fig)


# =============================================================================
# === SUMMARY PNG ===
# =============================================================================
def save_heat_summary(T_steady, solid, dx, k_field, pressure, vx, vy, vz,
                      thermal_m, flow_m, geo_name, png_path):
    nx, ny, nz = solid.shape
    void = ~solid
    T_3d = T_steady.reshape(nx, ny, nz)
    mid = ny // 2
    ext = [0, DOMAIN_M * 1e3, 0, DOMAIN_M * 1e3]
    solid_xz = solid[:, mid, :].T.astype(float)

    fig, axes = plt.subplots(2, 3, figsize=(20, 12))
    fig.suptitle(f"{geo_name} — Thermal & Flow Summary",
                 fontsize=14, fontweight="bold")

    # (0,0) Temperature XZ
    ax = axes[0, 0]
    im = ax.imshow(T_3d[:, mid, :].T, origin="lower", cmap="coolwarm",
                    vmin=T_COLD, vmax=T_HOT, aspect="equal", extent=ext)
    ax.contour(solid_xz, levels=[0.5], extent=ext, colors="gray", linewidths=0.5)
    plt.colorbar(im, ax=ax, shrink=0.8, label="T (°C)")
    ax.set_title("Temperature (y=mid)"); ax.set_xlabel("X mm"); ax.set_ylabel("Z mm")

    # (0,1) Temperature XY at z=mid
    ax = axes[0, 1]
    mid_k = nz // 2
    im2 = ax.imshow(T_3d[:, :, mid_k].T, origin="lower", cmap="coolwarm",
                     vmin=T_COLD, vmax=T_HOT, aspect="equal", extent=ext)
    ax.contour(solid[:, :, mid_k].T.astype(float), levels=[0.5],
               extent=ext, colors="gray", linewidths=0.5)
    plt.colorbar(im2, ax=ax, shrink=0.8, label="T (°C)")
    ax.set_title("Temperature (z=mid)"); ax.set_xlabel("X mm"); ax.set_ylabel("Y mm")

    # (0,2) T(z) profile
    ax = axes[0, 2]
    z_mm = np.linspace(dx / 2, DOMAIN_M - dx / 2, nz) * 1000
    ax.plot(z_mm, thermal_m["T_z_profile"], "k-", lw=2, label="Avg")
    ax.axhline(T_HOT, color="r", ls="--", lw=1, alpha=0.5)
    ax.axhline(T_COLD, color="b", ls="--", lw=1, alpha=0.5)
    ax.set_xlabel("Z mm"); ax.set_ylabel("T (°C)")
    ax.set_title("T(z) Profile"); ax.legend(); ax.grid(True, alpha=0.3)

    # (1,0) Pressure
    ax = axes[1, 0]
    if pressure is not None:
        void_xz = void[:, mid, :].T
        p_disp = np.where(void_xz, pressure[:, mid, :].T, np.nan)
        im3 = ax.imshow(p_disp, origin="lower", cmap="RdYlBu_r",
                         vmin=P_OUT, vmax=P_IN, aspect="equal", extent=ext)
        ax.contourf(solid_xz, extent=ext, levels=[0.5, 1.5],
                     colors=["gray"], alpha=0.3)
        plt.colorbar(im3, ax=ax, shrink=0.8, label="P (Pa)")
    else:
        ax.text(0.5, 0.5, "No flow path", ha="center", va="center",
                transform=ax.transAxes, fontsize=14)
    ax.set_title("Pressure (y=mid)"); ax.set_xlabel("X mm"); ax.set_ylabel("Z mm")

    # (1,1) Velocity magnitude
    ax = axes[1, 1]
    if pressure is not None:
        vm = np.sqrt(vx**2 + vy**2 + vz**2)
        void_xz = void[:, mid, :].T
        v_disp = np.where(void_xz, vm[:, mid, :].T, np.nan)
        vm_max = max(float(np.nanmax(v_disp)), 1e-10)
        im4 = ax.imshow(v_disp, origin="lower", cmap="plasma",
                         vmin=0, vmax=vm_max, aspect="equal", extent=ext)
        ax.contourf(solid_xz, extent=ext, levels=[0.5, 1.5],
                     colors=["gray"], alpha=0.3)
        plt.colorbar(im4, ax=ax, shrink=0.8, label="|v| (m/s)")
    else:
        ax.text(0.5, 0.5, "No flow data", ha="center", va="center",
                transform=ax.transAxes, fontsize=14)
    ax.set_title("Velocity (y=mid)"); ax.set_xlabel("X mm"); ax.set_ylabel("Z mm")

    # (1,2) Metrics
    ax = axes[1, 2]
    ax.axis("off")
    txt = (
        f"THERMAL\n{'─'*28}\n"
        f"k_eff     = {thermal_m['k_eff']:.4f} W/(m·K)\n"
        f"k_ratio   = {thermal_m['k_eff']/K_SOLID:.4f}\n"
        f"R_thermal = {thermal_m['R_thermal']:.4f} K/W\n"
        f"Q_total   = {thermal_m['q_total']:.3f} W\n\n"
        f"FLOW\n{'─'*28}\n"
        f"Porosity  = {flow_m['porosity']*100:.1f}%\n"
        f"Perm K    = {flow_m['permeability']:.2e} m²\n"
        f"v_avg     = {flow_m['avg_velocity']:.4f} m/s\n"
        f"Q_flow    = {flow_m['flow_rate']:.2e} m³/s\n\n"
        f"MATERIAL\n{'─'*28}\n"
        f"k_solid={K_SOLID} W/(m·K)\n"
        f"k_void={K_VOID} W/(m·K)\n"
        f"ΔT={T_HOT-T_COLD}°C  ΔP={P_IN-P_OUT} Pa"
    )
    ax.text(0.05, 0.95, txt, transform=ax.transAxes, fontsize=10,
            va="top", family="monospace",
            bbox=dict(boxstyle="round", fc="lightyellow", alpha=0.8))
    ax.set_title("Metrics")

    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(png_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"    Saved {os.path.basename(png_path)}", flush=True)


# =============================================================================
# === MAIN ===
# =============================================================================
if __name__ == "__main__":
    print("=" * 70)
    print("  TPMS Thermal & Flow Simulation")
    print("=" * 70)
    print(f"  Grid     : {SIM_RES}^3 = {SIM_RES**3:,} voxels")
    print(f"  Domain   : {DOMAIN_M*1e3:.1f} mm  |  {N_CELLS}^3 cells")
    print(f"  Thermal  : k_s={K_SOLID} k_v={K_VOID} W/(m·K)")
    print(f"  BCs      : T_hot={T_HOT}°C → T_cold={T_COLD}°C  "
          f"ΔP={P_IN-P_OUT} Pa")
    print(f"  Steps    : {N_THERMAL_STEPS} thermal  |  "
          f"{N_STREAMLINES} streamlines")
    print(f"  Date     : {DATE_STR}\n")

    # Find equations file
    eq_path = os.path.join(OUT, f"geometry_equations_{DATE_STR}.txt")
    if not os.path.exists(eq_path):
        eq_files = sorted([f for f in os.listdir(OUT)
                           if f.startswith("geometry_equations_")
                           and f.endswith(".txt")])
        if eq_files:
            eq_path = os.path.join(OUT, eq_files[-1])
            print(f"  Using: {eq_files[-1]}")
        else:
            print("  ERROR: No geometry_equations file found.")
            print("  Run tpms_compression.py first.")
            sys.exit(1)

    geometries = parse_geometries(eq_path)
    print(f"  Loaded {len(geometries)} geometries\n")

    t_total = time.time()
    all_results = []

    for geo in geometries:
        idx = geo["index"]
        name = geo["name"]
        tag = f"[GEO {idx:02d}/{len(geometries)}]"
        prefix = f"geo_{idx:03d}_{DATE_STR}"
        t_geo = time.time()

        eq = geo.get("equation", "")
        print(f"\n  {tag} {name}")
        print(f"  {tag} {eq}")

        # Voxelise
        print(f"  {tag} Voxelising...", flush=True)
        solid, dx = voxelize(geo, SIM_RES)
        ns = int(solid.sum())
        nt = SIM_RES ** 3
        print(f"  {tag} Solid: {ns} ({100*ns/nt:.1f}%)  "
              f"Void: {nt-ns} ({100*(nt-ns)/nt:.1f}%)")

        # Thermal
        print(f"  {tag} === THERMAL ===", flush=True)
        t_th = time.time()
        snapshots, dt_th, k_field = simulate_thermal(solid, dx, N_THERMAL_STEPS)
        t_th = time.time() - t_th
        print(f"  {tag} Thermal: {t_th:.1f}s")

        T_steady = snapshots[-1]
        th_m = compute_thermal_metrics(T_steady, solid, dx, k_field)
        print(f"  {tag} k_eff={th_m['k_eff']:.4f} W/(m·K)  "
              f"ratio={th_m['k_eff']/K_SOLID:.4f}  Q={th_m['q_total']:.3f} W")

        # Flow
        print(f"  {tag} === FLOW ===", flush=True)
        t_fl = time.time()
        pressure, vx, vy, vz, streamlines = simulate_flow(solid, dx)
        t_fl = time.time() - t_fl
        print(f"  {tag} Flow: {t_fl:.1f}s")

        fl_m = compute_flow_metrics(pressure, vx, vy, vz, ~solid, dx)
        print(f"  {tag} Porosity={fl_m['porosity']*100:.1f}%  "
              f"K={fl_m['permeability']:.2e} m²  "
              f"v_avg={fl_m['avg_velocity']:.4f} m/s")

        # Videos
        print(f"  {tag} Rendering thermal video...", flush=True)
        make_thermal_video(snapshots, solid, dx, name,
                           os.path.join(OUT, f"{prefix}_thermal.mp4"))

        print(f"  {tag} Rendering flow video...", flush=True)
        make_flow_video(pressure, vx, vy, vz, streamlines, solid, dx,
                        name, os.path.join(OUT, f"{prefix}_flow.mp4"))

        # Summary PNG
        save_heat_summary(T_steady, solid, dx, k_field,
                          pressure, vx, vy, vz,
                          th_m, fl_m, name,
                          os.path.join(OUT, f"{prefix}_heat_summary.png"))

        geo_time = time.time() - t_geo
        print(f"  {tag} Done ({geo_time:.1f}s)")

        all_results.append({
            "index": idx, "name": name, "equation": eq,
            "porosity": fl_m["porosity"],
            "k_eff": th_m["k_eff"],
            "k_ratio": th_m["k_eff"] / K_SOLID,
            "R_thermal": th_m["R_thermal"],
            "Q_total": th_m["q_total"],
            "permeability": fl_m["permeability"],
            "avg_velocity": fl_m["avg_velocity"],
            "flow_rate": fl_m["flow_rate"],
            "n_streamlines": len(streamlines),
            "thermal_time": t_th,
            "flow_time": t_fl,
        })

        del snapshots, T_steady, k_field
        del pressure, vx, vy, vz, streamlines, solid

    # === SUMMARY CSV ===
    sum_csv = os.path.join(OUT, f"thermal_flow_summary_{DATE_STR}.csv")
    with open(sum_csv, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["index", "name", "porosity_pct",
                     "k_eff_W_per_mK", "k_ratio", "R_thermal_K_per_W",
                     "Q_total_W", "permeability_m2", "avg_velocity_m_per_s",
                     "flow_rate_m3_per_s", "n_streamlines"])
        for r in all_results:
            w.writerow([r["index"], r["name"],
                         f"{r['porosity']*100:.1f}",
                         f"{r['k_eff']:.6f}", f"{r['k_ratio']:.6f}",
                         f"{r['R_thermal']:.4f}", f"{r['Q_total']:.4f}",
                         f"{r['permeability']:.6e}",
                         f"{r['avg_velocity']:.6f}",
                         f"{r['flow_rate']:.6e}", r["n_streamlines"]])
    print(f"\n  Saved {os.path.basename(sum_csv)}")

    # === SUMMARY PLOT ===
    if len(all_results) >= 2:
        labels = [f"S{r['index']:02d}" for r in all_results]
        colours = plt.cm.tab10(np.linspace(0, 1, len(all_results)))

        fig, axes = plt.subplots(2, 3, figsize=(20, 11))
        fig.suptitle(f"TPMS Thermal & Flow Comparison — {DATE_STR}",
                     fontsize=14, fontweight="bold")

        k_effs = [r["k_eff"] for r in all_results]
        axes[0, 0].barh(labels, k_effs, color=colours)
        axes[0, 0].set_xlabel("k_eff (W/(m·K))")
        axes[0, 0].set_title("Effective Thermal Conductivity")

        R_vals = [r["R_thermal"] for r in all_results]
        axes[0, 1].barh(labels, R_vals, color=colours)
        axes[0, 1].set_xlabel("R (K/W)")
        axes[0, 1].set_title("Thermal Resistance")

        Q_vals = [r["Q_total"] for r in all_results]
        axes[0, 2].barh(labels, Q_vals, color=colours)
        axes[0, 2].set_xlabel("Q (W)")
        axes[0, 2].set_title("Total Heat Flux")

        K_vals = [r["permeability"] for r in all_results]
        axes[1, 0].barh(labels, K_vals, color=colours)
        axes[1, 0].set_xlabel("K (m²)")
        axes[1, 0].set_title("Permeability")
        axes[1, 0].ticklabel_format(axis="x", style="scientific",
                                     scilimits=(0, 0))

        pors = [r["porosity"] * 100 for r in all_results]
        axes[1, 1].scatter(pors, k_effs, c=colours, s=100,
                            edgecolors="k", zorder=3)
        for i, r in enumerate(all_results):
            axes[1, 1].annotate(f"S{r['index']:02d}",
                                 (pors[i], k_effs[i]),
                                 fontsize=7, ha="center", va="bottom")
        axes[1, 1].set_xlabel("Porosity (%)")
        axes[1, 1].set_ylabel("k_eff (W/(m·K))")
        axes[1, 1].set_title("Porosity vs Conductivity")
        axes[1, 1].grid(True, alpha=0.3)

        axes[1, 2].scatter(pors, K_vals, c=colours, s=100,
                            edgecolors="k", zorder=3)
        for i, r in enumerate(all_results):
            axes[1, 2].annotate(f"S{r['index']:02d}",
                                 (pors[i], K_vals[i]),
                                 fontsize=7, ha="center", va="bottom")
        axes[1, 2].set_xlabel("Porosity (%)")
        axes[1, 2].set_ylabel("Permeability (m²)")
        axes[1, 2].set_title("Porosity vs Permeability")
        axes[1, 2].grid(True, alpha=0.3)
        axes[1, 2].ticklabel_format(axis="y", style="scientific",
                                     scilimits=(0, 0))

        fig.tight_layout(rect=[0, 0, 1, 0.95])
        sum_png = os.path.join(OUT, f"thermal_flow_summary_{DATE_STR}.png")
        fig.savefig(sum_png, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"  Saved {os.path.basename(sum_png)}")

    # === TABLE ===
    print(f"\n{'='*90}")
    print(f"  {'#':>3} {'Por%':>5} {'k_eff':>9} {'ratio':>7} "
          f"{'R(K/W)':>8} {'Q(W)':>7} {'Perm':>11} "
          f"{'v_avg':>7} {'Time':>6}")
    print("  " + "─" * 85)
    for r in all_results:
        print(f"  {r['index']:3d} {r['porosity']*100:5.1f} "
              f"{r['k_eff']:9.4f} {r['k_ratio']:7.4f} "
              f"{r['R_thermal']:8.4f} {r['Q_total']:7.3f} "
              f"{r['permeability']:11.2e} "
              f"{r['avg_velocity']:7.4f} "
              f"{r['thermal_time']+r['flow_time']:6.1f}s")

    print(f"\n  Total: {time.time()-t_total:.1f}s")
    print(f"  [DONE] {len(all_results)} geometries complete.")
