#!/usr/bin/env python3
# pip install numpy scipy matplotlib scikit-image
"""
TPMS Flow & Heat Simulation — Schwartz P
=========================================
Generates a Schwartz P TPMS shell, runs:
  1. Steady-state streamline flow simulation (potential flow approx)
  2. Transient heat conduction (explicit FD on solid voxels)
Displays interactive 3D windows and exports MP4 animations.
"""

import sys as _sys
if hasattr(_sys.stdout, 'reconfigure'):
    _sys.stdout.reconfigure(line_buffering=True)

import warnings
warnings.filterwarnings("ignore", category=RuntimeWarning)

import numpy as np
from scipy import sparse
from scipy.ndimage import gaussian_filter, laplace
from scipy.interpolate import RegularGridInterpolator
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, Normalize
from matplotlib.animation import FuncAnimation
from mpl_toolkits.mplot3d.art3d import Line3DCollection
from datetime import date
import os, time

try:
    from skimage.measure import marching_cubes
except ImportError:
    _sys.exit("ERROR: scikit-image required  (pip install scikit-image)")

# =============================================================================
# === PARAMETERS ===
# =============================================================================
N_CELLS       = 3
VOXEL_RES     = N_CELLS * 30        # 90
ISO_T         = 0.0
WALL_HALF     = 0.20
FREQ          = 1.0

L             = 2.0 * np.pi * N_CELLS
DX            = L / VOXEL_RES
DOMAIN_M      = 0.1524               # 6 inches physical size
DX_PHYS       = DOMAIN_M / VOXEL_RES # physical grid spacing (meters)

# Thermal (TPU)
K_COND        = 0.25                 # W/(m·K)
RHO           = 1200.0               # kg/m³
CP            = 1500.0               # J/(kg·K)
T_HOT         = 100.0                # °C
T_COLD        = 20.0                 # °C
N_HEAT_STEPS  = 8000
HEAT_SAVE_EVERY = 200                # → 40 frames

# Flow
BASE_VEL      = 1.0                  # m/s in +X
SEED_N        = 6                    # 6×6 grid on inlet
RK4_DS        = L / (VOXEL_RES * 4)
RK4_MAX       = 600
GAUSS_SIGMA   = 1.5
GAUSS_ITERS   = 5
FLOW_FRAMES   = 60
FLOW_FPS      = 24
HEAT_FPS      = 12
TRAIL_LEN     = 15

# View
ELEV          = 25
AZIM          = 35

DATE_STR      = date.today().strftime("%Y%m%d")
OUT           = os.getcwd()

# =============================================================================
# === COLOR MAP ===
# =============================================================================
_stops = [
    (0.0, "#0000FF"),
    (0.2, "#00BFFF"),
    (0.4, "#00FF80"),
    (0.6, "#FFFF00"),
    (0.8, "#FF8000"),
    (1.0, "#FF0000"),
]
_colors_rgb = []
_positions = []
for pos, hexc in _stops:
    _positions.append(pos)
    r = int(hexc[1:3], 16) / 255.0
    g = int(hexc[3:5], 16) / 255.0
    b = int(hexc[5:7], 16) / 255.0
    _colors_rgb.append((r, g, b))

kim_cmap = LinearSegmentedColormap.from_list(
    "kim_cmap",
    list(zip(_positions, _colors_rgb)),
    N=256,
)

# =============================================================================
# === GEOMETRY ===
# =============================================================================
print(f"[GEOMETRY] Generating Schwartz P {N_CELLS}x{N_CELLS}x{N_CELLS} "
      f"@ {VOXEL_RES}³ voxels...")
t0 = time.time()

x_1d = np.linspace(0, L, VOXEL_RES, endpoint=False)
X, Y, Z = np.meshgrid(x_1d, x_1d, x_1d, indexing="ij")
F = np.cos(FREQ * X) + np.cos(FREQ * Y) + np.cos(FREQ * Z) - ISO_T

# Boolean masks
solid_mask = np.abs(F) <= WALL_HALF          # shell
void_mask  = ~solid_mask                     # channels

# Marching cubes for shell surface (|F| - wall_half = 0)
G_shell = np.abs(F) - WALL_HALF
G_pad = np.pad(G_shell, 1, mode="constant", constant_values=G_shell.max() + 1.0)
verts, faces, _, _ = marching_cubes(G_pad, level=0.0, spacing=(DX, DX, DX))
verts -= DX  # undo padding offset
verts = np.clip(verts, 0, L - 1e-10)

# Laplacian smoothing (3 iterations, lambda=0.3)
def _laplacian_smooth(v, f, n_iter=3, lam=0.3):
    n = len(v)
    e = np.vstack([f[:, [0, 1]], f[:, [1, 2]], f[:, [0, 2]]])
    e = np.vstack([e, e[:, ::-1]])
    adj = sparse.coo_matrix(
        (np.ones(len(e)), (e[:, 0], e[:, 1])), shape=(n, n)
    ).tocsr()
    deg = np.array(adj.sum(axis=1)).ravel()
    deg[deg == 0] = 1
    bnd = np.zeros(n, dtype=bool)
    eps = L * 0.005
    for d in range(3):
        bnd |= (v[:, d] < eps) | (v[:, d] > L - eps)
    for _ in range(n_iter):
        avg = (adj @ v) / deg[:, None]
        delta = lam * (avg - v)
        delta[bnd] = 0.0
        v = v + delta
    return v

verts = _laplacian_smooth(verts, faces)

# STL export
stl_path = os.path.join(OUT, f"tpms_solid_{DATE_STR}.stl")
v_mm = verts.astype(np.float32)
tri = v_mm[faces]
fn = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
mag = np.linalg.norm(fn, axis=1, keepdims=True)
fn = (fn / np.where(mag > 1e-12, mag, 1.0)).astype(np.float32)
n_t = len(faces)
rec = np.zeros(n_t, dtype=[
    ("n", "<f4", 3), ("v0", "<f4", 3), ("v1", "<f4", 3),
    ("v2", "<f4", 3), ("a", "<u2")])
rec["n"] = fn; rec["v0"] = tri[:, 0]; rec["v1"] = tri[:, 1]; rec["v2"] = tri[:, 2]
with open(stl_path, "wb") as fp:
    fp.write(b"Schwartz P TPMS shell".ljust(80, b"\0"))
    fp.write(np.array(n_t, dtype="<u4").tobytes())
    fp.write(rec.tobytes())

print(f"[GEOMETRY] Vertices: {len(verts)} | Smoothing done | STL saved "
      f"({time.time()-t0:.1f}s)")

# Subsample shell mesh for 3D overlay
_step_f = max(1, len(faces) // 12000)
shell_faces_sub = faces[::_step_f]

# =============================================================================
# === FLOW SIMULATION ===
# =============================================================================
print("[FLOW] Building velocity field...")
t0 = time.time()

# Base uniform flow in +X
vx = np.full((VOXEL_RES, VOXEL_RES, VOXEL_RES), BASE_VEL, dtype=np.float64)
vy = np.zeros_like(vx)
vz = np.zeros_like(vx)

# Zero velocity inside solid (no-slip)
vx[solid_mask] = 0.0
vy[solid_mask] = 0.0
vz[solid_mask] = 0.0

# Smooth velocity field to create flow-around-obstacle
for _ in range(GAUSS_ITERS):
    vx = gaussian_filter(vx, sigma=GAUSS_SIGMA)
    vy = gaussian_filter(vy, sigma=GAUSS_SIGMA)
    vz = gaussian_filter(vz, sigma=GAUSS_SIGMA)
    # Re-enforce no-slip in solid
    vx[solid_mask] = 0.0
    vy[solid_mask] = 0.0
    vz[solid_mask] = 0.0

# Normalise so mean void velocity = 1.0 m/s
v_mag_field = np.sqrt(vx**2 + vy**2 + vz**2)
mean_void_v = float(v_mag_field[void_mask].mean())
if mean_void_v > 1e-12:
    scale = BASE_VEL / mean_void_v
    vx *= scale
    vy *= scale
    vz *= scale
    v_mag_field *= scale

# Build interpolators
interp_vx = RegularGridInterpolator((x_1d, x_1d, x_1d), vx,
                                     bounds_error=False, fill_value=0.0)
interp_vy = RegularGridInterpolator((x_1d, x_1d, x_1d), vy,
                                     bounds_error=False, fill_value=0.0)
interp_vz = RegularGridInterpolator((x_1d, x_1d, x_1d), vz,
                                     bounds_error=False, fill_value=0.0)
interp_F  = RegularGridInterpolator((x_1d, x_1d, x_1d), F,
                                     bounds_error=False, fill_value=0.0)

# Seed streamlines on x=0 plane (6×6 grid)
margin = L * 0.08
sy = np.linspace(margin, L - margin, SEED_N)
sz = np.linspace(margin, L - margin, SEED_N)
seeds = []
for yy in sy:
    for zz in sz:
        pt = np.array([DX * 0.5, yy, zz])
        f_val = interp_F(pt.reshape(1, 3)).item()
        if abs(f_val) > WALL_HALF:
            seeds.append(pt)

print(f"[FLOW] Seeding {SEED_N*SEED_N} streamlines | Valid seeds: {len(seeds)}")

# RK4 advection
def _rk4_step(pos, ds):
    def _vel(p):
        pp = p.reshape(1, 3)
        return np.array([interp_vx(pp).item(),
                         interp_vy(pp).item(),
                         interp_vz(pp).item()])
    k1 = _vel(pos)
    k2 = _vel(pos + 0.5 * ds * k1)
    k3 = _vel(pos + 0.5 * ds * k2)
    k4 = _vel(pos + ds * k3)
    return pos + (ds / 6.0) * (k1 + 2*k2 + 2*k3 + k4)


streamlines = []
stream_vmag = []

for seed in seeds:
    path = [seed.copy()]
    vmags = []
    pos = seed.copy()
    for _ in range(RK4_MAX):
        pp = pos.reshape(1, 3)
        vm = np.sqrt(interp_vx(pp).item()**2 +
                     interp_vy(pp).item()**2 +
                     interp_vz(pp).item()**2)
        vmags.append(vm)
        pos_new = _rk4_step(pos, RK4_DS)
        # Check bounds
        if np.any(pos_new < 0) or np.any(pos_new > L):
            break
        # Check solid
        f_val = interp_F(pos_new.reshape(1, 3)).item()
        if abs(f_val) <= WALL_HALF:
            break
        pos = pos_new
        path.append(pos.copy())
    if len(path) > 3:
        streamlines.append(np.array(path))
        stream_vmag.append(np.array(vmags[:len(path)]))

print(f"[FLOW] Advecting streamlines... done ({time.time()-t0:.1f}s)")

# Velocity range for coloring
all_vm = np.concatenate(stream_vmag) if stream_vmag else np.array([0, 1])
v_min_flow = float(all_vm.min())
v_max_flow = max(float(all_vm.max()), v_min_flow + 1e-6)
norm_flow = Normalize(vmin=v_min_flow, vmax=v_max_flow)

# =============================================================================
# === FLOW VISUALIZATION ===
# =============================================================================

def _draw_shell(ax):
    """Draw semi-transparent gray TPMS shell."""
    ax.plot_trisurf(
        verts[:, 0], verts[:, 1], verts[:, 2],
        triangles=shell_faces_sub,
        color=(0.5, 0.5, 0.5, 0.12),
        edgecolor="none",
        shade=False,
        zorder=0,
    )


def _draw_streamlines_colored(ax, alpha=1.0):
    """Draw all streamlines as colored line segments."""
    for sl, vm in zip(streamlines, stream_vmag):
        pts = sl
        colors = kim_cmap(norm_flow(vm[:len(pts)]))
        for i in range(len(pts) - 1):
            ax.plot(
                [pts[i, 0], pts[i+1, 0]],
                [pts[i, 1], pts[i+1, 1]],
                [pts[i, 2], pts[i+1, 2]],
                color=colors[i], lw=2.0, alpha=alpha,
            )


# --- Static interactive window ---
fig_flow = plt.figure("Flow Simulation", figsize=(14, 10))
ax_flow = fig_flow.add_subplot(111, projection="3d")
_draw_shell(ax_flow)
_draw_streamlines_colored(ax_flow, alpha=0.9)
ax_flow.set_xlim(0, L); ax_flow.set_ylim(0, L); ax_flow.set_zlim(0, L)
ax_flow.set_xlabel("X"); ax_flow.set_ylabel("Y"); ax_flow.set_zlabel("Z")
ax_flow.view_init(elev=ELEV, azim=AZIM)
ax_flow.set_title(f"TPMS Air Flow — Schwartz P | KimSurface {DATE_STR}",
                   fontsize=13, fontweight="bold")

# Colorbar
sm_flow = plt.cm.ScalarMappable(cmap=kim_cmap, norm=norm_flow)
sm_flow.set_array([])
cb_flow = fig_flow.colorbar(sm_flow, ax=ax_flow, shrink=0.6, pad=0.08)
cb_flow.set_label("Air Velocity (m/s)", fontsize=10)

# Flow direction arrow annotation
ax_flow.text(L * 0.05, L * 0.5, L * 1.02,
             "Flow →  +X", fontsize=10, color="red", fontweight="bold")

fig_flow.tight_layout()
flow_png = os.path.join(OUT, f"flow_static_{DATE_STR}.png")
fig_flow.savefig(flow_png, dpi=150, bbox_inches="tight")
print(f"[FLOW] Interactive window open | Exporting MP4...")
plt.show(block=False)
plt.pause(0.1)

# --- Flow MP4 animation ---
fig_fa = plt.figure("Flow Animation", figsize=(14, 10))
ax_fa = fig_fa.add_subplot(111, projection="3d")
ax_fa.view_init(elev=ELEV, azim=AZIM)
ax_fa.set_xlim(0, L); ax_fa.set_ylim(0, L); ax_fa.set_zlim(0, L)
ax_fa.set_xlabel("X"); ax_fa.set_ylabel("Y"); ax_fa.set_zlabel("Z")

# Static shell (drawn once)
_draw_shell(ax_fa)

# Static faint streamline paths
for sl, vm in zip(streamlines, stream_vmag):
    for i in range(len(sl) - 1):
        ax_fa.plot(
            [sl[i, 0], sl[i+1, 0]],
            [sl[i, 1], sl[i+1, 1]],
            [sl[i, 2], sl[i+1, 2]],
            color=(0.5, 0.5, 0.5), lw=0.4, alpha=0.15,
        )

# Colorbar
sm_fa = plt.cm.ScalarMappable(cmap=kim_cmap, norm=norm_flow)
sm_fa.set_array([])
cb_fa = fig_fa.colorbar(sm_fa, ax=ax_fa, shrink=0.6, pad=0.08)
cb_fa.set_label("Air Velocity (m/s)", fontsize=10)

# Pre-compute max path length
max_path_len = max((len(s) for s in streamlines), default=1)

# Particles and trails — stored as plot handles
_particle_plots = []
_trail_plots = []


def _flow_update(frame):
    # Remove previous particles and trails
    for p in _particle_plots:
        p.remove()
    _particle_plots.clear()
    for t in _trail_plots:
        t.remove()
    _trail_plots.clear()

    frac = frame / max(FLOW_FRAMES - 1, 1)

    for sl, vm in zip(streamlines, stream_vmag):
        n_pts = len(sl)
        # Current particle index along path
        p_idx = min(int(frac * (n_pts - 1)), n_pts - 1)

        # Particle dot
        c = kim_cmap(norm_flow(vm[p_idx]))
        sc = ax_fa.scatter(
            [sl[p_idx, 0]], [sl[p_idx, 1]], [sl[p_idx, 2]],
            color=c, s=40, zorder=10, edgecolors="none",
        )
        _particle_plots.append(sc)

        # Trail (last TRAIL_LEN positions)
        trail_start = max(0, p_idx - TRAIL_LEN)
        trail = sl[trail_start:p_idx + 1]
        trail_vm = vm[trail_start:p_idx + 1]
        for ti in range(len(trail) - 1):
            alpha_t = 0.1 + 0.7 * (ti / max(len(trail) - 1, 1))
            tc = kim_cmap(norm_flow(trail_vm[ti]))
            line, = ax_fa.plot(
                [trail[ti, 0], trail[ti+1, 0]],
                [trail[ti, 1], trail[ti+1, 1]],
                [trail[ti, 2], trail[ti+1, 2]],
                color=tc, lw=2.5, alpha=alpha_t,
            )
            _trail_plots.append(line)

    ax_fa.set_title(
        f"TPMS Air Flow — Schwartz P | KimSurface {DATE_STR}\n"
        f"Frame {frame+1}/{FLOW_FRAMES}",
        fontsize=12, fontweight="bold",
    )
    return []


anim_flow = FuncAnimation(fig_fa, _flow_update, frames=FLOW_FRAMES,
                           interval=1000 // FLOW_FPS, blit=False)

flow_mp4 = os.path.join(OUT, f"flow_{DATE_STR}.mp4")
flow_saved = False
try:
    from matplotlib.animation import FFMpegWriter
    anim_flow.save(flow_mp4, writer=FFMpegWriter(fps=FLOW_FPS), dpi=120)
    flow_saved = True
    print(f"[FLOW] Saved: {os.path.basename(flow_mp4)}")
except Exception as e:
    print(f"[FLOW] MP4 failed ({e}), trying GIF...")
    try:
        from matplotlib.animation import PillowWriter
        flow_gif = os.path.join(OUT, f"flow_{DATE_STR}.gif")
        anim_flow.save(flow_gif, writer=PillowWriter(fps=FLOW_FPS), dpi=100)
        print(f"[FLOW] Saved: {os.path.basename(flow_gif)}")
        flow_saved = True
    except Exception:
        print("[FLOW] WARNING: Could not save animation")

plt.close(fig_fa)

# =============================================================================
# === HEAT SIMULATION ===
# =============================================================================
alpha_th = K_COND / (RHO * CP)
# Stability: dt < dx_phys^2 / (6*alpha) for 3D explicit
dt_heat = 0.4 * DX_PHYS * DX_PHYS / (6.0 * alpha_th)

print(f"[HEAT] Running transient FD solver | "
      f"dt={dt_heat:.4f}s | {N_HEAT_STEPS} steps...")
t0 = time.time()

# Solid-only Laplacian: only sum contributions from solid neighbours,
# implementing zero-flux (Neumann) BC at solid-void interfaces.
def _solid_laplacian(T_field, smask):
    """Vectorized Laplacian counting only solid neighbours."""
    # Work with NaN-free copy (void = 0, won't be used)
    T_safe = np.where(smask, T_field, 0.0)
    lap = np.zeros_like(T_field)
    nx, ny, nz = T_field.shape
    # +x / -x
    lap[:-1] += np.where(smask[:-1] & smask[1:],  T_safe[1:] - T_safe[:-1], 0)
    lap[1:]  += np.where(smask[1:]  & smask[:-1], T_safe[:-1] - T_safe[1:], 0)
    # +y / -y
    lap[:, :-1] += np.where(smask[:, :-1] & smask[:, 1:],
                             T_safe[:, 1:] - T_safe[:, :-1], 0)
    lap[:, 1:]  += np.where(smask[:, 1:]  & smask[:, :-1],
                             T_safe[:, :-1] - T_safe[:, 1:], 0)
    # +z / -z
    lap[:, :, :-1] += np.where(smask[:, :, :-1] & smask[:, :, 1:],
                                T_safe[:, :, 1:] - T_safe[:, :, :-1], 0)
    lap[:, :, 1:]  += np.where(smask[:, :, 1:]  & smask[:, :, :-1],
                                T_safe[:, :, :-1] - T_safe[:, :, 1:], 0)
    return lap / (DX_PHYS * DX_PHYS)

# Temperature field — initialise to T_COLD everywhere
T = np.full((VOXEL_RES, VOXEL_RES, VOXEL_RES), np.nan)
T[solid_mask] = T_COLD

# Apply BCs
T[0, :, :] = np.where(solid_mask[0, :, :], T_HOT, np.nan)      # x=0 hot
T[-1, :, :] = np.where(solid_mask[-1, :, :], T_COLD, np.nan)    # x=L cold

# Mask for updatable voxels (solid, not on BC faces)
update_mask = solid_mask.copy()
update_mask[0, :, :] = False
update_mask[-1, :, :] = False

heat_snapshots = []
heat_times = []

for step in range(1, N_HEAT_STEPS + 1):
    lap_T = _solid_laplacian(T, solid_mask)

    T_new = T.copy()
    T_new[update_mask] = T[update_mask] + alpha_th * dt_heat * lap_T[update_mask]

    # Re-enforce BCs
    T_new[0, :, :] = np.where(solid_mask[0, :, :], T_HOT, np.nan)
    T_new[-1, :, :] = np.where(solid_mask[-1, :, :], T_COLD, np.nan)
    T_new[void_mask] = np.nan

    T = T_new

    if step % HEAT_SAVE_EVERY == 0:
        heat_snapshots.append(T.copy())
        heat_times.append(step * dt_heat)

    if step % 2000 == 0 or step == N_HEAT_STEPS:
        T_solid_vals = T[solid_mask & ~np.isnan(T)]
        T_mean = float(T_solid_vals.mean()) if len(T_solid_vals) > 0 else 0
        elapsed = time.time() - t0
        print(f"[HEAT] Step {step}/{N_HEAT_STEPS} | "
              f"T_mean_solid={T_mean:.1f}°C | "
              f"t_phys={step*dt_heat:.0f}s | {elapsed:.1f}s elapsed",
              flush=True)

# =============================================================================
# === HEAT VISUALIZATION ===
# =============================================================================
norm_heat = Normalize(vmin=T_COLD, vmax=T_HOT)

# Slice indices
iz_mid = VOXEL_RES // 2        # z = L/2
iy_mid = VOXEL_RES // 2        # y = L/2
ix_qtr = VOXEL_RES // 4        # x = L/4

# Coordinate arrays for slice plotting
xy_x, xy_y = np.meshgrid(x_1d, x_1d, indexing="ij")
xz_x, xz_z = np.meshgrid(x_1d, x_1d, indexing="ij")
yz_y, yz_z = np.meshgrid(x_1d, x_1d, indexing="ij")


def _draw_heat_slices(ax, T_field):
    """Draw three orthogonal temperature slice planes."""
    plots = []

    # Slice 1: XY at z=L/2
    T_xy = T_field[:, :, iz_mid].copy()
    T_xy[~solid_mask[:, :, iz_mid]] = np.nan
    mask_xy = ~np.isnan(T_xy)
    if mask_xy.any():
        xs = xy_x[mask_xy]; ys = xy_y[mask_xy]
        zs = np.full_like(xs, x_1d[iz_mid])
        cs = T_xy[mask_xy]
        sc = ax.scatter(xs, ys, zs, c=cs, cmap=kim_cmap, norm=norm_heat,
                         s=10, alpha=0.85, edgecolors="none", zorder=5)
        plots.append(sc)

    # Slice 2: XZ at y=L/2
    T_xz = T_field[:, iy_mid, :].copy()
    T_xz[~solid_mask[:, iy_mid, :]] = np.nan
    mask_xz = ~np.isnan(T_xz)
    if mask_xz.any():
        xs = xz_x[mask_xz]; zs = xz_z[mask_xz]
        ys = np.full_like(xs, x_1d[iy_mid])
        cs = T_xz[mask_xz]
        sc = ax.scatter(xs, ys, zs, c=cs, cmap=kim_cmap, norm=norm_heat,
                         s=10, alpha=0.85, edgecolors="none", zorder=5)
        plots.append(sc)

    # Slice 3: YZ at x=L/4
    T_yz = T_field[ix_qtr, :, :].copy()
    T_yz[~solid_mask[ix_qtr, :, :]] = np.nan
    mask_yz = ~np.isnan(T_yz)
    if mask_yz.any():
        ys = yz_y[mask_yz]; zs = yz_z[mask_yz]
        xs = np.full_like(ys, x_1d[ix_qtr])
        cs = T_yz[mask_yz]
        sc = ax.scatter(xs, ys, zs, c=cs, cmap=kim_cmap, norm=norm_heat,
                         s=10, alpha=0.85, edgecolors="none", zorder=5)
        plots.append(sc)

    return plots


# --- Static interactive window (final state) ---
fig_heat = plt.figure("Heat Simulation", figsize=(14, 10))
ax_heat = fig_heat.add_subplot(111, projection="3d")
_draw_shell(ax_heat)
_draw_heat_slices(ax_heat, heat_snapshots[-1])
ax_heat.set_xlim(0, L); ax_heat.set_ylim(0, L); ax_heat.set_zlim(0, L)
ax_heat.set_xlabel("X"); ax_heat.set_ylabel("Y"); ax_heat.set_zlabel("Z")
ax_heat.view_init(elev=ELEV, azim=AZIM)
ax_heat.set_title(
    f"TPMS Heat Conduction — Schwartz P | KimSurface {DATE_STR}",
    fontsize=13, fontweight="bold",
)

sm_heat = plt.cm.ScalarMappable(cmap=kim_cmap, norm=norm_heat)
sm_heat.set_array([])
cb_heat = fig_heat.colorbar(sm_heat, ax=ax_heat, shrink=0.6, pad=0.08)
cb_heat.set_label("Temperature (°C)", fontsize=10)

fig_heat.tight_layout()
heat_png = os.path.join(OUT, f"heat_static_{DATE_STR}.png")
fig_heat.savefig(heat_png, dpi=150, bbox_inches="tight")
print(f"[HEAT] Interactive window open | Exporting MP4...")
plt.show(block=False)
plt.pause(0.1)

# --- Heat MP4 animation ---
fig_ha = plt.figure("Heat Animation", figsize=(14, 10))
ax_ha = fig_ha.add_subplot(111, projection="3d")
ax_ha.view_init(elev=ELEV, azim=AZIM)
ax_ha.set_xlim(0, L); ax_ha.set_ylim(0, L); ax_ha.set_zlim(0, L)
ax_ha.set_xlabel("X"); ax_ha.set_ylabel("Y"); ax_ha.set_zlabel("Z")

# Static shell
_draw_shell(ax_ha)

# Colorbar
sm_ha = plt.cm.ScalarMappable(cmap=kim_cmap, norm=norm_heat)
sm_ha.set_array([])
cb_ha = fig_ha.colorbar(sm_ha, ax=ax_ha, shrink=0.6, pad=0.08)
cb_ha.set_label("Temperature (°C)", fontsize=10)

_heat_scatter_plots = []


def _heat_update(frame):
    for sc in _heat_scatter_plots:
        sc.remove()
    _heat_scatter_plots.clear()

    T_field = heat_snapshots[frame]
    t_sec = heat_times[frame]

    plots = _draw_heat_slices(ax_ha, T_field)
    _heat_scatter_plots.extend(plots)

    ax_ha.set_title(
        f"TPMS Heat Conduction — Schwartz P | KimSurface {DATE_STR}\n"
        f"t = {t_sec:.1f}s | Heat front propagating...",
        fontsize=12, fontweight="bold",
    )
    return []


n_heat_frames = len(heat_snapshots)
anim_heat = FuncAnimation(fig_ha, _heat_update, frames=n_heat_frames,
                           interval=1000 // HEAT_FPS, blit=False)

heat_mp4 = os.path.join(OUT, f"heat_{DATE_STR}.mp4")
try:
    from matplotlib.animation import FFMpegWriter
    anim_heat.save(heat_mp4, writer=FFMpegWriter(fps=HEAT_FPS), dpi=120)
    print(f"[HEAT] Saved: {os.path.basename(heat_mp4)}")
except Exception as e:
    print(f"[HEAT] MP4 failed ({e}), trying GIF...")
    try:
        from matplotlib.animation import PillowWriter
        heat_gif = os.path.join(OUT, f"heat_{DATE_STR}.gif")
        anim_heat.save(heat_gif, writer=PillowWriter(fps=HEAT_FPS), dpi=100)
        print(f"[HEAT] Saved: {os.path.basename(heat_gif)}")
    except Exception:
        print("[HEAT] WARNING: Could not save animation")

plt.close(fig_ha)

# =============================================================================
# === SUMMARY ===
# =============================================================================
print(f"\n{'='*60}")
print(f"  [DONE] All outputs saved.")
print(f"{'='*60}")
print(f"  tpms_solid_{DATE_STR}.stl")
print(f"  flow_static_{DATE_STR}.png")
print(f"  flow_{DATE_STR}.mp4")
print(f"  heat_static_{DATE_STR}.png")
print(f"  heat_{DATE_STR}.mp4")
print(f"{'='*60}")

# Keep interactive windows open
plt.show()
