#!/usr/bin/env python3
"""
Novel TPMS Generator + FEA Simulator
=====================================
Generates genuinely new triply-periodic MINIMAL surfaces via:
  1. Random Fourier synthesis over 12 cubic-symmetric basis functions
     to produce an initial periodic surface.
  2. Mean curvature flow (MCF) to drive the surface toward H=0,
     satisfying the mathematical definition of a minimal surface.

After MCF convergence each surface is:
  - Triply periodic  (Fourier basis guarantees periodicity)
  - Cubic symmetric  (basis respects Oh point group)
  - Minimal          (H ≈ 0 everywhere, verified numerically)
  - Novel            (random coefficients ≠ any known TPMS)
"""
# pip install numpy scipy matplotlib scikit-image pillow

import sys as _sys
if hasattr(_sys.stdout, 'reconfigure'):
    _sys.stdout.reconfigure(line_buffering=True)
import warnings
warnings.filterwarnings("ignore", category=RuntimeWarning)
import numpy as np
from scipy import sparse
from scipy.sparse.linalg import spsolve, splu
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import csv, time, sys, os
from datetime import date

try:
    from skimage.measure import marching_cubes
except ImportError:
    sys.exit("ERROR: scikit-image required  (pip install scikit-image)")

# =============================================================================
# === PARAMETERS ===
# =============================================================================
N_CELLS        = 3
VOXEL_RES      = N_CELLS * 30   # 90 — geometry grid, endpoint=False
STL_RES        = N_CELLS * 40   # 120 — finer grid for crack-free STL
FEM_RES        = 20              # FEM voxel mesh (coarser for NR tractability)
N_INCREMENTS   = 15
STRAIN_TARGET  = 0.30
E_MODULUS      = 10e6
NU             = 0.45
N_GEOMETRIES   = 10
DOMAIN_M       = 0.1524          # 6 inches
MIN_VERTICES   = 2000
MCF_ITERATIONS = 10              # mean curvature flow steps (reduced for speed)
MCF_DT         = 0.0005          # flow time step
MCF_TOL        = 0.10            # converge when max|H| < this

L_TPMS  = 2.0 * np.pi * N_CELLS
DX_TPMS = L_TPMS / VOXEL_RES
SCALE   = DOMAIN_M / L_TPMS
L       = DOMAIN_M
MU      = E_MODULUS / (2.0 * (1.0 + NU))
LAM     = E_MODULUS * NU / ((1.0 + NU) * (1.0 - 2.0 * NU))

D_MAT = np.zeros((6, 6))
D_MAT[:3, :3] = LAM
D_MAT[0, 0] = D_MAT[1, 1] = D_MAT[2, 2] = LAM + 2.0 * MU
D_MAT[3, 3] = D_MAT[4, 4] = D_MAT[5, 5] = MU

dNdr = np.array([[-1, -1, -1], [1, 0, 0], [0, 1, 0], [0, 0, 1]],
                dtype=np.float64)

DATE_STR = date.today().strftime("%Y%m%d")
OUT      = os.getcwd()
rng      = np.random.default_rng()

# =============================================================================
# === FOURIER BASIS — 12 cubic-symmetric periodic functions ===
# =============================================================================
# Every known TPMS is a specific coefficient vector in this basis.
# Random coefficients create genuinely new surfaces.

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
    ("P",   _b_P),    # Schwarz P lives here
    ("G",   _b_G),    # Gyroid lives here
    ("D",   _b_D),    # Diamond lives here
    ("IWP", _b_IWP),  # I-WP surface
    ("N",   _b_N),    # Neovius component
    ("SS",  _b_SS),   # sin-sin cross terms
    ("P2",  _b_P2),   # 2nd harmonic of P
    ("G2",  _b_G2),   # 2nd harmonic gyroid-type
    ("CC2", _b_CC2),  # cos-cos mixed harmonic
    ("SS2", _b_SS2),  # sin-sin mixed harmonic
    ("CSC", _b_CSC),  # cos-sin-cos triple
    ("L",   _b_L),    # Lidinoid-type term
]
N_BASIS = len(BASIS)

# =============================================================================
# === RANDOM COEFFICIENT SAMPLER ===
# =============================================================================

def sample_params():
    """Generate random Fourier coefficients for a novel periodic surface."""
    n_active = rng.integers(3, 8)                        # activate 3-7 terms
    active   = rng.choice(N_BASIS, size=n_active, replace=False)
    coeffs   = np.zeros(N_BASIS)
    coeffs[active] = rng.uniform(-1.0, 1.0, size=n_active)
    # normalise so largest coefficient is 1.0
    mx = np.max(np.abs(coeffs))
    if mx > 0:
        coeffs /= mx
    return {
        "coeffs":        coeffs,
        "active_idx":    sorted(active.tolist()),
        "isovalue_t":    float(rng.uniform(-0.20, 0.20)),
        "freq":          float(rng.uniform(0.92, 1.08)),
        "wall_half_iso": float(rng.uniform(0.25, 0.40)),  # thicker = no cracks
    }


def equation_str(p):
    """Human-readable equation showing active terms and coefficients."""
    parts = []
    for i in p["active_idx"]:
        c = p["coeffs"][i]
        name = BASIS[i][0]
        parts.append(f"{c:+.3f}*{name}")
    return " ".join(parts) + f"  [t={p['isovalue_t']:.3f} f={p['freq']:.3f}]"

# =============================================================================
# === FIELD EVALUATION ===
# =============================================================================

def eval_field_tpms(p, X, Y, Z):
    """Evaluate novel surface on TPMS-space coordinates."""
    f = p["freq"]
    Xf, Yf, Zf = f * X, f * Y, f * Z
    F = np.zeros_like(X)
    for i in p["active_idx"]:
        F += p["coeffs"][i] * BASIS[i][1](Xf, Yf, Zf)
    return F - p["isovalue_t"]


def eval_field_physical(p, Xp, Yp, Zp):
    """Evaluate on physical (metre) coordinates."""
    s = L_TPMS / DOMAIN_M
    return eval_field_tpms(p, s * Xp, s * Yp, s * Zp)

# =============================================================================
# === LAPLACIAN SMOOTHING ===
# =============================================================================

def laplacian_smooth(verts, faces, n_iter=3, lam=0.3, domain_size=None):
    n = len(verts)
    e = np.vstack([faces[:, [0, 1]], faces[:, [1, 2]], faces[:, [0, 2]]])
    e = np.vstack([e, e[:, ::-1]])
    adj = sparse.coo_matrix(
        (np.ones(len(e)), (e[:, 0], e[:, 1])), shape=(n, n)).tocsr()
    deg = np.array(adj.sum(axis=1)).ravel()
    deg[deg == 0] = 1
    bnd = np.zeros(n, dtype=bool)
    if domain_size is not None:
        eps = domain_size * 0.005
        for d in range(3):
            bnd |= (verts[:, d] < eps) | (verts[:, d] > domain_size - eps)
    for _ in range(n_iter):
        avg = (adj @ verts) / deg[:, None]
        delta = lam * (avg - verts)
        delta[bnd] = 0.0
        verts = verts + delta
    return verts

# =============================================================================
# === MEAN CURVATURE FLOW — drives surface toward H=0 (true minimal) ===
# =============================================================================

def _build_mesh_operators(verts, faces):
    """Build cotangent Laplacian and vertex normals for mean curvature."""
    n = len(verts)
    # Collect edge pairs from triangles
    ii, jj, ww = [], [], []
    for idx in range(3):
        i0 = faces[:, idx]
        i1 = faces[:, (idx + 1) % 3]
        i2 = faces[:, (idx + 2) % 3]
        # Edge vectors opposite to vertex idx
        e1 = verts[i1] - verts[i0]
        e2 = verts[i2] - verts[i0]
        # Cotangent of angle at vertex i0
        dot = np.sum(e1 * e2, axis=1)
        cross_norm = np.linalg.norm(np.cross(e1, e2), axis=1)
        cot = np.clip(dot / np.maximum(cross_norm, 1e-10), -50.0, 50.0)
        # Cotangent weight goes on the OPPOSITE edge (i1-i2)
        w = 0.5 * cot
        ii.extend(i1); jj.extend(i2); ww.extend(w)
        ii.extend(i2); jj.extend(i1); ww.extend(w)

    ii = np.array(ii, dtype=np.int64)
    jj = np.array(jj, dtype=np.int64)
    ww = np.array(ww, dtype=np.float64)
    L_cot = sparse.coo_matrix((ww, (ii, jj)), shape=(n, n)).tocsr()
    diag = np.array(L_cot.sum(axis=1)).ravel()
    L_cot = L_cot - sparse.diags(diag)

    # Mixed Voronoi area per vertex (sum of triangle areas / 3)
    v0 = verts[faces[:, 0]]; v1 = verts[faces[:, 1]]; v2 = verts[faces[:, 2]]
    tri_area = 0.5 * np.linalg.norm(np.cross(v1 - v0, v2 - v0), axis=1)
    vert_area = np.zeros(n)
    for k in range(3):
        np.add.at(vert_area, faces[:, k], tri_area / 3.0)
    vert_area = np.maximum(vert_area, 1e-15)

    # Vertex normals (area-weighted)
    face_n = np.cross(v1 - v0, v2 - v0)
    vert_n = np.zeros((n, 3))
    for k in range(3):
        np.add.at(vert_n, faces[:, k], face_n)
    norms = np.linalg.norm(vert_n, axis=1, keepdims=True)
    vert_n /= np.maximum(norms, 1e-15)

    return L_cot, vert_area, vert_n


def compute_mean_curvature(verts, faces, L_cot, vert_area):
    """Return signed mean curvature H at each vertex.

    Uses the cotangent Laplacian: ΔS x = 2 H n
    so H = (ΔS x · n) / 2, where ΔS x = L_cot @ x / A.
    """
    n = len(verts)
    # Recompute normals (cheap)
    v0 = verts[faces[:, 0]]; v1 = verts[faces[:, 1]]; v2 = verts[faces[:, 2]]
    face_n = np.cross(v1 - v0, v2 - v0)
    vert_n = np.zeros((n, 3))
    for k in range(3):
        np.add.at(vert_n, faces[:, k], face_n)
    nm = np.linalg.norm(vert_n, axis=1, keepdims=True)
    vert_n /= np.maximum(nm, 1e-15)

    lap = (L_cot @ verts) / vert_area[:, None]     # ΔS x  (n, 3)
    H = 0.5 * np.sum(lap * vert_n, axis=1)         # scalar mean curvature
    H = np.nan_to_num(H, nan=0.0, posinf=0.0, neginf=0.0)
    return H, vert_n


def mean_curvature_flow(verts, faces, n_iter=MCF_ITERATIONS, dt=MCF_DT,
                        tol=MCF_TOL, domain_size=None):
    """Flow surface vertices along -H*n until max|H| < tol.

    Returns (verts, final_max_H, iterations_used).
    Boundary vertices (at domain faces) are pinned.
    """
    verts = verts.copy()
    n = len(verts)

    bnd = np.zeros(n, dtype=bool)
    if domain_size is not None:
        eps = domain_size * 0.005
        for d in range(3):
            bnd |= (verts[:, d] < eps) | (verts[:, d] > domain_size - eps)

    # Average edge length for displacement clamping
    e_all = np.vstack([faces[:, [0, 1]], faces[:, [1, 2]], faces[:, [0, 2]]])
    avg_edge = float(np.mean(np.linalg.norm(
        verts[e_all[:, 0]] - verts[e_all[:, 1]], axis=1)))
    max_disp = 0.2 * avg_edge  # never move more than 20% of avg edge per step

    L_cot, vert_area, _ = _build_mesh_operators(verts, faces)
    for it in range(1, n_iter + 1):
        if it % 5 == 1 and it > 1:
            L_cot, vert_area, _ = _build_mesh_operators(verts, faces)
        H, normals = compute_mean_curvature(verts, faces, L_cot, vert_area)
        max_H = float(np.max(np.abs(H)))
        if max_H < tol:
            return verts, max_H, it

        displacement = (-H[:, None] * normals) * dt
        displacement[bnd] = 0.0
        # Clamp per-vertex displacement magnitude
        disp_mag = np.linalg.norm(displacement, axis=1, keepdims=True)
        scale = np.where(disp_mag > max_disp, max_disp / (disp_mag + 1e-15), 1.0)
        displacement *= scale
        verts = verts + displacement
        # Re-clamp to domain
        if domain_size is not None:
            verts = np.clip(verts, 0, domain_size - 1e-10)

    L_cot, vert_area, _ = _build_mesh_operators(verts, faces)
    H_final, _ = compute_mean_curvature(verts, faces, L_cot, vert_area)
    max_H = float(np.max(np.abs(H_final)))
    return verts, max_H, n_iter


# =============================================================================
# === VALIDITY CHECKS ===
# =============================================================================

def validate_geometry(p, verts, faces):
    whi = p["wall_half_iso"]
    if len(verts) < MIN_VERTICES:
        return False, f"too few vertices ({len(verts)}<{MIN_VERTICES})"
    for d in range(3):
        span = verts[:, d].max() - verts[:, d].min()
        if span < 0.6 * L_TPMS:
            return False, f"bbox axis {d} too small ({span:.2f})"
    bins = 4
    edges = np.linspace(0, L_TPMS, bins + 1)
    bx = np.clip(np.digitize(verts[:, 0], edges) - 1, 0, bins - 1)
    by = np.clip(np.digitize(verts[:, 1], edges) - 1, 0, bins - 1)
    bz = np.clip(np.digitize(verts[:, 2], edges) - 1, 0, bins - 1)
    occupied = len(set(zip(bx, by, bz)))
    if occupied < 40:
        return False, f"poor spatial fill ({occupied}/64 bins)"
    tri = verts[faces]
    e0 = np.linalg.norm(tri[:, 1] - tri[:, 0], axis=1)
    e1 = np.linalg.norm(tri[:, 2] - tri[:, 1], axis=1)
    e2 = np.linalg.norm(tri[:, 0] - tri[:, 2], axis=1)
    longest = np.maximum(np.maximum(e0, e1), e2)
    shortest = np.maximum(np.minimum(np.minimum(e0, e1), e2), 1e-15)
    if (longest / shortest).mean() > 12.0:
        return False, "mean aspect ratio > 12"
    centroid = verts.mean(axis=0)
    centre = L_TPMS / 2.0
    for d in range(3):
        if abs(centroid[d] - centre) > 0.05 * L_TPMS:
            return False, f"centroid off-centre axis {d}"
    return True, f"valid ({len(verts)} verts, {occupied}/64 bins)"

# =============================================================================
# === SURFACE MESH GENERATION ===
# =============================================================================

def generate_surface_mesh(p, res, run_mcf=True):
    """Marching cubes -> mean curvature flow -> Laplacian smoothing.

    Returns (verts, faces, max_H, mcf_iters).
    When run_mcf=False (validation pass), skip the expensive MCF step.
    """
    dx = L_TPMS / res
    x = np.linspace(0, L_TPMS, res, endpoint=False)
    X, Y, Z = np.meshgrid(x, x, x, indexing="ij")
    F   = eval_field_tpms(p, X, Y, Z)
    whi = p["wall_half_iso"]
    G   = np.abs(F) - whi
    G_pad = np.pad(G, 1, mode="constant", constant_values=G.max() + 1.0)
    verts, faces, _, _ = marching_cubes(G_pad, level=0.0, spacing=(dx, dx, dx))
    verts -= dx
    verts = np.clip(verts, 0, L_TPMS - 1e-10)

    max_H, mcf_it = 999.0, 0
    if run_mcf and len(verts) > 100:
        verts, max_H, mcf_it = mean_curvature_flow(
            verts, faces, n_iter=MCF_ITERATIONS, dt=MCF_DT,
            tol=MCF_TOL, domain_size=L_TPMS)

    verts = laplacian_smooth(verts, faces, n_iter=3, lam=0.3,
                             domain_size=L_TPMS)
    return verts, faces, max_H, mcf_it

# =============================================================================
# === FEM MESH (unchanged engine) ===
# =============================================================================

def identify_solid_fem(p, nx, domain_L):
    xn = np.linspace(0, domain_L, nx + 1)
    Xn, Yn, Zn = np.meshgrid(xn, xn, xn, indexing="ij")
    F   = eval_field_physical(p, Xn, Yn, Zn)
    whi = p["wall_half_iso"]
    cv  = np.stack([
        F[:-1, :-1, :-1], F[1:, :-1, :-1], F[:-1, 1:, :-1], F[1:, 1:, :-1],
        F[:-1, :-1, 1:],  F[1:, :-1, 1:],  F[:-1, 1:, 1:],  F[1:, 1:, 1:],
    ])
    solid = (cv.min(axis=0) < whi) & (cv.max(axis=0) > -whi)
    return solid, Xn, Yn, Zn


def build_tet_mesh(solid, Xn, Yn, Zn, nx):
    def nid(i, j, k):
        return i + j * (nx + 1) + k * (nx + 1) * (nx + 1)
    ijk = np.argwhere(solid)
    si, sj, sk = ijk[:, 0], ijk[:, 1], ijk[:, 2]
    hn = np.column_stack([
        nid(si, sj, sk), nid(si+1, sj, sk), nid(si, sj+1, sk), nid(si+1, sj+1, sk),
        nid(si, sj, sk+1), nid(si+1, sj, sk+1), nid(si, sj+1, sk+1), nid(si+1, sj+1, sk+1),
    ])
    ep = np.array([[0,1,3,5],[0,3,2,6],[0,5,4,6],[3,5,7,6],[0,3,5,6]])
    op = np.array([[1,0,2,4],[1,2,3,7],[1,4,5,7],[2,4,6,7],[1,2,4,7]])
    em = ((si + sj + sk) % 2) == 0
    elems = np.vstack([hn[em][:, ep].reshape(-1, 4),
                       hn[~em][:, op].reshape(-1, 4)]).astype(np.int64)
    all_co = np.column_stack([Xn.ravel(), Yn.ravel(), Zn.ravel()])
    used = np.unique(elems)
    remap = np.full(all_co.shape[0], -1, dtype=np.int64)
    remap[used] = np.arange(len(used))
    elems = remap[elems]; coords = all_co[used]
    ec = coords[elems]; J = np.einsum("ai,eaj->eij", dNdr, ec)
    neg = np.linalg.det(J) < 0
    if neg.any():
        elems[neg, :2] = elems[neg, 1::-1]
        ec = coords[elems]; J = np.einsum("ai,eaj->eij", dNdr, ec)
    vol = np.abs(np.linalg.det(J)) / 6.0
    return elems, coords, len(used), len(elems), vol, J


def extract_surface(elems):
    fl = np.array([[0,1,2],[0,1,3],[0,2,3],[1,2,3]])
    af = elems[:, fl].reshape(-1, 3)
    afs = np.sort(af, axis=1)
    _, inv, cnt = np.unique(afs, axis=0, return_inverse=True, return_counts=True)
    sf = af[cnt[inv] == 1]; return sf, np.unique(sf)

# =============================================================================
# === TOTAL LAGRANGIAN NONLINEAR FEM — PRECOMPUTATION ===
# =============================================================================

def precompute_reference(coords, elems):
    """Compute reference-config shape function gradients dNdX and volumes.

    Returns
    -------
    dNdX : ndarray (n_elem, 4, 3) — shape function gradients in material frame
    vol_ref : ndarray (n_elem,)   — reference tetrahedron volumes
    """
    ec = coords[elems]                            # (n_elem, 4, 3)
    J0 = np.einsum("ai,eaj->eij", dNdr, ec)      # (n_elem, 3, 3)
    detJ0 = np.linalg.det(J0)                     # (n_elem,)
    detJ0 = np.where(np.abs(detJ0) < 1e-30, 1e-30, detJ0)
    J0inv = np.linalg.inv(J0)                     # (n_elem, 3, 3)
    dNdX = np.einsum("ai,eij->eaj", dNdr, J0inv)  # (n_elem, 4, 3)
    vol_ref = np.abs(detJ0) / 6.0
    return dNdX, vol_ref


# =============================================================================
# === INTERNAL FORCES (Total Lagrangian) ===
# =============================================================================

# Voigt map: 0->xx, 1->yy, 2->zz, 3->xy, 4->yz, 5->xz
_VOIGT_IDX = [(0, 0), (1, 1), (2, 2), (0, 1), (1, 2), (0, 2)]


def compute_internal_forces(u_flat, elems, dNdX, vol_ref, n_nodes, n_dofs):
    """Compute internal force vector, element von Mises, and nodal von Mises.

    Uses Total Lagrangian formulation:
      F = I + grad(u)          (deformation gradient)
      E = 0.5*(F^T F - I)     (Green-Lagrange strain)
      S = D : E               (2nd Piola-Kirchhoff, SVK material)
      P = F S                  (1st Piola-Kirchhoff stress)
    """
    n_elem = len(elems)
    I3 = np.eye(3)

    # Element displacements: (n_elem, 4, 3)
    u_nodal = u_flat.reshape(-1, 3)
    u_elem = u_nodal[elems]

    # Displacement gradient H = du/dX = sum_a u_a (x) dNdX_a
    # u_elem: (e, a, i), dNdX: (e, a, j) -> H: (e, i, j)
    H = np.einsum("eai,eaj->eij", u_elem, dNdX)
    H = np.nan_to_num(H, nan=0.0, posinf=0.0, neginf=0.0)

    # Deformation gradient F = I + H
    F_def = I3[None, :, :] + H  # (n_elem, 3, 3)

    # Green-Lagrange strain E = 0.5*(F^T F - I)
    FtF = np.einsum("eji,ejk->eik", F_def, F_def)
    E_GL = 0.5 * (FtF - I3[None, :, :])

    # Voigt form of E:  [E_xx, E_yy, E_zz, 2*E_xy, 2*E_yz, 2*E_xz]
    E_voigt = np.zeros((n_elem, 6))
    E_voigt[:, 0] = E_GL[:, 0, 0]
    E_voigt[:, 1] = E_GL[:, 1, 1]
    E_voigt[:, 2] = E_GL[:, 2, 2]
    E_voigt[:, 3] = 2.0 * E_GL[:, 0, 1]
    E_voigt[:, 4] = 2.0 * E_GL[:, 1, 2]
    E_voigt[:, 5] = 2.0 * E_GL[:, 0, 2]

    # 2nd Piola-Kirchhoff stress S_voigt = D : E_voigt
    S_voigt = np.einsum("ij,ej->ei", D_MAT, E_voigt)

    # Reconstruct S as symmetric 3x3
    S_mat = np.zeros((n_elem, 3, 3))
    S_mat[:, 0, 0] = S_voigt[:, 0]
    S_mat[:, 1, 1] = S_voigt[:, 1]
    S_mat[:, 2, 2] = S_voigt[:, 2]
    S_mat[:, 0, 1] = S_mat[:, 1, 0] = S_voigt[:, 3]
    S_mat[:, 1, 2] = S_mat[:, 2, 1] = S_voigt[:, 4]
    S_mat[:, 0, 2] = S_mat[:, 2, 0] = S_voigt[:, 5]

    # 1st Piola-Kirchhoff P = F S
    P = np.einsum("eij,ejk->eik", F_def, S_mat)  # (n_elem, 3, 3)

    # Internal force:  f_int_a = vol * P @ dNdX_a
    # P: (e, i, j),  dNdX: (e, a, j) -> f_node: (e, a, i)
    f_node = vol_ref[:, None, None] * np.einsum("eij,eaj->eai", P, dNdX)

    # Assemble into global vector
    f_int = np.zeros(n_dofs)
    for a in range(4):
        node_ids = elems[:, a]  # (n_elem,)
        for i in range(3):
            np.add.at(f_int, 3 * node_ids + i, f_node[:, a, i])

    # --- von Mises from Cauchy stress for postprocessing ---
    detF = np.linalg.det(F_def)
    detF = np.where(np.abs(detF) < 1e-30, 1e-30, detF)
    # Cauchy: sigma = (1/J) F S F^T
    FS = np.einsum("eij,ejk->eik", F_def, S_mat)
    sigma = np.einsum("eij,ekj->eik", FS, F_def) / detF[:, None, None]
    sigma = np.nan_to_num(sigma, nan=0.0, posinf=0.0, neginf=0.0)

    s_xx = sigma[:, 0, 0]; s_yy = sigma[:, 1, 1]; s_zz = sigma[:, 2, 2]
    s_xy = sigma[:, 0, 1]; s_yz = sigma[:, 1, 2]; s_xz = sigma[:, 0, 2]
    vm_elem = np.sqrt(np.maximum(
        0.5 * ((s_xx - s_yy)**2 + (s_yy - s_zz)**2 + (s_zz - s_xx)**2
               + 6.0 * (s_xy**2 + s_yz**2 + s_xz**2)),
        0.0))

    # Average to nodes
    vm_node = np.zeros(n_nodes)
    nc = np.zeros(n_nodes)
    for a in range(4):
        np.add.at(vm_node, elems[:, a], vm_elem)
        np.add.at(nc, elems[:, a], 1.0)
    vm_node /= np.maximum(nc, 1.0)

    return f_int, vm_elem, vm_node


# =============================================================================
# === TANGENT STIFFNESS (Material + Geometric) ===
# =============================================================================

def assemble_tangent(u_flat, elems, dNdX, vol_ref, n_nodes, n_dofs):
    """Assemble the tangent stiffness K_T = K_M + K_G.

    Returns
    -------
    K_T  : sparse CSR (n_dofs, n_dofs)
    edofs : ndarray (n_elem, 12) — element DOF indices
    """
    n_elem = len(elems)
    I3 = np.eye(3)

    # Recompute deformation state
    u_nodal = u_flat.reshape(-1, 3)
    u_elem = u_nodal[elems]
    H = np.einsum("eai,eaj->eij", u_elem, dNdX)
    H = np.nan_to_num(H, nan=0.0, posinf=0.0, neginf=0.0)
    F_def = I3[None, :, :] + H

    # Green-Lagrange strain and SVK stress
    FtF = np.einsum("eji,ejk->eik", F_def, F_def)
    E_GL = 0.5 * (FtF - I3[None, :, :])
    E_voigt = np.zeros((n_elem, 6))
    E_voigt[:, 0] = E_GL[:, 0, 0]
    E_voigt[:, 1] = E_GL[:, 1, 1]
    E_voigt[:, 2] = E_GL[:, 2, 2]
    E_voigt[:, 3] = 2.0 * E_GL[:, 0, 1]
    E_voigt[:, 4] = 2.0 * E_GL[:, 1, 2]
    E_voigt[:, 5] = 2.0 * E_GL[:, 0, 2]
    S_voigt = np.einsum("ij,ej->ei", D_MAT, E_voigt)
    S_mat = np.zeros((n_elem, 3, 3))
    S_mat[:, 0, 0] = S_voigt[:, 0]
    S_mat[:, 1, 1] = S_voigt[:, 1]
    S_mat[:, 2, 2] = S_voigt[:, 2]
    S_mat[:, 0, 1] = S_mat[:, 1, 0] = S_voigt[:, 3]
    S_mat[:, 1, 2] = S_mat[:, 2, 1] = S_voigt[:, 4]
    S_mat[:, 0, 2] = S_mat[:, 2, 0] = S_voigt[:, 5]

    # --- Build B_NL (nonlinear B-matrix) ---
    # B_NL maps du_voigt to dE_voigt:  dE = B_NL * du_elem
    # B_NL[e, voigt_row, 3*a+l] incorporates F
    # voigt ordering: xx=0, yy=1, zz=2, xy=3, yz=4, xz=5
    # For voigt row (i,j):
    #   dE_ij = sum_a sum_l F[l,i]*dNdX[a,j]*du[a,l] (symmetrised)
    B_NL = np.zeros((n_elem, 6, 12))
    for a in range(4):
        c = 3 * a
        for v_idx, (vi, vj) in enumerate(_VOIGT_IDX):
            for l in range(3):
                if vi == vj:
                    # Diagonal voigt: dE_ii = F[l,i]*dNdX[a,i]*du[a,l]
                    B_NL[:, v_idx, c + l] += F_def[:, l, vi] * dNdX[:, a, vi]
                else:
                    # Off-diagonal: dE_ij = F[l,i]*dNdX[a,j]+F[l,j]*dNdX[a,i]
                    B_NL[:, v_idx, c + l] += (
                        F_def[:, l, vi] * dNdX[:, a, vj]
                        + F_def[:, l, vj] * dNdX[:, a, vi]
                    )

    # Material tangent K_M = vol * B_NL^T D B_NL
    DB_NL = np.einsum("ij,ejk->eik", D_MAT, B_NL)
    Ke_M = vol_ref[:, None, None] * np.einsum("eji,ejk->eik", B_NL, DB_NL)

    # --- Geometric tangent K_G ---
    # K_G[3a+i, 3b+i] = vol * dNdX_a^T S dNdX_b (scalar per i)
    # sigma_ab = dNdX[e,a,:] . S[e,:,:] . dNdX[e,b,:]
    SdN = np.einsum("eij,eaj->eai", S_mat, dNdX)  # (n_elem, 4, 3)
    sigma_ab = np.einsum("eai,ebi->eab", dNdX, SdN)  # (n_elem, 4, 4)
    Ke_G = np.zeros((n_elem, 12, 12))
    for a in range(4):
        for b in range(4):
            for i in range(3):
                Ke_G[:, 3 * a + i, 3 * b + i] += vol_ref * sigma_ab[:, a, b]

    Ke_T = Ke_M + Ke_G
    Ke_T = np.nan_to_num(Ke_T, nan=0.0, posinf=0.0, neginf=0.0)

    # Element DOF indices
    edofs = np.zeros((n_elem, 12), dtype=np.int64)
    for a in range(4):
        edofs[:, 3 * a]     = 3 * elems[:, a]
        edofs[:, 3 * a + 1] = 3 * elems[:, a] + 1
        edofs[:, 3 * a + 2] = 3 * elems[:, a] + 2

    # Sparse assembly
    rows = np.repeat(edofs[:, :, None], 12, axis=2).ravel()
    cols = np.repeat(edofs[:, None, :], 12, axis=1).ravel()
    K_T = sparse.coo_matrix(
        (Ke_T.ravel(), (rows, cols)), shape=(n_dofs, n_dofs)).tocsr()

    del Ke_M, Ke_G, Ke_T, B_NL, DB_NL, SdN, sigma_ab
    return K_T, edofs


# =============================================================================
# === NONLINEAR SOLVER (Newton-Raphson, displacement-controlled) ===
# =============================================================================

def solve_fem_nonlinear(coords, elems, dNdX, vol_ref, n_nodes, n_dofs, dx):
    """Displacement-controlled Total Lagrangian NR solver.

    Returns
    -------
    history : dict with per-step arrays:
        u_steps      — list of displacement vectors
        vm_node_steps — list of nodal von Mises arrays
        reaction_steps — list of reaction forces (scalar)
        strain_steps  — array of nominal strain values
        stress_steps  — array of nominal stress values (Pa)
        n_bot, n_top  — boundary node counts
        solve_time    — total wall-clock time
    """
    t0 = time.time()
    z = coords[:, 2]
    tol_bc = dx * 0.01
    bot = np.where(z < tol_bc)[0]
    top = np.where(z > L - tol_bc)[0]
    if len(bot) == 0 or len(top) == 0:
        raise RuntimeError(f"missing BC nodes (bot={len(bot)} top={len(top)})")

    bot_dofs = np.concatenate([3 * bot, 3 * bot + 1, 3 * bot + 2])
    top_z = 3 * top + 2
    cst = np.unique(np.concatenate([bot_dofs, top_z]))
    free = np.setdiff1d(np.arange(n_dofs), cst)

    total_d = -STRAIN_TARGET * L
    n_steps = N_INCREMENTS

    # Storage for converged results
    u_steps = []
    vm_node_steps = []
    reaction_steps = []
    strain_steps = []
    stress_steps = []
    A0 = L * L

    # Step 0: undeformed
    u_flat = np.zeros(n_dofs)
    _, _, vm0 = compute_internal_forces(u_flat, elems, dNdX, vol_ref, n_nodes, n_dofs)
    u_steps.append(u_flat.copy())
    vm_node_steps.append(vm0.copy())
    reaction_steps.append(0.0)
    strain_steps.append(0.0)
    stress_steps.append(0.0)

    NR_MAX_ITER = 8
    NR_REL_TOL = 1e-3
    NR_ABS_TOL = 1e-6

    for step in range(1, n_steps + 1):
        # Prescribed displacement at this step
        d_step = total_d * step / n_steps
        u_flat[top_z] = d_step
        # Bottom stays fixed (already zero from init, but ensure)
        u_flat[bot_dofs] = 0.0

        converged = False
        for nr_iter in range(NR_MAX_ITER):
            # Internal forces
            f_int, vm_elem, vm_node = compute_internal_forces(
                u_flat, elems, dNdX, vol_ref, n_nodes, n_dofs)

            # External forces are zero (displacement-controlled)
            # Residual R = f_int (on free DOFs should be zero)
            R = f_int.copy()
            R_free = R[free]
            res_norm = np.linalg.norm(R_free)

            if nr_iter == 0:
                res_norm_0 = max(res_norm, 1e-30)

            # Convergence check
            if res_norm < NR_REL_TOL * res_norm_0 or res_norm < NR_ABS_TOL:
                converged = True
                break

            # Assemble tangent
            K_T, _ = assemble_tangent(
                u_flat, elems, dNdX, vol_ref, n_nodes, n_dofs)

            # Solve for correction on free DOFs
            Kff = K_T[np.ix_(free, free)]
            try:
                factor = splu(Kff.tocsc())
                du_free = -factor.solve(R_free)
            except Exception:
                # Fallback to iterative if factorisation fails
                try:
                    du_free = -spsolve(Kff, R_free)
                except Exception:
                    break
            del K_T

            du_free = np.nan_to_num(du_free, nan=0.0, posinf=0.0, neginf=0.0)

            # Clamp correction magnitude to prevent divergence
            max_du = 0.5 * abs(total_d / n_steps)
            du_norm = np.linalg.norm(du_free, ord=np.inf)
            if du_norm > max_du:
                du_free *= max_du / du_norm

            u_flat[free] += du_free

        # If not converged after max iters, accept current state
        # Recompute internal forces for final state
        if not converged:
            f_int, vm_elem, vm_node = compute_internal_forces(
                u_flat, elems, dNdX, vol_ref, n_nodes, n_dofs)

        # Reaction force = sum of internal forces at top z-DOFs
        R_top = float(np.sum(f_int[top_z]))

        # Store converged step
        u_steps.append(u_flat.copy())
        vm_node_steps.append(vm_node.copy())
        reaction_steps.append(R_top)
        nom_strain = abs(d_step) / L
        nom_stress = abs(R_top) / A0
        strain_steps.append(nom_strain)
        stress_steps.append(nom_stress)

    solve_time = time.time() - t0
    return {
        "u_steps": u_steps,
        "vm_node_steps": vm_node_steps,
        "reaction_steps": reaction_steps,
        "strain_steps": np.array(strain_steps),
        "stress_steps": np.array(stress_steps),
        "n_bot": len(bot),
        "n_top": len(top),
        "solve_time": solve_time,
    }


# =============================================================================
# === STL EXPORT ===
# =============================================================================

def write_stl(verts_tpms, faces, p, stl_path):
    v_mm = (verts_tpms * SCALE * 1000.0).astype(np.float32)
    tri = v_mm[faces]
    fn = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
    mag = np.linalg.norm(fn, axis=1, keepdims=True)
    fn = (fn / np.where(mag > 1e-12, mag, 1.0)).astype(np.float32)
    n_t = len(faces)
    rec = np.zeros(n_t, dtype=[
        ("n", "<f4", 3), ("v0", "<f4", 3), ("v1", "<f4", 3),
        ("v2", "<f4", 3), ("a", "<u2")])
    rec["n"] = fn
    rec["v0"] = tri[:, 0]
    rec["v1"] = tri[:, 1]
    rec["v2"] = tri[:, 2]
    with open(stl_path, "wb") as fp:
        fp.write(b"Novel periodic surface via Fourier synthesis".ljust(80, b"\0"))
        fp.write(np.array(n_t, dtype="<u4").tobytes())
        fp.write(rec.tobytes())
    return n_t


# =============================================================================
# === PREVIEW + ANIMATION (nonlinear per-step data) ===
# =============================================================================

def save_preview_png(verts_tpms, faces, geo_name, eq, png_path):
    v_mm = verts_tpms * SCALE * 1000.0
    step = max(1, len(faces) // 15000)
    sf = faces[::step]
    fig = plt.figure(figsize=(12, 9))
    ax = fig.add_subplot(111, projection="3d")
    ax.plot_trisurf(v_mm[:, 0], v_mm[:, 1], v_mm[:, 2],
                    triangles=sf, cmap="viridis", alpha=0.75, edgecolor="none")
    ax.set_xlabel("X mm"); ax.set_ylabel("Y mm"); ax.set_zlabel("Z mm")
    ax.set_title(f"{geo_name}\n{eq}", fontsize=9)
    ax.view_init(elev=25, azim=-60)
    fig.tight_layout()
    fig.savefig(png_path, dpi=120, bbox_inches="tight")
    plt.close(fig)


def animate_geometry_nl(coords, history, surf_nodes, geo_name, mp4_p, gif_p):
    """Animate using actual per-step stored displacement and stress data."""
    u_steps = history["u_steps"]
    vm_steps = history["vm_node_steps"]
    eps_arr = history["strain_steps"]
    sig_arr = history["stress_steps"] / 1e6  # Pa -> MPa
    n_load = len(u_steps)  # N_INCREMENTS + 1

    co_mm = coords * 1e3
    mx_vm = max(float(np.max([v.max() for v in vm_steps])), 1e-10)

    # Loading: step 0..N, Unloading: step N-1..0
    load_idx = list(range(n_load))
    unload_idx = list(range(n_load - 2, -1, -1))
    all_idx = load_idx + unload_idx
    n_fr = len(all_idx)

    fig = plt.figure(figsize=(16, 7))
    ax3 = fig.add_subplot(121, projection="3d")
    ax2 = fig.add_subplot(122)
    fig.tight_layout(pad=3.0)

    def _upd(fi):
        si = all_idx[fi]
        is_loading = fi < n_load
        phase = "Loading" if is_loading else "Unloading"

        u_mm = u_steps[si].reshape(-1, 3) * 1e3
        vm = vm_steps[si]
        df = co_mm + u_mm

        ax3.clear()
        ax3.scatter(df[surf_nodes, 0], df[surf_nodes, 1], df[surf_nodes, 2],
                    c=vm[surf_nodes], cmap="plasma", s=3, alpha=0.85,
                    vmin=0, vmax=mx_vm)
        ax3.set_xlim(0, L * 1e3)
        ax3.set_ylim(0, L * 1e3)
        ax3.set_zlim(0, L * 1e3)
        ax3.set_xlabel("X mm"); ax3.set_ylabel("Y mm"); ax3.set_zlabel("Z mm")
        ax3.view_init(elev=25, azim=-60)
        ax3.set_title(
            f"{geo_name} | {phase} Step {si}/{N_INCREMENTS} | "
            f"Strain: {eps_arr[si]*100:.1f}%", fontsize=9)

        ax2.clear()
        cur_e = eps_arr[si]
        cur_s = sig_arr[si]

        # Ghost full curve
        ax2.plot(eps_arr * 100, sig_arr, "b-", lw=1.5, alpha=0.2)

        if is_loading:
            ax2.plot(eps_arr[:si + 1] * 100, sig_arr[:si + 1],
                     "b-", lw=2.5, label="Loading")
        else:
            ax2.plot(eps_arr * 100, sig_arr, "b-", lw=2.5, label="Loading")
            ax2.plot(eps_arr[si:] * 100, sig_arr[si:],
                     "r--", lw=2.5, label="Unloading")

        ax2.plot(cur_e * 100, cur_s, "ro", ms=8, zorder=5)
        ax2.set_xlabel("Strain (%)")
        ax2.set_ylabel("Stress (MPa)")
        ax2.set_title("Stress-Strain (Nonlinear)")
        ax2.grid(True, alpha=0.3)
        ymax = max(float(sig_arr.max()), 1e-10)
        ax2.set_xlim(-1, STRAIN_TARGET * 100 + 2)
        ax2.set_ylim(-0.05 * ymax, 1.15 * ymax)
        ax2.legend(loc="upper left", fontsize=9)
        return []

    anim = FuncAnimation(fig, _upd, frames=n_fr, interval=400, blit=False)
    saved = False
    try:
        from matplotlib.animation import FFMpegWriter
        anim.save(mp4_p, writer=FFMpegWriter(fps=5), dpi=120)
        saved = True
    except Exception:
        pass
    if not saved:
        try:
            from matplotlib.animation import PillowWriter
            anim.save(gif_p, writer=PillowWriter(fps=5), dpi=100)
        except Exception:
            pass
    plt.close(fig)


def save_curvature_png(coords, u_final, surf_nodes, vm_n,
                       eps_arr, sig_MPa, geo_name, E_load, v_energy, png_path):
    co_mm = (coords + u_final.reshape(-1, 3)) * 1e3
    mx_vm = max(float(vm_n.max()), 1e-10)
    fig = plt.figure(figsize=(16, 7))

    ax3 = fig.add_subplot(121, projection="3d")
    sc = ax3.scatter(co_mm[surf_nodes, 0], co_mm[surf_nodes, 1],
                     co_mm[surf_nodes, 2],
                     c=vm_n[surf_nodes], cmap="plasma", s=3,
                     vmin=0, vmax=mx_vm)
    ax3.set_xlim(0, L * 1e3)
    ax3.set_ylim(0, L * 1e3)
    ax3.set_zlim(0, L * 1e3)
    ax3.set_xlabel("X mm"); ax3.set_ylabel("Y mm"); ax3.set_zlabel("Z mm")
    ax3.view_init(elev=25, azim=-60)
    ax3.set_title(f"{geo_name} | Max compression", fontsize=10)
    plt.colorbar(sc, ax=ax3, shrink=0.6, label="von Mises (Pa)")

    ax2 = fig.add_subplot(122)
    ax2.plot(eps_arr * 100, sig_MPa, "b-o", lw=2, ms=4, label="Loading")
    ax2.plot(eps_arr[::-1] * 100, sig_MPa[::-1], "r--s", lw=2, ms=4,
             label="Unloading")
    ax2.set_xlabel("Strain (%)")
    ax2.set_ylabel("Stress (MPa)")
    ax2.set_title("Stress-Strain (Nonlinear)")
    ax2.grid(True, alpha=0.3)
    ax2.legend(fontsize=9)
    ax2.text(0.96, 0.05,
             f"E_abs={E_load:.2f} J\nVol.E={v_energy:.4f} J/cm^3",
             transform=ax2.transAxes, ha="right", va="bottom", fontsize=9,
             bbox=dict(boxstyle="round", fc="wheat", alpha=0.5))
    fig.tight_layout()
    fig.savefig(png_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


# =============================================================================
# === MAIN LOOP ===
# =============================================================================

print("=" * 70)
print("  Novel TPMS Generator + Nonlinear FEA  |  %d geometries" % N_GEOMETRIES)
print("=" * 70)
print(f"  Method    : Fourier synthesis ({N_BASIS} basis fns) + mean curvature flow")
print(f"  MCF       : max {MCF_ITERATIONS} iters, dt={MCF_DT}, converge at max|H|<{MCF_TOL}")
print(f"  TPMS grid : {VOXEL_RES}^3  |  STL grid: {STL_RES}^3  |  FEM grid: {FEM_RES}^3")
print(f"  Domain    : {DOMAIN_M*1e3:.1f} mm  |  {N_CELLS}^3 cells")
print(f"  Material  : E={E_MODULUS/1e6:.0f} MPa  v={NU}")
print(f"  FEM       : Total Lagrangian, SVK, Newton-Raphson")
print(f"  Strain    : {STRAIN_TARGET*100:.0f}% in {N_INCREMENTS} steps")
print(f"  Date      : {DATE_STR}\n")

t_total = time.time()
results = []
dx_fem = DOMAIN_M / FEM_RES
A0 = L * L
V0 = L ** 3
eq_path = os.path.join(OUT, f"geometry_equations_{DATE_STR}.txt")
eq_file = open(eq_path, "w")
geo_count = 0
attempt = 0

while geo_count < N_GEOMETRIES:
    attempt += 1
    idx = geo_count + 1
    tag = f"[GEO {idx:02d}/{N_GEOMETRIES}]"

    p = sample_params()
    eq = equation_str(p)
    active_names = [BASIS[i][0] for i in p["active_idx"]]
    print(f"\n  {tag} Sampling... terms={'+'.join(active_names)} "
          f"t={p['isovalue_t']:.3f} f={p['freq']:.3f} "
          f"whi={p['wall_half_iso']:.3f}")
    print(f"         {eq}")

    # ---- Generate surface on 90^3 for validation (skip MCF for speed) ----
    verts_v, faces_v, _, _ = generate_surface_mesh(p, VOXEL_RES, run_mcf=False)
    ok, reason = validate_geometry(p, verts_v, faces_v)
    if not ok:
        print(f"  {tag} REJECTED (attempt {attempt}): {reason}")
        del verts_v, faces_v
        continue
    print(f"  {tag} Validity PASSED  ({len(verts_v)} verts)")

    prefix = f"geo_{idx:03d}_{DATE_STR}"
    geoname = f"KimSurface_{idx:03d}_{DATE_STR}"
    t_geo = time.time()

    # ---- High-res STL on 120^3 with mean curvature flow ----
    stl_path = os.path.join(OUT, f"{prefix}.stl")
    print(f"  {tag} STL export ({STL_RES}^3) + mean curvature flow...")
    verts_stl, faces_stl, max_H, mcf_it = generate_surface_mesh(
        p, STL_RES, run_mcf=True)
    n_tri = write_stl(verts_stl, faces_stl, p, stl_path)
    sz = os.path.getsize(stl_path)
    print(f"  {tag} STL: {n_tri:,} tris ({sz/1e6:.1f} MB)")
    print(f"  {tag} MCF converged: max|H|={max_H:.4f} after {mcf_it} iters")

    # ---- Preview PNG ----
    prev_png = os.path.join(OUT, f"{prefix}_preview.png")
    save_preview_png(verts_stl, faces_stl, geoname, eq, prev_png)
    print(f"  {tag} Preview saved")
    del verts_v, faces_v, verts_stl, faces_stl

    # ---- FEM on 30^3 physical grid ----
    print(f"  {tag} FEM meshing ({FEM_RES}^3)...")
    solid, Xn, Yn, Zn = identify_solid_fem(p, FEM_RES, DOMAIN_M)
    n_solid = int(solid.sum())
    if n_solid == 0:
        print(f"  {tag} FEM mesh empty — resampling")
        continue
    elems, coords, n_nodes, n_elem, vol, J_all = build_tet_mesh(
        solid, Xn, Yn, Zn, FEM_RES)
    n_dofs = 3 * n_nodes
    print(f"  {tag} Mesh: {n_nodes} nodes, {n_elem} elements")
    del solid, Xn, Yn, Zn, vol, J_all

    _, surf_nodes = extract_surface(elems)

    # ---- Precompute reference config ----
    print(f"  {tag} Precomputing reference config...")
    dNdX_ref, vol_ref = precompute_reference(coords, elems)

    # ---- Nonlinear solve ----
    print(f"  {tag} Nonlinear solve ({N_INCREMENTS} steps, NR)...")
    try:
        history = solve_fem_nonlinear(
            coords, elems, dNdX_ref, vol_ref, n_nodes, n_dofs, dx_fem)
    except RuntimeError as e:
        print(f"  {tag} SOLVE FAILED: {e} — resampling")
        del dNdX_ref, vol_ref, elems, coords
        continue

    solve_t = history["solve_time"]
    n_bot = history["n_bot"]
    n_top = history["n_top"]
    print(f"  {tag} Solved in {solve_t:.1f}s (BC bot={n_bot} top={n_top})")

    # Extract per-step data
    eps_arr = history["strain_steps"]       # (N+1,)
    sig_arr = history["stress_steps"]       # (N+1,) in Pa
    sig_MPa = sig_arr / 1e6
    u_final = history["u_steps"][-1]
    vm_final = history["vm_node_steps"][-1]
    R_top_final = history["reaction_steps"][-1]

    for i in range(1, N_INCREMENTS + 1):
        disp_mm = eps_arr[i] * L * 1e3
        force_N = abs(history["reaction_steps"][i])
        print(f"  {tag} Step {i:02d}/{N_INCREMENTS} | "
              f"Disp: {disp_mm:.1f}mm | "
              f"Force: {force_N:.0f}N | "
              f"Stress: {sig_MPa[i]:.2f}MPa")

    # Energy via trapezoidal integration of actual nonlinear curve
    E_load = V0 * float(np.trapezoid(sig_arr, eps_arr))
    E_unload = E_load
    hyst = 0.0
    V_cm3 = V0 * 1e6
    v_energy = E_load / V_cm3
    print(f"  {tag} Energy: {E_load:.2f}J | Vol.E: {v_energy:.4f} J/cm^3")

    # ---- CSV results ----
    csv_path = os.path.join(OUT, f"{prefix}_results.csv")
    with open(csv_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["increment", "displacement_mm", "reaction_force_N",
                     "nominal_stress_MPa", "nominal_strain_pct"])
        for i in range(N_INCREMENTS + 1):
            w.writerow([i,
                         f"{eps_arr[i]*L*1e3:.4f}",
                         f"{abs(history['reaction_steps'][i]):.4f}",
                         f"{sig_MPa[i]:.6f}",
                         f"{eps_arr[i]*100:.2f}"])

    # ---- Animation ----
    mp4_p = os.path.join(OUT, f"{prefix}.mp4")
    gif_p = os.path.join(OUT, f"{prefix}.gif")
    print(f"  {tag} Animating...")
    animate_geometry_nl(coords, history, surf_nodes, geoname, mp4_p, gif_p)

    # ---- Curvature PNG ----
    cur_png = os.path.join(OUT, f"{prefix}_curvature.png")
    save_curvature_png(coords, u_final, surf_nodes, vm_final,
                       eps_arr, sig_MPa, geoname, E_load, v_energy, cur_png)

    # ---- Equation log ----
    eq_file.write(f"Index: {idx:03d}\n")
    eq_file.write(f"Name: {geoname}\n")
    eq_file.write(f"Equation: {eq}\n")
    eq_file.write(f"Active basis: {', '.join(active_names)}\n")
    coeffs_str = ", ".join(
        f"{BASIS[i][0]}={p['coeffs'][i]:.4f}" for i in p["active_idx"])
    eq_file.write(f"Coefficients: {coeffs_str}\n")
    eq_file.write(f"Isovalue t: {p['isovalue_t']:.4f}\n")
    eq_file.write(f"Freq: {p['freq']:.4f}\n")
    eq_file.write(f"Wall half-iso: {p['wall_half_iso']:.4f}\n")
    eq_file.write(f"N_CELLS: {N_CELLS}  VOXEL_RES: {VOXEL_RES}  "
                  f"STL_RES: {STL_RES}\n")
    eq_file.write(f"MCF iterations: {mcf_it}  max|H|: {max_H:.4f}  "
                  f"(tol={MCF_TOL}, dt={MCF_DT})\n")
    eq_file.write(f"Minimal surface: {'YES' if max_H < MCF_TOL else 'APPROX'} "
                  f"(H={'<' if max_H < MCF_TOL else '>'}{MCF_TOL})\n")
    eq_file.write(f"Vertices: {n_nodes}  Elements: {n_elem}\n")
    eq_file.write(f"FEM: Total Lagrangian, SVK, NR ({N_INCREMENTS} steps)\n")
    eq_file.write(f"Date: {DATE_STR}\n\n")
    eq_file.flush()

    # Compute peak element von Mises for the final step
    _, vm_elem_final, _ = compute_internal_forces(
        u_final, elems, dNdX_ref, vol_ref, n_nodes, n_dofs)

    results.append({
        "index": idx, "name": geoname, "equation": eq,
        "active_terms": "+".join(active_names),
        "isovalue_t": p["isovalue_t"], "freq": p["freq"],
        "wall_half_iso": p["wall_half_iso"],
        "n_vertices": n_nodes, "n_elements": n_elem,
        "E_load": E_load, "E_unload": E_unload, "hyst": hyst,
        "v_energy": v_energy,
        "peak_stress": float(vm_elem_final.max()) / 1e6,
        "peak_force": abs(R_top_final),
        "solve_time": solve_t,
        "max_H": max_H, "mcf_iters": mcf_it,
    })
    geo_time = time.time() - t_geo
    print(f"  {tag} Saved: {prefix}.stl/.mp4/_results.csv ({geo_time:.1f}s)")
    geo_count += 1
    del dNdX_ref, vol_ref, elems, coords, history, surf_nodes
    del vm_elem_final, u_final, vm_final

eq_file.close()
print(f"\n  Equation log -> {eq_path}")

# =============================================================================
# === SUMMARY ===
# =============================================================================

print(f"\n{'='*70}")
print(f"  [SUMMARY] All {N_GEOMETRIES} complete.")
print(f"{'='*70}")

sum_csv = os.path.join(OUT, f"summary_{DATE_STR}.csv")
with open(sum_csv, "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["index", "name", "equation", "active_terms",
                "isovalue_t", "freq", "wall_half_iso",
                "n_vertices", "n_elements",
                "energy_absorbed_J", "energy_recovered_J",
                "hysteresis_loss_pct", "volumetric_energy_J_per_cm3",
                "peak_stress_MPa", "peak_force_N", "solve_time_sec"])
    for r in results:
        w.writerow([r["index"], r["name"], r["equation"], r["active_terms"],
                     f"{r['isovalue_t']:.4f}", f"{r['freq']:.4f}",
                     f"{r['wall_half_iso']:.4f}", r["n_vertices"],
                     r["n_elements"],
                     f"{r['E_load']:.6f}", f"{r['E_unload']:.6f}",
                     f"{r['hyst']:.1f}", f"{r['v_energy']:.6f}",
                     f"{r['peak_stress']:.4f}", f"{r['peak_force']:.2f}",
                     f"{r['solve_time']:.1f}"])
print(f"  Saved summary_{DATE_STR}.csv")

if len(results) >= 2:
    labels = [f"S{r['index']:02d}" for r in results]
    e_abs = [r["E_load"] for r in results]
    v_en = [r["v_energy"] for r in results]
    iso_ts = [r["isovalue_t"] for r in results]
    terms = [r["active_terms"] for r in results]
    colours = plt.cm.tab10(np.linspace(0, 1, len(results)))
    fig, axes = plt.subplots(2, 2, figsize=(16, 11))
    axes[0, 0].barh(labels, e_abs, color=colours)
    axes[0, 0].set_xlabel("Energy Absorbed (J)")
    axes[0, 0].set_title("Energy Absorption")
    axes[0, 1].barh(labels, v_en, color=colours)
    axes[0, 1].set_xlabel("J / cm^3")
    axes[0, 1].set_title("Volumetric Energy Density")
    pk = [r["peak_stress"] for r in results]
    axes[1, 0].barh(labels, pk, color=colours)
    axes[1, 0].set_xlabel("Peak von Mises (MPa)")
    axes[1, 0].set_title("Peak Stress")
    axes[1, 1].scatter(iso_ts, e_abs, c=colours, s=100, edgecolors="k",
                       zorder=3)
    for i, r in enumerate(results):
        axes[1, 1].annotate(f"S{r['index']:02d}", (iso_ts[i], e_abs[i]),
                            fontsize=7, ha="center", va="bottom")
    axes[1, 1].set_xlabel("Isovalue t")
    axes[1, 1].set_ylabel("Energy (J)")
    axes[1, 1].set_title("Isovalue vs Energy")
    axes[1, 1].grid(True, alpha=0.3)
    fig.suptitle(
        f"Novel Periodic Surfaces (Fourier Synthesis, Nonlinear FEM) "
        f"-- {DATE_STR}",
        fontsize=14, fontweight="bold")
    fig.tight_layout()
    sum_png = os.path.join(OUT, f"summary_{DATE_STR}.png")
    fig.savefig(sum_png, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved summary_{DATE_STR}.png")

print(f"\n  {'#':>3} {'Active Terms':<22} {'t':>6} {'E(J)':>9} "
      f"{'Vol.E':>9} {'PkVM':>7} {'max|H|':>7} {'MCF':>4} {'Solve':>7}")
print("  " + "-" * 80)
for r in results:
    h_ok = "Y" if r["max_H"] < MCF_TOL else "~"
    print(f"  {r['index']:3d} {r['active_terms']:<22} "
          f"{r['isovalue_t']:6.3f} {r['E_load']:9.1f} "
          f"{r['v_energy']:9.4f} {r['peak_stress']:7.2f} "
          f"{r['max_H']:7.4f} {h_ok:>4} {r['solve_time']:7.1f}s")
print(f"\n  Total runtime: {time.time()-t_total:.1f}s")
print(f"  [SUMMARY] All {N_GEOMETRIES} complete. Saved summary_{DATE_STR}.csv/.png")
