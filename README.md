# Novel TPMS Metamaterials — Generative Geometry & Multi-Physics Simulation

<p align="center">
  <img src="outputs/previews/hero_render.png" alt="Generated TPMS lattice" width="400">
</p>

A Python framework for generating **genuinely new triply-periodic minimal surfaces (TPMS)** and simulating their mechanical, thermal, and fluid-flow behavior end-to-end.

Every known TPMS — Schwarz P, Gyroid, Diamond, I-WP, Neovius — is a specific point in a 12-dimensional Fourier basis of cubic-symmetric periodic functions. This project samples **random** points in that basis, drives each candidate toward zero mean curvature via curvature flow, then runs nonlinear FEA compression, transient heat conduction, and pressure-driven flow on the resulting structures.

## Gallery

Ten novel TPMS surfaces generated in a single run. Each is unique — random Fourier coefficients followed by mean curvature flow:

| | | | | |
|---|---|---|---|---|
| ![geo 001](outputs/previews/geo_001_20260406_preview.png) | ![geo 002](outputs/previews/geo_002_20260406_preview.png) | ![geo 003](outputs/previews/geo_003_20260406_preview.png) | ![geo 004](outputs/previews/geo_004_20260406_preview.png) | ![geo 005](outputs/previews/geo_005_20260406_preview.png) |
| ![geo 006](outputs/previews/geo_006_20260406_preview.png) | ![geo 007](outputs/previews/geo_007_20260406_preview.png) | ![geo 008](outputs/previews/geo_008_20260406_preview.png) | ![geo 009](outputs/previews/geo_009_20260406_preview.png) | ![geo 010](outputs/previews/geo_010_20260406_preview.png) |

Mean curvature heatmaps (verifying H ≈ 0):

| | | | | |
|---|---|---|---|---|
| ![curv 001](outputs/curvature/geo_001_20260406_curvature.png) | ![curv 002](outputs/curvature/geo_002_20260406_curvature.png) | ![curv 003](outputs/curvature/geo_003_20260406_curvature.png) | ![curv 004](outputs/curvature/geo_004_20260406_curvature.png) | ![curv 005](outputs/curvature/geo_005_20260406_curvature.png) |

Thermal + flow dashboards (steady-state heat conduction and Darcy flow through the void network):

| | | |
|---|---|---|
| ![heat 001](outputs/heat/geo_001_20260406_heat_summary.png) | ![heat 002](outputs/heat/geo_002_20260406_heat_summary.png) | ![heat 003](outputs/heat/geo_003_20260406_heat_summary.png) |

Aggregate comparison across all geometries:

![summary](outputs/summary/summary_20260406.png)

## How it works

**1. Fourier basis.** Twelve cubic-symmetric periodic functions span the space of "things that could plausibly be a TPMS":

```
P, G, D, IWP, N, SS, P², G², CC², SS², CSC, L
```

Every known TPMS is a known coefficient vector in this basis. Random coefficients → a new periodic surface candidate.

**2. Mean curvature flow.** A minimal surface satisfies H = 0 everywhere. The candidate surface is iteratively evolved along its mean-curvature normal until |H| drops below tolerance, mathematically converging toward minimality.

**3. Marching cubes → STL.** A high-resolution voxel grid is contoured to produce a watertight triangular mesh, then Laplacian-smoothed.

**4. Nonlinear FEA compression.** Total-Lagrangian formulation with a Saint Venant–Kirchhoff hyperelastic model, solved via Newton–Raphson over 15 load increments to 30% nominal strain. Stress, strain energy, and force-displacement curves recorded per increment.

**5. Heat + flow.** Transient heat conduction (implicit FDM, heterogeneous conductivity for solid vs. void) and pressure-driven creeping flow (Darcy regime) are simulated on the static structure.

## Stack

- **Language:** Python 3.10+
- **Math:** NumPy, SciPy (sparse linear algebra, FFT)
- **Geometry:** scikit-image (marching cubes), custom Laplacian smoothing
- **Visualization:** Matplotlib (static + animated)
- **FEA:** hand-rolled — Total Lagrangian, SVK material, Newton–Raphson

No external FEA library. Everything from the basis evaluation to the stiffness matrix assembly is implemented from scratch.

## Repository layout

```
.
├── src/                    # the three pipeline scripts
│   ├── tpms_compression.py
│   ├── tpms_thermal_flow.py
│   └── tpms_flow_heat.py
└── outputs/
    ├── previews/           # static surface renders (one per geometry)
    ├── curvature/          # mean curvature heatmaps
    ├── heat/               # thermal + flow dashboards
    ├── data/               # per-geometry FEA CSVs + Fourier coefficients
    ├── summary/            # aggregate comparison plots
    ├── animations/         # MP4s — not committed (too large)
    └── meshes/             # STLs — not committed (too large)
```

## Run it

```bash
pip install -r requirements.txt

# 1. Generate 10 random TPMS + run compression FEA on each
python src/tpms_compression.py

# 2. Heat + flow simulation on the generated structures
python src/tpms_thermal_flow.py

# 3. Standalone Schwarz P demo (no dependency on step 1)
python src/tpms_flow_heat.py
```

Each script writes its outputs (STL, MP4, PNG, CSV) into the current working directory tagged with today's date. The committed `outputs/` tree was sorted by file type after the fact for browsing.

## Outputs per geometry

- `geo_XXX_*.stl` — watertight triangle mesh, 3D-printable
- `geo_XXX_*.mp4` — compression animation (deforming under load)
- `geo_XXX_*_preview.png` — static surface render
- `geo_XXX_*_curvature.png` — mean-curvature heatmap
- `geo_XXX_*_thermal.mp4` — transient heat conduction
- `geo_XXX_*_flow.mp4` — fluid streamlines
- `geo_XXX_*_heat_summary.png` — steady-state thermal + flow dashboard
- `geo_XXX_*_results.csv` — per-increment FEA metrics
- `geometry_equations_*.txt` — exact Fourier coefficients for every generated surface

STL and MP4 outputs are not committed to the repo (they regenerate from the scripts and are large) — see `.gitignore`.

## License

MIT — see [LICENSE](LICENSE).
