# DeployableKirigami

## Introduction
This repository documents and shares the deployable kirigami project of the Mecawet group at the PMMH Laboratory (ESPCI Paris).
Deployable kirigamis are designed to work mainly in bending: the cuts turn the sheet into a series of connected ribbons. Here we create cutting patterns that deploy into 3D surfaces.

Given a target surface z(x, y) with a single maximum, the code computes a cutting pattern for a flat sheet. When the sheet is held at its rim and its centre is pulled up, it deploys into an approximation of the target. The pipeline has four stages:

1. **Slicing** (`lib/isolines.py`): M horizontal planes cut the target mesh, and each contour line is fitted with a closed spline.
2. **Cutting pattern** (`lib/cutting_pattern.py`): N section lines split every contour into arcs. Alternate arcs become cuts that overlap their neighbours by a length D.
3. **Cut mesh** (`lib/cut_mesh.py`): the flat sheet is triangulated with the cuts as edges, and the mesh is split along the cuts.
4. **Linear deployment** (`lib/ribbon_model.py`): every cut contour is a rigid level and the ribbons between two levels are clamped beams. The deployed shape and the pulling force come from one small linear system, in under a second.

The folder [`fem_deployment/`](fem_deployment/README.md) deploys the same cut mesh with a nonlinear shell finite-element model (Naghdi shell in FEniCSx) and compares it with the linear model.

## Installation

Clone the repository and install the requirements:

```
pip install -r requirements.txt
```

The `igl` module comes from the `libigl` package on PyPI. Only `cut_mesh.py` needs `triangle` and `shapely`. The FEM deployment needs its own FEniCSx environment, described in [fem_deployment/README.md](fem_deployment/README.md).

## Quick start

The code is in `lib/`, the example meshes in `data/` and the examples in `examples/`.

```python
import sys; sys.path.insert(0, 'lib')
import numpy as np, igl
from isolines import isolines
from cutting_pattern import cutting_patern, plot_cutting_pattern
from cut_mesh import kirigami_mesh
from ribbon_model import beams_from_pattern, RibbonModel

v, f = igl.read_triangle_mesh('data/cone.obj')          # target surface, z up, units in mm
z = v[:, 2]
heights = np.linspace(z.min() + 0.05*np.ptp(z), z.min() + 0.9*np.ptp(z), 15)  # the M slices, bottom to top
iso = isolines(v, f)
lines = [iso.isolinecoords([np.array([0, 0, 1]), np.array([0, 0, h])]) for h in heights]

spans = cutting_patern(lines, heights, N=6)              # the cuts, as spline parameter ranges
plot_cutting_pattern(lines, spans)
sheet = kirigami_mesh(lines, heights, N=6)               # flat triangle mesh, split along the cuts

model = RibbonModel(beams_from_pattern(lines, spans, 6), young=2100.0, thickness=0.1)  # PET, MPa and mm
top = heights[model.n_levels - 1] - heights[0]
q, F = model.pull_to(sheet['V'][sheet['center'], :2], top)   # move the centre to the top height; F is the force
part, coord = model.classify(sheet['V'], sheet['F'], lines, spans)
X = model.deform(sheet['V'], sheet['F'], part, coord, q)     # the deployed sheet
igl.write_triangle_mesh('cone_deployed.obj', X, sheet['F'])
```

## Examples

Each script writes its results to `examples/output/<example>/` (not tracked by git) and prints a summary. By default the sheet is PET, 0.1 mm thick, with E = 2.1 GPa (`--young`, `--thickness`). Every script lists its options with `--help`.

Every example is also a Jupyter notebook with the same name (`examples/0X_*.ipynb`), with its parameters in the first cell and the figures inline. The notebooks are saved with their outputs, so they can be read on GitHub without running them. To run them, start Jupyter in `examples/` (`pip install notebook`). They take between 5 s and 1 min.

| script | what it does | time |
|---|---|---|
| `01_target_structures.py` | Slices the targets of `data/` (cone, parabole, offcenter, trumpet, free_curve) into 15 contours and builds their cutting patterns and flat cut sheets (`<name>_pattern.png`, `<name>_flat.obj`, `<name>_sheet.npz`). It warns about ribbons narrower than 1 mm. | 15 s |
| `02_ribbon_model.py` | The ribbon model on concentric circles with a constant overlap D. It compares the central stiffness with Eq. 1 of the paper summed over the levels, pulls off centre to show the levels tilting, and draws the deformed sheet. | 2 s |
| `03_ribbon_targets.py` | The targets of example 1 deployed with the ribbon model by displacement, with every level compared with the height of its contour. The cone matches to 0.05 mm rms, at 1.86 N. | 10 s |
| `04_deploy_stl.py` | Any STL/OBJ target, sliced so that no ribbon is narrower than `--wmin`. It writes the cuts as an SVG for the laser cutter, checks the bed size (`--bed`, 220 × 220 mm by default, xTool F1 Ultra) and deploys with the ribbon model. `--scale` scales the target first. | 20 s |
| `05_test_samples.py` | Circular test samples for tensile tests (D, N and W series) as SVGs for the laser cutter, with their linear stiffness from the ribbon model and from the closed form k = E h³ W N / (8 (1 − ν²) M D³). | 5 s |

```
python examples/01_target_structures.py
python examples/03_ribbon_targets.py
python examples/04_deploy_stl.py path/to/fuji.stl --scale 0.45 --zmin 4.5 --zmax 78.75 --delta 2 --step 0.5
```

With `--scale 0.45`, the Mount Fuji model (a 524 × 540 mm tile; the STL is not included) gives 19 contours, ribbons 2.01 to 2.5 mm wide, 72 cuts and a 200 × 208 mm sheet. With the centre moved to 66.5 mm, the force is 2.8 N.

## Requirements on the target shape

- The surface must be a height field z(x, y): every vertical line meets it at most once. Overhangs give intersecting contour lines and vertical walls (a cylinder) give contour lines on top of each other.
- It must have a single maximum, which becomes the pulling point.
- Every contour line must be star-shaped as seen from the pulling point, so that each section line crosses it once (see stage 2).
- Only the longest contour of each level is used to build the pattern, so a level that splits into several loops (for example `data/double.obj`) is reduced to its largest loop.
- Ribbons narrower than about 1 mm cannot be laser cut reliably. `cut_mesh.ribbon_widths` returns the narrowest distance between each pair of consecutive contour lines and can be used to check a design before cutting.

Units follow the mesh. The example meshes are in millimetres; the ribbon model and the FEM deployment use millimetres, newtons and MPa.

## Procedure

### 1. Slicing the target surface (`isolines.py`)

`isolines(v, f, k=3)` holds the target triangle mesh and the degree `k` of the fitted splines. `isolinecoords(plane)` with `plane = [normal, point]` returns every curve where that plane cuts the mesh, as a list of `[spline, length]`, longest curve first. The code always slices with horizontal planes, `normal = (0, 0, 1)`.

1. **Edge crossings** (`cross_surface`). The signed distance of every vertex to the plane is d = (v − p)·n. A vertex with d ≤ 0 counts as below the plane, so a vertex lying exactly on the plane never creates a degenerate crossing. A triangle is crossed when its vertices are not all on the same side, and it has exactly two crossed edges. The crossing point on edge (a, b) is the exact linear interpolation v_a + t (v_b − v_a) with t = d_a / (d_a − d_b). Each crossing is stored once per edge, so neighbouring triangles share it.
2. **Chaining** (`chain_edges`). Each crossed triangle links its two crossed edges. Following these links gives the curves: a chain that starts and ends on the mesh boundary is open, any other chain is a closed loop. Open chains are walked from their boundary end.
3. **Orientation** (`orient_curves`). Closed curves are oriented counter-clockwise around the plane normal, using the sign of their vector area ½ Σ pᵢ × pᵢ₊₁. All levels then run in the same direction, which the cutting pattern relies on.
4. **Spline fit** (`interpolate`). Repeated points (when the plane passes through a vertex) are dropped. The curve is parametrised by its normalised chord length s ∈ [0, 1], starting exactly at 0. Closed curves are fitted with a **periodic** interpolating spline (`scipy.interpolate.make_interp_spline` with `bc_type='periodic'`), so the position and its first k−1 derivatives are continuous across the seam and `spline(s + 1) == spline(s)`. Open curves use natural end conditions. The returned length is the arc length of the spline, measured on 20 samples per data point.

Because the closed splines are periodic with period 1, any later step can evaluate them at parameters outside [0, 1] (for example a cut that runs through the seam from s = 0.9 to s = 1.1).

### 2. Section lines (`cutting_pattern.cutting_sections`)

The pattern is split into N angular sectors around a centre c, which is the pulling point. By default c is the mean of 100 points sampled on the top contour line; an explicit `center` can be passed to place the pulling point elsewhere (an off-centred pattern).

For every level j and every section line i = 0 … N−1 at angle θᵢ = 2πi/N, `spline_line_intersection` finds the spline parameter s_ij where the contour crosses the ray from c at angle θᵢ:

- initial guess: the sample (out of 1000) with the largest projection on the ray direction (cos θᵢ, sin θᵢ);
- refinement: `scipy.optimize.fsolve` on the cross product (x − c_x) sin θᵢ − (y − c_y) cos θᵢ = 0;
- the root is wrapped into [0, 1).

### 3. Overlaps and cuts (`cutting_pattern.cutting_patern`)

Consecutive levels j and j+1 bound a ring of material. At section line i this ring has

- height step δ_j = z_{j+1} − z_j,
- ribbon width W_ij = |C_{j+1}(s_{j+1,i}) − C_j(s_ij)|, the in-plane distance between the two contour lines along the section line.

From the beam model of the ribbons (Eq. 3 of the paper, δ/W = f (1 − ν²) D³ / (E h³ W²)), a ribbon of width W carrying the force f = F/N deflects by δ when its length is D = (B δ W / f)^(1/3), with B = E h³ / (1 − ν²). The code uses the scale-free form

    D_ij = (δ_j W_ij)^(1/3)

which is the formula above with N B / F = 1 (in mesh units). `cutting_patern(..., overlap=D)` (and `kirigami_mesh(..., overlap=D)`) uses a constant overlap D for every beam instead. In the linear regime this constant does not change the deployed shape, only the force needed to reach it and the ratio δ/D (see Limitations).

The overlap is converted to a spline parameter by dividing by the contour length, Δs_ij = D_ij / L_j. Cuts lie on the contour lines of levels 0 … M−2 (the top contour has no cut: the region inside it is the uncut cap that is pulled). On level j the cuts cover every other sector, alternating between levels:

- j even: sectors [θ_{2i}, θ_{2i+1}], i = 0 … N/2 − 1;
- j odd: sectors [θ_{2i+1}, θ_{2i+2}].

Each cut is extended by D past both of its section lines. Near every section line the cut of level j and the cut of level j+1 therefore overlap over a length 2D centred on the line, which forms the two ribbons of length D shown in Fig. 1 of the paper. N must be even.

The result `cut_spans[j]` is a list of N/2 pairs `(from_s, to_s)` with `from_s < to_s`; when a cut crosses the seam of the contour, `to_s` is larger than 1 and the periodic spline is evaluated past 1.

### 4. Cut mesh of the sheet (`cut_mesh.kirigami_mesh`)

`kirigami_mesh(lines, levels, N=6, h=None, margin=None, center=None)` builds the flat sheet as a triangle mesh in the plane z = 0 whose faces are disconnected across every cut.

1. **Mesh size.** `ribbon_widths` estimates the width of each ring as the minimum distance between 400 points on contour j and 400 on contour j+1. The target edge length h defaults to a third of the narrowest ring, so there are about three triangles across each ribbon.
2. **Cut polylines.** Every cut span is sampled on its spline with segments of length close to h.
3. **Outer boundary.** The sheet is the outermost contour grown outwards by `margin` (default: the outermost ring width) with `shapely`'s `buffer`, resampled at spacing h. Without this margin the outermost cuts would lie on the edge of the sheet.
4. **Triangulation.** A constrained Delaunay triangulation (`triangle`, switches `pq30a`) of the boundary and the cut polylines, with a minimum angle of 30° and a maximum triangle area of h²/2. The cuts are constrained segments, so they are made of mesh edges; the pulling point c is inserted as a vertex. Segment markers record which edges belong to the rim (1) and to cut i (2 + i).
5. **Splitting along the cuts** (`split_along_cuts`). For every vertex on a cut, the triangles around it are grouped by union–find, joining two triangles when they share an edge that is not a cut. Every group after the first receives its own copy of the vertex. Interior cut vertices therefore get two copies (one per side of the cut) while the two tips of a cut have a single group and stay shared, so the sheet remains in one piece.

Two checks are provided: `face_components(F)` must be 1 (one piece of material) and `boundary_loops(F)` must be 1 + the number of cuts (the rim plus one slit per cut).

The returned dictionary contains:

| key | content |
| --- | --- |
| `V`, `F` | vertices (z = 0) and triangles of the split mesh |
| `parent` | for every vertex, the vertex of the unsplit mesh it was copied from |
| `rim` | vertices on the outer boundary |
| `center` | vertex at the pulling point |
| `cut_edges`, `cut_id` | edges of the unsplit mesh that lie on a cut, and the index of their cut |
| `cut_spans` | the cutting pattern of stage 3 |
| `widths`, `h` | ring widths and mesh size |

For `data/parabole.obj` with 15 levels and N = 6 this gives 42 cuts, about 20 400 vertices and 36 700 triangles, one component and 43 boundary loops.

### 5. Linear deployment: the ribbon model (`ribbon_model.py`)

`beams_from_pattern(lines, spans, N)` reads the beams from the cutting pattern. Near every section line, the cut of contour j and the cut of contour j+1 overlap, and the strip between their tips is one beam: length L = 2D, width W, clamped into level j at one tip and into level j+1 at the other.

`RibbonModel(beams, young, thickness, poisson)` treats every cut contour as a rigid level that moves out of plane by a height w_j and two small rotations (θx, θy). A point (x, y) of level j then rises by w_j + θx y − θy x. Level 0 (the outer contour, with the rim) is clamped. The unknowns are q = (w_j, θx_j, θy_j) for j = 1 … M−1. Every beam has a 6 × 6 stiffness built from clamped-clamped bending (B W/L³ terms, B = E h³ / 12(1 − ν²)) and torsion (G W h³ / 3L). It is mapped to the motions of its two levels and assembled into K, which is block tridiagonal. The model is linear and its kinematics are small-slope: x and y do not move.

- `pull(point, force)` applies a force at a point of the top level and solves K q = Q.
- `pull_to(point, height)` deploys **by displacement**, which is the default everywhere. It moves the point to the given height, leaves the rotations free and solves the bordered system [[K, g], [gᵀ, 0]] [q; −F] = [0; H] with g = (1, y, −x) on the top level. The force F is an unknown, the Lagrange multiplier of the constraint.
- `classify(V, F, lines, spans)` and `deform(V, F, part, coord, q)` project the solution onto the cut mesh. A panel follows the rigid motion of its level, and a beam bends as the clamped cubic between its two ends with a linear twist. The result is a deployed mesh for viewing or exporting.
- `fit_levels(V, X, F, part, n_levels)` fits the rigid motion of every level to any deployed mesh X, for example a FEM result.
- `stiffness`, `beam_forces` and `max_stress` give the linear stiffness, the shear force in every beam and the peak bending stress.

The Supplementary Material of the paper (Section S3) derives the model step by step.

## FEM deployment

[`fem_deployment/`](fem_deployment/README.md) exports the cut mesh and its ribbon-model deployment, deploys the same mesh with a geometrically nonlinear Naghdi shell (FEniCSx, DOLFINx 0.11) by displacement of the centre, and compares the two. On the cone (4.7k vertices) the shell run takes about 3 to 6 min. The levels sit 2.4 mm rms below the ribbon model at 39.5 mm, and the force is 3.4 N against 1.86 N. The shell is softer at small heights and stiffens once the ribbons have to stretch.

## Limitations

- The ribbon model is linear. It holds while the rise of a level is small compared with its ribbon length 2D. `04_deploy_stl.py` prints that ratio. Above about 1 the real sheet, and the shell, stiffen.
- The overlap rule D = (δW)^(1/3) comes from linear beam theory, which assumes δ ≪ D.
- Every ribbon of a level is assumed to carry the same force F/N. The torque balance of Eq. 2 of the paper (off-centred patterns) is not implemented.
- Only one pulling point and horizontal slices are supported.

## License

MIT, see [LICENSE](LICENSE).
