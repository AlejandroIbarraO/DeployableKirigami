""" Step 2 (FEniCSx, DOLFINx 0.11): deploy a cut sheet written by
export_case.py with the nonlinear Naghdi shell of the FEniCSx-Shells demo
(demo_nonlinear-naghdi-semi-cylinder): 3D displacement in P2 + cubic bubble,
two director rotations in P2, membrane and shear locking relieved by partial
selective reduced integration (PSRI). The rim is clamped (displacement and
rotations zero), the centre vertex is pulled up to the height of the top
contour in steps (displacement prescribed, rotations free); plain SNES Newton
(the backtracking line search stalls here), a failed step is cut in half. The pulling force is
the residual at the centre's vertical displacement.
The first guess of the first step is the ribbon model deployment scaled to
that height; --flat-start starts from the flat sheet instead (needed when the
ribbon model is far from the shell, as for the Fuji pattern).
Writes <case>_naghdi.csv (height, force, Newton iterations) and
<case>_naghdi_<step>.npy (deformed vertex positions) for every step.
    python naghdi_deploy.py output/case_cone_h1.npz --steps 12
    python naghdi_deploy.py output/case_fuji_x0.45_h1.npz --steps 24 --flat-start """
import argparse
import glob
import os
import time

import basix.ufl
import numpy as np
import ufl
from dolfinx import default_scalar_type, fem, mesh as dmesh
from dolfinx.fem.petsc import NonlinearProblem
from mpi4py import MPI

parser = argparse.ArgumentParser(description = __doc__,formatter_class = argparse.RawDescriptionHelpFormatter)
parser.add_argument('case',help = 'case_<name>_h<h>.npz written by export_case.py')
parser.add_argument('--steps',type = int,default = 12,help = 'largest step = top height / steps')
parser.add_argument('--flat-start',action = 'store_true',help = 'start from the flat sheet, not the ribbon model')
parser.add_argument('--young',type = float,default = 2100.0,help = 'MPa')
parser.add_argument('--poisson',type = float,default = 0.4)
parser.add_argument('--thickness',type = float,default = 0.1,help = 'mm')
args = parser.parse_args()
case,steps = args.case,args.steps
E_,nu_,t_ = args.young,args.poisson,args.thickness
name = os.path.splitext(case)[0]
data = np.load(case)
V,F = data['V'],data['F'].astype(np.int64)
rim,c,top = data['rim'],int(data['center']),float(data['top'])
guess,normals = data['guess'],data['normals']

# ---- the cut sheet as a dolfinx mesh (vertices duplicated along the cuts)
domain = dmesh.create_mesh(MPI.COMM_SELF,F,ufl.Mesh(basix.ufl.element("Lagrange","triangle",1,shape = (2,))),V[:,:2])
tdim = 2
domain.topology.create_connectivity(1,0)
domain.topology.create_connectivity(0,2)
domain.topology.create_connectivity(tdim-1,tdim)
nv = domain.topology.index_map(0).size_local
# original index of every dolfinx vertex (dolfinx renumbers the input)
orig = domain.geometry.input_global_indices[dmesh.entities_to_geometry(domain,0,np.arange(nv,dtype = np.int32)).ravel()]
is_rim = np.zeros(len(V),dtype = bool)
is_rim[rim] = True
bnd = dmesh.exterior_facet_indices(domain.topology)
f2v = domain.topology.connectivity(1,0)
rim_facets = np.array([f for f in bnd if is_rim[orig[f2v.links(f)]].all()],dtype = np.int32)
v_centre = np.flatnonzero(orig == c).astype(np.int32)

# ---- Naghdi shell (as the FEniCSx-Shells demo, flat reference: phi0 = (x, y, 0), d0 = e3)
P2 = basix.ufl.element("Lagrange","triangle",2,shape = (3,))
B3 = basix.ufl.element("Bubble","triangle",3,shape = (3,))
P2B3 = basix.ufl.enriched_element([P2,B3])
P2r = basix.ufl.element("Lagrange","triangle",2,shape = (2,))
W = fem.functionspace(domain,basix.ufl.mixed_element([P2B3,P2r]))
q,dq,q_ = fem.Function(W),ufl.TrialFunction(W),ufl.TestFunction(W)
u,theta = ufl.split(q)
x = ufl.SpatialCoordinate(domain)
phi0 = ufl.as_vector([x[0],x[1],0.0])
director = lambda th: ufl.as_vector([ufl.sin(th[1])*ufl.cos(th[0]),-ufl.sin(th[0]),ufl.cos(th[1])*ufl.cos(th[0])])
d0 = ufl.as_vector([0.0,0.0,1.0])
F0 = ufl.grad(phi0)
Fd = ufl.grad(phi0+u)
d = director(theta)
eps = 0.5*(Fd.T*Fd-F0.T*F0)
kappa = 0.5*(Fd.T*ufl.grad(d)+ufl.grad(d).T*Fd)
gamma = Fd.T*d-F0.T*d0
E,nu,t = fem.Constant(domain,E_),fem.Constant(domain,nu_),fem.Constant(domain,t_)
mu = E/(2.0*(1.0+nu))
lmbda = 2.0*mu*nu/(1.0-2.0*nu)
lmbda_s = 2.0*mu*lmbda/(2.0*mu+lmbda) # plane stress
A = lambda X: 2.0*mu*X+lmbda_s*ufl.tr(X)*ufl.Identity(2)
psi_m = 0.5*t*ufl.inner(A(eps),eps)
psi_b = 0.5*(t**3/12.0)*ufl.inner(A(kappa),kappa)
psi_s = 0.5*t*mu*ufl.inner(gamma,gamma)
# PSRI: membrane and shear split by alpha = t^2/h^2, the (1 - alpha) part under-integrated
h = ufl.CellDiameter(domain)
alpha = t**2/h**2
dx_f = ufl.Measure("dx",domain = domain,metadata = {"quadrature_degree":4})
dx_r = ufl.Measure("dx",domain = domain,metadata = {"quadrature_degree":2})
Pi = psi_b*dx_f+alpha*(psi_m+psi_s)*dx_f+(1.0-alpha)*(psi_m+psi_s)*dx_r
Fr = ufl.derivative(Pi,q,q_)
J = ufl.derivative(Fr,q,dq)

# ---- supports: clamped rim, centre vertex pulled to height z
W0,_ = W.sub(0).collapse()
W1,_ = W.sub(1).collapse()
zero_u,zero_t,pull = fem.Function(W0),fem.Function(W1),fem.Function(W0)
dofs_rim_u = fem.locate_dofs_topological((W.sub(0),W0),1,rim_facets)
dofs_rim_t = fem.locate_dofs_topological((W.sub(1),W1),1,rim_facets)
dofs_c = fem.locate_dofs_topological((W.sub(0),W0),0,v_centre)
bcs = [fem.dirichletbc(zero_u,dofs_rim_u,W.sub(0)),fem.dirichletbc(zero_t,dofs_rim_t,W.sub(1)),
       fem.dirichletbc(pull,dofs_c,W.sub(0))]
dof_wc,pull_z = dofs_c[0][2],dofs_c[1][2] # the centre's vertical displacement in W and in W0 (dofs x, y, z)

# ---- first guess: the ribbon model, rotations from its normals
V1 = fem.functionspace(domain,("Lagrange",1,(3,)))
V1r = fem.functionspace(domain,("Lagrange",1,(2,)))
cells_v = domain.topology.connectivity(2,0)
v2dof = np.empty(nv,dtype = np.int32) # P1 dof of every vertex
for cell in range(domain.topology.index_map(2).size_local):
    v2dof[cells_v.links(cell)] = V1.dofmap.cell_dofs(cell)
n = normals*np.sign(normals[:,2:3]+1e-300)
beta0 = np.column_stack([-np.arcsin(np.clip(n[:,1],-1,1)),np.arctan2(n[:,0],n[:,2])]) # director(beta) = n

def p1(space, values, k):
    f = fem.Function(space)
    a = f.x.array.reshape(-1,k)
    a[v2dof] = values[orig]
    return f

ug,tg = p1(V1,guess-V,3),p1(V1r,beta0,2)
W0s,map0 = W.sub(0).collapse()
W1s,map1 = W.sub(1).collapse()
map0,map1 = np.asarray(map0).ravel(),np.asarray(map1).ravel()
ug0,tg0 = fem.Function(W0s),fem.Function(W1s)
ug0.interpolate(ug)
tg0.interpolate(tg)

problem = NonlinearProblem(Fr,q,bcs = bcs,J = J,petsc_options_prefix = "naghdi_",
                           petsc_options = {"snes_type":"newtonls","snes_linesearch_type":"basic",
                                            "snes_rtol":1e-8,"snes_atol":1e-8,"snes_stol":0.0,"snes_max_it":40,
                                            "ksp_type":"preonly","pc_type":"lu","pc_factor_mat_solver_type":"mumps"})
snes = problem.solver
residual = fem.form(Fr)
V1_out = fem.Function(V1)

for f in glob.glob('%s_naghdi_*.npy' % name): # shapes of an earlier run
    os.remove(f)
rows = []
start = time.time()
z_old,z_step = 0.0,top/steps
x_old = None # the last converged solution
while z_old < top-1e-9 and z_step > top/steps/64:
    z = min(z_old+z_step,top)
    pull.x.array[pull_z] = z
    if x_old is None and args.flat_start:
        q.x.array[:] = 0.0
    elif x_old is None:
        q.x.array[map0] = ug0.x.array*z/top
        q.x.array[map1] = tg0.x.array*z/top
    else: # predictor: scale the last converged shape to the new height
        q.x.array[:] = x_old*(z/z_old)
    problem.solve()
    reason,it = snes.getConvergedReason(),snes.getIterationNumber()
    if reason <= 0 or not np.isfinite(q.x.array).all(): # cut the step and try again
        z_step /= 2
        print('height %.2f mm failed (reason %d), step cut to %.3f mm' % (z,reason,z_step),flush = True)
        q.x.array[:] = x_old if x_old is not None else 0.0
        continue
    x_old,z_old = q.x.array.copy(),z
    if it < 8:
        z_step = min(1.5*z_step,top/steps)
    force = fem.assemble_vector(residual).array[dof_wc]
    rows.append((z,force,it))
    V1_out.interpolate(q.sub(0).collapse())
    X = V.copy()
    X[orig] += V1_out.x.array.reshape(-1,3)[v2dof]
    np.save('%s_naghdi_%02d.npy' % (name,len(rows)),X)
    print('height %.2f mm: force %.4f N, %d Newton its, %.0f s' % (z,force,it,time.time()-start),flush = True)
np.savetxt('%s_naghdi.csv' % name,np.array(rows),delimiter = ',',header = 'height_mm,force_N,newton_its',comments = '')
