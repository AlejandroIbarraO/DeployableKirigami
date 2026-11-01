""" Slit annular plate (Sze et al. 2004) with the DOLFINx (FEniCSx) Naghdi shell
of naghdi_deploy.py: P2 + bubble displacement, P2 rotations, PSRI.
Annulus Ri = 6, Ro = 10, h = 0.03, E = 21e6, slit along theta = 0: the lip
theta = 2 pi clamped, the lip theta = 0 loaded by a vertical line force q up to
0.8 (dead load). Reference at q = 0.8, nu = 0: w_A = 13.891 (r = 6), w_B = 17.528
(r = 10). Writes slit_naghdi_<nr>x<nt>_nu<nu>.csv (q, w_A, w_B).
    python check_slit_plate.py nr nt [steps] [nu] """
import sys
import time

import basix.ufl
import numpy as np
import ufl
from dolfinx import fem, mesh as dmesh
from dolfinx.fem.petsc import NonlinearProblem
from mpi4py import MPI

nr,nt = int(sys.argv[1]),int(sys.argv[2])
steps = int(sys.argv[3]) if len(sys.argv) > 3 else 40
nu_ = float(sys.argv[4]) if len(sys.argv) > 4 else 0.0
Ri,Ro,t_,E_,q_max = 6.0,10.0,0.03,21e6,0.8
domain = dmesh.create_rectangle(MPI.COMM_SELF,[[Ri,0.0],[Ro,2*np.pi]],[nr,nt],dmesh.CellType.triangle,
                                diagonal = dmesh.DiagonalType.crossed)
fdim = 1
loaded = dmesh.locate_entities_boundary(domain,fdim,lambda x: np.isclose(x[1],0.0))
clamped = dmesh.locate_entities_boundary(domain,fdim,lambda x: np.isclose(x[1],2*np.pi))
ent = np.r_[loaded,clamped]
order = np.argsort(ent) # meshtags needs sorted entities
tags = dmesh.meshtags(domain,fdim,ent[order],np.r_[np.full(len(loaded),1),np.full(len(clamped),2)].astype(np.int32)[order])
vA = dmesh.locate_entities(domain,0,lambda x: np.isclose(x[0],Ri)&np.isclose(x[1],0.0)) # before mapping:
vB = dmesh.locate_entities(domain,0,lambda x: np.isclose(x[0],Ro)&np.isclose(x[1],0.0)) # the lips coincide after
domain.topology.create_connectivity(0,2)
X = domain.geometry.x
r,th = X[:,0].copy(),X[:,1].copy()
X[:,0],X[:,1] = r*np.cos(th),r*np.sin(th)

P2 = basix.ufl.element("Lagrange","triangle",2,shape = (3,))
B3 = basix.ufl.element("Bubble","triangle",3,shape = (3,))
P2r = basix.ufl.element("Lagrange","triangle",2,shape = (2,))
W = fem.functionspace(domain,basix.ufl.mixed_element([basix.ufl.enriched_element([P2,B3]),P2r]))
q,dq,q_ = fem.Function(W),ufl.TrialFunction(W),ufl.TestFunction(W)
u,theta = ufl.split(q)
x = ufl.SpatialCoordinate(domain)
phi0 = ufl.as_vector([x[0],x[1],0.0])
d0 = ufl.as_vector([0.0,0.0,1.0])
F0 = ufl.grad(phi0)
Fd = ufl.grad(phi0+u)
d = ufl.as_vector([ufl.sin(theta[1])*ufl.cos(theta[0]),-ufl.sin(theta[0]),ufl.cos(theta[1])*ufl.cos(theta[0])])
eps = 0.5*(Fd.T*Fd-F0.T*F0)
kappa = 0.5*(Fd.T*ufl.grad(d)+ufl.grad(d).T*Fd)
gamma = Fd.T*d-F0.T*d0
E,nu,t = fem.Constant(domain,E_),fem.Constant(domain,nu_),fem.Constant(domain,t_)
mu = E/(2.0*(1.0+nu))
lmbda = 2.0*mu*nu/(1.0-2.0*nu)
A = lambda Xs: 2.0*mu*Xs+(2.0*mu*lmbda/(2.0*mu+lmbda))*ufl.tr(Xs)*ufl.Identity(2)
psi_m = 0.5*t*ufl.inner(A(eps),eps)
psi_b = 0.5*(t**3/12.0)*ufl.inner(A(kappa),kappa)
psi_s = 0.5*t*mu*ufl.inner(gamma,gamma)
h = ufl.CellDiameter(domain)
alpha = t**2/h**2
dx_f = ufl.Measure("dx",domain = domain,metadata = {"quadrature_degree":4})
dx_r = ufl.Measure("dx",domain = domain,metadata = {"quadrature_degree":2})
ds = ufl.Measure("ds",domain = domain,subdomain_data = tags)
qload = fem.Constant(domain,0.0)
Pi = psi_b*dx_f+alpha*(psi_m+psi_s)*dx_f+(1.0-alpha)*(psi_m+psi_s)*dx_r-qload*u[2]*ds(1)
Fr = ufl.derivative(Pi,q,q_)
J = ufl.derivative(Fr,q,dq)
W0,_ = W.sub(0).collapse()
W1,_ = W.sub(1).collapse()
bcs = [fem.dirichletbc(fem.Function(W0),fem.locate_dofs_topological((W.sub(0),W0),fdim,clamped),W.sub(0)),
       fem.dirichletbc(fem.Function(W1),fem.locate_dofs_topological((W.sub(1),W1),fdim,clamped),W.sub(1))]
problem = NonlinearProblem(Fr,q,bcs = bcs,J = J,petsc_options_prefix = "slit_",
                           petsc_options = {"snes_type":"newtonls","snes_linesearch_type":"basic",
                                            "snes_rtol":1e-8,"snes_atol":1e-9,"snes_stol":0.0,"snes_max_it":40,
                                            "ksp_type":"preonly","pc_type":"lu","pc_factor_mat_solver_type":"mumps"})
dA = fem.locate_dofs_topological(W.sub(0),0,vA)[2]
dB = fem.locate_dofs_topological(W.sub(0),0,vB)[2]
rows = []
start = time.time()
for qi in q_max*np.arange(1,steps+1)/steps:
    qload.value = qi
    problem.solve()
    it,reason = problem.solver.getIterationNumber(),problem.solver.getConvergedReason()
    wA,wB = q.x.array[dA],q.x.array[dB]
    rows.append((qi,wA,wB))
    print('q %.3f: w_A %.4f, w_B %.4f, %d its, reason %d, %.0f s' % (qi,wA,wB,it,reason,time.time()-start),flush = True)
np.savetxt('slit_naghdi_%dx%d_nu%g.csv' % (nr,nt,nu_),np.array(rows),delimiter = ',',header = 'q,w_A,w_B',comments = '')
