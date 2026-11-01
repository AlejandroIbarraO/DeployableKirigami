""" Reduced model of a kirigami sheet: every cut isoline is a rigid level, and
the N beams between two neighbouring isolines (the ribbon of that level) are
clamped-clamped elastic beams joining the two levels.

A beam is the strip of the band between isolines j and j+1 where both are cut:
near every section line the cut of isoline j and the cut of isoline j+1 overlap,
the strip between their two tips bends, and the tips are clamped into level j
and level j+1. The model is linear and out of plane: a level moves by a height
w and two small rotations (ax, ay) about the x and y axes, so a point (x, y) of
level j rises by w + ax*y - ay*x. Level 0 (the outer isoline, with the rim) is
clamped, and the last level carries the centre of the sheet.
Units as in the rest of the code: mm, N, MPa. """
import numpy as np
from matplotlib.path import Path

from cutting_pattern import cutting_sections

def _closest(spline, point, n = 2000):
    # closest point of a closed spline to a point, by sampling
    p = spline(np.linspace(0,1,n,endpoint = False))[:,:2]
    return p[np.argmin(np.sum((p-point)**2,axis = 1))]

def _arc_distance(a, b):
    # distance between two parameters of a closed curve, periodic with period 1
    d = np.abs(a-b) % 1.0
    return np.minimum(d,1-d)

def beams_from_pattern(isolevels, cut_spans, N, center = None):
    """ The beams of a cutting pattern (isolevels and cut_spans as given to and
    returned by cutting_patern). Returns a dict of arrays, one row per beam:
    level (the beam joins level j to level j+1), a and b (the clamped ends on
    level j and level j+1, xy), length L, width W and tangent t (from a to b). """
    sections = cutting_sections(isolevels,N,center)
    tips = [np.array([s for span in spans for s in span]) % 1.0 for spans in cut_spans]
    rows = {'level':[],'a':[],'b':[],'L':[],'W':[],'t':[]}
    for j in range(len(cut_spans)-1):
        lower,upper = isolevels[j][0][0],isolevels[j+1][0][0]
        for i in range(N):
            # the tip of each isoline next to section line i
            sa = tips[j][np.argmin(_arc_distance(tips[j],sections[j][i]))]
            sb = tips[j+1][np.argmin(_arc_distance(tips[j+1],sections[j+1][i]))]
            pa,pb = lower(sa)[:2],upper(sb)[:2]
            qa,qb = _closest(upper,pa),_closest(lower,pb)
            a,b = (pa+qa)/2,(pb+qb)/2 # ends on the mid-line of the band
            L = np.linalg.norm(b-a)
            rows['level'].append(j)
            rows['a'].append(a)
            rows['b'].append(b)
            rows['L'].append(L)
            rows['W'].append((np.linalg.norm(pa-qa)+np.linalg.norm(pb-qb))/2)
            rows['t'].append((b-a)/L)
    return {k:np.array(v) for k,v in rows.items()}

def _panels(F, keep):
    # connected components of the kept faces, joined through shared edges; -1 elsewhere
    from scipy.sparse import coo_matrix
    from scipy.sparse.csgraph import connected_components
    idx = np.flatnonzero(keep)
    e = np.sort(F[idx][:,[[0,1],[1,2],[2,0]]].reshape(-1,2),axis = 1)
    face = np.repeat(np.arange(len(idx)),3)
    _,inv = np.unique(e,axis = 0,return_inverse = True)
    order = np.argsort(inv.ravel(),kind = 'stable')
    inv = inv.ravel()[order]
    pair = np.flatnonzero(inv[1:] == inv[:-1])
    a,b = face[order][pair],face[order][pair+1]
    graph = coo_matrix((np.ones(len(a)),(a,b)),shape = (len(idx),len(idx)))
    out = -np.ones(len(F),int)
    out[idx] = connected_components(graph,directed = False)[1]
    return out

def _end_map(point, t):
    # rows: height, slope along t and twist about t of a level point, from (w, ax, ay)
    x,y = point
    return np.array([[1.0,y,-x],
                     [0.0,t[1],-t[0]],
                     [0.0,t[0],t[1]]])

def beam_stiffness(L, W, young, thickness, poisson):
    """ Stiffness of a beam on (w1, slope1, twist1, w2, slope2, twist2): Hermite
    bending with the plate rigidity B*W and Saint-Venant torsion G*W*h^3/3. """
    B = young*thickness**3/(12*(1-poisson**2))
    EI = B*W
    GJ = young/(2*(1+poisson))*W*thickness**3/3
    k = np.zeros((6,6))
    bend = EI/L**3*np.array([[12,6*L,-12,6*L],
                             [6*L,4*L**2,-6*L,2*L**2],
                             [-12,-6*L,12,-6*L],
                             [6*L,2*L**2,-6*L,4*L**2]])
    idx = [0,1,3,4]
    k[np.ix_(idx,idx)] = bend
    k[np.ix_([2,5],[2,5])] = GJ/L*np.array([[1,-1],[-1,1]])
    return k

class RibbonModel:
    """ Rigid levels joined by beams. beams is the output of beams_from_pattern,
    or any dict with the same arrays. """
    def __init__(self, beams, young = 2100.0, thickness = 0.1, poisson = 0.4):
        self.beams = beams
        self.n_levels = int(beams['level'].max())+2
        self.young,self.thickness,self.poisson = young,thickness,poisson
        self.K = self.assemble()

    def _maps(self, k):
        # the 6 beam end values from the 6 values of its two levels
        j = self.beams['level'][k]
        t = self.beams['t'][k]
        T = np.zeros((6,6))
        T[:3,:3] = _end_map(self.beams['a'][k],t)
        T[3:,3:] = _end_map(self.beams['b'][k],t)
        dofs = np.r_[3*j:3*j+6]
        return T,dofs

    def assemble(self):
        K = np.zeros((3*self.n_levels,3*self.n_levels))
        for k in range(len(self.beams['L'])):
            T,dofs = self._maps(k)
            kb = beam_stiffness(self.beams['L'][k],self.beams['W'][k],self.young,self.thickness,self.poisson)
            K[np.ix_(dofs,dofs)] += T.T@kb@T
        return K

    def solve(self, loads):
        """ loads: list of (level, (x, y), Fz) point forces. Level 0 is clamped.
        Returns q, the (n_levels, 3) motions (w, ax, ay) of the levels. """
        Q = np.zeros(3*self.n_levels)
        for j,(x,y),fz in loads:
            Q[3*j:3*j+3] += fz*np.array([1.0,y,-x])
        free = np.r_[3:3*self.n_levels]
        q = np.zeros(3*self.n_levels)
        q[free] = np.linalg.solve(self.K[np.ix_(free,free)],Q[free])
        return q.reshape(-1,3)

    def height(self, q, level, point):
        # height of a point of a level
        w,ax,ay = q[level]
        return w+ax*point[1]-ay*point[0]

    def pull(self, point, force = 1.0):
        """ Pull the last level up at point. Returns the level motions and the
        height of the pulled point. """
        q = self.solve([(self.n_levels-1,point,force)])
        return q,self.height(q,self.n_levels-1,point)

    def pull_to(self, point, height, level = None):
        """ Move the point of a level (the last one by default) to the given
        height, its rotations free: displacement control, with the force F as
        an unknown. Solves the bordered system [[K, g], [g^T, 0]] [q; -F] =
        [0; height], g = (1, y, -x) on the unknowns of that level, F being the
        Lagrange multiplier of the constraint. Returns the level motions and F. """
        j = self.n_levels-1 if level is None else level
        n = 3*self.n_levels
        g = np.zeros(n)
        g[3*j:3*j+3] = [1.0,point[1],-point[0]]
        free = np.r_[3:n]
        m = len(free)
        A = np.zeros((m+1,m+1))
        A[:m,:m] = self.K[np.ix_(free,free)]
        A[:m,m] = A[m,:m] = g[free]
        rhs = np.zeros(m+1)
        rhs[m] = height
        x = np.linalg.solve(A,rhs)
        q = np.zeros(n)
        q[free] = x[:m]
        return q.reshape(-1,3),-x[m]

    def stiffness(self, point):
        # force per unit height when the last level is pulled at point, N/mm
        return 1.0/self.pull(point,1.0)[1]

    def beam_forces(self, q):
        """ Shear force carried by every beam (N, positive when it pulls its
        upper level down) and the largest bending moment along it (N mm). """
        f,m = [],[]
        for k in range(len(self.beams['L'])):
            T,dofs = self._maps(k)
            kb = beam_stiffness(self.beams['L'][k],self.beams['W'][k],self.young,self.thickness,self.poisson)
            e = kb@(T@q.ravel()[dofs])
            f.append(e[3])
            m.append(max(abs(e[1]),abs(e[4])))
        return np.array(f),np.array(m)

    def max_stress(self, q):
        # peak bending stress of every beam, sigma = 6 M / (W h^2), MPa
        _,m = self.beam_forces(q)
        return 6*m/(self.beams['W']*self.thickness**2)

    def deployed_lines(self, isolevels, q, n = 200):
        """ 3D polylines of the cut isolines, each moved by the rigid motion of
        its level: a list of (n, 3) arrays. """
        s = np.linspace(0,1,n)
        lines = []
        for j in range(self.n_levels):
            p = isolevels[j][0][0](s)[:,:2]
            w,ax,ay = q[j]
            lines.append(np.column_stack([p,w+ax*p[:,1]-ay*p[:,0]]))
        return lines

    def classify(self, V, F, isolevels, cut_spans, n = 2000):
        """ Which part of the sheet every face of a flat cut mesh belongs to:
        part[f] = (level, -1) for a panel of that level, or (level, k) for
        beam k. Returns part and, for the beam faces, their coordinates
        along (s, 0 to 1) and across (e, mm) the beam. """
        c = V[F,:2].mean(axis = 1)
        n_cut = len(cut_spans)
        s = np.linspace(0,1,n,endpoint = False)
        curves = [isolevels[j][0][0](s)[:,:2] for j in range(n_cut+1)]
        inside = np.array([Path(p).contains_points(c) for p in curves])
        band = inside.sum(axis = 0)-1 # -1 outside the outer isoline, j between isolines j and j+1
        level = np.clip(band,0,n_cut-1)
        beam = -np.ones(len(F),int)
        coord = np.zeros((len(F),2))
        b = self.beams
        for k in range(len(b['L'])):
            j = b['level'][k]
            nrm = np.array([-b['t'][k][1],b['t'][k][0]])
            sk = (c-b['a'][k])@b['t'][k]/b['L'][k]
            ek = (c-b['a'][k])@nrm
            hit = (band == j)&(sk >= 0)&(sk <= 1)&(np.abs(ek) < b['W'][k])
            beam[hit] = k
            coord[hit] = np.column_stack([sk,ek])[hit]
        # a panel face of band j hangs from level j where isoline j is not cut, else from level j+1
        for j in range(n_cut-1):
            sel = (band == j)&(beam < 0)
            if not np.any(sel):
                continue
            near = s[np.argmin(((c[sel,None,:]-curves[j][None])**2).sum(axis = 2),axis = 1)]
            cut = np.zeros(sel.sum(),bool)
            for a,e in cut_spans[j]:
                cut |= (near-a) % 1.0 <= e-a
            level[np.flatnonzero(sel)[cut]] = j+1
        # the rule is unsure next to the cut tips, so every panel (a piece of the
        # sheet between beams) takes the level most of its faces were given
        panel = _panels(F,beam < 0)
        for c_ in np.unique(panel[panel >= 0]):
            sel = panel == c_
            level[sel] = np.bincount(level[sel]).argmax()
        level[beam >= 0] = b['level'][beam[beam >= 0]]
        return np.column_stack([level,beam]),coord

    def deform(self, V, F, part, coord, q):
        """ Heights of the vertices of the flat cut mesh: panels move with the
        rigid motion of their level, beams bend as the Hermite cubic between
        their clamped ends and twist linearly. Linear kinematics, so x and y
        stay where they are. Returns the deployed vertex positions. """
        X = V.copy()
        for f,(j,k) in enumerate(part):
            p = V[F[f],:2]
            if k < 0:
                w,ax,ay = q[j]
                X[F[f],2] = w+ax*p[:,1]-ay*p[:,0]
                continue
            T,dofs = self._maps(k)
            w1,d1,psi1,w2,d2,psi2 = T@q.ravel()[dofs]
            a,t,L = self.beams['a'][k],self.beams['t'][k],self.beams['L'][k]
            sv = np.clip((p-a)@t/L,0,1)
            ev = (p-a)@np.array([-t[1],t[0]])
            h1,h2,h3,h4 = 1-3*sv**2+2*sv**3,L*(sv-2*sv**2+sv**3),3*sv**2-2*sv**3,L*(sv**3-sv**2)
            X[F[f],2] = h1*w1+h2*d1+h3*w2+h4*d2+ev*((1-sv)*psi1+sv*psi2)
        return X

def fit_levels(V, X, F, part, n_levels):
    """ Best rigid motion (w, ax, ay) of every level of a deployed sheet X (from
    any simulation of the flat cut mesh V, F), fitted by least squares to the
    heights of the panel vertices of that level. part is RibbonModel.classify's.
    Returns an (n_levels, 3) array comparable with the q of RibbonModel. """
    q = np.zeros((n_levels,3))
    for j in range(n_levels):
        v = np.unique(F[(part[:,0] == j)&(part[:,1] < 0)])
        A = np.column_stack([np.ones(len(v)),V[v,1],-V[v,0]])
        q[j] = np.linalg.lstsq(A,X[v,2],rcond = None)[0]
    return q

def guided_stiffness(L, W, young, thickness, poisson):
    # stiffness of one beam with both ends kept flat, 12 B W / L^3 (paper Eq. 1 with D = L)
    return young*thickness**3*W/((1-poisson**2)*L**3)
