import os

import numpy as np
import igl
import triangle
from shapely.geometry import Polygon
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components

from isolines import isolines
from cutting_pattern import cutting_patern

DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'data')

def ribbon_widths(isolevels, n = 400):
    # distance between consecutive isolines (the ribbon width W of each level)
    s = np.linspace(0,1,n,endpoint = False)
    widths = []
    for j in range(len(isolevels)-1):
        p = isolevels[j][0][0](s)[:,:2]
        q = isolevels[j+1][0][0](s)[:,:2]
        d = np.linalg.norm(p[:,None,:]-q[None,:,:],axis = 2)
        widths.append(d.min())
    return np.array(widths)

def resample(points, h, closed = False):
    # resample a polyline with segments of length close to h
    if closed:
        points = np.vstack([points,points[:1]])
    l = np.r_[0,np.cumsum(np.linalg.norm(np.diff(points,axis = 0),axis = 1))]
    n = max(int(np.ceil(l[-1]/h)),3)
    t = np.linspace(0,l[-1],n+1)
    new = np.column_stack([np.interp(t,l,points[:,i]) for i in range(points.shape[1])])
    return new[:-1] if closed else new

def cut_polylines(isolevels, cut_spans, h):
    # sample every cut span (from_s, to_s) of the cutting pattern as a flat polyline
    cuts = []
    for j,spans in enumerate(cut_spans):
        spline,lenght = isolevels[j][0]
        for from_s,to_s in spans:
            n = max(int(np.ceil(lenght*(to_s-from_s)/h)),2)
            cuts.append(spline(np.linspace(from_s,to_s,n+1))[:,:2])
    return cuts

def outer_boundary(isolevels, margin, h):
    # the sheet is the outermost isoline grown by margin, so the cuts on it stay inside the sheet
    s = np.linspace(0,1,1000,endpoint = False)
    polygon = Polygon(isolevels[0][0][0](s)[:,:2]).buffer(margin,quad_segs = 32)
    return resample(np.array(polygon.exterior.coords)[:-1],h,closed = True)

def triangulate_sheet(boundary, cuts, center, h):
    """ Constrained Delaunay triangulation of the flat sheet with the cuts as
    internal segments. Segment markers: 1 for the outer boundary, 2+i for cut i. """
    vertices = [boundary]
    segments = []
    markers = []
    nb = len(boundary)
    segments.append(np.column_stack([np.arange(nb),(np.arange(nb)+1)%nb]))
    markers.append(np.ones(nb,dtype = int))
    offset = nb
    for i,cut in enumerate(cuts):
        n = len(cut)
        vertices.append(cut)
        segments.append(offset+np.column_stack([np.arange(n-1),np.arange(1,n)]))
        markers.append(np.full(n-1,2+i))
        offset += n
    vertices.append(np.reshape(center,(1,2))) # the pulling point is a vertex of the mesh
    pslg = dict(vertices = np.vstack(vertices),
                segments = np.vstack(segments),
                segment_markers = np.concatenate(markers)[:,None])
    mesh = triangle.triangulate(pslg,'pq30a%f' % (0.5*h*h))
    V = mesh['vertices']
    F = mesh['triangles']
    E = mesh['segments']
    M = mesh['segment_markers'].ravel()
    center_vertex = np.argmin(np.linalg.norm(V-center,axis = 1))
    return V,F,E,M,center_vertex

def split_along_cuts(V, F, cut_edges):
    """ Duplicate the vertices along the cuts so the faces on either side of a
    cut no longer share vertices. Around each cut vertex the incident faces are
    grouped by the non-cut edges they share; every group after the first gets
    its own copy of the vertex. The tips of a cut have a single group, so the
    sheet stays connected there. Returns the new vertices, faces and, for every
    vertex, the index of the original vertex it was copied from. """
    F = F.copy()
    cut_set = set(map(tuple,np.sort(cut_edges,axis = 1)))
    cut_vertices = np.unique(cut_edges)
    vertex_faces = [[] for _ in range(len(V))]
    for fi,face in enumerate(F):
        for v in face:
            vertex_faces[v].append(fi)
    parent = list(range(len(V)))
    new_vertices = []
    for v in cut_vertices:
        faces = vertex_faces[v]
        # union-find over the faces around v, joined through the edges (v,w) that are not cut
        group = {fi:fi for fi in faces}
        def find(a):
            while group[a] != a:
                group[a] = group[group[a]]
                a = group[a]
            return a
        by_edge = {}
        for fi in faces:
            for w in F[fi]:
                if w != v:
                    by_edge.setdefault(w,[]).append(fi)
        for w,fs in by_edge.items():
            if len(fs) == 2 and (min(v,w),max(v,w)) not in cut_set:
                group[find(fs[0])] = find(fs[1])
        roots = sorted(set(find(fi) for fi in faces))
        for root in roots[1:]:
            copy = len(V)+len(new_vertices)
            new_vertices.append(V[v])
            parent.append(v)
            for fi in faces:
                if find(fi) == root:
                    F[fi][F[fi] == v] = copy
    V = np.vstack([V]+new_vertices) if new_vertices else V
    return V,F,np.array(parent)

def face_components(F):
    # number of pieces of the mesh, two faces are connected if they share an edge
    edges = np.vstack([F[:,[0,1]],F[:,[1,2]],F[:,[2,0]]])
    face_id = np.tile(np.arange(len(F)),3)
    key = np.sort(edges,axis = 1)
    order = np.lexsort((key[:,1],key[:,0]))
    key,face_id = key[order],face_id[order]
    same = np.all(key[1:] == key[:-1],axis = 1)
    a,b = face_id[:-1][same],face_id[1:][same]
    graph = coo_matrix((np.ones(len(a)),(a,b)),shape = (len(F),len(F)))
    return connected_components(graph,directed = False)[0]

def boundary_loops(F):
    # number of boundary loops, a boundary edge belongs to a single face
    edges = np.sort(np.vstack([F[:,[0,1]],F[:,[1,2]],F[:,[2,0]]]),axis = 1)
    unique,count = np.unique(edges,axis = 0,return_counts = True)
    boundary = unique[count == 1]
    vertices,index = np.unique(boundary,return_inverse = True)
    index = index.reshape(-1,2)
    graph = coo_matrix((np.ones(len(index)),(index[:,0],index[:,1])),shape = (len(vertices),len(vertices)))
    return connected_components(graph,directed = False)[0]

def kirigami_mesh(isolevels, height, N = 6, h = None, margin = None, center = None, overlap = None):
    """ Build the flat kirigami sheet for a cutting pattern: a triangle mesh in
    the plane z = 0 whose faces are disconnected along every cut.
    h is the target edge length (default a third of the narrowest ribbon) and
    margin the width of material outside the outermost isoline (default the
    outermost ribbon width). overlap is an optional constant overlap length
    (see cutting_patern). Returns a dict with the mesh and its labels. """
    cut_spans = cutting_patern(isolevels,height,N,center,overlap)
    widths = ribbon_widths(isolevels)
    h = h if h is not None else widths.min()/3
    margin = margin if margin is not None else widths[0]
    s = np.linspace(0,1,100)
    center = center if center is not None else np.mean(isolevels[-1][0][0](s),axis=0)[:2]
    cuts = cut_polylines(isolevels,cut_spans,h)
    boundary = outer_boundary(isolevels,margin,h)
    V2,F,E,M,center_vertex = triangulate_sheet(boundary,cuts,center,h)
    rim = np.unique(E[M == 1])
    cut_edges = E[M >= 2]
    V2,F,parent = split_along_cuts(V2,F,cut_edges)
    V = np.column_stack([V2,np.zeros(len(V2))])
    return dict(V = V, F = F, parent = parent, rim = rim, center = center_vertex,
                cut_edges = cut_edges, cut_id = M[M >= 2]-2, cut_spans = cut_spans,
                widths = widths, h = h)

def main():
    v, f = igl.read_triangle_mesh(os.path.join(DATA_DIR, 'parabole.obj')) # load a mesh
    h = v[:,2]
    vals = np.linspace(h.min()*1.05,h.max()*0.95,15)
    iso = isolines(v,f)
    lines = []
    for val in vals:
        lines.append(iso.isolinecoords([np.array([0,0,1]),np.array([0,0,val])]))
    sheet = kirigami_mesh(lines,vals,N = 6)
    V,F = sheet['V'],sheet['F']
    n_cuts = sum(len(spans) for spans in sheet['cut_spans'])
    n_loops = boundary_loops(F)
    print('%d vertices, %d faces, %d cuts' % (len(V),len(F),n_cuts))
    print('pieces: %d, boundary loops: %d (outer rim + one per cut)' % (face_components(F),n_loops))
    igl.write_triangle_mesh('parabole_kirigami.obj',V,F)

if __name__ == "__main__":
    main()
