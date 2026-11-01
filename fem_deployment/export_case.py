""" Step 1 (the environment of the repository, with libigl, triangle and
shapely): build the cut sheet of a target and its linear ribbon model
deployment, and write everything the FEniCSx shell needs to one .npz file.
The target is either a mesh of data/ (sliced as in examples/01_target_structures.py)
or any STL/OBJ mesh (sliced with a minimum ribbon width as in
examples/04_deploy_stl.py). The ribbon model is deployed by displacement: the
centre of the top level is moved to the height of its contour (RibbonModel.pull_to).
    python fem_deployment/export_case.py cone --h 1
    python fem_deployment/export_case.py path/to/fuji.stl --scale 0.45 --zmin 4.5 --zmax 78.75 --delta 2 --step 0.5 --N 8 --h 1
Writes fem_deployment/output/case_<name>_h<h>.npz with
  V (n,3) flat sheet, F (m,3) triangles (vertices duplicated along the cuts),
  rim clamped vertices, center pulled vertex, top height of the top level,
  guess (n,3) the ribbon model deployment (first guess of the shell),
  normals (n,3) its vertex normals, ribbon_force, width narrowest ribbon,
  part, coord, q, heights (to compare the shell with the ribbon model). """
import argparse
import importlib
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path[:0] = [os.path.join(HERE,'..','lib'),os.path.join(HERE,'..','examples')]
import numpy as np
import common # noqa: F401, sets the paths of the repository
from common import igl
from isolines import isolines
from cutting_pattern import cutting_patern
from cut_mesh import kirigami_mesh
from ribbon_model import beams_from_pattern, RibbonModel

example1 = importlib.import_module('01_target_structures')
example4 = importlib.import_module('04_deploy_stl')

def vertex_normals(X, F):
    n = np.cross(X[F[:,1]]-X[F[:,0]],X[F[:,2]]-X[F[:,0]]) # area weighted
    N = np.zeros_like(X)
    for k in range(3):
        np.add.at(N,F[:,k],n)
    return N/np.linalg.norm(N,axis = 1)[:,None]

def main():
    parser = argparse.ArgumentParser(description = __doc__,formatter_class = argparse.RawDescriptionHelpFormatter)
    parser.add_argument('target',help = 'a target of data/ (cone, parabole, offcenter, trumpet, free_curve) or a path to an STL/OBJ mesh')
    parser.add_argument('--h',type = float,default = 1.0,help = 'mesh size of the flat sheet, mm')
    parser.add_argument('--N',type = int,default = 6,help = 'number of sections (even)')
    parser.add_argument('--levels',type = int,default = 15,help = 'data/ targets: number of contour lines')
    parser.add_argument('--top',type = float,default = 0.9,help = 'data/ targets: top contour, fraction of the height')
    parser.add_argument('--scale',type = float,default = 1.0,help = 'mesh files: scale factor')
    parser.add_argument('--zmin',type = float,default = 10.0,help = 'mesh files: lowest contour, mm')
    parser.add_argument('--zmax',type = float,default = 175.0,help = 'mesh files: highest contour, mm')
    parser.add_argument('--delta',type = float,default = 2.0,help = 'mesh files: smallest height step, mm')
    parser.add_argument('--wmin',type = float,default = 2.0,help = 'mesh files: narrowest ribbon, mm')
    parser.add_argument('--step',type = float,default = 0.5,help = 'mesh files: a contour moves up by this until it is wmin wide, mm')
    parser.add_argument('--young',type = float,default = 2100.0,help = 'MPa')
    parser.add_argument('--thickness',type = float,default = 0.1,help = 'mm')
    args = parser.parse_args()
    if os.path.isfile(args.target):
        name = os.path.splitext(os.path.basename(args.target))[0]
        v,f = igl.read_triangle_mesh(args.target)
        v,_,_,f = igl.remove_duplicate_vertices(v,f,1e-6)
        v = v*args.scale
        if args.scale != 1.0:
            name += '_x%g' % args.scale
        heights,lines = example4.slice_min_width(isolines(v,f),args.zmin,args.zmax,args.delta,args.wmin,args.step)
    else:
        name = args.target
        v,lines,heights = example1.slice_target(name,args.levels,args.top)
    heights = np.asarray(heights)
    spans = cutting_patern(lines,heights,args.N)
    sheet = kirigami_mesh(lines,heights,args.N,h = args.h)
    V,F = sheet['V'],sheet['F']
    model = RibbonModel(beams_from_pattern(lines,spans,args.N),young = args.young,thickness = args.thickness)
    part,coord = model.classify(V,F,lines,spans)
    c = int(sheet['center'])
    top = (heights-heights[0])[model.n_levels-1]
    q,force = model.pull_to(V[c,:2],top) # by displacement: the force is an unknown
    guess = model.deform(V,F,part,coord,q)
    out = os.path.join(HERE,'output')
    os.makedirs(out,exist_ok = True)
    path = os.path.join(out,'case_%s_h%g.npz' % (name,args.h))
    np.savez(path,V = V,F = F,rim = sheet['rim'],center = c,top = top,guess = guess,
             normals = vertex_normals(guess,F),ribbon_force = force,width = sheet['widths'].min(),
             part = part,coord = coord,q = q,heights = heights)
    print('%s: %d vertices, %d triangles, ribbons from %.2f mm (%.1f rows of triangles), top level at %.2f mm, ribbon model %.3f N -> %s' % (
        name,len(V),len(F),sheet['widths'].min(),sheet['widths'].min()/(args.h*np.sqrt(3)/2),top,force,path))

if __name__ == '__main__':
    main()
