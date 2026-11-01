""" Example 4: a cutting pattern for any target mesh (STL or OBJ) with a
minimum ribbon width, deployed with the ribbon model.
The target is sliced between --zmin and --zmax. The contour lines are spaced
by --delta, but a contour is moved up (--step mm at a time) until it is at least
--wmin from the one below everywhere, so no ribbon is narrower than wmin and
the steep parts get a larger height step. The pattern (overlap D = (delta W)^(1/3))
is then deployed with the ribbon model by displacement: the centre of its top
contour is moved up to the height of that contour, the force being an unknown,
and every level is compared with the target. --scale scales the mesh first
(zmin and zmax are heights of the scaled mesh), and the size of the cut sheet is
checked against the bed of the laser cutter (--bed).
    python examples/04_deploy_stl.py path/to/fuji.stl --zmin 10 --zmax 175 --wmin 2
    python examples/04_deploy_stl.py path/to/fuji.stl --scale 0.42 --zmin 4.2 --zmax 73.5 --delta 2 --step 0.5
Writes examples/output/04_deploy_stl/<name>_: pattern.png, cuts.svg (the cuts
and the outline, in mm, for the laser cutter), flat.obj, deployed.obj,
levels.png and levels.csv. """
import argparse
import os

import numpy as np

from common import output_dir, plot_pattern, finish, plt, igl
from isolines import isolines
from cutting_pattern import cutting_patern
from cut_mesh import kirigami_mesh, ribbon_widths, face_components
from ribbon_model import beams_from_pattern, RibbonModel

def contour(iso, z):
    return iso.isolinecoords([np.array([0,0,1.0]),np.array([0,0,z])])

def slice_min_width(iso, zmin, zmax, delta, wmin, step = 1.0):
    """ Heights from zmin up to at most zmax, at least delta apart, each moved up
    by step until its contour is at least wmin from the previous one. """
    heights,lines = [zmin],[contour(iso,zmin)]
    z = zmin+delta
    while z <= zmax+1e-9:
        line = contour(iso,z)
        if ribbon_widths([lines[-1],line])[0] >= wmin:
            heights.append(z)
            lines.append(line)
            z += delta
        else:
            z += step
    return np.array(heights),lines

def write_svg(path, sheet, lines, cut_spans, margin):
    # cuts in red and the outline of the sheet in blue, in mm
    s = np.linspace(0,1,2000)
    from cut_mesh import outer_boundary
    outline = outer_boundary(lines,margin,1.0)
    paths = []
    for j,spans in enumerate(cut_spans):
        for a,b in spans:
            p = lines[j][0][0](np.linspace(a,b,max(int((b-a)*lines[j][0][1]),8)))[:,:2]
            paths.append(('red',p,False))
    paths.append(('blue',outline,True))
    allp = np.vstack([p for _,p,_ in paths])
    lo,hi = allp.min(axis = 0)-5,allp.max(axis = 0)+5
    with open(path,'w') as f:
        f.write('<svg xmlns="http://www.w3.org/2000/svg" width="%.1fmm" height="%.1fmm" viewBox="0 0 %.3f %.3f">\n' % (
            hi[0]-lo[0],hi[1]-lo[1],hi[0]-lo[0],hi[1]-lo[1]))
        for color,p,closed in paths:
            q = np.column_stack([p[:,0]-lo[0],hi[1]-p[:,1]]) # y down in svg
            d = 'M'+' L'.join('%.3f,%.3f' % (x,y) for x,y in q)+(' Z' if closed else '')
            f.write('<path d="%s" fill="none" stroke="%s" stroke-width="0.1"/>\n' % (d,color))
        f.write('</svg>\n')
    return allp.max(axis = 0)-allp.min(axis = 0) # size of the sheet, mm

def main():
    parser = argparse.ArgumentParser(description = __doc__,formatter_class = argparse.RawDescriptionHelpFormatter)
    parser.add_argument('mesh',help = 'target mesh, STL or OBJ, z up, in mm')
    parser.add_argument('--zmin',type = float,required = True,help = 'height of the lowest contour (the rim), mm')
    parser.add_argument('--zmax',type = float,required = True,help = 'highest allowed contour, below any crater or second summit, mm')
    parser.add_argument('--delta',type = float,default = 4.0,help = 'smallest height step, mm')
    parser.add_argument('--wmin',type = float,default = 2.0,help = 'narrowest ribbon allowed, mm')
    parser.add_argument('--N',type = int,default = 8,help = 'number of sections (even)')
    parser.add_argument('--step',type = float,default = 1.0,help = 'how far a contour is moved up at a time to widen a ribbon, mm')
    parser.add_argument('--scale',type = float,default = 1.0,help = 'scale factor applied to the mesh before slicing')
    parser.add_argument('--bed',type = float,nargs = 2,default = (220.0,220.0),help = 'working area of the laser cutter, mm')
    parser.add_argument('--h',type = float,default = 1.0,help = 'mesh size of the flat sheet, mm')
    parser.add_argument('--young',type = float,default = 2100.0,help = 'MPa')
    parser.add_argument('--thickness',type = float,default = 0.1,help = 'mm')
    args = parser.parse_args()
    name = os.path.splitext(os.path.basename(args.mesh))[0]
    out = output_dir('04_deploy_stl')
    stem = os.path.join(out,name+'_')
    v,f = igl.read_triangle_mesh(args.mesh)
    v,_,_,f = igl.remove_duplicate_vertices(v,f,1e-6) # STL stores every triangle on its own
    v = v*args.scale
    if args.scale != 1.0:
        name += '_x%g' % args.scale
        stem = os.path.join(out,name+'_')
    iso = isolines(v,f)
    heights,lines = slice_min_width(iso,args.zmin,args.zmax,args.delta,args.wmin,args.step)
    widths = ribbon_widths(lines)
    print('%d contours from %.1f to %.1f mm, height steps %.1f to %.1f mm, ribbons %.2f to %.1f mm wide' % (
        len(heights),heights[0],heights[-1],np.diff(heights).min(),np.diff(heights).max(),widths.min(),widths.max()))
    spans = cutting_patern(lines,heights,args.N)
    plot_pattern(lines,spans,stem+'pattern.png','%s, N = %d, W >= %g mm' % (name,args.N,args.wmin))
    sheet = kirigami_mesh(lines,heights,args.N,h = args.h)
    size = write_svg(stem+'cuts.svg',sheet,lines,spans,widths[0])
    fits = size[0] <= args.bed[0] and size[1] <= args.bed[1] or size[1] <= args.bed[0] and size[0] <= args.bed[1]
    print('cut sheet %.1f x %.1f mm: %s the %g x %g mm bed' % (size[0],size[1],'fits' if fits else 'does NOT fit',*args.bed))
    igl.write_triangle_mesh(stem+'flat.obj',sheet['V'],sheet['F'])
    print('%d cuts, flat sheet of %d vertices, %d piece(s)' % (sum(len(s) for s in spans),len(sheet['V']),face_components(sheet['F'])))

    model = RibbonModel(beams_from_pattern(lines,spans,args.N),young = args.young,thickness = args.thickness)
    point = sheet['V'][int(sheet['center']),:2]
    target = heights-heights[0]
    q,force = model.pull_to(point,target[model.n_levels-1]) # by displacement: the force is an unknown
    part,coord = model.classify(sheet['V'],sheet['F'],lines,spans)
    X = model.deform(sheet['V'],sheet['F'],part,coord,q)
    igl.write_triangle_mesh(stem+'deployed.obj',X,sheet['F'])
    s = np.linspace(0,1,400,endpoint = False)
    z = []
    for j,(w,ax,ay) in enumerate(q):
        p = lines[j][0][0](s)[:,:2]
        zz = w+ax*p[:,1]-ay*p[:,0]
        z.append((zz.mean(),zz.min(),zz.max()))
    z = np.array(z)
    t = target[:model.n_levels]
    tilt = np.degrees(np.hypot(q[:,1],q[:,2]))
    b = model.beams
    rise = np.diff(heights)[b['level']]/b['L']
    print('rise of a level over the ribbon length 2D: median %.2f, max %.2f (the linear model needs it well below 1)' % (np.median(rise),rise.max()))
    print('centre moved to %.1f mm (force %.3f N): level height error rms %.2f mm, largest spread along a contour %.2f mm, largest tilt %.1f deg, peak bending stress %.0f MPa' % (
        t[-1],force,np.sqrt(np.mean((z[:,0]-t)**2)),(z[:,2]-z[:,1]).max(),tilt.max(),model.max_stress(q).max()))
    np.savetxt(stem+'levels.csv',np.column_stack([np.arange(model.n_levels),heights[:model.n_levels],t,z,tilt,widths[:model.n_levels]]),
               fmt = '%.4f',delimiter = ',',header = 'level,contour_z_mm,target_mm,mean_mm,min_mm,max_mm,tilt_deg,ribbon_width_mm',comments = '')
    fig,ax = plt.subplots(figsize = (4,3.3))
    ax.plot(t,t,'k-',lw = 0.8,label = 'target')
    ax.errorbar(t,z[:,0],yerr = [z[:,0]-z[:,1],z[:,2]-z[:,0]],fmt = 'o',ms = 3,capsize = 2,label = 'ribbon model (min to max along the contour)')
    ax.set_xlabel('target height of the contour (mm)')
    ax.set_ylabel('deployed height (mm)')
    ax.set_title('%s, centre moved to %.1f mm' % (name,t[-1]),fontsize = 9)
    ax.legend(frameon = False,fontsize = 7)
    fig.tight_layout()
    finish(fig,stem+'levels.png')

if __name__ == "__main__":
    main()
