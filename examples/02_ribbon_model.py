""" Example 2: the reduced ribbon model (lib/ribbon_model.py) on a pattern of
concentric circles with the same overlap D for every beam. Every cut
isoline is a rigid level and the N beams between two isolines are clamped into
them, so a central pull gives the paper's Eq. 1 summed over the levels. The
script compares that linear stiffness with Eq. 1, then pulls the same pattern off centre to show the levels tilting.
    python examples/02_ribbon_model.py
Writes examples/output/02_ribbon_model/: force_height.png, off_centre.png,
deployment.png (the cut sheet moved by the model: panels rigid, beams bent) and
centre_XX.obj / off_centre_XX.obj for the viewers. """
import argparse
import os

import numpy as np

from common import output_dir, finish, plt, constant_overlap_sheet, igl
from ribbon_model import beams_from_pattern, RibbonModel, guided_stiffness

def main():
    parser = argparse.ArgumentParser(description = __doc__,formatter_class = argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--radius',type = float,default = 30.0,help = 'outer contour, mm')
    parser.add_argument('--width',type = float,default = 3.0,help = 'ribbon width W, mm')
    parser.add_argument('--overlap',type = float,default = 2.0,help = 'overlap D, mm')
    parser.add_argument('--N',type = int,default = 6)
    parser.add_argument('--young',type = float,default = 2100.0,help = 'MPa')
    parser.add_argument('--thickness',type = float,default = 0.1,help = 'mm')
    parser.add_argument('--offset',type = float,default = 5.0,help = 'distance of the off-centre pull, mm')
    parser.add_argument('--height',type = float,default = 10.0,help = 'pulled height of the 3D views, mm')
    args = parser.parse_args()
    out = output_dir('02_ribbon_model')
    lines,spans,sheet = constant_overlap_sheet(args.radius,args.width,overlap = args.overlap,N = args.N)
    beams = beams_from_pattern(lines,spans,args.N,center = (0.0,0.0))
    model = RibbonModel(beams,young = args.young,thickness = args.thickness)
    k = model.stiffness((0.0,0.0))
    k_beam = guided_stiffness(beams['L'],beams['W'],args.young,args.thickness,0.4)
    k_eq1 = 1/np.sum(1/np.bincount(beams['level'],k_beam))
    print('%d levels, %d beams, beam length %.2f mm (twice the overlap)' % (model.n_levels,len(beams['L']),beams['L'].mean()))
    print('central pull: %.4f N/mm (Eq. 1 summed over the levels: %.4f N/mm)' % (k,k_eq1))
    q,H = model.pull((0.0,0.0),k*10.0)
    print('at 10 mm: peak bending stress %.0f MPa' % model.max_stress(q).max())

    fig,ax = plt.subplots(figsize = (4.5,3.5))
    h = np.linspace(0,args.height,50)
    ax.plot(h,k*h,'-',label = 'ribbon model')
    ax.plot(h,k_eq1*h,'k:',label = 'Eq. 1 summed over the levels')
    ax.set_xlabel('height of the centre (mm)')
    ax.set_ylabel('pulling force (N)')
    ax.legend(frameon = False)
    fig.tight_layout()
    finish(fig,os.path.join(out,'force_height.png'))

    # off-centre pull: every level tilts towards the pulled side
    point = (args.offset,0.0)
    q,H = model.pull(point,1.0)
    tilt = np.degrees(np.hypot(q[:,1],q[:,2])*H**-1*10.0) # tilt at 10 mm of pulled height
    print('pull at x = %g mm: %.4f N/mm, top level tilted %.1f deg at 10 mm' % (args.offset,1/H,tilt[-1]))
    fig,(a1,a2) = plt.subplots(1,2,figsize = (8,3.3))
    for j,line in enumerate(model.deployed_lines(lines,q*10.0/H)):
        a1.plot(line[:,0],line[:,2],color = plt.cm.viridis(j/model.n_levels))
    a1.plot(point[0],10.0,'rx')
    a1.set_xlabel('x (mm)')
    a1.set_ylabel('z (mm)')
    a1.set_title('side view, pulled to 10 mm at x = %g mm' % args.offset,fontsize = 9)
    a2.plot(np.arange(model.n_levels),tilt,'o-')
    a2.set_xlabel('level (0 = rim)')
    a2.set_ylabel('tilt (deg)')
    fig.tight_layout()
    finish(fig,os.path.join(out,'off_centre.png'))
    plot_deployment(model,sheet,lines,spans,args.height,args.offset,out)

def draw_sheet(ax, X, F, part, n_levels, title, box = None):
    # panels shaded by level, beams in orange
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection
    keep = np.ones(len(F),bool) if box is None else np.all((X[F,0] > box[0])&(X[F,0] < box[1])&(X[F,1] > box[2])&(X[F,1] < box[3]),axis = 1)
    colors = np.where(part[:,1:2] >= 0,np.array([[0.95,0.5,0.1,1.0]]),plt.cm.Blues(0.3+0.6*part[:,0:1]/n_levels)[:,0])
    faces = Poly3DCollection(X[F[keep]],facecolors = colors[keep],edgecolor = 'none')
    ax.add_collection3d(faces)
    P = X[np.unique(F[keep])]
    lo,hi = P.min(axis = 0),P.max(axis = 0)
    ax.set_xlim(lo[0],hi[0])
    ax.set_ylim(lo[1],hi[1])
    ax.set_zlim(lo[2],hi[2])
    ax.set_box_aspect(np.maximum(hi-lo,1e-3))
    ax.view_init(elev = 28,azim = -60)
    ax.set_xlabel('x (mm)',fontsize = 7)
    ax.set_ylabel('y (mm)',fontsize = 7)
    ax.set_zlabel('z (mm)',fontsize = 7)
    ax.tick_params(labelsize = 6)
    ax.set_title(title,fontsize = 9)

def plot_deployment(model, sheet, lines, spans, height, offset, out):
    """ The flat cut sheet moved by the model at one pulled height: centred
    pull, off-centre pull, and a close-up of the centred pull. """
    V,F = sheet['V'],sheet['F']
    part,coord = model.classify(V,F,lines,spans)
    frames = {}
    for name,point in (('centre',(0.0,0.0)),('off_centre',(offset,0.0))):
        q,H = model.pull(point,1.0)
        frames[name] = model.deform(V,F,part,coord,q*height/H)
        igl.write_triangle_mesh(os.path.join(out,'%s_%02d.obj' % (name,round(height))),frames[name],F)
    fig = plt.figure(figsize = (11,3.8))
    draw_sheet(fig.add_subplot(1,3,1,projection = '3d'),frames['centre'],F,part,model.n_levels,'pulled at the centre to %g mm' % height)
    draw_sheet(fig.add_subplot(1,3,2,projection = '3d'),frames['off_centre'],F,part,model.n_levels,'pulled at x = %g mm to %g mm' % (offset,height))
    r = np.abs(V[:,:2]).max()
    draw_sheet(fig.add_subplot(1,3,3,projection = '3d'),frames['centre'],F,part,model.n_levels,'close-up: beams (orange) between rigid levels',box = (0,r,0,r))
    fig.tight_layout()
    finish(fig,os.path.join(out,'deployment.png'))

if __name__ == "__main__":
    main()
