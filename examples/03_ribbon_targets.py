""" Example 3: the target structures of example 1 deployed with the ribbon model.
Every target is sliced and cut as in example 1, the beams are read from its
cutting pattern, and the centre of the pattern (the centre of the top contour) is moved up to the
height of its contour line (deployment by displacement, the force an unknown).
Every level is then compared with the height of its contour line on the target.
No simulation is needed: all five targets take a few seconds.
    python examples/03_ribbon_targets.py
Writes examples/output/03_ribbon_targets/: <name>_ribbon.obj (the cut sheet
moved by the model), <name>_levels.png and levels.csv. """
import argparse
import importlib
import os

import numpy as np

from common import output_dir, finish, plt, igl
from cutting_pattern import cutting_patern
from cut_mesh import kirigami_mesh, ribbon_widths
from ribbon_model import beams_from_pattern, RibbonModel

example1 = importlib.import_module('01_target_structures')

def deploy_target(name, levels, top, N, young, thickness):
    """ Pull the pattern of a target with the ribbon model. Returns the sheet,
    the model, the level motions q and the heights of the contour lines,
    measured from the lowest one. """
    v,lines,heights = example1.slice_target(name,levels,top)
    spans = cutting_patern(lines,heights,N)
    sheet = kirigami_mesh(lines,heights,N,h = max(ribbon_widths(lines).min()/3,0.5))
    model = RibbonModel(beams_from_pattern(lines,spans,N),young = young,thickness = thickness)
    point = sheet['V'][int(sheet['center']),:2]
    target = heights-heights[0]
    q,force = model.pull_to(point,target[model.n_levels-1]) # by displacement, to the height of the top cut contour
    return sheet,lines,spans,model,q,force,target

def level_heights(lines, q, n = 400):
    # heights of every cut contour line moved by its level: (mean, min, max) per level
    s = np.linspace(0,1,n,endpoint = False)
    out = []
    for j,(w,ax,ay) in enumerate(q):
        p = lines[j][0][0](s)[:,:2]
        z = w+ax*p[:,1]-ay*p[:,0]
        out.append((z.mean(),z.min(),z.max()))
    return np.array(out)

def main():
    parser = argparse.ArgumentParser(description = __doc__,formatter_class = argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--targets',nargs = '+',default = example1.TARGETS)
    parser.add_argument('--levels',type = int,default = 15)
    parser.add_argument('--top',type = float,default = 0.9)
    parser.add_argument('--N',type = int,default = 6)
    parser.add_argument('--young',type = float,default = 2100.0,help = 'MPa')
    parser.add_argument('--thickness',type = float,default = 0.1,help = 'mm')
    args = parser.parse_args()
    out = output_dir('03_ribbon_targets')
    rows = []
    for name in args.targets:
        sheet,lines,spans,model,q,force,target = deploy_target(name,args.levels,args.top,args.N,args.young,args.thickness)
        V,F = sheet['V'],sheet['F']
        part,coord = model.classify(V,F,lines,spans)
        igl.write_triangle_mesh(os.path.join(out,name+'_ribbon.obj'),model.deform(V,F,part,coord,q),F)
        z = level_heights(lines,q)
        t = target[:model.n_levels]
        err = z[:,0]-t
        tilt = np.degrees(np.hypot(q[:,1],q[:,2]))
        print('%-10s %2d levels, force %.3f N, level height error rms %.2f mm (max %.2f mm) of %.1f mm, '
              'largest spread along a contour %.2f mm, largest tilt %.1f deg' % (
              name,model.n_levels,force,np.sqrt(np.mean(err**2)),np.abs(err).max(),t[-1],(z[:,2]-z[:,1]).max(),tilt.max()))
        for j in range(model.n_levels):
            rows.append((name,j,t[j],z[j,0],z[j,1],z[j,2],tilt[j]))
        fig,ax = plt.subplots(figsize = (4,3.3))
        ax.plot(t,t,'k-',lw = 0.8,label = 'target')
        ax.errorbar(t,z[:,0],yerr = [z[:,0]-z[:,1],z[:,2]-z[:,0]],fmt = 'o',ms = 3,capsize = 2,label = 'ribbon model (min to max along the contour)')
        ax.set_xlabel('target height of the contour (mm)')
        ax.set_ylabel('deployed height (mm)')
        ax.set_title('%s, pulled with %.3f N' % (name,force),fontsize = 9)
        ax.legend(frameon = False,fontsize = 7)
        fig.tight_layout()
        finish(fig,os.path.join(out,name+'_levels.png'))
    np.savetxt(os.path.join(out,'levels.csv'),np.array(rows,dtype = object),fmt = '%s',delimiter = ',',
               header = 'target,level,target_mm,mean_mm,min_mm,max_mm,tilt_deg',comments = '')

if __name__ == "__main__":
    main()
