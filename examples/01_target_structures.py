""" Example 1: cutting patterns for target structures.
Every target mesh in data/ is sliced into contour lines, turned into a cutting
pattern (overlap D = (delta W)^(1/3), see the README) and into the flat cut
sheet. Example 3 deploys them with the ribbon model.
    python examples/01_target_structures.py
Writes examples/output/01_target_structures/<name>_pattern.png, <name>_flat.obj
and <name>_sheet.npz. """
import argparse
import os

import numpy as np

from common import output_dir, plot_pattern, save_sheet, igl
from isolines import isolines
from cutting_pattern import cutting_patern
from cut_mesh import kirigami_mesh, ribbon_widths, face_components, DATA_DIR

TARGETS = ('cone','parabole','offcenter','trumpet','free_curve')

def slice_target(name, levels, top = 0.9):
    v,f = igl.read_triangle_mesh(os.path.join(DATA_DIR,name+'.obj'))
    z = v[:,2]
    heights = np.linspace(z.min()+0.05*np.ptp(z),z.min()+top*np.ptp(z),levels)
    iso = isolines(v,f)
    lines = [iso.isolinecoords([np.array([0,0,1]),np.array([0,0,h])]) for h in heights]
    return v,lines,heights

def main():
    parser = argparse.ArgumentParser(description = __doc__,formatter_class = argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--targets',nargs = '+',default = TARGETS)
    parser.add_argument('--levels',type = int,default = 15,help = 'number of contour lines')
    parser.add_argument('--top',type = float,default = 0.9,help = 'height of the top contour, as a fraction of the target height')
    parser.add_argument('--N',type = int,default = 6,help = 'number of sections (even)')
    parser.add_argument('--h',type = float,default = None,help = 'mesh size in mm (default a third of the narrowest ribbon, at least 0.5 mm)')
    args = parser.parse_args()
    out = output_dir('01_target_structures')
    for name in args.targets:
        v,lines,heights = slice_target(name,args.levels,args.top)
        spans = cutting_patern(lines,heights,args.N)
        plot_pattern(lines,spans,os.path.join(out,name+'_pattern.png'),name)
        widths = ribbon_widths(lines)
        if widths.min() < 1.0:
            print('  warning: the narrowest ribbon of %s is %.2f mm, below the 1 mm that can be laser cut' % (name,widths.min()))
        h = args.h if args.h is not None else max(widths.min()/3,0.5)
        sheet = kirigami_mesh(lines,heights,args.N,h = h)
        save_sheet(os.path.join(out,name+'_sheet.npz'),sheet)
        os.replace(os.path.join(out,name+'_sheet.obj'),os.path.join(out,name+'_flat.obj'))
        print('%-10s %d cuts, %d vertices, narrowest ribbon %.2f mm' % (name,sum(len(s) for s in spans),len(sheet['V']),sheet['widths'].min()))
        if face_components(sheet['F']) > 1:
            print('  warning: the cuts split the sheet into %d pieces (overlapping cuts on a small contour), try a lower --top' % face_components(sheet['F']))

if __name__ == "__main__":
    main()
