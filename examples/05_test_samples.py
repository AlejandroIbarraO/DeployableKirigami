""" Example 5: a set of test samples to check the ribbon theory with tensile
tests. Every sample is a pattern of concentric circles with the same overlap D
and ribbon width W on every level (as in example 2), with an uncut rim to clamp
and an uncut central pad to pull. The series change one parameter at a time
around a base sample, so the measured stiffness can be checked against
    k = E h^3 W N / (8 (1 - nu^2) M D^3)
(Supplementary Material, Section S3): D series (k ~ 1/D^3), N series (k ~ N) and W series
(k ~ W/M, the outer radius fixed). For every sample the script gives the
linear stiffness of the ribbon model and of the closed form above.
Every sample fits the 220 x 220 mm bed of an xTool F1 Ultra.
    python examples/05_test_samples.py
Writes examples/output/05_test_samples/: <name>.svg (cuts in red, outline and
holes in blue, label in black for engraving), patterns.png and
predictions.csv. """
import argparse
import os

import numpy as np

from common import output_dir, finish, plt
from isolines import circle_levels
from cutting_pattern import cutting_patern
from ribbon_model import beams_from_pattern, RibbonModel

BED = 220.0 # mm, the bed of an xTool F1 Ultra
BASE = dict(N = 6,D = 3.0,W = 4.0)
SERIES = [('D',[2.0,3.0,4.0,5.0]),('N',[4,6,8]),('W',[3.0,4.0,6.0])]

def samples(radius, inner):
    """ Names and parameters of the samples, the base sample once. """
    out = {}
    for key,values in SERIES:
        for v in values:
            p = dict(BASE,**{key:v})
            out['N%d_D%g_W%g' % (p['N'],p['D'],p['W'])] = dict(p,R = radius,inner = inner)
    return out

def pattern(p):
    # contour lines and cut spans of a sample
    radii = np.arange(p['R'],p['inner']-1e-9,-p['W'])
    lines = circle_levels(radii)
    spans = cutting_patern(lines,np.arange(len(radii),dtype = float),p['N'],overlap = p['D'])
    return radii,lines,spans

def connector_gap(radii, N, D):
    # shortest uncut arc between two cuts of the same contour (on the innermost cut contour)
    return 2*np.pi*radii[-2]/N-2*D

def circle(c, r, n = 120):
    t = np.linspace(0,2*np.pi,n,endpoint = False)
    return np.column_stack([c[0]+r*np.cos(t),c[1]+r*np.sin(t)])

def sample_paths(p, lines, spans, rim, holes):
    """ (colour, polyline, closed) of a sample centred at 0: the cuts, the outline
    of the rim, the clamping holes in the rim and the pulling holes in the pad. """
    paths = []
    for j,sp in enumerate(spans):
        for a,b in sp:
            paths.append(('red',lines[j][0][0](np.linspace(a,b,max(int((b-a)*lines[j][0][1]),8)))[:,:2],False))
    R = p['R']
    paths.append(('blue',circle((0,0),R+rim,360),True))
    for k in range(holes):
        t = 2*np.pi*(k+0.5)/holes
        paths.append(('blue',circle((0.5*(2*R+rim)*np.cos(t),0.5*(2*R+rim)*np.sin(t)),1.6,40),True))
    for x in (0.0,5.0,10.0): # pull at the centre, or 5 and 10 mm off centre
        if x+1.5 < p['inner']-1.0:
            paths.append(('blue',circle((x,0),1.5,40),True))
    return paths

def write_svg(path, groups, size):
    """ groups: list of (offset, paths, label, label position). Black text is meant
    to be engraved, red and blue lines to be cut (cuts first, outline last). """
    with open(path,'w') as f:
        f.write('<svg xmlns="http://www.w3.org/2000/svg" width="%.1fmm" height="%.1fmm" viewBox="0 0 %.3f %.3f">\n' % (size[0],size[1],size[0],size[1]))
        for off,paths,label,at in groups:
            for color in ('red','blue'):
                for c,q,closed in paths:
                    if c != color:
                        continue
                    q = np.column_stack([q[:,0]+off[0],size[1]-(q[:,1]+off[1])]) # y down in svg
                    d = 'M'+' L'.join('%.3f,%.3f' % (x,y) for x,y in q)+(' Z' if closed else '')
                    f.write('<path d="%s" fill="none" stroke="%s" stroke-width="0.1"/>\n' % (d,color))
            f.write('<text x="%.2f" y="%.2f" font-family="Arial" font-size="3" text-anchor="middle" fill="black">%s</text>\n' % (
                at[0]+off[0],size[1]-(at[1]+off[1]),label))
        f.write('</svg>\n')

def predictions(p, lines, spans, young, thickness):
    """ Linear stiffness of the ribbon model and of the closed form. """
    beams = beams_from_pattern(lines,spans,p['N'])
    linear = RibbonModel(beams,young = young,thickness = thickness)
    k = linear.stiffness(np.zeros(2))
    M = linear.n_levels-1
    k_formula = young*thickness**3*p['W']*p['N']/(8*(1-0.4**2)*M*p['D']**3)
    return dict(M = M,beam_length = beams['L'].mean(),k = k,k_formula = k_formula)

def main():
    parser = argparse.ArgumentParser(description = __doc__,formatter_class = argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--radius',type = float,default = 70.0,help = 'outer cut contour, mm')
    parser.add_argument('--inner',type = float,default = 14.0,help = 'radius of the central pad, mm')
    parser.add_argument('--rim',type = float,default = 12.0,help = 'width of the uncut rim to clamp, mm')
    parser.add_argument('--holes',type = int,default = 8,help = 'clamping holes in the rim (3.2 mm)')
    parser.add_argument('--young',type = float,default = 2100.0,help = 'MPa')
    parser.add_argument('--thickness',type = float,default = 0.1,help = 'mm')
    args = parser.parse_args()
    out = output_dir('05_test_samples')
    diameter = 2*(args.radius+args.rim)
    assert diameter <= BED-10, 'a sample is %.0f mm wide, the bed is %.0f mm' % (diameter,BED)
    rows,drawn = [],[]
    for name,p in samples(args.radius,args.inner).items():
        radii,lines,spans = pattern(p)
        gap = connector_gap(radii,p['N'],p['D'])
        if gap < p['W']/2:
            print('%s: skipped, the uncut connectors are only %.1f mm long' % (name,gap))
            continue
        paths = sample_paths(p,lines,spans,args.rim,args.holes)
        label = '%s h%g' % (name,args.thickness)
        write_svg(os.path.join(out,name+'.svg'),[((diameter/2+1,diameter/2+1),paths,label,(0,-(p['R']+args.rim/2)))],
                  (diameter+2,diameter+2))
        r = predictions(p,lines,spans,args.young,args.thickness)
        rows.append((name,p['N'],p['D'],p['W'],r['M'],r['beam_length'],gap,r['k'],r['k_formula']))
        drawn.append((name,paths))
        print('%s: M = %d levels, beams %.1f mm, connectors %.1f mm, k = %.2e N/mm (formula %.2e)' % (
            name,r['M'],r['beam_length'],gap,r['k'],r['k_formula']),flush = True)
    np.savetxt(os.path.join(out,'predictions.csv'),np.array(rows,dtype = object),fmt = '%s',delimiter = ',',
               header = 'name,N,D_mm,W_mm,M,beam_length_mm,connector_mm,k_ribbon_N_per_mm,k_formula_N_per_mm',
               comments = '')
    n = len(drawn)
    cols = 5
    fig,axes = plt.subplots(int(np.ceil(n/cols)),cols,figsize = (2.2*cols,2.3*np.ceil(n/cols)))
    for ax in axes.ravel():
        ax.axis('off')
    for ax,(name,paths) in zip(axes.ravel(),drawn):
        for c,q,closed in paths:
            q = np.vstack([q,q[:1]]) if closed else q
            ax.plot(q[:,0],q[:,1],'-',color = c,lw = 0.4)
        ax.set_aspect('equal')
        ax.set_title(name,fontsize = 7)
    fig.tight_layout()
    finish(fig,os.path.join(out,'patterns.png'))

if __name__ == "__main__":
    main()
