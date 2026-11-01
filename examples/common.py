""" Helpers shared by the examples: paths and plots. """
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0,os.path.join(HERE,'..','lib'))
import matplotlib
NOTEBOOK = 'ipykernel' in sys.modules
if not NOTEBOOK:
    matplotlib.use('Agg') # the scripts only write files, the notebooks also show the figures
import matplotlib.pyplot as plt
import igl


def output_dir(name):
    path = os.path.join(HERE,'output',name)
    os.makedirs(path,exist_ok = True)
    return path

def finish(fig, filename):
    # save the figure, and show it in a notebook
    fig.savefig(filename,dpi = 200)
    if NOTEBOOK:
        plt.show()
    else:
        plt.close(fig)

def plot_pattern(isolevels, cut_spans, filename, title = None):
    # contour lines in grey, cuts in red
    s = np.linspace(0,1,400)
    fig,ax = plt.subplots(figsize = (4,4))
    for level in isolevels:
        p = level[0][0](s)
        ax.plot(p[:,0],p[:,1],color = '0.8',lw = 0.5)
    for j,spans in enumerate(cut_spans):
        for a,b in spans:
            p = isolevels[j][0][0](np.linspace(a,b,200))
            ax.plot(p[:,0],p[:,1],'r-',lw = 1)
    ax.set_aspect('equal')
    ax.set_xlabel('x (mm)')
    ax.set_ylabel('y (mm)')
    if title:
        ax.set_title(title)
    fig.tight_layout()
    finish(fig,filename)

def save_sheet(path, sheet):
    np.savez(path,**{k:sheet[k] for k in ('V','F','rim','center','cut_edges','cut_id','parent')})
    igl.write_triangle_mesh(path.replace('.npz','.obj'),sheet['V'],sheet['F'])

def constant_overlap_sheet(radius = 30.0, width = 3.0, inner = 6.0, overlap = 2.0, N = 6, h = None):
    """ Concentric circular contour lines from radius down to inner, width apart,
    cut with the same overlap D for every beam. Returns the contour lines, the
    cutting pattern and the flat cut sheet (h defaults to width/3). """
    from isolines import circle_levels
    from cutting_pattern import cutting_patern
    from cut_mesh import kirigami_mesh
    radii = np.arange(radius,inner-1e-9,-width)
    lines = circle_levels(radii)
    heights = np.arange(len(radii),dtype = float) # not used with a constant overlap
    spans = cutting_patern(lines,heights,N,overlap = overlap)
    sheet = kirigami_mesh(lines,heights,N,h = h if h is not None else width/3,overlap = overlap)
    return lines,spans,sheet
