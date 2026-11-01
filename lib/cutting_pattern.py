import os

import numpy as np
import igl
import matplotlib.pyplot as plt
from scipy.optimize import fsolve

from isolines import isolines

DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'data')

def spline_line_intersection(spline, th,center=(0,0)):
    # This function computes the intersection between a spline and line with slope th
    # and returns the parametric values of the spline at the intersection points
    # because is a close loop, we look for two instersetions
    # spline1 is a tuple (bspline, length)
    # make a first guess then refine with bisect\
    s  = np.linspace(0,1,1000)
    x_vals = spline(s)[:,0]-center[0]
    y_vals = spline(s)[:,1]-center[1]
    #find the first intersection
    index = np.argmax(x_vals*np.cos(th)+y_vals*np.sin(th)) # find the point with maximum projection on the line using the dot product
    
    sg = s[index]
    #finde the root using fsolve 
    def equations(s): 
        # function to find the root using the cross product between the ray and the coordinates
        coords = spline(s)
        return (coords[:,0]-center[0])*np.sin(th) - (coords[:,1]-center[1])*np.cos(th)
    
    root = fsolve(equations, sg)[0]
    return root % 1.0 # the closed isolines are periodic in s with period 1


def cutting_sections(isolevels,N = 6,center = None):
    # Return the t parameter for each isolevel where the line is cutterd by N lines for each isolevel
    # the order of the t parameter is based the position of the lines and this lines are equally spaced in angle
    # also they are ordered from 0 to 2pi
    s = np.linspace(0,1,100)
    center = center if center is not None else np.mean(isolevels[-1][0][0](s),axis=0)[:2]
    cuts = []

    for j in range(len(isolevels)):
        spline = isolevels[j][0][0]                
        s_intersection = []
        for i in range(N):
            th = 2*i*np.pi/N
            root = spline_line_intersection(spline, th,center)
            s_intersection.append(root)
        cuts.append(s_intersection)
    return cuts

def cutting_patern(isolevels, height, N = 6, center = None, overlap = None):
    # Given a set of isolevels and heights, return a cutting patern
    # the isolevels are a set of splines representing the isolines at different heights
    # height is the height of the isolines
    # N is the number of cuts to generate
    # The isolevels and height should be order from bottom to top
    # Also in nessesary, to define the center of the pattern
    # If center is None, the center of the last isolevel is used
    # overlap is an optional constant overlap length D (in the units of the mesh)
    # used for every beam instead of D = (delta*W)^(1/3)
    cuts = cutting_sections(isolevels, N, center)

    # Now plot the cutting create the cutting pattern and add the overlap to the s_intersection variables
    cuts_with_overlap = []
    for j in range(len(height)-1):
        spline = isolevels[j][0][0]
        delta = height[j+1]-height[j]
        lenght = isolevels[j][0][1]
        sub_cuts_overlap = []
        for i in range(len(cuts[j])):
            w = np.sqrt(np.sum((isolevels[j+1][0][0](cuts[j+1][i])[:2]-isolevels[j][0][0](cuts[j][i])[:2])**2)) # distance between the two isolines at this cut
            D = (delta*w)**(1/3) if overlap is None else overlap # overlap distance we assume that F/(NB) = 1 
            ovelap_parameter =  D/lenght
            # The side parametre is 1 if we are going up, -1 if we are going down, depending on j and i
            side = 1 if i%2==0 else -1
            side = side * (1 if j%2==1  else -1)
            new_cut = cuts[j][i]+ovelap_parameter*side
            sub_cuts_overlap.append(new_cut)

        cuts_with_overlap.append(sub_cuts_overlap)
    cut_spans = []
    for j in range(len(height)-1):
        from_to_spans = []
        for i in range(int(N/2)):   
            if j%2 ==0:
                from_s = cuts_with_overlap[j][2*i]
                to_s = cuts_with_overlap[j][2*i+1]
            else:
                from_s = cuts_with_overlap[j][2*i+1]
                to_s = cuts_with_overlap[j][2*(i+1)%N]
            if to_s<from_s:
                to_s +=1
            from_to_spans.append((from_s, to_s))
        cut_spans.append(from_to_spans)
    return cut_spans

def plot_cutting_pattern(isolevels, cutting_lines):
    fig, ax = plt.subplots()
    for j in range(len(isolevels)-1):
        spline = isolevels[j][0][0]
        # plot the cutting lines
        for span in cutting_lines[j]:
            from_s, to_s = span
            span_s = np.linspace(from_s, to_s, 100)
            ax.plot(spline(span_s)[:,0], spline(span_s)[:,1], 'r-', linewidth=2)
    ax.set_aspect('equal')
    plt.show()
            
def main():
    # Example of usage
    v, f = igl.read_triangle_mesh(os.path.join(DATA_DIR, 'parabole.obj')) # load a mesh
    h = v[:,2]
    vals = np.linspace(h.min()*1.05,h.max()*0.95,15)
    iso = isolines(v,f)
    lines = []
    for val in vals:
        line =iso.isolinecoords([np.array([0,0,1]),np.array([0,0,val])])
        lines.append(line)
    cutting_lines = cutting_patern(lines, vals, N = 6)
    
    # Plot the cutting pattern
    plot_cutting_pattern(lines, cutting_lines)


if __name__ == "__main__":
    main()
