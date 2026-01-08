import numpy as np
import igl
from isolines import isolines
from scipy.optimize import bisect, fsolve
import matplotlib.pyplot as plt

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
    index = np.argmax(x_vals*np.cos(th)+y_vals*np.sin(th))
    
    sg = s[index]
    # the other should be in the oposite side, there fore if s1 more than 0.5 subtract 0.5, else add 0.5

    def equations(s):
        coords = spline(s)
        return (coords[:,0]-center[0])*np.sin(th) - (coords[:,1]-center[1])*np.cos(th)
    
    root = sg#fsolve(equations, sg)[0]
    return root

def cutting_patern(isolevels, height, N = 6, center = None):
    # Given a set of isolevels and heights, return a cutting patern
    # the isolevels are a set of splines representing the isolines at different heights
    # height is the height of the isolines
    # N is the number of cuts to generate
    # The isolevels and height should be order from bottom to top
    # Also in nessesary, to define the center of the pattern
    # If center is None, the center of the last isolevel is used
    s = np.linspace(0,1,100)
    center = center if center is not None else np.mean(isolevels[-1][0][0](s),axis=0)[:2]
    #print("center:", center)
    cuts = []

    for j in range(len(isolevels)):
        spline = isolevels[j][0][0]
        x_vals = spline(s)[:,0]
        y_vals = spline(s)[:,1]
                
        s_intersection = []
        for i in range(N):
            th = 2*i*np.pi/N
            root = spline_line_intersection(spline, th,center)
            s_intersection.append(root)
        cuts.append(s_intersection)
    # Now plot the cutting create the cutting pattern and add the overlap to the s_intersection variables
    cuts_with_overlap = []
    for j in range(len(height)-1):
        spline = isolevels[j][0][0]
        delta = height[j+1]-height[j]
        lenght = isolevels[j][0][1]
        sub_cuts_overlap = []
        for i in range(len(cuts[j])):
            #print(root1_low, root2_low)
            w = np.sqrt(np.sum((isolevels[j+1][0][0](cuts[j+1][i])[:2]-isolevels[j][0][0](cuts[j][i])[:2])**2)) # distance between the two isolines at this cut
            D = (delta*w)**(1/3) # overlap distance we assume that F/(NB) = 1 
            print(D)
            ovelap_parameter =  D/lenght
            # The side parametre is 1 if we are going up, -1 if we are going down, depending on j and i
            side = 1 if i%2==0 else -1
            side = side * (1 if j%2==1  else -1)
            new_cut = cuts[j][i]+ovelap_parameter*side
            sub_cuts_overlap.append(new_cut)

        cuts_with_overlap.append(sub_cuts_overlap)
    #print("cuts with overlap:", cuts_with_overlap)
    for j in range(len(height)-1):
        spline = isolevels[j][0][0]
        for i in range(int(N/2)):   
            if j%2 ==0:
                #print(j,i,2*i,2*i+1,cuts_with_overlap[j][2*i], cuts_with_overlap[j][2*i+1])
                from_s = cuts_with_overlap[j][2*i]
                to_s = cuts_with_overlap[j][2*i+1]
                if to_s<from_s:
                    to_s +=1
                s_span = np.linspace(from_s, to_s, 100)
                plt.plot(spline(s_span)[:,0],spline(s_span)[:,1],'-r',label = '{},{}'.format(j,i)) # add overlap to the cut
            else:
                #print(j,i,2*i+1,2*(i+1)%N,cuts_with_overlap[j][2*i+1], cuts_with_overlap[j][2*(i+1)%N])
                from_s = cuts_with_overlap[j][2*i+1]
                to_s = cuts_with_overlap[j][2*(i+1)%N]
                if to_s<from_s:
                    to_s +=1
                s_span = np.linspace(from_s, to_s, 100)
                plt.plot(spline(s_span)[:,0],spline(s_span)[:,1],'-b',label = '{},{}'.format(j,i)) # add overlap to the cut
            # add overlap to the cut
    plt.show()
    
            
def main():
    v, f = igl.read_triangle_mesh('../data/trumpet.obj') # load a mesh
    h = v[:,2]
    vals = np.linspace(h.min()*1.05,h.max()*0.95,15)
    iso = isolines(v,f)
    lines = []
    for val in vals:
        line =iso.isolinecoords([np.array([0,0,1]),np.array([0,0,val])])
        lines.append(line)
    cutting_patern(lines, vals, N = 6)
if __name__ == "__main__":
    main()
            

    