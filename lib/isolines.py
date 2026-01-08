# -*- coding: utf-8 -*-
"""
Created on Tue Mar  5 10:39:08 2024

@author: Alejandro
"""

import igl
import numpy as np
from scipy.interpolate import make_interp_spline


class isolines:
    def __init__(self,v,f):
        self.v = v
        self.f = f
        
    def cross_surface(self, plane):
        """  Given a plane defined by the normal and the center position calcuate 
        which triangles are cuted by the plane
        
        input:
        
        plane = [normal,center]
        
        output:
            
        crossed triangle indices and midpoint in the triangle where was crossed
        
        """
        d = np.dot((self.v-plane[1]),plane[0]) # distance of the point to a plane
        d = d/np.abs(d)>0 # normalice to get the side
        cross_clasification = d[self.f].sum(axis = 1) # eval in the triangles and clasifie the croessin (which side)
        crossed_triangles = np.logical_and(cross_clasification<3,cross_clasification>0) # check if the triangle is crossed by the plane

        dtri2 = d[self.f] 
        # search for the crossing point --> some clasification of vertices and geometry
        index_source = ~dtri2*(cross_clasification==2).reshape(-1,1)+dtri2*(cross_clasification==1).reshape(-1,1)
        index_end = dtri2*(cross_clasification==2).reshape(-1,1)+~dtri2*(cross_clasification==1).reshape(-1,1)
        index_source = self.f[index_source]
        index_end = self.f[index_end].reshape(-1,2)
        v_source = self.v[index_source,:]
        v_end_1 =self.v[index_end[:,0]]
        v_end_2 = self.v[index_end[:,1]]
        t1 = np.dot(plane[1]-v_source,plane[0])/np.dot(v_end_1-v_source,plane[0])
        t1 = t1.reshape(-1,1)
        t2 = np.dot(plane[1]-v_source,plane[0])/np.dot(v_end_2-v_source,plane[0])
        t2 = t2.reshape(-1,1)
        p_cross = 0.5*(v_end_1*t1+v_end_2*t2 + v_source*(2-t1-t2))
        
        data = {}
        for tri,cord in zip(np.nonzero(crossed_triangles)[0], p_cross):
            data[tri] = cord
        return p_cross,np.nonzero(crossed_triangles)[0],data

    def conections(self,triangles_indices):
        """ given a set of triangles give the conection between triangles """
        conections = {}
        triangles = self.f[triangles_indices]
        for i,tri in zip(triangles_indices,triangles):
            flags = (triangles == tri[0])+ (triangles == tri[1])+(triangles == tri[2])
            index = np.where(flags.sum(axis = 1) ==2)[0]
            conections[i] = triangles_indices[index]
        return conections
    
    def sort_split(self,connections):
        """ look for a open curve"""
        triangles = list(connections.keys())
        loose_ends = [ind for ind in connections.keys() if len(connections[ind])==1]
        is_open = False
        periodic = []
        curves = [[]] # save the triangeles indices in lists for each curve
        if len(loose_ends) == 0: # check if we have just loops or opn curves:
            curves[-1].append(triangles[0])
            triangles.remove(triangles[0])
            is_open = True
        else:
            curves[-1].append(loose_ends[0])
            triangles.remove(loose_ends[0])
            loose_ends.remove(loose_ends[0])
            is_open = False
        
        while len(triangles)>0:
            c_element = curves[-1][-1]
            if c_element in connections:
                next_element = None
                is_sadle= False
                if len(connections[c_element])>2:
                    is_sadle = True
                for el in connections[c_element]:
                    if el in triangles:
                        next_element = el                        
                        break

            if next_element in triangles:
                curves[-1].append(next_element)
                if is_sadle == False:
                    triangles.remove(next_element)
                if next_element in loose_ends:
                    loose_ends.remove(next_element)

            if len(triangles) == 0:
                if is_open == True:
                    curves[-1].append(curves[-1][0])
                    periodic.append(True)
                else:
                    periodic.append(False)
            if next_element == None:
                if is_open == True:
                    curves[-1].append(curves[-1][0])
                    periodic.append(True)
                else:
                    periodic.append(False)
                curves.append([])
                if len(loose_ends) == 0: # check if we have just loops or opn curves:
                    curves[-1].append(triangles[0])
                    triangles.remove(triangles[0])
                    is_open = True
                else:
                    curves[-1].append(loose_ends[0])
                    triangles.remove(loose_ends[0])
                    loose_ends.remove(loose_ends[0])
                    is_open = False
        return curves,periodic
    def orient_curves(self,curves,look_up,periodic):
        """ given a set of curves orient them to have positive circulation """
        oriented_curves = []
        
        for curve,periodic_flag in zip(curves,periodic):
            curve_oriented = []
            if periodic_flag == True:
                position = []
                for t in curve:
                    position.append(look_up[t])
                position = np.array(position)
                delta_vectors = np.diff(position,axis =0)
                cross = position[:-1,0]*delta_vectors[:,1]-position[:-1,1]*delta_vectors[:,0]
                total_area = np.sum(cross)
                if total_area<0:
                    curve_oriented.append(np.flip(position, axis=0))
                else: 
                    curve_oriented.append(position)
            oriented_curves.append(curve_oriented)
        return oriented_curves
                    
                
                
    def isolinecoords(self,plane):
        _, crossed_triangles,look_up = self.cross_surface(plane)
        conections = self.conections(crossed_triangles)
        curves,periodic = self.sort_split(conections)
        curves_oriented = self.orient_curves(curves,look_up,periodic) # evaluate the orientation of the curves and return the coordinates 
        for points in curves_oriented:
            data = self.interpolate(points,periodic)
        return data
        
    def interpolate(self,coordiantes,periodic):
        splines = []
        for p,s_points in zip(periodic,coordiantes):
            dpoints = np.gradient(s_points,axis = 0)**2 
            l = np.cumsum(np.sqrt(dpoints.sum(axis = 1))) # arclength parameter
            lenght = l[-1]
            l = l/lenght
            if p == True:
                bspl = make_interp_spline(l, s_points, k=5,bc_type = 'periodic')
            else:
                bspl = make_interp_spline(l, s_points, k=5,bc_type = 'natural')
            splines.append([bspl,lenght])
        return splines
            
def main():
    v, f = igl.read_triangle_mesh('../data/offcenter.obj') # load a mesh
    h = v[:,2]
    vals = np.linspace(h.min()*1.1,h.max()*0.9,10)
    iso = isolines(v,f)
    lines = []
    s = np.linspace(0,1,200)
    import matplotlib.pyplot as plt
    for vals in vals:
        data =iso.isolinecoords([np.array([0,0,1]),np.array([0,0,vals])])
        for bspl,l in data:
              coord = bspl(s)
              lines.append(coord)    
              plt.plot(coord[:,0],coord[:,1],'-')
    plt.show()
    
   #ps.init() 
   
   #ps_mesh = ps.register_surface_mesh("my mesh", v, f, smooth_shade=True)
   #ps_cloud = ps.register_point_cloud("my points", v_iso)
   #ps.show()
    
        
    
if __name__ == "__main__":
    main()