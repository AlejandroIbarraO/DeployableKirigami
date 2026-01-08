# -*- coding: utf-8 -*-
"""
Created on Tue Nov 29 15:05:22 2022

@author: Alejandro
"""
from datetime import datetime
import numpy as np
from stl import mesh
from matplotlib import pyplot
import multiprocessing
class cutplane:
    def __init__(self,mesh,offset = 0 ,pich = 0,roll = 0):
        self.mesh =mesh
        self.offset = offset
        self.pich = pich
        self.roll = roll
    
    def conectivity(self):
        start_time = datetime.now()
        pool = multiprocessing.Pool(12)
        self.conection= list(pool.map(self.look_neigbour,self.mesh.vectors))
        end_time = datetime.now()
        print('Duration: {}'.format(end_time - start_time))
    def cross_plane(self,triangle, plane):
        """ Check if a given triangle cross the plane, the triangle it's defined by 
        three points and the plane by a normal vector and the ofset vector"""
        side = []
        for point in triangle:
            d = np.dot((point-plane[1]),plane[0])
            d = d > 0
            side.append(d)
        side = np.array(side)
        if sum(side) == 3 or sum(side) == 0:
            return False,None,None
        else:
            if sum(side)==2:
                l0 = triangle[~side][0]
                pl = triangle[side]
            else:
                l0 = triangle[side][0]
                pl = triangle[~side]
            cross = []
            for vec in pl:
                d = np.dot(plane[1]-l0,plane[0])/np.dot(vec-l0,plane[0])
                cross.append((vec-l0)*d+l0)
            pcross = np.array(cross).mean(axis = 0)
            return True,pcross,cross
    def look_neigbour(self,ctriangle):
        neig = []
        for index,tri in enumerate(self.mesh.vectors):
            if self.is_neigbour(ctriangle, tri)==2:
                neig.append(index)
        return neig
            
    def is_neigbour(self,ctriangle,otriangle):
        i = 0
        for cpoint in ctriangle:
            for opoint in otriangle:
                equal = cpoint == opoint
                if equal.all():
                    i = i+1
        return i
    def stl_cross(self,plane):
        ids = [] 
        ps = []
        p1 = []
        p2 = []
        for i,tri in enumerate(self.mesh.vectors):
            side,p0,p0l = self.cross_plane(tri, plane) 
            if side == True:
                ids.append(i)
                ps.append(p0)
                p1.append(p0l[0])
                p2.append(p0l[1])
        return ids,ps,p1,p2
    
    def curves(self,ids):
        listpath = []
        unsorted_ids = ids.copy() 
        unsorted_ids.remove(ids[0])
        path = [ids[0]]
        while len(unsorted_ids) > 0:
            connected = True
            print(path[-1],self.conection[path[-1]])
            for el in self.conection[path[-1]]:
               if el in unsorted_ids:
                   path.append(el)
                   unsorted_ids.remove(el)
                   connected = True
                   break
               else:
                   connected = False
            if connected == False:
               listpath.append(path)
               if len(unsorted_ids) > 0:
                   path = [unsorted_ids[0]]
                   unsorted_ids.remove(unsorted_ids[0])
                   
        return listpath
                     
                   
    def isocurves(self,ids):
        listpath = []
        unsorted_ids = ids.copy()
        while len(unsorted_ids) > 0:
            sorted_ids = [unsorted_ids[0]]
            unsorted_ids = unsorted_ids[1:].copy()
            for cids in sorted_ids:
                for oids in unsorted_ids:
                    if self.is_neigbour(self.mesh.vectors[cids], self.mesh.vectors[oids]) == 2:
                        sorted_ids.append(oids)
                        unsorted_ids.remove(oids)
                        break
            listpath.append(sorted_ids)
        return listpath
    def sortedpads(self,paths, ids):
        sortedpaths = []
        for path in paths:
            sortpath = []
            for pid in path:
                for i,idx in enumerate(ids):
                    #print(i)
                    if idx == pid:
                        sortpath.append(i)
            sortedpaths.append(sortpath)
        return sortedpaths    
    def getplane(self,offset = 0 ,pich = 0,roll = 0):
        if offset != None:
            self.offset = offset
        if pich != None:
            self.pich = pich
        if roll != None:
            self.roll = roll
        plane = [np.array([np.sin(self.roll)*np.cos(self.pich),np.sin(self.roll)*np.sin(self.pich),np.cos(self.roll)]),np.array([0,0,self.offset])]
        ids,ps = self.stl_cross(plane)
        paths = self.curves(ids)
        sortpath = self.sortedpads(paths,ids)
        return sortpath,ps

def main():
   your_mesh = mesh.Mesh.from_file('cone.stl')
   print('conections ...')
   cp = cutplane(your_mesh)
   # cp.conectivity()
   zmax = your_mesh.z.max()
   zmin = your_mesh.z.min()
   zoffs = np.linspace(0,zmax*0.9,5)
   for off in zoffs:
       #paths, ps = cp.getplane(off,0,0)
       ids,ps,p1,p2 = cp.stl_cross([np.array([0,0,1]),np.array([0,0,off])])
       
       ps = np.array(ps)
       psc = ps.mean(axis =0)
       th = np.arctan2(ps[:,1]-psc[1],ps[:,0]-psc[0])
       indx = np.argsort(th)
       indx = np.append(indx, indx[0])
       p1 = np.array(p1)
       p2 = np.array(p2)
       pyplot.plot(ps[indx,0],ps[indx,1],'o')
       #pyplot.plot(p1[:,0],p1[:,1],'s')
       #pyplot.plot(p2[:,0],p2[:,1],'x')
   
   return cp,ps,p1,p2
   
if __name__ == "__main__":
    cp,ps,p1,p2 = main()