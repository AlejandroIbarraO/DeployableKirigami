# -*- coding: utf-8 -*-
"""
Created on Tue Mar  5 10:39:08 2024

@author: Alejandro
"""

import os

import igl
import numpy as np
from scipy.interpolate import make_interp_spline

DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'data')


class isolines:
    def __init__(self,v,f,k = 3):
        self.v = v
        self.f = f
        self.k = k # degree of the interpolating splines

    def cross_surface(self, plane):
        """ Given a plane defined by the normal and the center position calculate
        where the plane crosses the edges of the mesh

        input:

        plane = [normal,center]

        output:

        crossed triangle indices, for each crossed triangle the keys of its two
        crossed edges, and a look up table edge key -> crossing point
        """
        d = np.dot((self.v-plane[1]),plane[0]) # signed distance of the vertices to the plane
        side = d > 0 # a vertex lying on the plane counts as below, so every crossed edge has d[a] != d[b]
        n_v = len(self.v)

        crossed_triangles = np.nonzero(np.isin(side[self.f].sum(axis = 1),(1,2)))[0]
        edge_keys = []
        look_up = {}
        for tri in crossed_triangles:
            keys = []
            for a,b in ((0,1),(1,2),(2,0)):
                va,vb = self.f[tri,a],self.f[tri,b]
                if side[va] == side[vb]:
                    continue
                key = min(va,vb)*n_v+max(va,vb)
                if key not in look_up:
                    t = d[va]/(d[va]-d[vb])
                    look_up[key] = self.v[va]+t*(self.v[vb]-self.v[va]) # exact crossing point on the edge
                keys.append(key)
            edge_keys.append(keys)
        return crossed_triangles,edge_keys,look_up

    def chain_edges(self,edge_keys):
        """ Link the crossed edges through the crossed triangles into curves.
        Each crossed triangle joins its two crossed edges, so every edge has two
        neighbours inside the mesh and one on the boundary of the mesh. Curves
        that start and end on the boundary are open, the rest are closed loops. """
        neighbours = {}
        for e1,e2 in edge_keys:
            neighbours.setdefault(e1,[]).append(e2)
            neighbours.setdefault(e2,[]).append(e1)

        visited = set()
        curves = []
        closed = []
        # open curves first, starting from an edge on the boundary of the mesh
        starts = [e for e in neighbours if len(neighbours[e]) == 1]+list(neighbours)
        for start in starts:
            if start in visited:
                continue
            curve = [start]
            visited.add(start)
            current = start
            while True:
                next_edges = [e for e in neighbours[current] if e not in visited]
                if len(next_edges) == 0:
                    break
                current = next_edges[0]
                curve.append(current)
                visited.add(current)
            is_closed = len(curve) > 2 and start in neighbours[curve[-1]]
            curves.append(curve)
            closed.append(is_closed)
        return curves,closed

    def orient_curves(self,curves,look_up,closed,normal):
        """ given a set of curves return their coordinates, with the closed
        curves oriented to have positive circulation around the normal """
        oriented_curves = []
        for curve,closed_flag in zip(curves,closed):
            position = np.array([look_up[e] for e in curve])
            if closed_flag:
                area = np.cross(position,np.roll(position,-1,axis = 0)).sum(axis = 0)
                if np.dot(area,normal) < 0:
                    position = np.flip(position,axis = 0)
            oriented_curves.append(position)
        return oriented_curves

    def isolinecoords(self,plane):
        """ Return a list of [spline, length] for every curve where the plane
        cuts the mesh, the longest curve first. Splines are parametrised by
        the normalised arc length s; closed curves are periodic in s with
        period 1, so spline(s) and spline(s+1) are the same point. """
        _,edge_keys,look_up = self.cross_surface(plane)
        curves,closed = self.chain_edges(edge_keys)
        curves_oriented = self.orient_curves(curves,look_up,closed,plane[0])
        data = []
        for points,closed_flag in zip(curves_oriented,closed):
            spline = self.interpolate(points,closed_flag)
            if spline is not None:
                data.append(spline)
        data.sort(key = lambda spline: -spline[1])
        return data

    def interpolate(self,points,closed):
        """ Interpolate the points of a curve with a spline of degree k
        parametrised by the normalised chord length. Closed curves use periodic
        boundary conditions so the curve and its derivatives match at the seam. """
        # drop repeated points, for example when the plane passes through a vertex
        nxt = np.roll(points,-1,axis = 0) if closed else points[1:]
        seg = np.linalg.norm(nxt-points[:len(nxt)],axis = 1)
        keep = seg > 1e-9*seg.max()
        if not closed:
            keep = np.r_[keep,True] # always keep the last point of an open curve
        points = points[keep]
        if closed:
            points = np.vstack([points,points[:1]]) # close the curve, periodicity needs the first point repeated at the end
        if len(points) < self.k+2:
            return None
        l = np.r_[0,np.cumsum(np.linalg.norm(np.diff(points,axis = 0),axis = 1))] # chord length parameter, starts at 0
        lenght = l[-1]
        l = l/lenght
        if closed:
            bspl = make_interp_spline(l, points, k=self.k,bc_type = 'periodic')
        else:
            bspl = make_interp_spline(l, points, k=self.k,bc_type = 'natural' if self.k == 3 else None)
        s = np.linspace(0,1,20*len(points))
        lenght = np.linalg.norm(np.diff(bspl(s),axis = 0),axis = 1).sum() # arc length of the spline
        return [bspl,lenght]

def circle_levels(radii, center = (0.0,0.0), n = 200, k = 3):
    """ Concentric circles in the format of isolinecoords: one [[spline, length]]
    per radius, counter-clockwise and periodic in s with period 1. Give the radii
    from the outside in (bottom to top), to build a pattern without a target mesh. """
    s = np.linspace(0,1,n+1)
    levels = []
    for r in radii:
        points = np.column_stack([center[0]+r*np.cos(2*np.pi*s),center[1]+r*np.sin(2*np.pi*s),np.zeros(n+1)])
        points[-1] = points[0]
        levels.append([[make_interp_spline(s,points,k = k,bc_type = 'periodic'),2*np.pi*r]])
    return levels

def main():
    v, f = igl.read_triangle_mesh(os.path.join(DATA_DIR, 'offcenter.obj')) # load a mesh
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


if __name__ == "__main__":
    main()
