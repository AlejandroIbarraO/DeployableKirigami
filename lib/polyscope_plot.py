import numpy as np
import polyscope as ps
from isolines import isolines
import igl

def main():
    v, f = igl.read_triangle_mesh('../data/offcenter.obj') # load a mesh
    h = v[:,2]
    vals = np.linspace(h.min()*1.1,h.max()*0.9,10)
    iso = isolines(v,f)
    lines = []
    for vals in vals:
        data =iso.isolinecoords([np.array([0,0,1]),np.array([0,0,vals])])
        for bspl,l in data:
              coord = bspl(s)
              lines.append(coord)    
    
   #ps.init() 
   
   #ps_mesh = ps.register_surface_mesh("my mesh", v, f, smooth_shade=True)
   #ps_cloud = ps.register_point_cloud("my points", v_iso)
   #ps.show()
    
        
    
if __name__ == "__main__":
    main()