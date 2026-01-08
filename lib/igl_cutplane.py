# -*- coding: utf-8 -*-
"""
Created on Mon Feb 26 16:56:22 2024

@author: Alejandro
"""


import polyscope as ps
import igl
import numpy as np
import matplotlib.pyplot as plt
#from kirigami import isolines

            
        
v, f = igl.read_triangle_mesh('../data/parabole.obj')

#cp = isolines(v,f)
field = v[:,2]#np.sqrt(v[:,0]**2+v[:,1]**2)
grad_s = igl.grad(v,f)
hess = igl.hessian(v,f)
grad= grad_s.dot(field)
hess_field=hess.dot(field)
hess_field = np.reshape(hess_field,(len(v),3,3), order = 'F')
_,_,k1,k2 = igl.principal_curvature(v,f)
D = k1*k2

center = igl.barycenter(v, f)

grad_t = np.reshape(grad,(len(f),3), order = 'F')
grad = igl.average_onto_vertices(v,f,grad_t)

grad_m = np.linalg.norm(grad,axis = 1)

# l = np.linspace(0,1,25)
# levels = np.linspace(1,field.max()*0.9,5)
# isoV,isoE,az = igl.isolines(v, f, field, levels)
# info =igl.edges_to_path(isoE[az ==4])
idx = (field>field.mean())*(D>0)*(grad_m<0.1)
v_max = v[idx]
ps.init() 
ps_mesh = ps.register_surface_mesh("my mesh", v, f, smooth_shade=True)
ps_mesh.add_scalar_quantity("field vals", field, enabled=False)
ps_mesh.add_scalar_quantity("hessian det", D, enabled=False)
ps_mesh.add_scalar_quantity("gradient magnitude", grad_m, enabled=False)
ps_mesh.add_scalar_quantity("filter", grad_m*(D>0), enabled=False)

#ps_cloud = ps.register_point_cloud("my points", c)
#ps_cloud.add_vector_quantity("rand vecs", grad, enabled=True,vectortype='ambient')
#ps_cloud = ps.register_point_cloud("my min", v_max)
# for lv in levels:
#     pts,edges = cp.meandeirng_triangles(field, lv)
#     plt.plot(pts[:,0],pts[:,1],'xb')
#     ps.register_point_cloud("my points_"+str(lv),pts,radius=0.002)
ps.show()

# plt.plot(unipoints[:,0],unipoints[:,1],'.b')
