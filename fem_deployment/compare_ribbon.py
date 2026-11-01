""" Step 3 (the environment of the repository): compare the Naghdi shell
(naghdi_deploy.py) with the linear ribbon model on the same mesh. Both are
deployed by displacement of the centre of the top level; the ribbon model is
linear, so at every height the shell reached it is its export_case.py
deployment scaled to that height. Rigid motions are fitted to the panels of
every level of the shell (ribbon_model.fit_levels) to compare the levels.
    python fem_deployment/compare_ribbon.py fem_deployment/output/case_cone_h1.npz
Writes next to the case:
  <case>_compare.csv  one row per height: forces, distance between the two
                      sheets (rms, 95th percentile, max), level heights, tilts
  <case>_levels.csv   level heights and tilts at the last height
  <case>_compare.png  force and shape difference against height
  <case>_vs_ribbon.png (a) level heights, (b) height difference on the flat
                      sheet, (c, d) the two deployed sheets, at the last height """
import argparse
import glob
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path[:0] = [os.path.join(HERE,'..','lib')]
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.collections import PolyCollection
from matplotlib.colors import LightSource
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from ribbon_model import fit_levels

def level_means(V, F, part, q):
    # mean height of the panel vertices of every level moved by q
    z = []
    for j in range(len(q)):
        v = np.unique(F[(part[:,0] == j)&(part[:,1] < 0)])
        z.append(np.mean(q[j,0]+q[j,1]*V[v,1]-q[j,2]*V[v,0]))
    return np.array(z)

def draw(ax, X, F, H, V, title):
    # deployed sheet coloured by height and shaded
    tri = X[F]
    n = np.cross(tri[:,1]-tri[:,0],tri[:,2]-tri[:,0])
    n /= np.linalg.norm(n,axis = 1)[:,None]+1e-30
    shade = np.clip(np.abs(n@LightSource(azdeg = 315,altdeg = 45).direction),0.3,1.0)
    col = plt.cm.viridis(np.clip(tri[:,:,2].mean(axis = 1)/H,0,1))
    col[:,:3] *= shade[:,None]
    ax.add_collection3d(Poly3DCollection(tri,facecolors = col,edgecolors = 'none'))
    r = max(np.ptp(V[:,0]),np.ptp(V[:,1]))/2
    mid = (V[:,:2].max(axis = 0)+V[:,:2].min(axis = 0))/2
    ax.set_xlim(mid[0]-r,mid[0]+r)
    ax.set_ylim(mid[1]-r,mid[1]+r)
    ax.set_zlim(0,H)
    ax.set_box_aspect((1,1,H/(2*r)),zoom = 1.3)
    ax.view_init(22,-60)
    ax.set_axis_off()
    ax.set_title(title)

def main():
    parser = argparse.ArgumentParser(description = __doc__,formatter_class = argparse.RawDescriptionHelpFormatter)
    parser.add_argument('case',help = 'case_<name>_h<h>.npz written by export_case.py')
    args = parser.parse_args()
    stem = os.path.splitext(args.case)[0]
    d = np.load(args.case)
    V,F,part,q_top,top = d['V'],d['F'],d['part'],d['q'],float(d['top'])
    n_levels = len(q_top)
    target = (d['heights']-d['heights'][0])[:n_levels]
    shell = np.loadtxt(stem+'_naghdi.csv',delimiter = ',',skiprows = 1,ndmin = 2)
    shapes = sorted(glob.glob(stem+'_naghdi_[0-9][0-9].npy'))
    rows = []
    for (H,force,_),path in zip(shell,shapes):
        X = np.load(path)
        s = H/top
        Xr = V+(d['guess']-V)*s
        dist = np.linalg.norm(X-Xr,axis = 1)
        qs = fit_levels(V,X,F,part,n_levels)
        zs,zr = level_means(V,F,part,qs),level_means(V,F,part,q_top*s)
        tilt_s = np.degrees(np.hypot(qs[:,1],qs[:,2]))
        tilt_r = np.degrees(np.hypot(q_top[:,1],q_top[:,2])*s)
        rows.append((H,force,float(d['ribbon_force'])*s,np.sqrt(np.mean(dist**2)),np.percentile(dist,95),dist.max(),
                     np.sqrt(np.mean((zs-zr)**2)),np.abs(zs-zr).max(),np.abs(tilt_s-tilt_r).max()))
    rows = np.array(rows)
    np.savetxt(stem+'_compare.csv',rows,delimiter = ',',fmt = '%.5g',comments = '',
               header = 'height_mm,force_shell_N,force_ribbon_N,dist_rms_mm,dist_p95_mm,dist_max_mm,level_rms_mm,level_max_mm,tilt_diff_max_deg')
    np.savetxt(stem+'_levels.csv',np.column_stack([np.arange(n_levels),target*s,zr,zs,tilt_r,tilt_s]),delimiter = ',',fmt = '%.4f',
               comments = '',header = 'level,target_mm,ribbon_mm,shell_mm,ribbon_tilt_deg,shell_tilt_deg')
    H,force,F_rib = rows[-1,:3]
    print('at %.2f mm: force shell %.3f N, ribbon model %.3f N' % (H,force,F_rib))
    print('vertices apart rms %.2f mm, 95 %% %.2f mm, max %.2f mm; level heights rms %.2f mm, max %.2f mm; tilts differ by up to %.1f deg' % tuple(rows[-1,3:]))
    plt.rcParams.update({'font.size':8})

    fig,ax = plt.subplots(1,2,figsize = (9,3.5))
    ax[0].plot(rows[:,0],rows[:,2],label = 'linear ribbon model')
    ax[0].plot(rows[:,0],rows[:,1],label = 'Naghdi shell')
    ax[0].set(xlabel = 'height of the centre of the top level (mm)',ylabel = 'force (N)',title = '(a) pulling force')
    ax[0].legend(frameon = False)
    for k,label in ((3,'vertices, rms'),(4,'vertices, 95 %'),(6,'level heights, rms')):
        ax[1].plot(rows[:,0],rows[:,k],label = label)
    ax[1].set(xlabel = 'height of the centre of the top level (mm)',ylabel = 'shell $-$ ribbon model (mm)',title = '(b) shape difference')
    ax[1].legend(frameon = False)
    fig.tight_layout()
    fig.savefig(stem+'_compare.png',dpi = 160)

    Xs,Xr = np.load(shapes[-1]),V+(d['guess']-V)*s
    t = target*s
    fig = plt.figure(figsize = (10,6.2))
    a1 = fig.add_subplot(2,2,1)
    a1.plot(t,t,'k-',lw = 0.8,label = 'target')
    a1.plot(t,zr,'o',ms = 4,label = 'linear ribbon model')
    a1.plot(t,zs,'s',ms = 4,mfc = 'none',label = 'Naghdi shell')
    a1.set_xlabel('target height of the contour (mm)')
    a1.set_ylabel('mean height of the level (mm)')
    a1.legend(frameon = False)
    a1.set_title('(a) levels, centre moved to %.1f mm' % H)
    a2 = fig.add_subplot(2,2,2)
    diff = (Xs[:,2]-Xr[:,2])[F].mean(axis = 1)
    m = np.abs(diff).max()
    pc = PolyCollection(V[F,:2],array = diff,cmap = 'RdBu_r',edgecolors = 'none')
    pc.set_clim(-m,m)
    a2.add_collection(pc)
    a2.set_aspect('equal')
    a2.autoscale_view()
    a2.axis('off')
    fig.colorbar(pc,ax = a2,shrink = 0.8,label = 'height, shell $-$ ribbon model (mm)')
    a2.set_title('(b) difference of the heights on the flat sheet')
    draw(fig.add_subplot(2,2,3,projection = '3d'),Xr,F,H,V,'(c) linear ribbon model, %.2f N' % F_rib)
    draw(fig.add_subplot(2,2,4,projection = '3d'),Xs,F,H,V,'(d) Naghdi shell, %.2f N' % force)
    fig.tight_layout()
    fig.savefig(stem+'_vs_ribbon.png',dpi = 160)

if __name__ == '__main__':
    main()
