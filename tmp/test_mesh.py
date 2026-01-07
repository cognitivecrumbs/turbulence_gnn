import numpy as np

from scipy.spatial import Voronoi, voronoi_plot_2d,Delaunay
from matplotlib import pyplot as plt


rows = 40
cols = 40
points = np.empty((rows,cols,2))
points[:,:,1] = np.linspace(0,1,rows)[:,np.newaxis]
for i in range(rows):
    point_row = np.linspace(0,1,cols)
    points[i,:,0] = point_row + i%2*(1/cols/2) # stagger 
    # points[i,:,0] = point_row 

points = points.reshape(-1,2)

mesh = Voronoi(points)
# links = Delaunay(points)

vals = points[:,0]%0.2 + points[:,1]%0.2
vals = np.sin(points[:,0]*20)*np.cos(points[:,1]*20)

fig = voronoi_plot_2d(mesh)
fig.axes[0].set_aspect('equal')


fig2 = plt.figure()
ax = fig2.add_subplot(111)
ax.tricontourf(points[:,0], points[:,1], vals, levels=500,cmap="RdBu_r")
ax.scatter(points[:,0], points[:,1], c=vals, s = 4,cmap="RdBu_r")
ax.set_aspect('equal')
plt.show()


