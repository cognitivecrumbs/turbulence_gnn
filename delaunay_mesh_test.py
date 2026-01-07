import numpy as np
import scipy
import sys
# from scipy.interpolate import RegularGridInterpolator

sys.path.append('../')
# sys.path.append('../..')

# from src.mesh_util import delanay_mesh_constructor
from src.mesh_util import *


def delaunay_test():
    x_range = (0,1)
    y_range = (0,1)

    rows = 16
    cols = 16

    staggered_points = np.empty((rows,cols,2))

    # consistent spacing
    dx = (x_range[1]-x_range[0])/(rows)
    dy = (y_range[1]-y_range[0])/(cols)
    staggered_points[:,:,1] = np.arange(y_range[0],y_range[1]-dy/10,dy)[:,np.newaxis]
    for i in range(rows):
        point_row = np.arange(x_range[0],x_range[1]-6*dx/10,dx)
        staggered_points[i,:,0] = point_row + i%2*(dx/2) # stagger
        # staggered_points[i,:,0] = point_row 

    staggered_points = staggered_points.reshape(-1,2)

    spacing = (x_range[1]-x_range[0])/(rows-1)
    mesh, connectivity, connectivity_periodic, edge_attr, edge_attr_periodic = delanay_mesh_constructor(staggered_points,periodic=True,bidirectional=True,periodic_limits=np.array([[x_range[0],x_range[1]],[y_range[0],y_range[1]]]))


    # combine connectivities
    connectivity_combined = np.concatenate([connectivity,connectivity_periodic],axis=1)

    # combine relative position
    edge_attr_combined = np.concat([edge_attr,edge_attr_periodic],axis=0)

    for i in range(rows*cols):
        mask_src = connectivity_combined[0] == i
        mask_dst = connectivity_combined[1] == i

        assert mask_src.sum() == 6
        assert mask_dst.sum() == 6

        # no repeats
        assert len(set(connectivity_combined[1,mask_src])) == 6
        assert len(set(connectivity_combined[0,mask_dst])) == 6

        assert np.all(connectivity_combined[1,mask_src] != i)
        assert np.all(connectivity_combined[0,mask_dst] != i)

        assert edge_attr_combined.shape[0] == connectivity_combined.shape[1]


if __name__ == '__main__':
    delaunay_test()
    print('test passed')