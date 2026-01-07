import numpy as np
import scipy.spatial

from typing import Optional

def voronoi_to_vertex_links(mesh:scipy.spatial.Voronoi) -> np.ndarray:
    connections = []
    for ridge in mesh.ridge_vertices:
        if -1 not in ridge:
            connections.append(ridge)

    return np.array(connections)

def delanay_simplex_to_connectivity_array(delaunay_in:scipy.spatial.Delaunay,
                                          bidirectional:bool=True,
                                          mask_indices: Optional[np.ndarray]=None,
                                          replace_indices: Optional[np.ndarray]=None
                                          ) -> np.ndarray:
    delaunay_connections = []
    for simplex in delaunay_in.simplices:
        for i in range(len(simplex)):
            pair = [simplex[i],simplex[(i+1)%len(simplex)]]
            if not(pair in delaunay_connections or pair[::-1] in delaunay_connections):
                if mask_indices is not None:
                    if pair[0] in mask_indices or pair[1] in mask_indices:
                        delaunay_connections.append(pair)
                        if bidirectional:
                            delaunay_connections.append(pair[::-1])
                else:
                    delaunay_connections.append(pair)
                    if bidirectional:
                        delaunay_connections.append(pair[::-1])

    delaunay_connections = np.array(delaunay_connections).T
    dx = delaunay_in.points[delaunay_connections[1,:],:] - delaunay_in.points[delaunay_connections[0,:],:]

    if replace_indices is not None:
        # for i in range(len(delaunay_connections[0])):
        #     if delaunay_connections[0,i] in mask_indices:
        #         idx = np.where(mask_indices == delaunay_connections[0,i])[0][0]
        #         delaunay_connections[0,i] = replace_indices[idx]
        #     if delaunay_connections[1,i] in mask_indices:
        #         idx = np.where(mask_indices == delaunay_connections[1,i])[0][0]
        #         delaunay_connections[1,i] = replace_indices[idx]

        mask = np.arange(len(replace_indices)) == replace_indices
        mask_connections = mask[delaunay_connections].all(axis=0)
        mask_periodic_connections = mask[delaunay_connections].sum(axis=0) == 1

        delaunay_connections = replace_indices[delaunay_connections]

        # recheck overlaps due to replacement
        for i in range(1,delaunay_connections.shape[1]):
            if (delaunay_connections[:,i][:,np.newaxis] == delaunay_connections[:,:i]).all(0).any():
                delaunay_connections[:,i] = -1
                dx[i,:] = np.nan
                
                # delaunay_connections = np.concat([delaunay_connections[:,:i],delaunay_connections[:,i+1:]],axis=1)
        
        internal_connections = delaunay_connections[:,mask_connections]
        internal_connections = internal_connections[:,(internal_connections!=-1).all(0)]

        periodic_connections = delaunay_connections[:,mask_periodic_connections]
        periodic_connections = periodic_connections[:,(periodic_connections!=-1).all(0)]

        internal_dx = dx[mask_connections,:]
        internal_dx = internal_dx[~np.isnan(internal_dx).all(1),:]

        periodic_dx = dx[mask_periodic_connections,:]
        periodic_dx = periodic_dx[~np.isnan(periodic_dx).all(1),:]

        return internal_connections, periodic_connections, internal_dx, periodic_dx

    return delaunay_connections, None, dx, None

def delanay_mesh_constructor(points:np.ndarray,
                             bidirectional:bool=True,
                             periodic:bool=True,
                             periodic_limits: Optional[np.ndarray] = None
                             ) -> scipy.spatial.Delaunay:
    if periodic:
        # Create periodic copies of the points
        copies = []
        for dx in [0, -1, 1]:
            for dy in [0, -1, 1]:
                if dx == 0 and dy == 0:
                    continue
                shift = np.array([dx * (periodic_limits[0,1] - periodic_limits[0,0]),
                                  dy * (periodic_limits[1,1] - periodic_limits[1,0])])
                copies.append(points + shift)
        all_points = np.vstack([points] + copies)
    else:
        all_points = points

    delaunay = scipy.spatial.Delaunay(all_points)

    connectivity, connectivity_periodic, dx, dx_periodic = delanay_simplex_to_connectivity_array(delaunay,
                                                         bidirectional=bidirectional,
                                                         mask_indices=np.arange(len(points)) if periodic else None,
                                                         replace_indices=np.arange(9*len(points))%len(points) if periodic else None)
    
    return delaunay, connectivity, connectivity_periodic, dx, dx_periodic