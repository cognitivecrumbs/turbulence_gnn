import numpy as np
import xarray
import os

def calculate_vorticity(model: xarray.Dataset) -> None:
#   return (model.dims,np.gradient(model.v,axis=-2)/np.gradient(model.x)[np.newaxis,np.newaxis,np.newaxis,:] - np.gradient(model.u,axis=-1)/np.gradient(model.y)[np.newaxis,np.newaxis,:,np.newaxis])
#   return (model.dims,np.gradient(model.v,axis=-2)/np.gradient(model.x)[np.newaxis,np.newaxis,:,np.newaxis] - np.gradient(model.u,axis=-1)/np.gradient(model.y)[np.newaxis,np.newaxis,np.newaxis,:])
  model['vorticity'] = (model.dims,np.gradient(model.v,axis=-2)/np.gradient(model.x)[np.newaxis,np.newaxis,:,np.newaxis] - np.gradient(model.u,axis=-1)/np.gradient(model.y)[np.newaxis,np.newaxis,np.newaxis,:])

def load_dataset(datset_names: dict,path:str = 'data') -> dict:
    dataset = {}
    for k, v in datset_names.items():
      dataset[k] = xarray.open_dataset(os.path.join(path,v), chunks={'time': '100MB'})
    #   dataset[k]['vorticity'] = calculate_vorticity(dataset[k])
      calculate_vorticity(dataset[k])
    
    return dataset