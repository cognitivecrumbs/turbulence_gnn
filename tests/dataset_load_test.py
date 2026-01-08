import xarray
import numpy as np
from matplotlib import pyplot as plt
from cmap import Colormap

cm = Colormap('seaborn:icefire')  # case insensitive

# def correlation(x, y):
#   state_dims = ['x', 'y']
#   p  = xru.normalize(x, state_dims) * xru.normalize(y, state_dims)
#   return p.sum(state_dims)

def calculate_vorticity(model):
#   return (model.dims,np.gradient(model.v,axis=-2)/np.gradient(model.x)[np.newaxis,np.newaxis,np.newaxis,:] - np.gradient(model.u,axis=-1)/np.gradient(model.y)[np.newaxis,np.newaxis,:,np.newaxis])
  return (model.dims,np.gradient(model.v,axis=-2)/np.gradient(model.x)[np.newaxis,np.newaxis,:,np.newaxis] - np.gradient(model.u,axis=-1)/np.gradient(model.y)[np.newaxis,np.newaxis,np.newaxis,:])
#   return (model.dims,np.gradient(model.v,axis=-1)/np.gradient(model.x)[np.newaxis,np.newaxis,:,np.newaxis] - np.gradient(model.u,axis=-2)/np.gradient(model.y)[np.newaxis,np.newaxis,np.newaxis,:])
#   return (model.dims,np.gradient(model.v,axis=-1)/np.gradient(model.x)[np.newaxis,np.newaxis,np.newaxis,:] - np.gradient(model.u,axis=-2)/np.gradient(model.y)[np.newaxis,np.newaxis,:,np.newaxis])

def calculate_time_until(vorticity_corr):
  threshold = 0.95
  return (vorticity_corr.mean('sample') >= threshold).idxmin('time').rename('time_until')

def calculate_time_until_bootstrap(vorticity_corr, bootstrap_samples=10000):
  rs = np.random.RandomState(0)
  indices = rs.choice(16, size=(10000, 16), replace=True)
  boot_vorticity_corr = vorticity_corr.isel(
      sample=(('boot', 'sample2'), indices)).rename({'sample2': 'sample'})
  return calculate_time_until(boot_vorticity_corr)

def calculate_upscaling(time_until):
  slope = ((np.log(16) - np.log(8))
          / (time_until.sel(model='baseline_1024')
              - time_until.sel(model='baseline_512')))
  x = time_until.sel(model='learned_interp_64')
  x0 = time_until.sel(model='baseline_512')
  intercept = np.log(8)
  factor = np.exp(slope * (x - x0) + intercept)
  return factor

def calculate_speedup(time_until):
  runtime_baseline_8x = 44.053293
  runtime_baseline_16x = 412.725656
  runtime_learned = 1.155115
  slope = ((np.log(runtime_baseline_16x) - np.log(runtime_baseline_8x))
          / (time_until.sel(model='baseline_1024')
              - time_until.sel(model='baseline_512')))
  x = time_until.sel(model='learned_interp_64')
  x0 = time_until.sel(model='baseline_512')
  intercept = np.log(runtime_baseline_8x)
  speedups = np.exp(slope * (x - x0) + intercept) / runtime_learned
  return speedups


baseline_filenames = {
    f'baseline_{r}': f'baseline_{r}x{r}.nc'
    for r in [64, 128, 256, 512, 1024, 2048]
}
learned_filenames = {
    f'learned_interp_{r}': f'learned_{r}x{r}.nc'
    for r in [32, 64, 128]
}
long_eval_filenames = {
    f'long_eval{r}': f'long_eval_{r}x{r}_64x64.nc'
    # for r in [64, 128, 256, 512, 1024, 2048]
    for r in [64]
}

decaying_filenames = {
    f'long_eval{r}': f'eval_{r}x{r}_64x64.nc'
    # for r in [64, 128, 256, 512, 1024, 2048]
    for r in [64]
}

models = {}
# for k, v in baseline_filenames.items():
#   print(f'data/kolmogorov_re_1000_fig1/{v}')
#   models[k] = xarray.open_dataset(f'data/kolmogorov_re_1000_fig1/{v}', chunks={'time': '100MB'})
#   models[k]['vorticity'] = calculate_vorticity(models[k])

# for k, v in learned_filenames.items():
#   ds = xarray.open_dataset(f'data/kolmogorov_re_1000_fig1/{v}', chunks={'time': '100MB'})
#   models[k] = ds.reindex_like(models['baseline_64'], method='nearest')
#   models[k]['vorticity'] = calculate_vorticity(models[k])

# for k, v in long_eval_filenames.items():
#   models[k] = xarray.open_dataset(f'data/kolmogorov_re_1000/{v}', chunks={'time': '100MB'})
#   models[k]['vorticity'] = calculate_vorticity(models[k])

for k, v in decaying_filenames.items():
  print(f'data/kolmogorov_re_1000_fig1/{v}')
  models[k] = xarray.open_dataset(f'data/decaying/{v}', chunks={'time': '100MB'})
  models[k]['vorticity'] = calculate_vorticity(models[k])
pass

data = 'long_eval64'
# data = 'baseline_64'
# data = 
fig = plt.figure()
# for i,data in enumerate(long_eval_filenames):
#     ax = fig.add_subplot(1,len(baseline_filenames),i+1)
#     # ax.contourf(models[data].x.data, models[data].y.data, models[data].vorticity.mean('time')[0], levels=100, cmap='RdBu_r')
#     plot = ax.contourf(models[data].y.data, models[data].x.data, models[data].vorticity[0,0], levels=1000, cmap=cm.to_matplotlib(),vmin=-8,vmax=8,extend='neither')
#     ax.set_aspect('equal')
#     plt.colorbar(plot)


for i in range(16):
    ax = fig.add_subplot(4,4,i+1)
    # ax.contourf(models[data].x.data, models[data].y.data, models[data].vorticity.mean('time')[0], levels=100, cmap='RdBu_r')
    plot = ax.contourf(models[data].y.data, models[data].x.data, models[data].vorticity[i,0].to_numpy().T, levels=1000, cmap=cm.to_matplotlib(),vmin=-8,vmax=8,extend='neither')
    # plot = ax.contourf(models[data].y.data, models[data].x.data, models[data].u[i,0], levels=1000, cmap=cm.to_matplotlib(),vmin=-2,vmax=2,extend='neither')
    ax.set_aspect('equal')
    plt.colorbar(plot) 



sample_ind = -1

conv1 = models[data].u.differentiate('x')*models[data].u + models[data].u.differentiate('y')*models[data].v
dudt = models[data].u.differentiate('time')
ratio = -conv1/dudt
print(ratio[sample_ind,-1].mean().values)
print(ratio[sample_ind,-1].std().values)

dudx = np.gradient(models[data].u,axis=-2)/np.gradient(models[data].x)[np.newaxis,np.newaxis,:,np.newaxis]
dudy = np.gradient(models[data].u,axis=-1)/np.gradient(models[data].y)[np.newaxis,np.newaxis,np.newaxis,:]
dvdy = np.gradient(models[data].v,axis=-1)/np.gradient(models[data].y)[np.newaxis,np.newaxis,np.newaxis,:]

# dudx = np.gradient(models[data].u,axis=-1)/np.gradient(models[data].x)[np.newaxis,np.newaxis,:,np.newaxis]
# dudy = np.gradient(models[data].u,axis=-2)/np.gradient(models[data].y)[np.newaxis,np.newaxis,np.newaxis,:]
# dvdy = np.gradient(models[data].u,axis=-2)/np.gradient(models[data].y)[np.newaxis,np.newaxis,np.newaxis,:]
conv1 = dudx*models[data].u.to_numpy() + dudy*models[data].v.to_numpy()

uu = models[data].u**2
uv = models[data].u*models[data].v
duudx = np.gradient(uu,axis=-2)/np.gradient(models[data].x)[np.newaxis,np.newaxis,:,np.newaxis]
duvdy = np.gradient(uv,axis=-1)/np.gradient(models[data].y)[np.newaxis,np.newaxis,np.newaxis,:]

conv1 = duudx + duvdy
dudt = (models[data].u[sample_ind,1:].to_numpy() - models[data].u[sample_ind,:-1].to_numpy())/(models[data].time[1]-models[data].time[0]).data

ratio = -conv1[sample_ind,:-1]/dudt
difference = dudt+conv1[sample_ind,:-1]
# difference = dudt+conv1[:,1:]
print(ratio[-1].mean())
print(ratio[-1].std())

divergence = dudx+dvdy
print('divergence')
print(np.abs(divergence[sample_ind,-1]).mean())
print(divergence[sample_ind,-1].std())

fig1 = plt.figure()
ax1 = fig1.add_subplot(111)
ax1.hist(divergence[sample_ind,-1,:,:].flatten(),100)

fig2 = plt.figure()
ax2 = fig2.add_subplot(411)
ax2.hist(dudt[sample_ind].flatten(),100)

ax3 = fig2.add_subplot(412)
ax3.hist(-conv1[sample_ind,-1,:,:].flatten(),100)

# ax3 = fig2.add_subplot(413)
# ax3.hist(ratio[-1,:,:].flatten(),100)

ax4 = fig2.add_subplot(414)
ax4.hist(difference[-1,:,:].flatten(),100)
print('check std')
print(dudt[-1].std())
print(difference[-1].std())

energy = models[data].u.to_numpy()**2 + models[data].v.to_numpy()**2
plot_energy = energy[sample_ind].reshape(energy.shape[1],np.prod(energy.shape[2:]))
fig3 = plt.figure()
ax31 = fig3.add_subplot(111)
ax31.plot(plot_energy.mean(-1))


plt.show()