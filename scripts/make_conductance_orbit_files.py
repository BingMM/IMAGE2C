#%% Import 

import os
import numpy as np
import glob
from secsy import CSgrid, CSprojection
from scipy.io import netcdf_file
from datetime import datetime, timedelta
import apexpy
from tqdm import tqdm
from tqdm.contrib.concurrent import process_map  # tqdm-compatible multiprocessing
from functools import partial

from icbuilder import PreImage
from icbuilder import BinnedImage
from icbuilder import ConductanceImage

#%% Fun

def find_common_times_with_indices(list1, list2, list3, tolerance=timedelta(seconds=2)):
    # Sort the lists and keep track of the original indices
    sorted_list1 = sorted((time, i) for i, time in enumerate(list1))
    sorted_list2 = sorted((time, i) for i, time in enumerate(list2))
    sorted_list3 = sorted((time, i) for i, time in enumerate(list3))
    
    # Initialize pointers for each list
    i, j, k = 0, 0, 0
    common_times = []
    indices = []
    
    # Traverse the lists
    while i < len(sorted_list1) and j < len(sorted_list2) and k < len(sorted_list3):
        # Find the maximum and minimum times among the current elements
        max_time = max(sorted_list1[i][0], sorted_list2[j][0], sorted_list3[k][0])
        min_time = min(sorted_list1[i][0], sorted_list2[j][0], sorted_list3[k][0])
        
        # Check if the times are within the tolerance
        if max_time - min_time <= tolerance:
            common_times.append(max_time)
            indices.append((sorted_list1[i][1], sorted_list2[j][1], sorted_list3[k][1]))
            i += 1
            j += 1
            k += 1
        else:
            # Move the pointer of the list with the smallest time
            if sorted_list1[i][0] == min_time:
                i += 1
            elif sorted_list2[j][0] == min_time:
                j += 1
            else:
                k += 1
    
    return np.array(common_times), np.array(indices)

def safe_apex_convert(apex, glat_row, glon_row, height=120):
    valid = ~np.isnan(glat_row) & ~np.isnan(glon_row)
    mlat = np.full_like(glat_row, np.nan)
    mlon = np.full_like(glon_row, np.nan)
    if np.any(valid):
        mlat_valid, mlon_valid = apex.convert(glat_row[valid], glon_row[valid], 'geo', 'apex', height=height)
        mlat[valid] = mlat_valid
        mlon[valid] = mlon_valid
    return mlat, mlon

def process_orbit(orbit, p_wic_nc, p_s12_nc, p_s13_nc, p_out, grid_w, grid_s, f_thres=0.1):
    try:
        # Load nc orbit files
        wic_nc = netcdf_file(p_wic_nc + f'wic_or{orbit:04d}.nc', 'r')
        s12_nc = netcdf_file(p_s12_nc + f's12_or{orbit:04d}.nc', 'r')
        s13_nc = netcdf_file(p_s13_nc + f's13_or{orbit:04d}.nc', 'r')

        # Get start time for each instrument
        t0_wic = datetime.strptime(b''.join(wic_nc.variables['t_start'].data).decode('utf8'), '%Y-%m-%dT%H:%M:%S')
        t0_s12 = datetime.strptime(b''.join(s12_nc.variables['t_start'].data).decode('utf8'), '%Y-%m-%dT%H:%M:%S')
        t0_s13 = datetime.strptime(b''.join(s13_nc.variables['t_start'].data).decode('utf8'), '%Y-%m-%dT%H:%M:%S')
        
        # Get time for each sensor
        t_wic = [t0_wic + timedelta(seconds=int(sec)) for sec in wic_nc.variables['date'][:]]
        t_s12 = [t0_s12 + timedelta(seconds=int(sec)) for sec in s12_nc.variables['date'][:]]
        t_s13 = [t0_s13 + timedelta(seconds=int(sec)) for sec in s13_nc.variables['date'][:]]
        
        # Get common times and their indices
        t, indices = find_common_times_with_indices(t_wic, t_s12, t_s13)
        
        # Remove completely empty frames
        empty = np.all(np.isnan(wic_nc.variables['mlat'][indices[:, 0]]), axis=(1,2))
        empty |= np.all(np.isnan(s12_nc.variables['mlat'][indices[:, 1]]), axis=(1,2))
        empty |= np.all(np.isnan(s13_nc.variables['mlat'][indices[:, 2]]), axis=(1,2))
        t, indices = t[~empty], indices[~empty, :]
        
        # Extract data
        wic_pI = PreImage(wic_nc, indices[:, 0])
        s12_pI = PreImage(s12_nc, indices[:, 1])
        s13_pI = PreImage(s13_nc, indices[:, 2])
        
        # Close files
        wic_nc.close()
        s12_nc.close()
        s13_nc.close()
        
        # APEX conversion
        for i, ti in enumerate(t):
            apex = apexpy.Apex(ti)
            wic_pI.mlat[i], wic_pI.mlon[i] = safe_apex_convert(apex, wic_pI.glat[i], wic_pI.glon[i])
            s12_pI.mlat[i], s12_pI.mlon[i] = safe_apex_convert(apex, s12_pI.glat[i], s12_pI.glon[i])
            s13_pI.mlat[i], s13_pI.mlon[i] = safe_apex_convert(apex, s13_pI.glat[i], s13_pI.glon[i])
        
        # Fullness check
        wic_p = wic_pI.percent_full(grid_w)
        s12_p = s12_pI.percent_full(grid_s)
        s13_p = s13_pI.percent_full(grid_s)
        f = (wic_p >= f_thres) & (s12_p >= f_thres) & (s13_p >= f_thres)
        
        t = t[f]
        wic_pI.discard(f)
        s12_pI.discard(f)
        s13_pI.discard(f)
        
        # Bin and conductance images
        wic_bI = BinnedImage(wic_pI, grid_w, inflate_uncertainty=True)
        s12_bI = BinnedImage(s12_pI, grid_s, inflate_uncertainty=True, target_grid=grid_w, interpolate=True)
        s13_bI = BinnedImage(s13_pI, grid_s, inflate_uncertainty=True, target_grid=grid_w, interpolate=True)
        cI = ConductanceImage(wic_bI, s12_bI, s13_bI, time=t)
        
        # Save to netCDF
        out_path = f'{p_out}or_{orbit:04d}.nc'
        cI.to_nc(out_path)
        return orbit  # Success
    except Exception as e:
        return f"Failed orbit {orbit}: {e}"

def run_all_orbits(o, p_wic_nc, p_s12_nc, p_s13_nc, p_out, grid_w, grid_s, parallel=True, n_processes=None):

    kwargs = {
            'p_wic_nc': p_wic_nc,
            'p_s12_nc': p_s12_nc,
            'p_s13_nc': p_s13_nc,
            'p_out': p_out,
            'grid_w': grid_w,  # must be defined in your namespace
            'grid_s': grid_s,  # must be defined in your namespace
            'f_thres': 0.1
            }
    
    print(f'Pulling WIC data from {p_wic_nc}')
    print(f'Pulling SI12 data from {p_s12_nc}')
    print(f'Pulling SI13 data from {p_s13_nc}\n')
    
    print(f'Outputting conductance objects in {p_out}\n')
    
    func = partial(process_orbit, **kwargs)
    
    if parallel:
        results = process_map(func, o, max_workers=n_processes, chunksize=1, desc='Loop over orbits')
    else:
        results = []
        for orbit in tqdm(o, desc='Loop over orbits'):
            results.append(func(orbit))

    return results

#%% Paths

base = '/Home/siv32/mih008/repos/icBuilder/example_data/'

p_wic_nc = base + 'wic/'
p_s12_nc = base + 's12/'
p_s13_nc = base + 's13/'

p_out = base + 'conductance/'

#%% Fetch orbits available in all nc files

# Fetch all orbits
o_wic = [int(o[-7:-3]) for o in sorted(glob.glob(p_wic_nc + '*.nc'))]
o_s12 = [int(o[-7:-3]) for o in sorted(glob.glob(p_s12_nc + '*.nc'))]
o_s13 = [int(o[-7:-3]) for o in sorted(glob.glob(p_s13_nc + '*.nc'))]

# Insert code that loads .npy orbit flags and discard orbits without flag=1
wic_avail = np.load(base + 'wic_avail_orbit.npy')
s12_avail = np.load(base + 's12_avail_orbit.npy')
s13_avail = np.load(base + 's13_avail_orbit.npy')

print(f'WIC: {wic_avail.shape[0]} nc files found. {int(np.sum(wic_avail[:, 1]==1))} orbits useable.')
print(f'S12: {s12_avail.shape[0]} nc files found. {int(np.sum(s12_avail[:, 1]==1))} orbits useable.')
print(f'S13: {s13_avail.shape[0]} nc files found. {int(np.sum(s13_avail[:, 1]==1))} orbits useable.\n')

o_wic = [ow for ow, keep in zip(o_wic, wic_avail[:, 1] == 1) if keep]
o_s12 = [ow for ow, keep in zip(o_s12, s12_avail[:, 1] == 1) if keep]
o_s13 = [ow for ow, keep in zip(o_s13, s13_avail[:, 1] == 1) if keep]

# Create list of all overlapping orbits
o = set(o_wic) & set(o_s12) & set(o_s13)

print(f'There are {len(list(o))} common orbtis between WIC, S12, and S13\n')

#%% Define grids

position = (0, 90) # lon, lat
orientation = (0, 1) # east, north
L, Lres = 20000e3, 225e3
grid_w = CSgrid(CSprojection(position, orientation), L, L, Lres, Lres, R = 6481.2e3)

target_Lres = 450e3
dist = grid_w.Lres*grid_w.shape[0]
steps = np.round(dist / target_Lres).astype(int)
dxi = np.diff(np.linspace(grid_w.xi.min(), grid_w.xi.max(), steps)).mean()
deta = np.diff(np.linspace(grid_w.eta.min(), grid_w.eta.max(), steps)).mean()
xi_e = np.linspace(grid_w.xi.min()-dxi/2, grid_w.xi.max()+dxi/2, steps+1)
eta_e = np.linspace(grid_w.eta.min()-deta/2, grid_w.eta.max()+deta/2, steps+1)
Lres = dist / steps
grid_s = CSgrid(CSprojection(position, orientation), L, L, Lres, Lres, edges = (xi_e, eta_e), R = 6481.2e3)
print('Fine grid resolution is: ' + str(grid_w.Lres/1e3) + ' km')
print('Coarse grid target resolution is: ' + str(target_Lres/1e3) + ' km')
print('Coarse grid resolution is: ' + str(np.round(Lres/1e3, 2)) + ' km\n')

#%%

results = run_all_orbits(o, p_wic_nc, p_s12_nc, p_s13_nc, p_out, grid_w, grid_s, parallel=False)
#results = run_all_orbits(o, p_wic_nc, p_s12_nc, p_s13_nc, p_out, grid_w, grid_s, parallel=True, n_processes=96)


