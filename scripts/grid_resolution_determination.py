#%% Import

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from secsy import CSgrid, CSprojection
import fuvpy
import glob
from tqdm import tqdm

#%% Data location

si12_path = '/Data/ift/ift_romfys1/IMAGE_FUV/si12_data/'
si13_path = '/Data/ift/ift_romfys1/IMAGE_FUV/si13_data/'
wic_path  = '/Data/ift/ift_romfys1/IMAGE_FUV/wic/'

#%% Get file names from 1 day
print('Get file names for doy 004 in year 2001.')
print('WIC: Getting files names')
wicfiles = glob.glob(wic_path + 'wic2001004*.idl')
print('SI13: Getting files names')
si13files = glob.glob(si13_path + 's132001004*.idl')
print('SI12: Getting files names')
si12files = glob.glob(si12_path + 's122001004*.idl')

#%% Load wic images
print('WIC: Read data')
wic = fuvpy.read_idl(wicfiles, dzalim=75)
print('SI13: Read data')
si13 = fuvpy.read_idl(si13files, dzalim=75, reflat=False)
print('SI12: Read data')
si12 = fuvpy.read_idl(si12files, dzalim=75, reflat=False)

#%% Background removal
print('WIC: Bacground removal')
wic = fuvpy.backgroundmodel_BS(wic,tukeyVal=5,tOrder=0)
print('SI13: Bacground removal')
si13 = fuvpy.backgroundmodel_BS(si13,tukeyVal=5,tOrder=0)
print('SI12: Bacground removal')
si12 = fuvpy.backgroundmodel_BS(si12,tukeyVal=5,tOrder=0)

#%% Loop over grid resolutions

position = (0, 90) # lon, lat
orientation = (0, 1) # east, north

res = np.linspace(50, 1000, 100)

threshold = [2, 5, 10, 20, 30] # Threshold on the number of pixels in a bin

p_filled_wic = np.zeros((len(threshold), wic.dgimg.shape[0], res.size))
p_filled_si13 = np.zeros((len(threshold), si13.dgimg.shape[0], res.size))
p_filled_si12 = np.zeros((len(threshold), si12.dgimg.shape[0], res.size))

print('Start loop')
for i, resi in tqdm(enumerate(res), total=res.size):
    L, W, Lres, Wres = 9000e3, 9000e3, resi*1e3, resi*1e3 # dimensions and resolution of grid
    grid = CSgrid(CSprojection(position, orientation), L, W, Lres, Wres, R = 6481.2e3)
    
    for j in range(wic.dgimg.shape[0]):
        f = grid.ingrid(wic.glon[j].values, wic.glat[j].values)
        counts = grid.count(wic.glon[j].values[f], wic.glat[j].values[f])
        for k, thres in enumerate(threshold):
            p_filled_wic[k, j, i] = np.sum(counts <= thres) / grid.size

    for j in range(si13.dgimg.shape[0]):
        f = grid.ingrid(si13.glon[j].values, si13.glat[j].values)
        counts = grid.count(si13.glon[j].values[f], si13.glat[j].values[f])
        for k, thres in enumerate(threshold):
            p_filled_si13[k, j, i] = np.sum(counts <= thres) / grid.size

    for j in range(si12.dgimg.shape[0]):
        f = grid.ingrid(si12.glon[j].values, si12.glat[j].values)
        counts = grid.count(si12.glon[j].values[f], si12.glat[j].values[f])
        for k, thres in enumerate(threshold):
            p_filled_si12[k, j, i] = np.sum(counts <= thres) / grid.size

#%% WIC
print('WIC: Make plot')
plt.ioff()
plt.figure(figsize=(10, 10))
for k, thres in enumerate(threshold):
    median = np.median(p_filled_wic[k], axis=0)
    q25 = np.quantile(p_filled_wic[k], .25, axis=0)
    q75 = np.quantile(p_filled_wic[k], .75, axis=0)
    
    plt.plot(res, median, label='{}'.format(thres), zorder=1)
    plt.fill_between(res, q25, q75, color='grey', alpha=.3, zorder=0)
        
ax = plt.gca()
plt.xlim([50, 500])
plt.xlabel('Grid resolution [km]', fontsize=18)
ax.set_xticks([50, 100, 200, 300, 400, 500, 600])
ax.set_xticklabels(['50', '100', '200', '300', '400', '500', '600'], fontsize=14)
plt.ylabel('Cells below threshold [\%]', fontsize=18)
ax.set_yticks([0, .2, .4, .6, .8, 1])
ax.set_yticklabels(['0', '.2', '.4', '.6', '.8', '1'], fontsize=14)
plt.title('WIC: Grid resolution test', fontsize=20)
plt.legend(fontsize=14)
plt.savefig('/Home/siv32/mih008/repos/icAurora/figure/wic_resolution_test.png', bbox_inches='tight')
plt.close('all')
plt.ion()

#%% SI13
print('SI13: Make plot')
plt.ioff()
plt.figure(figsize=(10, 10))
for k, thres in enumerate(threshold):
    median = np.median(p_filled_si13[k], axis=0)
    q25 = np.quantile(p_filled_si13[k], .25, axis=0)
    q75 = np.quantile(p_filled_si13[k], .75, axis=0)
    
    plt.plot(res, median, label='{}'.format(thres), zorder=1)
    plt.fill_between(res, q25, q75, color='grey', alpha=.3, zorder=0)
        
ax = plt.gca()
plt.xlim([50, 1000])
plt.xlabel('Grid resolution [km]', fontsize=18)
ax.set_xticks([100, 200, 300, 400, 500, 600, 700, 800, 900, 1000])
ax.set_xticklabels(['100', '200', '300', '400', '500', '600', '700', '800', '900', '1000'], fontsize=14)
plt.ylabel('Cells below threshold [\%]', fontsize=18)
ax.set_yticks([0, .2, .4, .6, .8, 1])
ax.set_yticklabels(['0', '.2', '.4', '.6', '.8', '1'], fontsize=14)
plt.title('SI13: Grid resolution test', fontsize=20)
plt.legend(fontsize=14)
plt.savefig('/Home/siv32/mih008/repos/icAurora/figure/si13_resolution_test.png', bbox_inches='tight')
plt.close('all')
plt.ion()

#%% SI12
print('SI12: Make plot')
plt.ioff()
plt.figure(figsize=(10, 10))
for k, thres in enumerate(threshold):
    median = np.median(p_filled_si12[k], axis=0)
    q25 = np.quantile(p_filled_si12[k], .25, axis=0)
    q75 = np.quantile(p_filled_si12[k], .75, axis=0)
    
    plt.plot(res, median, label='{}'.format(thres), zorder=1)
    plt.fill_between(res, q25, q75, color='grey', alpha=.3, zorder=0)
        
ax = plt.gca()
plt.xlim([50, 1000])
plt.xlabel('Grid resolution [km]', fontsize=18)
ax.set_xticks([100, 200, 300, 400, 500, 600, 700, 800, 900, 1000])
ax.set_xticklabels(['100', '200', '300', '400', '500', '600', '700', '800', '900', '1000'], fontsize=14)
plt.ylabel('Cells below threshold [\%]', fontsize=18)
ax.set_yticks([0, .2, .4, .6, .8, 1])
ax.set_yticklabels(['0', '.2', '.4', '.6', '.8', '1'], fontsize=14)
plt.title('SI12: Grid resolution test', fontsize=20)
plt.legend(fontsize=14)
plt.savefig('/Home/siv32/mih008/repos/icAurora/figure/si12_resolution_test.png', bbox_inches='tight')
plt.close('all')
plt.ion()
