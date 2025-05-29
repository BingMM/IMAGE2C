#%% Import

import pandas as pd
import glob

#%% Premade orbit file from WIC, not complete list only published corrected data

orbit_dates = '/disk/IMAGE_FUV/fuv/orbitdates.csv'

#%% Import orbit file

print('Reading orbit date from NIRD file')
orbits = pd.read_csv(orbit_dates)

# Convert str to dt
orbits['dt_start'] = pd.to_datetime(orbits['date_start'],format='%Y-%m-%d %H:%M:%S')
orbits['dt_end'] = pd.to_datetime(orbits['date_end'],format='%Y-%m-%d %H:%M:%S')

#%% Path to SI12 and SI13

wic_path = '/Data/ift/ift_romfys1/IMAGE_FUV/wic/'
s12_path = '/Data/ift/ift_romfys1/IMAGE_FUV/si12_data/'
s13_path = '/Data/ift/ift_romfys1/IMAGE_FUV/si13_data/'

#%% get file names

print('Getting wic file names')
wic_fn = sorted(glob.glob(wic_path + '*.idl'))
print('Getting s12 file names')
s12_fn = sorted(glob.glob(s12_path + '*.idl'))
print('Getting s13 file names')
s13_fn = sorted(glob.glob(s13_path + '*.idl'))

#%% functions

def generate_files_file(fn):
    # Allocate space
    df = pd.DataFrame()

    # Date and filename
    df['date'] = [pd.to_datetime(file[-15:-4],format='%Y%j%H%M') for file in fn]
    df['filename']=[file[-18:] for file in fn]

    # Orbit number
    df['orbit'] = -1
    df['orbit'] = df['orbit'].astype('int64')

    for _, row in orbits.iterrows():
        mask = (df['date'] >= row['dt_start']) & (df['date'] <= row['dt_end'])
        df.loc[mask, 'orbit'] = row['orbit_number']
    df = df.loc[df['orbit']!=-1, :].reset_index().drop(columns='index')

    df.set_index('date', inplace=True)
    
    return df

#%% wic

print('Generating wic files file')
df = generate_files_file(wic_fn)
df.to_hdf('/disk/IMAGE_FUV/fuv/wicfiles.h5', key='data', mode='w')

#%% s12

print('Generating s12 files file')
df = generate_files_file(s12_fn)
df.to_hdf('/disk/IMAGE_FUV/fuv/s12files.h5', key='data', mode='w')

#%% s13

print('Generating s13 files file')
df = generate_files_file(s13_fn)
df.to_hdf('/disk/IMAGE_FUV/fuv/s13files.h5', key='data', mode='w')
