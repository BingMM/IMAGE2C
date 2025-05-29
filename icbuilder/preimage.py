#%% Imports

import numpy as np
from scipy.io.netcdf import NetCDFFile
from typing import Union, Optional
from numpy.typing import NDArray
from secsy import CSgrid

#%% Class for storing orbit files

class PreImage:
    __slots__ = ("mlat", "mlon", "glat", "glon", "dgimg", "shimg", "dgmodel", "shape", "index")

    def __init__(self,
                 ncdf: NetCDFFile,
                 index: Optional[Union[list[int], NDArray[np.int_]]] = None):
        """
        Load orbit file data (e.g., WIC/SI13/SI12) from a NetCDF file.

        Parameters
        ----------
        ncdf : NetCDFFile
            NetCDF file containing image and coordinate data.
        index : Optional[list[int] or np.ndarray]
            Frame indices to load. If None, all frames are loaded.
        """
        self.index = index

        var_names = ['mlat', 'mlon', 'glat', 'glon', 'dgimg', 'shimg', 'dgmodel']
        for name in var_names:
            var_data = ncdf.variables[name]
            data = var_data[...] if index is None else var_data[index, :, :]
            setattr(self, name, np.copy(data))

        self.shape = self.mlat.shape

    def get_shimg(self, i: int) -> NDArray[np.float_]:
        return self.shimg[i, :, :]

    def get_dgimg(self, i: int) -> NDArray[np.float_]:
        return self.dgimg[i, :, :]

    def get_model(self, i: int) -> NDArray[np.float_]:
        return self.dgmodel[i, :, :]

    def get_mcoords(self, i: int) -> tuple[NDArray[np.float_], NDArray[np.float_]]:
        return self.mlat[i, :, :], self.mlon[i, :, :]

    def get_gcoords(self, i: int) -> tuple[NDArray[np.float_], NDArray[np.float_]]:
        return self.glat[i, :, :], self.glon[i, :, :]

    def discard(self, f: Union[list[int], NDArray[np.int_]]) -> None:
        """Keep only the selected frames."""
        for name in ['mlat', 'mlon', 'glat', 'glon', 'dgimg', 'shimg', 'dgmodel']:
            setattr(self, name, getattr(self, name)[f, :, :])

        self.shape = self.mlat.shape

    def percent_full(self, grid: CSgrid) -> float:
        counts = np.zeros((self.shape[0], grid.shape[0], grid.shape[1]))
        for i in range(self.shape[0]):
            mlat, mlon = self.get_mcoords(i)
            f = grid.ingrid(mlon, mlat)
            counts[i] = grid.count(mlon[f], mlat[f])
        f = counts != 0
        return np.sum(f, axis=(1,2)) / grid.size