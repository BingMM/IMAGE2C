#%% Imports

import os
import numpy as np
from typing import Union, Optional
from numpy.typing import NDArray
from secsy import CSgrid
from copy import deepcopy as dcopy
from scipy.stats import t, chi2
from scipy.interpolate import griddata
from .preimage import PreImage

#%% BinnedImage class

class BinnedImage:
    def __init__(self,
                 pI: PreImage,
                 grid: CSgrid,
                 target_grid: Optional[CSgrid] = None,
                 inflate_uncertainty: bool = False,
                 interpolate: bool = False
                 ):
        """
        Bin statistics from a PreImage object into a CSgrid.

        Parameters
        ----------
        pI : PreImage
            Input image data to bin.
        grid : CSgrid
            Grid to bin onto.
        target_grid : CSgrid, optional
            If provided, interpolate results onto this grid.
        inflate_uncertainty : bool
            If True, inflates uncertainties using t and chi² statistics.
        interpolate : bool
            If True, interpolate binned results to the target grid.
        """
        self.grid = dcopy(grid)

        time_len, ny, nx = pI.shape[0], grid.shape[0], grid.shape[1]
        self.counts = np.zeros((time_len, ny, nx))
        self.mu = np.full_like(self.counts, np.nan)
        self.sigma = np.full_like(self.counts, np.nan)
        self.shape = self.counts.shape

        for i in range(time_len):
            lat, lon = pI.get_mcoords(i)
            f = grid.ingrid(lon, lat)
            self.counts[i] = grid.count(lon[f], lat[f])

            shimg = pI.get_shimg(i)
            j, k = grid.bin_index(lon, lat)

            for jj in range(ny):
                for kk in range(nx):
                    mask = (j == jj) & (k == kk)
                    if np.any(mask) and self.counts[i, jj, kk] > 1:
                        values = shimg.flatten()[mask]
                        median_val = np.median(values)
                        self.mu[i, jj, kk] = max(median_val, 0)  # zero if negative
                        self.sigma[i, jj, kk] = np.std(values)

        if inflate_uncertainty:
            self._inflate_uncertainty()

        if interpolate and target_grid is not None:
            self._interpolate(target_grid)

    def _inflate_uncertainty(self, 
                             alpha_mean: float = 0.32, 
                             alpha_std: float = 0.32):
        """Inflate the uncertainty using t and chi² distribution."""
        for i in range(self.shape[0]):
            for j in range(self.shape[1]):
                for k in range(self.shape[2]):
                    df = self.counts[i, j, k] - 1
                    if df < 1:
                        continue
                    t_multiplier = t.ppf(1 - alpha_mean / 2, df)
                    mean_unc = t_multiplier * self.sigma[i, j, k] / np.sqrt(self.counts[i, j, k])
                    chi2_lower = chi2.ppf(alpha_std / 2, df)
                    std_inflation = self.sigma[i, j, k] * np.sqrt(df / chi2_lower)
                    self.sigma[i, j, k] = np.sqrt(mean_unc**2 + std_inflation**2)

    def percent_full(self) -> NDArray[np.float_]:
        """Returns the fraction of bins filled at each time step."""
        return np.sum(self.counts != 0, axis=(1, 2)) / self.counts[0].size

    def _interpolate(self, target_grid: CSgrid):
        """Interpolates mu and sigma to a new grid."""
        self.mu_ = np.copy(self.mu)
        self.sigma_ = np.copy(self.sigma)

        time_len = self.mu.shape[0]
        ny, nx = target_grid.shape

        self.mu = np.full((time_len, ny, nx), np.nan)
        self.sigma = np.full((time_len, ny, nx), np.nan)

        for i in range(time_len):
            # Interpolate mu
            mask = ~np.isnan(self.mu_[i])
            self.mu[i] = griddata(
                (self.grid.xi[mask], self.grid.eta[mask]), self.mu_[i][mask],
                (target_grid.xi, target_grid.eta), method='linear', fill_value=np.nan
            )
            # Interpolate sigma
            mask = ~np.isnan(self.sigma_[i])
            self.sigma[i] = griddata(
                (self.grid.xi[mask], self.grid.eta[mask]), self.sigma_[i][mask],
                (target_grid.xi, target_grid.eta), method='linear', fill_value=np.nan
            )

        self.grid = dcopy(target_grid)

    def discard(self, f: Union[list[int], NDArray[np.int_]]):
        """Discard time steps not in f."""
        self.counts = self.counts[f]
        self.mu = self.mu[f]
        self.sigma = self.sigma[f]
        self.shape = self.mu.shape
