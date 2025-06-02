#%% Imports

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
    """
    A class to bin IMAGE data onto a CSgrid.

    This class processes a `PreImage` object and computes binned statistics
    (median and standard deviation) for each grid cell, optionally inflating the
    uncertainties or interpolating to a different grid.

    Attributes
    ----------
    grid : CSgrid
        The CSgrid to which data is binned (or interpolated).
    counts : np.ndarray
        Number of samples contributing to each grid cell.
    mu : np.ndarray
        Mean (median) of the binned values in each grid cell.
    sigma : np.ndarray
        Standard deviation of the values in each grid cell.
    shape : tuple
        Shape of the binned data arrays.
    """
    
    def __init__(self,
                 pI: PreImage,
                 grid: CSgrid,
                 target_grid: Optional[CSgrid] = None,
                 inflate_uncertainty: bool = False
                 ):
        """
        Bin statistics from a PreImage object into a CSgrid.

        Parameters
        ----------
        pI : PreImage
            Input IMAGE data to bin.
        grid : CSgrid
            Cubbed sphere grid to bin onto.
        target_grid : CSgrid, optional
            If provided, interpolate results onto this cubbed sphere grid.
        inflate_uncertainty : bool
            If True, inflates uncertainties using t and chi² statistics.
            Should be used when less than 30 counts per bin.
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

        if target_grid is not None:
            self._interpolate(target_grid)
            self.shape = self.mu.shape

    def _inflate_uncertainty(self, 
                             alpha_mean: float = 0.32, 
                             alpha_std: float = 0.32):
        """
        Inflate the uncertainty estimates using Student's t-distribution
        and the chi-squared distribution.

        Parameters
        ----------
        alpha_mean : float
            Confidence level for inflating the uncertainty on the mean. .32 = 68% = 1 std
        alpha_std : float
            Confidence level for inflating the standard deviation. .32 = 68% = 1 std
        """
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

    def _interpolate(self, 
                     target_grid: CSgrid):
        """
        Interpolate `mu` and `sigma` fields from the current grid to a new grid.

        Parameters
        ----------
        target_grid : CSgrid
            The grid to interpolate onto.
        """
        self.mu_ = np.copy(self.mu)
        self.sigma_ = np.copy(self.sigma)

        time_len = self.shape[0]
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
        """
        Discard all time steps NOT listed in `f`.

        Parameters
        ----------
        f : list[int] or NDArray[np.int_]
            Indices of time steps to retain.
        """
        self.counts = self.counts[f]
        self.mu = self.mu[f]
        self.sigma = self.sigma[f]
        self.shape = self.mu.shape
