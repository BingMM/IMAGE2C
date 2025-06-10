#%% Import

import numpy as np
from typing import Union, Optional
from numpy.typing import NDArray
from copy import deepcopy as dcopy
from scipy.io import netcdf_file
from datetime import datetime

# External dependencies
from .imagesat_e0_eflux_estimates import E0_eflux_propagated as EF_fun
from .robinson import ped, hall, peduncertainty, halluncertainty
from .binnedimage import BinnedImage

#%% Conductance Image class

class ConductanceImage:
    """
    Constructs ionospheric conductance from IMAGE satellite binned counts.

    Attributes
    ----------
    time : datetime, optional
        Time associated with the dataset.
    Ep : float
        Mean electron energy [keV].
    dEp : float
        Uncertainty in the mean energy [keV].
    grid : CSgrid
        Cubbed sphere grid from the BinnedImage input.
    shape : tuple
        Shape of the input binned images.

    The following arrays are populated:
        E0, dE0 : Characteristic energy and its uncertainty.
        Fe, dFe : Energy flux and its uncertainty.
        R, dR   : Ratio and its uncertainty.
        P, H    : Pedersen and Hall conductance [mho].
        dP, dH  : Uncertainty based on error propagation [mho].
        dP2, dH2: Alternative uncertainty.
    """

    def __init__(
        self,
        wic: BinnedImage,
        s12: BinnedImage,
        s13: BinnedImage,
        time: Optional[Union[NDArray[datetime], list[datetime]]] = None,
        Ep: Union[int, float] = 2,
        dEp: Union[int, float] = 0,
        out_fn: Optional[str] = None
    ):
        """
        Initialize a ConductanceImage from three binned images.

        Parameters
        ----------
        wic : BinnedImage
            WIC binned image (counts).
        s12 : BinnedImage
            SI12 binned image (counts).
        s13 : BinnedImage
            SI13 binned image (counts).
        time : array or list of datetime, optional
            Timestamp of the images.
        Ep : float, optional
            Mean electron energy [keV], default 2 keV.
        dEp : float, optional
            Uncertainty in Ep, default 0 keV.
        out_fn : str, optional
            If given, the resulting conductance image will be saved to this NetCDF file.
        """
        if not (wic.shape == s12.shape == s13.shape):
            raise ValueError('wic, s12, and s13 have to have the same shape.')

        self.time = time
        self.Ep = Ep
        self.dEp = dEp
        self.grid = dcopy(wic.grid)
        self.shape = wic.shape
        
        self._store_binned_statistics(wic, s12, s13)
        self._initialize_arrays()
        self._compute_conductance()

        if out_fn:
            self.to_nc(out_fn)

    def _store_binned_statistics(self, wic: BinnedImage, s12: BinnedImage, s13: BinnedImage):
        """
        Stores the average and standard deviation from each BinnedImage.
        """
        self.wic_avg = wic.mu
        self.wic_std = wic.sigma
        self.s12_avg = s12.mu
        self.s12_std = s12.sigma
        self.s13_avg = s13.mu
        self.s13_std = s13.sigma

    def _initialize_arrays(self):
        """
        Initializes empty arrays for all estimated quantities.
        """
        self.E0     = np.full(self.shape, np.nan)
        self.dE0    = np.full(self.shape, np.nan)
        self.Fe     = np.full(self.shape, np.nan)
        self.dFe    = np.full(self.shape, np.nan)
        self.R      = np.full(self.shape, np.nan)
        self.dR     = np.full(self.shape, np.nan)
        self.P      = np.full(self.shape, np.nan)
        self.H      = np.full(self.shape, np.nan)
        self.dP     = np.full(self.shape, np.nan)
        self.dH     = np.full(self.shape, np.nan)
        self.dP2    = np.full(self.shape, np.nan)
        self.dH2    = np.full(self.shape, np.nan)

    def _compute_conductance(self):
        """
        Loops through all pixels and computes conductance estimates.
        """
        for i in range(self.shape[0]):
            for j in range(self.shape[1]):
                for k in range(self.shape[2]):
                    self._compute_pixel(i, j, k)

    def _compute_pixel(self, i: int, j: int, k: int):
        """
        Compute E0, Fe, R, and conductances at a single pixel.

        Parameters
        ----------
        i, j, k : int
            Indices into the 3D array (time, lat, lon).
        """
        W, T, S = self.wic_avg[i, j, k], self.s12_avg[i, j, k], self.s13_avg[i, j, k]
        dW, dT, dS = self.wic_std[i, j, k], self.s12_std[i, j, k], self.s13_std[i, j, k]

        counts = [W, T, S]
        dayglow = [0, 0, 0]  # Assume zero dayglow
        uncertainties = [dW, dT, dS]
        
        if np.all(~np.isnan(counts)) and np.all(~np.isnan(uncertainties)):
            E0, Fe, dE0, dFe, R, dR = EF_fun(counts, dayglow, uncertainties, self.Ep, self.dEp)

            self.E0[i, j, k], self.dE0[i, j, k] = E0, dE0
            self.Fe[i, j, k], self.dFe[i, j, k] = Fe, dFe
            self.R[i, j, k],  self.dR[i, j, k]  = R,  dR 

            varE0Fe = 0  # Placeholder for covariance
            P, H = ped(E0, Fe), hall(E0, Fe)

            if Fe == 0:
                dP = dP2 = 0.4 * P
                dH = dH2 = 0.4 * H
            else:
                dP, dP2 = peduncertainty(E0, Fe, dE0, dFe, varE0Fe)
                dH, dH2 = halluncertainty(E0, Fe, dE0, dFe, varE0Fe)
            
            self.P[i, j, k], self.H[i, j, k] = P, H
            self.dP[i, j, k], self.dH[i, j, k] = dP, dH
            self.dP2[i, j, k], self.dH2[i, j, k] = dP2, dH2

    def to_nc(self, filename: str):
        """
        Save conductance image to a NetCDF file.
        Can be read/rebuilt using the icReader library.

        Parameters
        ----------
        filename : str
            Full path to output NetCDF file.
        """
        with netcdf_file(filename, 'w') as nc:
            t, y, x = self.shape
            nc.createDimension('time', t)
            nc.createDimension('dim1', y)
            nc.createDimension('dim2', x)

            def save_var(name, data):
                var = nc.createVariable(name, float, ('time', 'dim1', 'dim2'))
                var[:] = data

            for attr in ['wic_avg', 's12_avg', 's13_avg', 'wic_std', 's12_std', 's13_std', 
                         'E0', 'dE0', 'Fe', 'dFe', 'R', 'dR', 'P', 'H', 'dP', 'dH', 
                         'dP2', 'dH2']:
                save_var(attr, getattr(self, attr))

            nc.Ep = float(self.Ep)
            nc.dEp = float(self.dEp)

            if self.time is not None:
                ref_time = datetime(2000, 1, 1)
                time_seconds = np.array([(t - ref_time).total_seconds() for t in self.time], dtype=np.int32)
                nc.createVariable("time", np.int32, ("time",))[:] = time_seconds
                nc.reference_time = ref_time.strftime("%Y-%m-%dT%H:%M:%S")

            if self.grid and hasattr(self.grid, "projection"):
                nc.position     = self.grid.projection.position.astype(float)
                nc.orientation  = self.grid.projection.orientation
                nc.L    = self.grid.L
                nc.W    = self.grid.W
                nc.Lres = self.grid.Lres
                nc.Wres = self.grid.Wres
                nc.gridR    = self.grid.R
