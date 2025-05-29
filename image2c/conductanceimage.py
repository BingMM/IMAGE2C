import os
import numpy as np
from typing import Union, Optional
from numpy.typing import NDArray
from secsy import CSgrid, CSprojection
from copy import deepcopy as dcopy
from scipy.io import netcdf_file
from datetime import datetime, timedelta

# External dependencies
os.chdir('/Home/siv32/mih008/repos/icAurora/scripts')
import imagesat_e0_eflux_estimates as conFun
from robinson import ped, hall, peduncertainty, halluncertainty
from binnedimage import BinnedImage

#%%

class ConductanceImage:
    def __init__(
        self,
        wic: Optional[BinnedImage] = None,
        s12: Optional[BinnedImage] = None,
        s13: Optional[BinnedImage] = None,
        time: Optional[datetime] = None,
        Ep: Union[int, float] = 2,
        dEp: Union[int, float] = 0,
        in_fn: Optional[str] = None,
        out_fn: Optional[str] = None
    ):
        # Sanity checks
        if in_fn is not None and any(x is not None for x in [wic, s12, s13]):
            raise ValueError("Provide either an input filename or WIC, SI12, and SI13 images, not both.")
        if in_fn is None and not all(x is not None for x in [wic, s12, s13]):
            raise ValueError("Provide all three: WIC, SI12, and SI13 images if not using input file.")

        self.time = time
        self.Ep = Ep
        self.dEp = dEp
        self.grid = None

        if in_fn:
            self.load_nc(in_fn)
        else:
            self.grid = dcopy(wic.grid)
            self._shape = wic.mu.shape
            self._initialize_arrays()
            self._compute_conductance(wic, s12, s13)

        if out_fn:
            self.to_nc(out_fn)

    @property
    def shape(self):
        return self._shape

    def _initialize_arrays(self):
        shp = self.shape
        self.E0     = np.full(shp, np.nan)
        self.dE0    = np.full(shp, np.nan)
        self.Fe     = np.full(shp, np.nan)
        self.dFe    = np.full(shp, np.nan)
        self.R      = np.full(shp, np.nan)
        self.dR     = np.full(shp, np.nan)
        self.P      = np.full(shp, np.nan)
        self.H      = np.full(shp, np.nan)
        self.dP     = np.full(shp, np.nan)
        self.dH     = np.full(shp, np.nan)
        self.dP2    = np.full(shp, np.nan)
        self.dH2    = np.full(shp, np.nan)

    def _compute_conductance(self, 
                             wic: BinnedImage, 
                             s12: BinnedImage, 
                             s13: BinnedImage):
        for i in range(self.shape[0]):
            for j in range(self.shape[1]):
                for k in range(self.shape[2]):
                    self._compute_pixel(i, j, k, wic, s12, s13)

    def _compute_pixel(self, 
                       i: int, 
                       j: int, 
                       k: int, 
                       wic: BinnedImage, 
                       s12: BinnedImage, 
                       s13: BinnedImage):
        W, T, S = wic.mu[i, j, k], s12.mu[i, j, k], s13.mu[i, j, k]
        dW, dT, dS = wic.sigma[i, j, k], s12.sigma[i, j, k], s13.sigma[i, j, k]

        counts          = [W, T, S]
        dayglow         = [0, 0, 0]
        uncertainties   = [dW, dT, dS]
        
        if np.all(~np.isnan(counts)) and np.all(~np.isnan(uncertainties)):
            
            E0, Fe, dE0, dFe, R, dR = conFun.E0_eflux_propagated(counts, dayglow, 
                                                                 uncertainties, 
                                                                 self.Ep, self.dEp)
            self.E0[i, j, k], self.dE0[i, j, k] = E0, dE0
            self.Fe[i, j, k], self.dFe[i, j, k] = Fe, dFe
            self.R[i, j, k],  self.dR[i, j, k]  = R,  dR 
            
            varE0Fe = 0  # Placeholder for covariance, assumed 0 here
            P, H = ped(E0, Fe), hall(E0, Fe)
            
            if Fe == 0:
                dP, dP2 = .4*P, .4*P
                dH, dH2 = .4*H, .4*H
            else:
                dP, dP2 = peduncertainty(E0, Fe, dE0, dFe, varE0Fe)
                dH, dH2 = halluncertainty(E0, Fe, dE0, dFe, varE0Fe)
            
            self.P[i, j, k],    self.H[i, j, k]     = P, H
            self.dP[i, j, k],   self.dH[i, j, k]    = dP, dH
            self.dP2[i, j, k],  self.dH2[i, j, k]   = dP2, dH2

    def to_nc(self, filename: str):
        with netcdf_file(filename, 'w') as nc:
            t, y, x = self.shape
            nc.createDimension('time', t)
            nc.createDimension('dim1', y)
            nc.createDimension('dim2', x)

            def save_var(name, data):
                var = nc.createVariable(name, float, ('time', 'dim1', 'dim2'))
                var[:] = data

            for attr in ["E0", "dE0", "Fe", "dFe", "R", "dR", "P", "H", "dP", "dH", "dP2", "dH2"]:
                save_var(attr, getattr(self, attr))

            nc.Ep = float(self.Ep)
            nc.dEp = float(self.dEp)

            if self.time is not None:
                ref_time = datetime(2000, 1, 1)
                time_seconds = np.array([(t - ref_time).total_seconds() for t in self.time], dtype=np.int32)
                nc.createVariable("time", np.int32, ("time",))[:] = time_seconds
                nc.reference_time = ref_time.strftime("%Y-%m-%dT%H:%M:%S")

            if self.grid and hasattr(self.grid, "projection"):
                nc.position = self.grid.projection.position.astype(float)
                nc.orientation = self.grid.projection.orientation
                nc.L = self.grid.L
                nc.W = self.grid.W
                nc.Lres = self.grid.Lres
                nc.Wres = self.grid.Wres

    def load_nc(self, filename: str):
        with netcdf_file(filename, 'r') as nc:
            def load_var(name):
                return np.copy(nc.variables[name][:])

            for attr in ["E0", "dE0", "Fe", "dFe", "R", "dR", "P", "H", "dP", "dH", "dP2", "dH2"]:
                setattr(self, attr, load_var(attr))

            self.Ep = float(nc.Ep)
            self.dEp = float(nc.dEp)
            self._shape = self.E0.shape

            if "time" in nc.variables:
                ref = datetime.strptime(nc.reference_time.decode(), "%Y-%m-%dT%H:%M:%S")
                self.time = np.array([ref + timedelta(seconds=int(s)) for s in nc.variables["time"][:]])

            self.grid = CSgrid(
                CSprojection(nc.position, nc.orientation),
                nc.L, nc.W, nc.Lres, nc.Wres
                )
