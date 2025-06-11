#%% Import 

import numpy as np

#%% Conductance fun

def ped(x1,x2):
    """
    Equation (3) in Robinson et al (1987)

    Input
    =====
    x1     : e- average energy [keV]
    x2     : e- energy flux    [ergs/cm², or equivalently mW/m²]

    Output
    ======
    Sigmap : Pedersen conductance [mho, or equiv. siemens, S]

    """
    Sigmap = 40*x1/(16+x1**2)*np.sqrt(x2)
    return Sigmap


def hall(x1,x2):
    """
    Equation (4) in Robinson et al (1987)

    Input
    =====
    x1     : e- average energy [keV]
    x2     : e- energy flux    [ergs/cm², or equivalently mW/m²]

    Output
    ======
    Sigmah : Hall conductance [mho, or equiv. siemens, S]
    """
    Sigmah = 18*x1**(1.85)/(16+x1**2)*np.sqrt(x2) 
    return Sigmah


def peduncertainty(x1,x2,dx1,dx2,varx1x2):
    """
    dSigmaP = peduncertainty(x1,x2,dx1,dx2,varx1x2)

    Calc uncertainty in Pedersen conductance given by Equation (3) in Robinson et al (1987)

    Input
    =====
    x1      : e- average energy                           [keV]
    x2      : e- energy flux                              [ergs/cm², or equivalently mW/m²]
    dx1     : Uncertainty/std deviation of e- avg energy  [keV]
    dx2     : Uncertainty/std deviation of e- energy flux [ergs/cm²]
    varx1x2 : Covariance of e- avg energy and energy flux [keV-ergs/cm²]

    Output
    ======
    dSigmap : Uncertainty in Sigmap [mho, or equiv. S, siemens]
    """

    if x2 == 0:
        _, dSigmap = min_pedersenuncertainty(x1, dx1, dx2, varx1x2)
    else:
        # derivative of Sigmap wrt average energy
        denom = 16+x1**2
        dsp_dx1 = (40/denom - 80*(x1/denom)**2)*np.sqrt(x2)
        dsp_dx2 = 40*x1/denom/2/np.sqrt(x2)
        dSigmap = np.sqrt(dsp_dx1**2 * dx1**2 + dsp_dx2**2 * dx2**2 + 2 * dsp_dx1 * dsp_dx2 * varx1x2)

    return dSigmap

def min_pedersenuncertainty(x1, dx1, dx2, varx1x2):
    """
    Compute the Fe (x2) that minimizes the propagated uncertainty
    in Pedersen conductance, and the corresponding uncertainty.

    Parameters
    ----------
    x1        : float
        Average electron energy [keV]
    dx1       : float
        Standard deviation of average energy [keV]
    dx2       : float
        Standard deviation of energy flux [ergs/cm²]
    cov_x1x2  : float, optional
        Covariance of x1 and x2 [keV·ergs/cm²]. Default is 0 (uncorrelated)

    Returns
    -------
    x2_min    : float
        Energy flux [ergs/cm²] that minimizes the uncertainty
    dSigmap   : float
        Minimum propagated uncertainty in Pedersen conductance [mho]
    """
    # Constants A and B as functions of x1
    A = 40 * x1 / (16 + x1**2)
    B = 40 * (16 - x1**2) / (16 + x1**2)**2

    # x2 (energy flux) that minimizes the uncertainty
    x2_min = (A * dx2) / (2 * B * dx1)

    # Components of uncertainty at the minimum
    term1 = B**2 * x2_min * dx1**2
    term2 = (A**2 / (4 * x2_min)) * dx2**2
    term3 = A * B * varx1x2

    dSigmap = np.sqrt(term1 + term2 + term3)

    return x2_min, dSigmap


def halluncertainty(x1,x2,dx1,dx2,varx1x2):
    """
    dSigmaH = halluncertainty(x1,x2,dx1,dx2,varx1x2)

    Calc uncertainty in Hall conductance given by Equation (4) in Robinson et al (1987)

    Input
    =====
    x1      : e- average energy                           [keV]
    x2      : e- energy flux                              [ergs/cm², or equivalently mW/m²]
    dx1     : Uncertainty/std deviation of e- avg energy  [keV]
    dx2     : Uncertainty/std deviation of e- energy flux [ergs/cm²]
    varx1x2 : Covariance of e- avg energy and energy flux [keV-ergs/cm²]

    Output
    ======
    dSigmah : Uncertainty in Sigmah [mho, or equiv. S, siemens]
    """

    if x2 == 0:
        _, dSigmah = min_halluncertainty(x1, dx1, dx2, varx1x2)
    else:
        # derivative of Sigmah wrt average energy
        denom = 16 + x1**2
        dsh_dx1 = 18 * x1**(0.85) / denom * (1.85 - 2 * x1**2 / denom) * np.sqrt(x2)
        dsh_dx2 =  9 * x1**(1.85) / denom / np.sqrt(x2)
        dSigmah = np.sqrt(dsh_dx1**2 * dx1**2 + dsh_dx2**2 * dx2**2 + 2 * dsh_dx1 * dsh_dx2 * varx1x2)

    return dSigmah

def min_halluncertainty(x1, dx1, dx2, varx1x2):
    """
    Compute the value of x2 (energy flux) that minimizes the propagated uncertainty
    in the Hall conductance, and return the minimum uncertainty.

    Parameters
    ----------
    x1        : average electron energy [keV]
    dx1       : uncertainty in average energy [keV]
    dx2       : uncertainty in energy flux [ergs/cm²]
    varx1x2   : covariance of x1 and x2 [keV * ergs/cm²]

    Returns
    -------
    x2_star   : energy flux that minimizes uncertainty [ergs/cm²]
    dsigmah_min : minimum propagated uncertainty in Hall conductance [S]
    """

    denom = 16 + x1**2

    # A and B constants from the derivation
    A = 18 * x1**1.85 / denom
    B = 18 * x1**0.85 / denom * (1.85 - 2 * x1**2 / denom)

    # Optimal x2 value (where dSigmah is minimized)
    x2_min = (A * dx2) / (2 * B * dx1)

    # Plug into full expression for propagated uncertainty at the minimum
    term1 = B**2 * x2_min * dx1**2
    term2 = (A**2 / (4 * x2_min)) * dx2**2
    term3 = A * B * varx1x2

    dSigmah = np.sqrt(term1 + term2 + term3)

    return x2_min, dSigmah
