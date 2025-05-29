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

    # derivative of Sigmap wrt average energy
    denom = 16+x1**2
    dsp_dx1 = (40/denom - 80*(x1/denom)**2)*np.sqrt(x2)
    dsp_dx2 = 40*x1/denom/2/np.sqrt(x2)

    dSigmap = np.sqrt(dsp_dx1**2 * dx1**2 + dsp_dx2**2 * dx2**2 + 2 * dsp_dx1 * dsp_dx2 * varx1x2)

    # Alternate expression
    dSigmap2 = np.sqrt( (40/denom * (1-2*x1**2/denom))**2*x2*dx1**2 
                        + (20*x1/denom)**2 * dx2**2/x2 
                        + 1600 * x1 / denom**2 * (1 - 2*x1**2/denom) * varx1x2 )

    return dSigmap, dSigmap2


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

    # derivative of Sigmah wrt average energy
    denom = 16 + x1**2
    dsh_dx1 = 18 * x1**(0.85) / denom * (1.85 - 2 * x1**2 / denom) * np.sqrt(x2)
    dsh_dx2 =  9 * x1**(1.85) / denom / np.sqrt(x2)

    dSigmah = np.sqrt(dsh_dx1**2 * dx1**2 + dsh_dx2**2 * dx2**2 + 2 * dsh_dx1 * dsh_dx2 * varx1x2)

    # Alternate expression
    dSigmah2 = np.sqrt( (18 * x1**(0.85) / denom * (1.85 - 2 * x1**2 / denom) )**2 * x2 * dx1**2 
                        + (9 * x1**(1.85) / denom )**2 * dx2**2 / x2 
                        + 18**2 * x1**(2.7) / denom**2 * (1.85 - 2*x1**2/denom) * varx1x2 )

    return dSigmah, dSigmah2
