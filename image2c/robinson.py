import numpy as np

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

#E0,eflux,sigE0,sigeflux,varE0eflux):

if __name__ == '__main__':
    
    E0,dE0 = 5,1
    eflux,deflux = 2,0.5
    varE0eflux = 0
    
    p = ped(E0,eflux)
    h = hall(E0,eflux)

    dp, dp2 = peduncertainty(E0,eflux,dE0,deflux,varE0eflux)
    dh, dh2 = halluncertainty(E0,eflux,dE0,deflux,varE0eflux)

    print("INPUTS")
    print("======")
    print("{:20s} [{:8s}]: {:10.1f} ± {:5.1f}".format("E0","keV",E0,dE0))
    print("{:20s} [{:8s}]: {:10.1f} ± {:5.1f}".format("eflux","erg/cm²",eflux,deflux))
    print("")
    print("OUTPUTS")
    print("=======")
    print("{:20s} [{:8s}]: {:10.1f} ± {:12.8f}".format("SigmaP(alt1)","S",p,dp))
    print("{:20s} [{:8s}]: {:10.1f} ± {:12.8f}".format("SigmaP(alt2)","S",p,dp2))
    print("")
    print("{:20s} [{:8s}]: {:10.1f} ± {:12.8f}".format("SigmaH(alt1)","S",h,dh))
    print("{:20s} [{:8s}]: {:10.1f} ± {:12.8f}".format("SigmaH(alt2)","S",h,dh2))

    
def E_and_uncertainty(je, jn, sigp, sigh, dsigp, dsigh):
    """
    je, jn, sigp, and sigh all in SI units (A/m for current, mhos for conductance)
    
    """

    sigtotsq = sigp**2+sigh**2
    Ee = ( sigp * je - sigh * jn ) / sigtotsq
    En = ( sigp * jn + sigh * je ) / sigtotsq

    # Derivative of eastward E-field component wrt Sigma_P
    dEe_dsigp =  je / sigtotsq * (1 - 2 * (sigp/sigtot)**2 )
    dEn_dsigp =  jn / sigtotsq * (1 - 2 * (sigp/sigtot)**2 )
    
    dEe_dsigh = -jn / sigtotsq * (1 - 2 * (sigh/sigtot)**2 )
    dEn_dsigh =  je / sigtotsq * (1 - 2 * (sigh/sigtot)**2 )

    DEe = np.sqrt( (dEe_dsigp * dsigp)**2 + (dEe_dsigh * dsigh)**2)
    DEn = np.sqrt( (dEn_dsigp * dsigp)**2 + (dEn_dsigh * dsigh)**2)

    return Ee, En, DEe, DEn

def v_and_uncertainty(je, jn, sigp, sigh, dsigp, dsigh,
                      B0=5e-5):
    """
    je, jn, sigp, sigh, and B0 all in SI units (A/m for current, mhos for conductance)
    
    B0 is, by default, 50,000 nT
    """

    Ee, En, DEe, DEn = E_and_uncertainty(je, jn, sigp, sigh, dsigp, dsigh)

    ve = -En * B0
    vn =  Ee * B0

    Dve = np.sqrt((B0*DEn)**2)
    Dvn = np.sqrt((B0*DEe)**2)

    return ve, vn, Dve, Dvn

