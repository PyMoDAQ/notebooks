import lmfit
import numpy as np
import pymodaq_data.data as pdd

def initial_guess_single_peak(f, pl, w0):
    """
    Find starting parameters for the fit to a single lorentzian.
    
    Parameters
    ----------
    f: 1D ndarray
        Frequency list (Hz)
    pl: 1D ndarray
        PL data
    w0: float
        Initial guess for the width

    Returns
    -------
    init_guess: dict of dicts
        Initial set of parameters, with value, min and max for lmfit
    """
    # PL off resonance
    init_guess = {}
    average = (pl + np.roll(pl, 1) + np.roll(pl, -1))/3
    init_guess["c"] = dict(value=np.quantile(average, 0.8),
                           min=np.min(average),
                           max=np.max(average))
    # min freq, obtained by averaging over 3 px and taking the min value
    init_guess["center"] = dict(value=f[np.argmin(average[1:-1]) + 1],
                                min=f[0]-w0,
                                max=f[-1]+w0)
    # contrast
    init_guess["amplitude"] = dict(value=-(np.quantile(average, 0.8) - np.min(pl))*w0*np.pi/2,
                                   min=-(np.quantile(average, 0.8) - np.min(pl))*w0*np.pi,
                                   max=0)
    # FWHM
    init_guess["sigma"] = dict(value=w0/2, min=0, max=f[-1]-f[0])
    
    return init_guess

def fit_single_lorentzian(f, pl, w0, std=None, **kwargs):
    """
    Fits the data to a lorentzian peak.
    
    Parameters
    ----------
    f: 1D ndarray
        Frequency list (Hz)
    pl: 1D ndarray
        PL data
    w0: float
        Initial guess for the width
    
    Returns
    -------
    fit_result: lm.ModelResult
  
        Result of the fit
    """
    
    sl = lmfit.models.LorentzianModel() + lmfit.models.ConstantModel()
    init_guess = initial_guess_single_peak(f, pl, w0)
    
    # we will use the additional attribute user_data to store units
    # in each Parameter
    params = sl.make_params(**init_guess)
    params = set_params_units(params)
   
    weights = None
    if std is not None:
        try:
            weights = 1.0/std
        except:
            weights = None
        
    fit_result = sl.fit(pl, params, x=f, weights=weights, nan_policy='omit')
    return fit_result

def fit_single_gaussian(f, pl, w0, std=None, **kwargs):
    """
    Fits the data to a gaussian peak.

    Parameters
    ----------
    f: 1D ndarray
        Frequency list (Hz)
    pl: 1D ndarray
        PL data
    w0: float
        Initial guess for the width

    Returns
    -------
    fit_result: lm.ModelResult
        Result of the fit
    """

    sg = lmfit.models.GaussianModel() + lmfit.models.ConstantModel()
    init_guess = initial_guess_single_peak(f, pl, w0)

    # we will use the additional attribute user_data to store units
    # in each Parameter
    params = sg.make_params(**init_guess)
    params = set_params_units(params)

    weights = None
    if std is not None:
        weights = 1.0/std

    fit_result = sg.fit(pl, params, x=f, weights=weights)
    
    return fit_result

def initial_guess_double_peak(f, pl, w0, D):
    """
    Find starting parameters for the fit to 2 lorentzian peaks.
    
    Parameters
    ----------
    f: 1D ndarray
        Frequency list (Hz)
    pl: 1D ndarray
        PL data
    w0: float
        Initial guess for the width
    D: float
        Zero-field splitting (Hz)

    Returns
    -------
    init_guess: dict of dicts
        Initial set of parameters, with value, min and max for lmfit
    """
    # PL off resonance
    init_guess = {}
    average = (pl + np.roll(pl, 1) + np.roll(pl, -1))/3
    left_pl = average[:len(pl) // 2]
    right_pl = average[len(pl) // 2:]

    init_guess["c"] = dict(value = np.quantile(average, 0.8),
                           min = np.min(average), max = np.max(average))

    # min freq, obtained by averaging over 3 px and taking the min value
    init_guess['p1_center'] = dict(value=f[np.argmin(average[1:-1])+1],
                                   min=f[0], max=f[-1])
    if init_guess['p1_center']['value'] < D: # might be suboptimal for thermo measurement
        init_guess['p1_center']['max'] = D
        init_guess['p2_center'] = dict(value = 2*D-init_guess['p1_center']['value'],
                                       min=D, max=f[-1])
    else:
        init_guess['p2_center'] = dict(value=init_guess['p1_center']['value'],
                                       min=f[0], max=D)
        init_guess['p1_center']['value'] = dict(value=2*D-init_guess['p2_center']['value'],
                                                min=D, max=f[-1])        
    # contrast
    init_guess["p1_amplitude"] = dict(value = -(np.max(left_pl)-np.min(left_pl))*w0*np.pi/2,
                                      min = -(np.max(left_pl)-np.min(left_pl))*w0*np.pi, max = 0)
    init_guess["p2_amplitude"] = dict(value =-(np.max(right_pl)-np.min(right_pl))*w0*np.pi/2,
                                      min = -(np.max(right_pl)-np.min(right_pl))*w0*np.pi, max = 0)

    # FWHM
    init_guess["p1_sigma"] = dict(value = w0/2, min = (f[-1]-f[0])/50, max = 0.5*(f[-1]-f[0]))
    init_guess["p2_sigma"] = dict(value = w0/2, min = (f[-1]-f[0])/50, max = 0.5*(f[-1]-f[0]))


    return init_guess

def fit_double_lorentzian(f, pl, w0, std=None, D=2.87e9):
    """
    Fits the data to 2 lorentzian peaks.
    
    Parameters
    ----------
    f: 1D ndarray
        Frequency list (Hz)
    pl: 1D ndarray
        PL data
    w0: float
        Initial guess for the width
    D: float
        Zero-field splitting (Hz)

    Returns
    -------
    fit_result: lm.ModelResult
        Result of the fit
    """
    
    dl = lmfit.models.LorentzianModel(prefix='p1_') + \
         lmfit.models.LorentzianModel(prefix='p2_') + \
         lmfit.models.ConstantModel()

    init_guess = initial_guess_double_peak(f, pl, w0, D)
    
    # we will use the additional attribute user_data to store units
    # in each Parameter
    params = dl.make_params(**init_guess)
    params = set_params_units(params)
   
    weights = None
    if std is not None:
        weights = 1.0/std

    fit_result = dl.fit(pl, params, x=f, weights=weights)
    return fit_result
    
def fit_double_gaussian(f, pl, w0, std=None, D=2.87e9):
    """
    Fits the data to 2 gaussian peaks.

    Parameters
    ----------
    f: 1D ndarray
        Frequency list (Hz)
    pl: 1D ndarray
        PL data
    w0: float
        Initial guess for the width
    D: float
        Zero-field splitting (Hz)

    Returns
    -------
    fit_result: lm.ModelResult
        Result of the fit
    """

    dg = lmfit.models.GaussianModel(prefix='p1_')+ \
         lmfit.models.GaussianModel(prefix='p2_') + \
         lmfit.models.ConstantModel()

    init_guess = initial_guess_double_peak(f, pl, w0, D)
    params = dg.make_params(**init_guess)

    # we will use the additional attribute user_data to store units
    # in each Parameter
    params = set_params_units(params)

    weights = None
    if std is not None:
        weights = 1.0/std

    fit_result = dg.fit(pl, params, x=f, weights=weights)
    return fit_result


def set_params_units(params):
    """
    Sets the units of the fit parameters.
    """
    for k in params.keys():
        if 'center' in k or 'sigma' in k or 'fwhm' in k:
            params[k].user_data = 'Hz'
        elif 'height' in k or k=='c':
            params[k].user_data = 'counts/s' 
       
    return params


available_functions = {'single_lorentzian': fit_single_lorentzian,
                       'single_gaussian': fit_single_gaussian,
                       'double_lorentzian': fit_double_lorentzian,
                       'double_gaussian': fit_double_gaussian}
