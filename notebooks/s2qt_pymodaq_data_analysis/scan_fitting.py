import numpy as np
import tqdm
from joblib import Parallel, delayed
import scipy.constants as cst
from pymodaq_data.data import DataRaw, DataCalculated, DataToExport
import s2qt_pymodaq_data_analysis.fit_utils as fut


def fit_raw_scan(spectra_data: DataRaw, w0, fit_function, D=2.87e9, n_jobs=4):
    """
    Fits the spectrum at each pixel and outputs maps.
    
    Parameters
    ----------
    spectra_data: DataRaw
        Object to fit (again)
    w0: float
        Typical width of the resonance for the fit
    fit_function: str
        Name of the fitting function to use, should be present
        in fit_utils.available_functions
    D: float
        Zero field splitting (Hz), used only for init guess when fitting
        both resonances

    Returns
    -------
    fitted_spectra: DataRaw
        Object containing the fitted spectra
    raw_fitted_scan: DataToExport
        Object containing the maps of all the fit parameters and their fit error
    """
    # First we check the fit function
    try:
        fit_func = fut.available_functions[fit_function]
    except:
        print('Unknown fit function, aborting!')
        return
    
    # Then we get the data as np arrays, we first need to find their index
    raw_idx = spectra_data.labels.index('raw')
    odmr_data = spectra_data[raw_idx]
    freqs = spectra_data.get_axis_from_label('Frequency').get_data()
    data_shape = odmr_data.shape

    # we do one fit to get the output of the fit and create a DataRaw object
    # for each param
    fit_result = fit_func(freqs, odmr_data[0, 0], w0)
    dwa_dict = {}
    for ch in fit_result.params.keys():
        dwa_dict[ch] = DataRaw(ch,
                               # preparing two arrays
                               # one for the values and one for the fit error
                               data=[np.zeros(data_shape[:-1]),
                                     np.zeros(data_shape[:-1])],
                               # units of the parameters should be stored
                               # in their user_data attribute
                               unit=fit_result.params[ch].user_data,
                               axes=spectra_data.axes[:-1],
                               labels=[ch, ch + ' std'])
        
    # Adding the spectra
    data_list = [odmr_data, np.zeros(data_shape), np.zeros(data_shape)]
    labels_list = ['raw', 'fit', 'init']

    # Check if we had already a fit performed during the acquisition
    try:
        fit_idx = spectra_data.labels.index('fit')
        data_list.append(spectra_data[fit_idx])
        labels_list.append('instr_fit')
    except:
        print('No fit from the instrument found.')
        pass

    # Check if we have std error on the PL in the raw data object
    std_idx = -1
    try:
        std_idx = spectra_data.labels.index('std')
        data_list.append(spectra_data[std_idx])
        labels_list.append('std')
    except:
        print('No std from the instrument found.')
        pass
    
    fitted_spectra = DataRaw('ODMR spectra', data=data_list,
                             unit='counts/s', axes=spectra_data.axes,
                             labels=labels_list)
    fitted_spectra.nav_indexes = (0, 1)

    # Loop over all pixels for fitting
    for i in tqdm.tqdm(range(data_shape[0])):
        # for j in range(data_shape[1]):

        #     if std_idx > -1:
        #         std = spectra_data[std_idx][i, j]
        #     else:
        #         std = None

        #     fit_result = fit_func(freqs, odmr_data[i, j], w0, std=std, D=D)
        #     fitted_spectra.data[1][i, j] = fit_result.best_fit
        #     fitted_spectra.data[2][i, j] = fit_result.init_fit

        #     for ch in fit_result.params.keys():
        #         dwa_dict[ch].data[0][i, j] = fit_result.params[ch].value
        #         dwa_dict[ch].data[1][i, j] = fit_result.params[ch].stderr

        ## Why is this not faster???
        fit_results = Parallel(n_jobs=n_jobs)(delayed(fit_wrapper)(j, i, fit_func,
                                    std_idx, spectra_data, freqs, odmr_data, w0, D)
                                    for j in range(data_shape[1]))

        for j, fr in enumerate(fit_results):
            fitted_spectra.data[1][i, j] = fr.best_fit
            fitted_spectra.data[2][i, j] = fr.init_fit

            for ch in fr.params.keys():
                dwa_dict[ch].data[0][i, j] = fr.params[ch].value
                dwa_dict[ch].data[1][i, j] = fr.params[ch].stderr

    raw_fitted_scan = DataToExport(name='fit result',
                                   data=[dwa_dict[ch] for ch in dwa_dict.keys()])
    
    return fitted_spectra, raw_fitted_scan


def fit_wrapper(j, i, fit_func, std_idx, spectra_data, freqs, odmr_data, w0, D):
    """ Perform the fit"""
    
    if std_idx > 0:
        std = spectra_data[std_idx][i, j]
    else:
        std = None

    return fit_func(freqs, odmr_data[i, j], w0, std=None, D=D)


def post_treat_fitted_scan(raw_fitted_scan, output_maps, additional_dwas,
                           metadata, ref_freq=0.0):
    """
    Takes the output from the fit and formats it into a nice DataToExport
    for further manipulations and for saving.

    Parameters
    ----------
    raw_fitted_scan: DataToExport
        Result from the fitting, contains all the maps for each parameter.
        This is in principle the output from fit_raw_scan.
    output_maps: list of str
        List of strings specifying the channels the user wishes to obtain
        as output. For example: 'B field', 'Contrast', 'Linewidth',
        'Fitted PL', etc.
    additional_dwas: list of DataRaw
        Additional channels that should be included in the outputed
        DataToExport, for example the topography. 
    metadata: dict
        Metadata loaded from the data files.
    ref_freq: float, default 0 Hz, or list of 2 floats if both resonances
        are tracked. Reference position for the magnetic resonance(s). 
      

    Returns
    -------
    treated_maps: DataToExport
        Contains the maps specified in the output_maps list, and the
        additional ones.
    """
    treated_maps = DataToExport('Treated scan')
    axes = raw_fitted_scan.data[0].axes
    
    if len(ref_freq) == 2:
        two_peaks = True
    else:
        two_peaks = False
    
    for ch in output_maps:
        if ch=='B field':
            gamma_NV = cst.value('electron gyromag. ratio in MHz/T')*1e6
            if two_peaks:
                f1 = raw_fitted_scan.get_data_from_name('p1_center')[0]
                f2 = raw_fitted_scan.get_data_from_name('p2_center')[0]
                b_map = [(f2-f1)-(ref_freq[1]-ref_freq[0])]/(2*gamma_NV)
            else:
                b_map = (raw_fitted_scan.get_data_from_name('center')[0]-ref_freq)/gamma_NV
            treated_maps.append(DataCalculated(ch, data=[b_map], unit='T',
                                               axes=axes, labels=['B_NV'] ))
        elif ch=='Fitted PL':
            pl = raw_fitted_scan.get_data_from_name('c')[0]
            treated_maps.append(DataCalculated(ch, data=[pl], unit='counts/s',
                                               axes=axes, labels=['PL']))
        elif ch=='Linewidth':
            if two_peaks:
                lw1 = raw_fitted_scan.get_data_from_name('p1_fwhm')[0]
                lw2 = raw_fitted_scan.get_data_from_name('p2_fwhm')[0]
                treated_maps.append(DataCalculated(ch, data=[lw1, lw2], unit='Hz',
                                                   axes=axes,labels=['Lower resonance',
                                                                     'Upper resonance']))
            else:
                lw = raw_fitted_scan.get_data_from_name('fwhm')[0]
                treated_maps.append(DataCalculated(ch, data=[lw], unit='Hz',
                                                   axes=axes, labels=['Linewidth']))
        elif ch=='Contrast':
            pl = raw_fitted_scan.get_data_from_name('c')[0]
            if two_peaks:
                c1 = -100*raw_fitted_scan.get_data_from_name('p1_height')[0]/pl
                c2 = -100*raw_fitted_scan.get_data_from_name('p2_height')[0]/pl
                treated_maps.append(DataCalculated(ch, data=[c1, c2], unit='%',
                                                   axes=axes,labels=['Lower resonance',
                                                                     'Upper resonance']))
            else:
                c = raw_fitted_scan.get_data_from_name('fwhm')[0]/pl
                treated_maps.append(DataCalculated(ch, data=[c], unit='%',
                                                   axes=axes, labels=['Contrast']))
        else:
            print(f'Output {ch} not implemented yet.')

    # We add the additional channels too
    treated_maps.append(additional_dwas)
    # We add the metadata to all dwas inside the dte
    for dwa in treated_maps.data:
        dwa.add_extra_attribute(**metadata)

    return treated_maps
            
    
