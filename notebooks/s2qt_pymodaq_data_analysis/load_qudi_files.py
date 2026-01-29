import glob
import numpy as np

import pymodaq_data.data as pdd
from pymodaq_data import Q_, ureg, Unit
from pymodaq_utils.enums import enum_checker

import s2qt_pymodaq_data_analysis.utils as ut
import s2qt_pymodaq_data_analysis.fit_utils as fut

def load_header_qudi(content):
    """
    Get the header part for a Qudi data file.

    Parameters
    ----------
    filename: str
        Path to the file.

    Returns
    -------
    header_content: str
        List of the text lines before the data starts.
    """
    header_content = []
    with open(filename, 'r') as f:
        line = f.readline()
        while not '#Data:' in line:
            header_content.append(line)
            line = f.readline()
            
    return header_content


def get_metadata_qudi(content):
    """
    Extract useful info from the header of Qudi files.

    Parameters
    ----------
    content: list of str
        List of str containing the header information read from the file.

    Returns
    -------
    info: dict
        Interpreted header info.
    ch_list: list of str
        List of data channels acquired during the scan, if relevant,
        otherwise empty list.
    """
    info = {}
    for l in content[5:]:
        if ':' in l:
            header_key, header_value = l[1:].split(':')
            try:
                header_value = float(header_value)
            except:
                   pass     
            info[header_key] = header_value
                
    info["X resolution (px)"] = int(info["X resolution (px)"])
    info["Y resolution (px)"] = int(info["Y resolution (px)"])
    return info
    

def load_qudi_odmr(path, timestamp, tag='', fit=False, fitmodel='single_lorentzian'):
    """Load a raw ODMR scan measured with qudi.

    Parameters
    ----------
    path: str
        Path to the folder containing all the raw data files
    timestamp: str, "yyyymmdd-hhmm-ss"
        Unique timestamp of the file
    fit: bool
        Indicate if the data should be fitted
    fitmodel: str
        Specify which fit model to use, should be a key of the
        available_functions dict from fit_utils

    Returns
    -------
    odmr_scan: pdd.DataRaw
        The spectrum and its fit if asked for.
    fit_result: lmfit.ModelResult
        Result of the fit if it was performed
    """
    data = np.loadtxt(f'{path}/{timestamp}_ODMR_data_ch0{tag}.dat')
    axis = pdd.Axis('freq_axis', units='Hz', data=data[:,0], index=0)
    data_list = [data[:,1]]
    # label_list = ['raw PL']
    #
    # fit_result = None
    #
    # if fit:
    #    try:
    #        fit_func = fut.available_functions[fitmodel]
    #        fit_result = fit_func(axis.get_data(), data[:,1], 10e6)
    #    except:
    #        print('Unknown fit function! No fit will be provided.')

    # if fit_result is not None:
    #     data_list.append(fit_result.best_fit)
    #     label_list.append('Fit')
    #     data_list.append(fit_result.init_fit)
    #     label_list.append('Initial guess')
    
    # odmr_scan = pdd.DataRaw('ODMR scan', dim=pdd.DataDim['Data1D'],
    #                        data=data_list, labels=label_list,
    #                        units='counts/s', axes=[axis])
    
    odmr_scan, fit_result = lut.fit_and_format_odmr(axis, data_list, fit, fitmodel)
    
    return odmr_scan, fit_result


def load_qudi_scan(path, timestamp, experiment_type):
    """ Load a raw scan measured with qudi.

    Parameters
    ----------
    path: str
        Path to the folder containing all the raw data files
    timestamp: str, "yyyymmdd-hhmm-ss"
        Unique timestamp of the file
    experiment_type: ut.ExperimentType
        Enum specifying the experiment type
    
    Returns
    -------
    raw_scan: pdd.DataToExport
        Data object containing all the image data channels
    all_headers: dict
        All the info extracted from the headers
    spectra_data: pdd.DataRaw (or None)
        3D data containing the spectra and the fits from qudi if present
    """
    data_files = glob.glob(f'{path}/{timestamp}*.dat')
    
    all_headers = {}
    data_arrays = {}
    spectra = {}
    dwa_dict = {}
    freq_list = None
    spectra_data = None
    
    for f in data_files:
        if f.endswith('raw.dat'):
            data_array = np.loadtxt(f) # col 0 = X, col 1 = Y, col 2 = data
            header_content = load_header_qudi(f)
            header = get_metadata_qudi(header_content)
            for ch in ch_list:
                if ch in f:
                    all_headers[ch] = header
                    data_arrays[ch] = data_array
                if experiment_type == ut.ExperimentType.FullB and freq_list is None:
                    freq_list = np.linspace(header['ESR Frequency start (Hz)'],
                                            header['ESR Frequency stop (Hz)'],
                                            int(header['ESR Frequency points (#)']))
                    freq_axis = pdd.Axis('Frequency', data=freq_list,
                                         index=2, units="Hz")
                    

        elif experiment_type == ut.ExperimentType.FullB and 'esr' in f:
            data_array = np.loadtxt(f)
            if 'fit' in f:
                spectra['fit'] = data_array
            elif 'std' in f:
                spectra['std'] = data_array
            else:
                spectra['raw'] = data_array
                

    for ch in ch_list:
        # total nb of pixels
        N = all_headers[ch]['Y resolution (px)']*all_headers[ch]['X resolution (px)']
        
        data2D = data_arrays[ch][:N, 2].reshape((all_headers[ch]['Y resolution (px)'],
                                                 all_headers[ch]['X resolution (px)']))
        
        x_axis = pdd.Axis('x', data=1e6*np.linspace(np.nanmin(data_arrays[ch][:N, 0]),
                                                np.nanmax(data_arrays[ch][:N, 0]),
                                                all_headers[ch]['X resolution (px)']),
                          index=1, units='µm')
        
        y_axis = pdd.Axis('y', data=1e6*np.linspace(np.nanmin(data_arrays[ch][:N, 1]),
                                                np.nanmax(data_arrays[ch][:N, 1]),
                                                all_headers[ch]['Y resolution (px)']),
                          index=0, units='µm')
        
        dwa_dict[ch] = pdd.DataRaw(ch, data=data2D, unit=all_headers[ch]['Unit'],
                                   axes=[x_axis, y_axis], labels=[ch])
        

    raw_scan = pdd.DataToExport(name=timestamp,
                                data=[dwa_dict[ch] for ch in ch_list])

    if experiment_type == ut.ExperimentType.FullB and freq_list is not None:

        Nx = all_headers[ch_list[0]]['X resolution (px)']
        Ny = all_headers[ch_list[0]]['Y resolution (px)']
        Nf = len(freq_list)
        
        spectra_data = pdd.DataRaw("Spectra",
                                   data=[spectra[k].reshape((Ny, Nx, Nf)) for k in spectra.keys()],
                                   units="counts/s", labels=[k for k in spectra.keys()],
                                   axes=[x_axis, y_axis, freq_axis], nav_indexes=(0, 1))

    return raw_scan, all_headers, spectra_data

