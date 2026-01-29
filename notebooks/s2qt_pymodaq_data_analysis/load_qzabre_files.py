import glob
import numpy as np
import h5py
import json

import pymodaq_data.data as pdd
from pymodaq_data import Q_, ureg, Unit

import s2qt_pymodaq_data_analysis.utils as ut
import s2qt_pymodaq_data_analysis.fit_utils as fut
import s2qt_pymodaq_data_analysis.load_utils as lut

def load_metadata(path):
    """
    Get the header part for a Qnami data file.

    Parameters
    ----------
    path: str
        Path to the folder with the file.

    Returns
    -------
    metadata: dict
        
    """
    metadata_path = path+"imageMeta.json"
    with open(metadata_path, 'r') as f:
        metadata = json.load(f)
    return metadata



def load_qzabre_odmr(path, timestamp, odmr_type='odmr_hardware', 
                     fit=False, fitmodel='single_lorentzian'):
    """Load a raw ODMR scan measured with the QSM from Qzabre.

    Parameters
    ----------
    path: str
        Path to the folder containing all the raw data files
    timestamp: str, "yyyy-mm-dd-hh-mm-ss"
        Unique timestamp of the file
    odmr_type: str
        Type of odmr measurement ('hardware_odmr' for example)
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
    data_path = f"{path}{timestamp}-{odmr_type}/"  
    plfile = h5py.File(data_path+"seq0/integratedCounts.h5", "r")
    freqs = h5py.File(data_path+"seq0/eval/odmr.h5", "r")

    freq = np.squeeze(freqs['data']['frequency'])
    pl = np.squeeze(plfile['data']["counts"])*1e-3
    axis = pdd.Axis('freq_axis', units='Hz', data=freq, index=0)
    data_list = [pl]
    
    odmr_scan, fit_result = lut.fit_and_format_odmr(axis, data_list, fit, fitmodel)
    ut.change_timestamp(odmr_scan, timestamp)
    
    return odmr_scan, fit_result



def load_qzabre_scan(path, timestamp, experiment_type):
    """ Load a raw scan measured with the QSM from Qzabre.

    Parameters
    ----------
    path: str
        Path to the folder containing all the raw data
    timestamp: str
        Timestamp identifier of the file
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
    data_path = f'{path}/{timestamp}-{ut.ExperimentType.qzabre_name(experiment_type)}/'
    all_headers = load_metadata(data_path)
    h5file = h5py.File(data_path+"scalarData.h5", "r")
    dwa_dict = {}
    spectra_data = None
    
    x_line = np.linspace(all_headers['rect']['center'][0] - 0.5*all_headers['rect']['size'][0],
                         all_headers['rect']['center'][0] + 0.5*all_headers['rect']['size'][0],
                         all_headers['resolution'][0])
    x_axis = pdd.Axis('x', data=1e6*x_line, index=1, units='µm')

    y_line = np.linspace(all_headers['rect']['center'][1] - 0.5*all_headers['rect']['size'][1],
                         all_headers['rect']['center'][1] + 0.5*all_headers['rect']['size'][1],
                         all_headers['resolution'][1])
    y_axis = pdd.Axis('x', data=1e6*y_line, index=0, units='µm')


    for ch in all_headers['units'].keys():
        data2D = h5file['data'][ch]
        dwa_dict[ch] = pdd.DataRaw(ch, data=data2D, unit=all_headers['units'][ch],
                                   axes=[x_axis, y_axis], labels=[ch])
        ut.change_timestamp(dwa_dict[ch], timestamp)

        
    raw_scan = pdd.DataToExport(name=timestamp, 
                                data=[dwa_dict[ch] for ch in all_headers['units'].keys()])

    if experiment_type == ut.ExperimentType['FullB']:
        odmrfile = h5py.File(data_path+"seq0/eval/odmr.h5", "r")
        freq_axis = pdd.Axis('Frequency', units='Hz', index=2, 
                             data=odmrfile['data']['frequency'][0,0])

        spectra_data = pdd.DataRaw(name='Spectra', units='counts/s',
                                   data=[odmrfile['data']['sig']], 
                                   axes=[x_axis, y_axis, freq_axis],
                                   nav_indexes = (0,1),
                                   labels=["raw"])
        ut.change_timestamp(spectra_data, timestamp)
        
    return raw_scan, all_headers, spectra_data    
        
