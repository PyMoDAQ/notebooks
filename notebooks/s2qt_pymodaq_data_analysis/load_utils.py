import numpy as np
import s2qt_pymodaq_data_analysis.utils as ut
import s2qt_pymodaq_data_analysis.fit_utils as fut
import pymodaq_data.data as pdd

def fit_and_format_odmr(axis, data_list, fit, fitmodel):
    
    label_list = ['raw PL']
    fit_result = None

    if fit:
        fit_func = fut.available_functions[fitmodel]
        fit_result = fit_func(axis.get_data(), data_list[0], 10e6)
        try:
            fit_func = fut.available_functions[fitmodel]
            fit_result = fit_func(axis.get_data(), data_list[0], 10e6)
        except Exception as e:
            print('Unknown fit function! No fit will be provided.')
            print(e)

    if fit_result is not None:
        data_list.append(fit_result.best_fit)
        label_list.append('Fit')
        data_list.append(fit_result.init_fit)
        label_list.append('Initial guess')
    
    odmr_scan = pdd.DataRaw('ODMR scan', dim=pdd.DataDim['Data1D'],
                            data=data_list, labels=label_list,
                            units='counts/s', axes=[axis])
    
    return odmr_scan, fit_result
