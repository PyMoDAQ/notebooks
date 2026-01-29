import numpy as np
from scipy.optimize import leastsq
import pymodaq_data.data as pdd

def subtract_average(data, mean_to_zero=True, direction='row'):
    """
    Data correction function. Substract to the values of each row or columns
    their mean value.
    
    Parameters
    ----------
    data: np.ndarray
        The 2D-ndarray to correct
    mean_to_zero: bool
        If True, after use of this function, the mean value of each row (resp. col)
        is zero. If False, the original mean value of the image is set again at the end
    direction: str
        'row' or 'col', default 'row'
    
    Returns
    -------
    corr_data: np.ndarray
        Corrected data. The array is duplicated, so the modif is not done in place.
    """
    m = np.nanmean(data)
    
    if direction == 'row':
        corr_data = np.copy(data) - np.nanmean(data, axis=1, keepdims=True)
    elif direction == 'col':
        corr_data = np.copy(data) - np.nanmean(data, axis=0, keepdims=True)
    else:
        print('Unknown scan direction !')
        return

    if not mean_to_zero:
        corr_data = corr_data - np.mean(corr_data) + m
        
    return corr_data

def median_diff(data, direction='row'):
    """
    Data correction function. Substract to the value of each row or column the median of
    differences between their neighboring rows or columns.
    
    
    Parameters
    ----------
    data: np.ndarray
        The 2D-ndarray to correct
   
    direction: str
        'row' or 'col', default 'row'
    
    Returns
    -------
    corr_data: np.ndarray
        Corrected data. The array is duplicated, so the modif is not done in place.
    """
    corr_data = np.copy(data)
    
    if direction == 'row':
        for i in range(1, np.size(data, axis=0)):
            med_diff = np.nanmedian(data[i, :]-data[i-1, :])
            corr_data[i, :] = corr_data[i, :] - med_diff
    elif direction == 'col':
        for i in range(1, np.size(data, axis=1)):
            med_diff = np.nanmedian(data[:, i]-data[:, i-1])
            corr_data[:, i] = corr_data[:, i] - med_diff
    else:
        print('Unknown scan direction !')
        return
    
    return corr_data

def median(data, direction='row'):
    """
    Data correction function. Substract to each row or column the value of their median.
    
    Parameters
    ----------
    data: np.ndarray
        The 2D-ndarray to correct
   
    direction: str
        'row' or 'col', default 'row'
    
    Returns
    -------
    corr_data: np.ndarray
        Corrected data. The array is duplicated, so the modif is not done in place.
    """
    corr_data = np.copy(data)
    
    if direction == 'row':
        for i in range(0, np.size(data, axis=0)):
            med = np.nanmedian(data[i, :])
            corr_data[i, :] = corr_data[i, :] - med
    elif direction == 'col':
        for i in range(0, np.size(data, axis=1)):
            med = np.nanmedian(data[:, i])
            corr_data[:, i] = corr_data[:, i] - med
    else:
        print('Unknown scan direction !')
        return
    
    return corr_data

def median_div(data, direction='row'):
    """
    Data correction function. Divide each row or column by the value of their median.
    Parameters
    ----------
    data: np.ndarray
        The 2D-ndarray to correct
   
    direction: str
        'row' or 'col', default 'row'
    
    Returns
    -------
    corr_data: np.ndarray
        Corrected data. The array is duplicated, so the modif is not done in place.
    """
    corr_data = np.copy(data)
    
    if direction == 'row':
        for i in range(0, np.size(data, axis=0)):
            med = np.nanmedian(data[i, :])
            corr_data[i, :] = corr_data[i, :]/med
    elif direction == 'col':
        for i in range(0, np.size(data, axis=1)):
            med = np.nanmedian(data[:, i])
            corr_data[:, i] = corr_data[:, i]/med
    else:
        print('Unknown scan direction !')
        return
    
    return corr_data

# Equation of a plane from 2 array of coordinates x,y
def plane(x, y, a, b, c):
    return a*x+b*y+c

# Compute the difference between a 2D array zz and the plane
# with parameters a, b, c in params
def errorfunc_plane(params, *args):
    zz = args[0]
    xx = args[1]
    yy = args[2]
    out = zz-plane(xx, yy, *params)
    return out.flatten()

def plane_fit(array):
    """
    Data correction function. Fits a plane to the data and subtracts it.
   
    Parameters
    ----------
    data: np.ndarray
        The 2D-ndarray to correct
    
    Returns
    -------
    corr_data: np.ndarray
        Corrected data. The array is duplicated, so the modif is not done in place.
    """
    # pb with incomplete pictures,
    # we need to fit only the part containing data.
    clean = array[~np.isnan(array).any(axis=1), :]
    
    N = np.size(clean, axis=0)
    M = np.size(clean, axis=1)
    xx, yy = np.meshgrid(np.arange(M), np.arange(N))
    result = leastsq(errorfunc_plane, [1, 1, 1], args=(clean, xx, yy))
    pfit = result[0]
    zz = clean-plane(xx, yy, *pfit)

    # back to initial shape
    corr_data = np.full(array.shape, np.nan)
    
    # start missing
    if np.isnan(array).any(axis=1)[0]:
        i = 0
        while np.isnan(array).any(axis=1)[i]:
            i = i+1
        corr_data[i:, :] = zz
    # end missing
    elif np.isnan(array).any(axis=1)[-1]:
        n = np.size(array, axis=0)-1
        i = n
        while np.isnan(array).any(axis=1)[i]:
            i = i-1
        corr_data[:i+1, :] = zz
    else:
        corr_data = zz
        
    return corr_data

def three_pts_plane(x0, y0, x1, y1, x2, y2, data, n=15):
    """
    Data correction function. Subtracts the plane defined by the mean values
    around the 3 points indicated.
    
    Parameters:
    ----
    x0, y0: int, int
        Coordinates (in px) of the 1st point.
    x1, y1: int, int
        Coordinates (in px) of the 2nd point.
    x2, y2: int, int
        Coordinates (in px) of the 3rd point.
    data: np.ndarray
        The 2D-ndarray to correct
    n: int
        Radius of the averaging zone around the points
    
    Returns
    -------
    corr_data: np.ndarray
        Corrected data. The array is duplicated, so the modif is not done in place.
    """
    z0 = np.mean(data[y0-n:y0+n, x0-n:x0+n])
    z1 = np.mean(data[y1-n:y1+n, x1-n:x1+n])
    z2 = np.mean(data[y2-n:y2+n, x2-n:x2+n])
    
    u = np.array([x1-x0, y1-y0, z1-z0])
    v = np.array([x2-x0, y2-y0, z2-z0])
    w = np.cross(u, v)

    z = np.zeros(np.shape(data))
    yy, xx = np.meshgrid(range(np.size(z, axis=1)), range(np.size(z, axis=0)))
    z = z0-(w[1]*(xx-x0)+w[0]*(yy-y0))/w[2]
    corr_data = np.copy(data) - z

    return corr_data

def crop_data_raw(data_raw: pdd.DataRaw, coordinates, nav_axis = True):
    """
    The PyMoDAQ crop_data_raw function doesn't seem to be working so here is a working one
    
    Parameters
    ----------
    data_raw: DataRaw Object to crop
    coordinates: list of tuples as [(axis 1 start, axis 1 stop), (axis 2 start, axis 2 stop)]
    nav_axis: boolean, True if we crop along a  navigation axes
    
    Returns
    -------
    DataRaw object with cropped data and axes
    """
    slices = []
    for axis_index, coord in enumerate(coordinates):
        axis = data_raw.get_axis_from_index(axis_index)[0]
        indexes = axis.find_indexes(coord)
        indexes[-1] += 1 # Python convention, the slice index must be one above the last index to include
        slices.append(slice(*indexes))
    if not nav_axis:
        slices[0], slices[2] = slices[2], slices[0]

    return data_raw._slicer(tuple(slices), is_navigation = nav_axis)


def remove_line_with_nan(raw_fullb_data: pdd.DataRaw, scan_axis=0):
    """
    Remove lines containing lines from 2D data
    
    Parameters
    ----------
    raw_fullb_data: FullB spectra as a DataRaw object
    scan_axis: axis that will be modified
    Returns
    -------
    DataRaw object of the FullB without the incomplete lines
    """
    if raw_fullb_data.data[0].ndim == 3:
        data_array_2d = np.mean(raw_fullb_data.data[0], axis=2)
    else:
        data_array_2d = raw_fullb_data.data[0].T
    mask = np.min(data_array_2d, axis=scan_axis) != 0

    new_axis = raw_fullb_data.axes[scan_axis].get_data()[mask]
    coords = [1e5, 1e5]
    coords[scan_axis] = (new_axis[0], new_axis[-1])
    if raw_fullb_data.data[0].ndim == 2:
        coords.reverse()
    return crop_data_raw(raw_fullb_data, coords[::-1])
