from pymodaq_utils.enums import BaseEnum
from datetime import datetime

class ExperimentType(BaseEnum):
    """Enum for experiment types."""
    PLscan = 0
    IsoB = 1
    FullB = 2
    FastFullB = 3
    Multiple = 4
    Other = 5

    @classmethod
    def qzabre_name(cls, exp_type):
        if exp_type is cls.PLscan:
            return 'quench'
        elif exp_type is cls.IsoB:
            return 'isob'
        elif exp_type is cls.FullB:
            return 'odmr_hardware'
        elif exp_type is cls.FastFullB:
            return 'odmr_quick'
        elif exp_type is cls.Multiple:
            return 'multiple'
        else:
            return ''
        
        
def change_timestamp(data, new_timestamp):
    """
    Changes the timestamp of a DataWithAxis, to use the date of the measurement
    rather than the date of the data reading.

    Parameters
    ----------
    data: DataWithAxis (or a child class)

    new_timestamp: str, specifying the measurement time, either Qnami or Qzabre style.
    """
    try:
        new_date = datetime.strptime(new_timestamp, '%Y%m%d-%H%M-%S') # Qnami style
    except:
        try:
            new_date = datetime.strptime(new_timestamp, '%Y-%m-%d-%H-%M-%S') # Qzabre style
        except:
            print('Unknown timestamp format!')
            return
    data._timestamp = new_date.timestamp()
    return 
    
