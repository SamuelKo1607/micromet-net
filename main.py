import numpy as np
import sys
import cdflib

from conversions import tt2000_to_date
from classify import is_dust_dummy
from scipy import signal


def pad(wf,where_to_start=None):
    """
    A function to pad the given array and return an array of double the lenght. 

    Parameters
    ----------
    wf : array of number
        An array to be padded.
    where_to_start : int or None, optional
        There in the array of double length will the original data start. 
        The default is None, in which case it starts in the middle.

    Returns
    -------
    wf_padded : array of number
        The padded array.
    """

    if where_to_start is None:
        where_to_start = len(wf)//2
    
    #noise background
    centered = wf-np.mean(wf)
    usual = np.sort(centered)[int(len(centered)*0.1):int(len(centered)//1.1)]
    wf_padded = np.random.normal(np.mean(wf),np.std(usual),2*len(wf))
    
    #softened signal
    softening_mask = np.hstack((np.arange(0,1,0.1),
                                np.ones(len(wf)-20),
                                np.arange(0.9,-0.1,-0.1)))
    wf_soft = wf*softening_mask
    
    #inverse softening mask to cut out a part of the noise
    mask = range(where_to_start,where_to_start+len(wf))
    wf_padded[mask] *= 1-softening_mask
    
    #sum
    wf_padded[mask] += wf_soft
    
    return wf_padded


def subsample(wf,samples=4096):
    """
    A wrapper to do the right subsampling that fits our needs.

    Parameters
    ----------
    wf : np.array of float
        The signal to be subsampled.
    samples : int, optional
        The target number of samples. The default is 4096.

    Returns
    -------
    resampled
        The subsampled signal.

    """
    resampled = signal.resample(wf,samples)
    return resampled


def preprocess(cdf,i):
    """
    The purpose is to prepare ch1,ch2,ch3 in monopole equivalent and in the
    right format for the CNN classifier "is_dust". 

    Parameters
    ----------
    cdf_file : cdflib.cdfread.CDF
        The L2 TSWF-E file of interest.
    i : int
        The event index of interest.

    Returns
    -------
    ch1 : numpy.ndarray
        A 1D array of len = 4096, containing the waveform for channel 1
        in such format that it can be classified with CNN classfier "is_dust".
    ch2 : numpy.ndarray
        A 1D array of len = 4096, containing the waveform for channel 2
        in such format that it can be classified with CNN classfier "is_dust".
    ch3 : numpy.ndarray
        A 1D array of len = 4096, containing the waveform for channel 3
        in such format that it can be classified with CNN classfier "is_dust".

    """
    e               = cdf.varget("WAVEFORM_DATA_VOLTAGE")[i,:,:]
    sampling_rate   = cdf.varget("SAMPLING_RATE")[i]
    epoch           = cdf.varget("EPOCH")[i]
    date_time       = tt2000_to_date(epoch)
    epoch_offset    = cdf.varget("EPOCH_OFFSET")[i]
    channel_ref     = cdf.varget("CHANNEL_REF")[i]
    sw              = cdf.attget("Software_version",entry=0).Data
    sw              = int(sw.replace(".",""))
    time_step       = epoch_offset[1]-epoch_offset[0]

    is_lo_sampling = np.isclose(sampling_rate, 262137.5, rtol=1e-03)
    is_hi_sampling = np.isclose(sampling_rate, 524275.0, rtol=1e-03)

    is_xld1 = min(channel_ref == [13, 21, 20])
    is_se1  = min(channel_ref == [10, 20, 30])


    fail_reasons = []
    if sw<211:
        fail_reasons.append("sw < 211")
    if not is_lo_sampling+is_hi_sampling:
        fail_reasons.append("unknown sampling")
    if not is_xld1+is_se1:
        fail_reasons.append("neither SE1 nor XLD1")
    if len(fail_reasons):
        raise Exception(fail_reasons)


    if is_xld1:
        ch1 = e[2,:]-e[1,:]
        ch2 = e[2,:]
        ch3 = e[2,:]-e[1,:]-e[0,:]
    elif is_se1:
        ch1 = e[0,:]
        ch2 = e[1,:]
        ch3 = e[2,:]



    # TBD treatment



    #subsample (compress)
    ch1 = subsample(ch1)
    ch2 = subsample(ch2)
    ch3 = subsample(ch3)

    #normalize
    norm = np.abs(np.vstack([ch1,ch2,ch3])).max()
    ch1 /= norm
    ch2 /= norm
    ch3 /= norm

    return ch1,ch2,ch3



def main(input_cdf,
         output_cnn_flags):
    """
    The main wrapper funcrtion to load a cdf, preporcess, analyze and 
    save the resulting CNN flag and other mtedadata in a new file.

    Parameters
    ----------
    input_cdf : str
        The L2 TSWF-E file to analyze.
    output_flags_file : str
        The output file with the CNN flag..

    Returns
    -------
    None.

    """
    print(input_cdf)
    print(output_cnn_flags)

    #input_cdf = "data\\rpw_tds_surv_tswf_e\\solo_L2_rpw-tds-surv-tswf-e_20230530_V01.cdf"

    cdf = cdflib.CDF(input_cdf)
    events_count = len(cdf.varget("Epoch"))
    for i in range(events_count):
        try:
            ch1, ch2, ch3 = preprocess(cdf,i)
        except Exception as e:
            print(e)
            cnn_flag = np.nan
        else:
            cnn_flag = is_dust_dummy(ch1,ch2,ch3)

        print(cnn_flag)

    #save


#%%
if __name__ == "__main__":
    main(sys.argv[1],sys.argv[2])    
