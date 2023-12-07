import numpy as np
import sys
import os
import cdflib
import glob
import pandas as pd

from conversions import tt2000_to_date
from classify import is_dust_dummy as is_dust
from scipy import signal
from scipy import interpolate

from paths import cnn_flags_location
from paths import preprocessed_location


def pad(wf,where_to_start=None):
    """
    A function to pad the given array and 
    return an array of double the lenght. 

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

    if where_to_start is None or where_to_start>len(wf)//2:
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


def subsample(wf,
              samples=4096,
              style="interpolate",
              denoise=7):
    """
    A wrapper to do the right subsampling that fits our needs.

    Parameters
    ----------
    wf : np.array of float
        The signal to be subsampled.
    samples : int, optional
        The target number of samples. The default is 4096.
    style : str, optional
        The algorithm to be used to subsample the data.
        The valid options are: "interpolate", "fourier", and "decimate". 
        The default is "interpolate".
        They are comparably fast, but the "interpolate" maintains the noise
        level, while the other two supress the noise even without additional 
        filtering. Beware that the "fourier" method adds jittery 
        start and end, so it is advisable to cut a small margin if used. 
    denoise : int, optional
        Adds a median filter of this length before subsampling. The number, 
        if not 0, is rounded up to the next odd integer, not smaller than 3.
        The default is 7.

    Returns
    -------
    subsampled
        The subsampled signal.

    """
    if denoise:
        round_up_odd = lambda x : max([x-x%2+1,3])
        wf = signal.medfilt(wf, round_up_odd(denoise))
    else:
        pass

    if style == "fourier":
        subsampled = signal.resample(wf,samples)
    elif style == "decimate":
        q = len(wf)//samples
        subsampled = signal.decimate(wf,q,ftype="fir")
        if len(subsampled) != samples:
            subsampled = subsample(subsampled,samples=samples,
                                   style="interpolate",
                                   denoise=0)
    elif style == "interpolate":
        times_original = np.arange(0,1,1/len(wf))
        times_subsampled = np.arange(0,1,1/samples)
        interpolated = interpolate.interp1d(times_original, wf-np.median(wf),
                                            kind='linear')
        subsampled = interpolated(times_subsampled)
    else:
        raise Exception(f"unknown subsampling style: {style}")
    return subsampled


def preprocess(cdf,i,
               cached=False,
               cache_folder=preprocessed_location):
    """
    The purpose is to prepare ch1,ch2,ch3 in monopole equivalent and in the
    right format for the CNN classifier "is_dust". 

    Parameters
    ----------
    cdf_file : cdflib.cdfread.CDF
        The L2 TSWF-E file of interest.
    i : int
        The event index of interest.
    cached : bool, optional
        If True, then the processed ch1, ch2, ch3 are saved in the process.
        They are also looked for, and if found then the loaded cache
        is used and nothing is computed. 
        The default is False.
    cache_folder : str, optional
        The folder where to put the cached results. 
        The default is preprocessed_location from "paths.py". 

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

    if cached:
        cdf_file_path = str(cdf.file)
        file_no_extension = cdf_file_path[cdf_file_path.find("solo_L2"):-4]
        cache_name = file_no_extension+f"_{i:06}.csv"
        hits = glob.glob(os.path.join(cache_folder,cache_name))
        if len(hits):
            hit = hits[0]
        else:
            hit = None
    else:
        hit = None

    if cached and hit:
        data = pd.read_csv(hit)
        ch1 = np.array(data["ch1"])
        ch2 = np.array(data["ch2"])
        ch3 = np.array(data["ch3"])
    else:
        e               = cdf.varget("WAVEFORM_DATA_VOLTAGE")[i,:,:]
        sampling_rate   = cdf.varget("SAMPLING_RATE")[i]
        channel_ref     = cdf.varget("CHANNEL_REF")[i]
        sw              = cdf.attget("Software_version",entry=0).Data
        sw              = int(sw.replace(".",""))
    
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
        else:
            raise Exception("unexpected: neither SE1 nor XLD1")
    
        #pad the high frequency data
        if is_hi_sampling:
            # The seed is set as the date, file version,
            # and the index to ensure reproducibility.
            np.random.seed(int( str(cdf.file)[-16:-8] #date
                               +str(cdf.file)[-6:-4]  #file version
                               +str(i)))              #event index
            start = np.random.randint(len(ch1))
            ch1 = pad(ch1,start)
            ch2 = pad(ch2,start)
            ch3 = pad(ch3,start)
        elif is_lo_sampling:
            pass
        else:
            raise Exception("unexpected: neither hi_sampling nor lo_sampling")
    
        #remove offset
        ch1 -= np.median(ch1)
        ch2 -= np.median(ch2)
        ch3 -= np.median(ch3)
    
        #filter (medfilt) and subsample (compress)
        ch1 = subsample(ch1)
        ch2 = subsample(ch2)
        ch3 = subsample(ch3)
    
        #normalize
        norm = np.abs(np.vstack([ch1,ch2,ch3])).max()
        ch1 /= norm
        ch2 /= norm
        ch3 /= norm

    if cached and not hit:
        write_preprocessed_waveforms_file(ch1,ch2,ch3,cache_name)

    return ch1,ch2,ch3


def write_cnn_flags_file(events,
                         cnn_flags,
                         name):
    """
    The function that writes the CNN classified flags.

    Parameters
    ----------
    events : iterable
        The 1D array of he indices from the cdf file of interest.
    cnn_flags : itearble
        The ID array of CNN classification flags 
        for the cdf file of interest.
    name : str
        The name of the file to be written. Usually the name of the 
        underlying cdf file without the ".cdf".

    Returns
    -------
    filepath : str
        The location of the file that was just written.

    """
    filepath = os.path.join(name)

    with open(filepath,"w") as file:
        file.write("index,cnn_flag\n")
        [file.write(f"{i},{flag}\n") for i, flag in zip(events,cnn_flags)]

    return filepath


def write_preprocessed_waveforms_file(ch1,
                                      ch2,
                                      ch3,
                                      name,
                                      location=preprocessed_location,
                                      precision=np.half):
    """
    The function to write the chache .csv file 
    containting the preprocessed waveforms that 
    can be directly used by "is_dust" function 
    of "classify.py".

    Parameters
    ----------
    ch1 : np.ndarray
        1D array with the waveform data for ch1.
    ch2 : np.ndarray
        1D array with the waveform data for ch2.
    ch3 : np.ndarray
        1D array with the waveform data for ch3.
    name : str
        The name of the file to be written. Usually the name of the 
        underlying cdf file with the index.
    location : str, optional
        Path to the directorry where the file will be written. 
        The default is preprocessed_location from "paths.py".

    Returns
    -------
    filepath : str
        The location of the file that was just written.
    """

    filepath = os.path.join(location,name)

    ch1 = ch1.astype(precision)
    ch2 = ch2.astype(precision)
    ch3 = ch3.astype(precision)

    with open(filepath,"w") as file:
        file.write("ch1,ch2,ch3\n")
        for v1,v2,v3 in zip(ch1,ch2,ch3):
            file.write(f"{v1},{v2},{v3}\n")

    return filepath


def main(input_cdf,
         output_cnn_flags=None):
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

    if output_cnn_flags is None:
        output_cnn_flags = input_cdf[input_cdf.find("solo_L2"):-4]+".csv"

    print(input_cdf)
    print(output_cnn_flags)

    #input_cdf = "data\\rpw_tds_surv_tswf_e\\solo_L2_rpw-tds-surv-tswf-e_20230530_V01.cdf"

    cdf = cdflib.CDF(input_cdf)
    events_count = len(cdf.varget("Epoch"))
    events = np.arange(events_count)
    cnn_flags = np.zeros(events_count)
    for i in events:
        try:
            ch1, ch2, ch3 = preprocess(cdf,i,cached=True)
        except Exception as e:
            print(e)
            cnn_flags[i] = -1.
        else:
            cnn_flags[i] = is_dust(ch1,ch2,ch3)

    write_cnn_flags_file(events,cnn_flags,output_cnn_flags)


#%%
if __name__ == "__main__":
    main(sys.argv[1],sys.argv[2])    
