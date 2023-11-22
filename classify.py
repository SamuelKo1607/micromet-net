import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt
from scipy.signal import medfilt
from scipy.interpolate import interp1d


# %%
def is_dust_dummy(ch1,ch2,ch3):
    """
    This is a dumy function that imitates the actual "is_dust" function. 
    The working of is_dust function is that it gets three arrays representing
    the three TSWF-E channels and gives a verdict in the form of float between
    0 and 1, 1 meaning it is a certain dust impact.

    Parameters
    ----------
    ch1 : np.array of float
        Monopole 1, compressed, normalized, resampled.
    ch2 : np.array of float
        Monopole 2, compressed, normalized, resampled.
    ch3 : np.array of float
        Monopole 3, compressed, normalized, resampled.

    Returns
    -------
    prob : flaot
        The certainty that the given waveforms represent a dust impact.
        the convention is that 0 means certain no-dust and 1 means 
        a certain dust.

    """

    prob = np.random.random()

    return prob


# %%
def so_preproc(data_raw, data_length, compression, sampling_rate):
    """
    Process raw data to the format expected by the dust classifier.

    Parameters:
    data_raw (array): Raw antenna voltage [1xN] samples.
    data_length (scalar): Length (N) of data_raw.
    compression (scalar): Compression factor used for CNN (e.g., 8).
    sampling_rate (scalar): Instrument sampling rate (e.g., 2.6213750e+05).

    Returns:
    data_processed (array): Processed voltage samples [1x(N/compression)].
    times_processed (array): Processed time steps [1x(N/compression)].
    """

    # 1 - Determine raw and processed time steps
    times_raw = np.arange(0, data_length) / sampling_rate
    times_processed = np.linspace(times_raw[0], times_raw[-1], round(data_length / compression))

    # 2 - Heavy median filtering of the data for bias removal
    F = 21
    data_filtered = medfilt(data_raw, kernel_size=F)

    # 3 - Determine the median of the data (the potential bias)
    data_median = np.median(data_filtered)

    # 4 - Step 1: Remove the potential bias
    data_nobias = data_raw - data_median

    # 5 - Step 2: Filter the data using a median filter of length 7
    data_filtered = medfilt(data_nobias, kernel_size=7)

    # 6 - Step 3: Compress the signal with factor: compression
    interpolate_function = interp1d(times_raw, data_filtered, kind='linear')
    data_compressed = interpolate_function(times_processed)

    # 7 - Step 4: Normalize the data with respect to the maximum absolute value
    mx = np.max(np.abs(data_compressed))
    data_processed = data_compressed / mx

    return data_processed, mx, times_processed


#%%
if __name__ == "__main__":

    # This block of code is ran if the script is called as a main process,
    # but not if a function is imported from the script.


    # %%
    
    # Read the content of the file
    file_path = '/Users/akv020/Projects/micromet-net/data/preprocessed/20211025_0463.csv'

    #
    # i accidantelly tried running the script but I couldnt because
    # of the absolute path. I suggest the following line should work
    # as well and independent of the computer
    #
    # file_path = 'data/preprocessed/20211025_0463.csv'


    # Reading the file into a DataFrame
    data = pd.read_csv(file_path, header=None, names=['ch1', 'ch2', 'ch3'])
    
    # Dropping empty rows (if any)
    data.dropna(inplace=True)
    
    # Extracting each column into separate variables
    ch1 = data['ch1'].tolist()
    ch2 = data['ch2'].tolist()
    ch3 = data['ch3'].tolist()
    
    
    # %%
    data_length = 16384
    compression = 4
    sampling_rate = 2.6213750e+05
    d1, mx1, times_processed = so_preproc(ch1,data_length,compression,sampling_rate)
    d2, mx2, times_processed = so_preproc(ch2,data_length,compression,sampling_rate)
    d3, mx3, times_processed = so_preproc(ch3,data_length,compression,sampling_rate)
    maxi = max(mx1,mx2,mx3)
    
    #%%
    TD1 = (d1*mx1)/maxi
    TD2 = (d2*mx2)/maxi
    TD3 = (d3*mx3)/maxi
    signals = np.column_stack((TD1, TD2, TD3))
    
    # %%
    # Creating a line plot for ch1, ch2, and ch3
    plt.figure(figsize=(12, 6))
    plt.plot(times_processed*1000,signals[:,0], label='ch1')
    plt.plot(times_processed*1000,signals[:,1], label='ch2')
    plt.plot(times_processed*1000,signals[:,2], label='ch3')
    # Adding labels and title
    plt.xlabel('time [ms]')
    plt.ylabel('Value')
    plt.title('Line Plot of ch1, ch2, and ch3')
    plt.legend()
    
    # Display the plot
    plt.show()
    
    
    
    
    
    
    
