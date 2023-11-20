import numpy as np
import sys
from classify import is_dust_dummy


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

    ch1, ch2, ch3 = np.zeros(10),np.zeros(10),np.zeros(10)

    is_dust = is_dust_dummy(ch1,ch2,ch3)

    #save


#%%
if __name__ == "__main__":
    main(sys.argv[1],sys.argv[2])    
