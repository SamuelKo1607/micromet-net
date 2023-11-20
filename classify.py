import numpy as np




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

    prob = np.random.rand()

    return prob



