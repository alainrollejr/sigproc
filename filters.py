
# Author: ARO

import numpy as np
import matplotlib.pyplot as plt

__all__=['rrcosfilter']



def rrcosfilter(L, alpha, R, Fs):
    """
    Generates a root raised cosine (RRC) filter (FIR) impulse response.

    Parameters
    ----------
    L : int
        Length of the filter in symbols.

    alpha : float
        Roll off factor (Valid values are [0, 1]).

    R : float
        Symbol rate

    Fs : float
        Sampling Rate in Hz.

    Returns
    ---------

    h_rrc : 1-D ndarray of floats
        Impulse response of the root raised cosine filter.

    time_idx : 1-D ndarray of floats
        Array containing the time indices, in seconds, for
        the impulse response.
    """
    N = L*int(np.round(float(Fs)/float(R)))
    T = 1.0/float(R)
    T_delta = 1/float(Fs)
    time_idx = ((np.arange(N)-N/2))*T_delta
    sample_num = np.arange(N)
    h_rrc = np.zeros(N, dtype=float)

    for x in sample_num:
        t = (x-N/2)*T_delta
        if t == 0.0:
            h_rrc[x] = 1.0 - alpha + (4*alpha/np.pi)
        elif alpha != 0 and t == T/(4*alpha):
            h_rrc[x] = (alpha/np.sqrt(2))*(((1+2/np.pi)* \
                    (np.sin(np.pi/(4*alpha)))) + ((1-2/np.pi)*(np.cos(np.pi/(4*alpha)))))
        elif alpha != 0 and t == -T/(4*alpha):
            h_rrc[x] = (alpha/np.sqrt(2))*(((1+2/np.pi)* \
                    (np.sin(np.pi/(4*alpha)))) + ((1-2/np.pi)*(np.cos(np.pi/(4*alpha)))))
        else:
            h_rrc[x] = (np.sin(np.pi*t*(1-alpha)/T) +  \
                    4*alpha*(t/T)*np.cos(np.pi*t*(1+alpha)/T))/ \
                    (np.pi*t*(1-(4*alpha*t/T)*(4*alpha*t/T))/T)

    return time_idx, h_rrc

def main(argv):
    time_idx, h_rrc = rrcosfilter(32, 0.05, 1e6, 16e6)
    
    plt.plot(time_idx, h_rrc,'.-')
    plt.grid()
    
    
if __name__ == "__main__":
    main(sys.argv)
