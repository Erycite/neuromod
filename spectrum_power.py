import neo
import pickle
import numpy as np
import matplotlib
import scipy.signal as signal
matplotlib.use('Agg')

import matplotlib.pyplot as plt

ne = pickle.load(open("py{'Populations.ext.cellparams.duration': 5000}.pkl", "rb"))
data = ne.segments[0]

dt = 0.1
Fs = 1/dt # sampling per millisecond !! => *1000 for seconds
sr = Fs*1000 # sample rate in s



def LFP(data):
    v = data.filter(name="v")[0]
    g = data.filter(name="gsyn_exc")[0]
    # We produce the current for each cell for this time interval, with the Ohm law:
    # I = g(V-E), where E is the equilibrium for exc, which usually is 0.0 (we can change it)
    # (and we also have to consider inhibitory condictances)
    i = g*(v) #AMPA
    # the LFP is the result of cells' currents
    avg_i_by_t = np.sum(i,axis=1)/i.shape[0] #
    sigma = 0.1 # [0.1, 0.01] # Dobiszewski_et_al2012.pdf
    lfp = (1/(4*np.pi*sigma)) * avg_i_by_t
    return lfp

def bandpower(data, sf, band, max_freq, method='welch', window_sec=None, relative=False):
    """Compute the average power of the signal x in a specific frequency band.
    from https://raphaelvallat.com/bandpower.html
    uses MNE-Python >= 0.14.
    Parameters
    ----------
    data : 1d-array
      Input signal in the time-domain.
    sf : float
      Sampling frequency of the data.
    band : list
      Lower and upper frequencies of the band of interest.
    method : string
      Periodogram method: 'welch' or 'multitaper'
    max_freq : int
      Maximum frequency considered. Usually 200 (Hz)
    window_sec : float
      Length of each window in seconds. Useful only if method == 'welch'.
      If None, window_sec = (1 / min(band)) * 2.
    relative : boolean
      If True, return the relative power (= divided by the total power of the signal).
      If False (default), return the absolute power.
    Return
    ------
    bp : float
      Absolute or relative band power.
    """
    from scipy.signal import welch
    from scipy.integrate import simps
    band = np.asarray(band)
    low, high = band
    # Compute the modified periodogram (Welch)
    if method == 'welch':
        if window_sec is not None:
            nperseg = window_sec * sf
        else:
            nperseg = (2 / low) * sf
        freqs, psd = welch(data, sf, nperseg=nperseg)
    if max_freq is not None:
        if max(freqs) > max_freq:
            freqs = freqs[np.where(freqs < max_freq)]
            psd = psd[np.where(freqs < max_freq)]
    # Frequency resolution
    freq_res = freqs[1] - freqs[0]
    # Find index of band in frequency vector
    idx_band = np.logical_and(freqs >= low, freqs <= high)
    # Integral approximation of the spectrum using parabola (Simpson's rule)
    bp = simps(psd[idx_band], dx=freq_res)
    if relative:
        bp /= simps(psd, dx=freq_res)
    return bp

# Get the LFP data
lfp = LFP(data)
lfp = lfp[5000:] # Supress the first 500ms to look after the initial kick

# Compute the power spectrum 'classical way', with 2sec temporal window and 1sec overlap
freq, P = signal.welch(lfp, sr, window='hamming', nperseg=20000, noverlap=10000)

# Select frequency bands to compute the power from
bands = [[0.5,2], [2,4], [4,12], [12,30], [30,80], [80,140]]

# Compute specific bands for relative power ratio
# The computed band power will be relative to the total power of the lfp in the range [0-150Hz]
delta_p = bandpower(lfp, int(sr), [2,4], max_freq=150., method='welch', window_sec=2, relative=True)
beta_p = bandpower(lfp, int(sr), [12,30], max_freq=150., method='welch', window_sec=2, relative=True)

delta_beta_ratio = delta_p/beta_p
txt_ratio = ('delta/beta ratio= \n' + str(round(delta_beta_ratio,3)))


xtick_pos = np.arange(len(bands))+0.5 # for plotting purpose
# Get the list of power according to the bands list we created before
powerlist = []
for band in bands:
    bp = bandpower(lfp, int(sr), band, max_freq=150., method='welch', window_sec=2, relative=True)
    powerlist.append(bp)

### Plotting the spectrogram and a barplot showing the relative band power
fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(7, 4))
# Fs = 1 / params['dt']  # sampling frequency
# plot different spectrum types:
axes[0].set_title("Log. Magnitude Spectrum")
axes[0].semilogx(freq, P, color = 'r')
axes[0].set_xlabel('Frequency (Hz)')
axes[0].set_ylabel('Power spectrum (µV**2)')
#axes[0].set_xlim(0.5,100)
axes[1].set_title("relative band power")
axes[1].bar(xtick_pos, powerlist,width=1, color = 'r')
axes[1].set_xticks(xtick_pos)
axes[1].set_xticklabels(bands, rotation = 45)
axes[1].text(2,0.4, txt_ratio)
fig.tight_layout()
fig.savefig('Spectrum_show.png')
fig.clear()
