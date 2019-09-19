import neo
import pickle
import numpy as np
import matplotlib
matplotlib.use('Agg')
import scipy.stats as stats
import scipy.signal as signal
import matplotlib.pyplot as plt

####
# Use: put the script in neuromod folder, go to the simulation_result folder, run: python ../netw_model_analysis.py
####
def pcklfiles(str):
    """
    Get the result (pkl) files present in the folder
    :param str: name of the neuron pop we got the results from
    :return:
    """
    import os
    lf = os.listdir()
    lf2=[lfI.find(".pkl") for lfI in lf]
    lpkl = []
    for lki, lk in enumerate(lf2):
       if lk != -1:
           lpkl.append(lf[lki])
    lp2 = [f.find(str)for f in lpkl]
    files = []
    for lki, lk in enumerate(lp2):
       if lk == 0:
           files.append(lpkl[lki])
    return(files)

def rate(rec_duration, spiketrains, bin_size=10 ):
    """
    Binned-time Firing firing rate
    """
    if spiketrains == [] :
        return NaN
    # create bin edges based on number of times and bin size
    bin_edges = np.arange( 0, rec_duration, bin_size )
    #print "bin_edges",bin_edges.shape
    # binning absolute time, and counting the number of spike times in each bin
    hist = np.zeros( bin_edges.shape[0]-1 )
    for spike_times in spiketrains:
        hist = hist + np.histogram( spike_times, bin_edges )[0]
    return hist / len(spiketrains)

def cc(rec_duration, spiketrains, bin_size=10 ):
    """
    Binned-time Cross-correlation
    """
    if spiketrains == [] :
        return NaN
    # create bin edges based on number of times and bin size
    # binning absolute time, and counting the number of spike times in each bin
    bin_edges = np.arange( 0, rec_duration, bin_size )
    #print "bin_edges",bin_edges.shape
    CC = []
    for n, spike_times_i in enumerate(spiketrains):
        for spike_times_j in spiketrains:
            itrain = np.histogram( spike_times_i, bin_edges )[0] # spike count of bin_size bins
            jtrain = np.histogram( spike_times_j, bin_edges )[0]
            CC.append( np.corrcoef(itrain, jtrain)[0,1] )
    return np.nanmean(CC)

def isi( spiketrains ):
    """
    Inter-Spike Intervals histogram for all spiketrains
    """
    if np.count_nonzero(np.array(spiketrains)) > 1:
        # print(spiketrains)
        return np.diff( spiketrains )
    else:
        return None

def cv( spiketrains ):
    """
    Coefficient of variation
    """
    if np.count_nonzero(np.array(spiketrains)) > 1:
        return np.std(isi(spiketrains)) / np.mean(isi(spiketrains))
    else:
        return None

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

def plot_Vm_distrib(py_data, pltinfo):
    vm = py_data.filter(name = 'v')[0]
    m_vm = np.mean(vm,1)
    # Vm histogram
    plt.subplot(pltinfo[0], 2, pltinfo[1]*2+1)
    n,bins,patches = plt.hist(m_vm,50)
    plt.title('histogram of neam vm across simulation_duration')
    # qqplot
    ks = stats.kstest(m_vm, 'norm', stats.norm.fit(m_vm))
    meanmvm = np.mean(m_vm)
    medmvm = np.median(m_vm)
    m_dif = meanmvm - medmvm
    if m_dif <0:
        kstxt = ('Left skewed distrib (mean-med) = ' + str(round(m_dif ,2))+'\nKSstat = ' + str(round(ks[0], 3)) + '\npval = ' + str(ks[1]))
    else:
        kstxt = ('Right skewed distrib (mean-med) = ' + str(round(m_dif ,2))+'\nKSstat = ' + str(round(ks[0], 3)) + '\npval = ' + str(ks[1]))
    #kstxt = ('KSstat = '+ str(round(ks[0],3)) + '\npval = ' + str(ks[1]))
    plt.subplot(pltinfo[0], 2, pltinfo[1]*2+2)
    stats.probplot(m_vm, dist="norm", plot=plt)
    plt.title('qqplot')
    plt.text(-4, -60, kstxt)

def plot_spectrums(py_data, sr, bands, pltinfo):
    # Get the LFP data
    lfp = LFP(py_data)
    lfp = lfp[5000:] # Supress the first 500ms to look after the initial kick
    # Compute the power spectrum 'classical way', with 2sec temporal window and 1sec overlap
    freq, P = signal.welch(lfp, sr, window='hamming', nperseg=20000, noverlap=10000)
    # The computed band power will be relative to the total power of the lfp in the range [0-150Hz]
    delta_p = bandpower(lfp, int(sr), [2,4], max_freq=150., method='welch', window_sec=2, relative=True)
    beta_p = bandpower(lfp, int(sr), [12,30], max_freq=150., method='welch', window_sec=2, relative=True)
    tetha_p = bandpower(lfp, int(sr), [4,12], max_freq=150., method='welch', window_sec=2, relative=True)
    delta_beta_ratio = delta_p/beta_p
    tetha_delta_ratio = tetha_p/delta_p
    txt_ratio = ('delta/beta ratio= ' + str(round(delta_beta_ratio,3)) + '\ntetha/delta ratio = ' + str(round(tetha_delta_ratio,3)))
    xtick_pos = np.arange(len(bands))+0.5 # for plotting purpose
    # Get the list of power according to the bands list we created before
    powerlist = []
    for band in bands:
        bp = bandpower(lfp, int(sr), band, max_freq=150., method='welch', window_sec=2, relative=True)
        powerlist.append(bp)
    ### Plotting the spectrogram and a barplot showing the relative band power
    # plot different spectrum types:
    plt.subplot(pltinfo[0], 2, pltinfo[1]*2 + 1)
    plt.title("Log. Magnitude Spectrum")
    plt.semilogx(freq, P, color = 'r')
    plt.xlim(0.5,150)
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('Power spectrum (µV**2)')
    # plot power bands
    plt.subplot(pltinfo[0], 2, pltinfo[1]*2 + 2)
    plt.title("relative band power")
    plt.bar(xtick_pos, powerlist,width=1, color = 'r')
    plt.xticks(xtick_pos, bands, rotation=45)
    plt.text(2,0.3, txt_ratio)

def plot_lfp_and_frate(py_data, inh_data, time, thresh, binsize, pltinfo):
    data_py = py_data.segments[0]
    data_inh = inh_data.segments[0]
    # plot lfp
    lfp = LFP(py_data)
    cc_f100_py = cc(rec_duration, data_py.spiketrains[:100], bin_size=10)
    #cv_f100_py = cv(data_py.spiketrains[100]) ######################################################" np.nanmean(CC)
    Cv_py = [cv(data_py.spiketrains[i]) for i in range(100)]
    for i in range(sum([Cv_py[i] == None for i in range(len(Cv_py))])):
        Cv_py.remove(None)
    Cv_py = np.mean(Cv_py)
    cc_f100_inh = cc(rec_duration, data_inh.spiketrains[:100], bin_size=10)
    Cv_inh = [cv(data_inh.spiketrains[i]) for i in range(100)]
    for i in range(sum([Cv_inh[i] == None for i in range(len(Cv_inh))])):
        Cv_inh.remove(None)
    Cv_inh = np.mean(Cv_inh)
    coeftxt = ('100 pyra neurons: \nCc = ' + str(round(cc_f100_py,5))+'\nCv = ' + str(round(Cv_py, 5))+ '\n100 pyra neurons: \nCc = ' + str(round(cc_f100_inh,5))+'\nCv = ' + str(round(Cv_inh, 5)))
    plt.subplot(pltinfo[0], 2, pltinfo[1] * 2 + 1)
    plt.plot(time, lfp)
    plt.text(100,-0.008, coeftxt)
    plt.title('lfp signal')
    plt.xlabel('time (ms)')
    plt.ylabel('v')
    # compute firing rate of the pop
    fr_py = rate(rec_duration, data_py.spiketrains, bin_size=binsize)
    fr_inh = rate(rec_duration, data_inh.spiketrains, bin_size=binsize)
    # portion_up_py = sum(fr_py >= thresh) / len(fr_py)
    portion_down_py = sum(fr_py < thresh) / len(fr_py)
    # portion_up_inh = sum(fr_inh >= thresh) / len(fr_inh)
    portion_down_inh = sum(fr_inh < thresh) / len(fr_inh)
    txt = ('pyra down state = ' +str(round(portion_down_py,3)*100) + '%  \ninhib down state = ' + str(round(portion_down_inh,3)*100) + ' %')
    # plotting firing rate and stuffs
    plt.subplot(pltinfo[0], 2, pltinfo[1]*2 + 2)
    plt.plot(fr_py, 'b')
    plt.plot(fr_inh,'r')
    plt.title('firing thingy')
    plt.ylim([.0,1.])
    plt.ylabel('proportion of cell firing')
    plt.xlabel('time bins with bin size = '+str(binsize) + '(ms)')
    plt.legend(['py_neurons','inh_neurons'])
    plt.hlines(thresh, 0, len(fr_py), color = 'k', linestyle='--')
    plt.text(2,0.8, txt)


#########################################
# General setup for the analysis profile:
rec_duration = 5000.#ms
dt = 0.1
Fs = 1/dt # sampling per millisecond !! => *1000 for seconds
sr = Fs*1000 # sample rate in s
time = np.arange(0,rec_duration+dt, dt)
# Select frequency bands to compute the power from
bands = [[0.5,2], [2,4], [4,12], [12,30], [30,80], [80,140]]
# select thresh
thresh = 0.01  # less than 1% of cells firing in the specified time bin
# select bin (time) size
binsize = 10
#########################################

analysis = ['firing_rate', 'Spectrums', 'Vm_distrib'] #options to be plotted on the profile plot

# look at the different result files (1 py+1 inh / simulation) in the folder
pyra = pcklfiles('py')
inh = pcklfiles('inh')
#### Get the data ####
for r_file in range(len(pyra)):
    py_data = pickle.load(open(pyra[r_file], "rb"))
    inh_data = pickle.load(open(inh[r_file], "rb"))
    suptitle = pyra[r_file][pyra[r_file].find('Populations'): pyra[r_file].find('}.pkl')]
    # plt the profile picture of the simulation (data loaded)
    plt.subplots(len(analysis), 2 , figsize=(14,12))
    plt.suptitle(suptitle)
    for n_analysis, analy in enumerate(analysis):
        pltinfo = [len(analysis), n_analysis]
        if analy == 'firing_rate':
            plot_lfp_and_frate(py_data, inh_data, time, thresh, binsize, pltinfo)
        elif analy == 'Spectrums':
            plot_spectrums(py_data, sr, bands, pltinfo)
        elif analy == 'Vm_distrib':
            plot_Vm_distrib(py_data, pltinfo)
    plt.tight_layout()
    plt.savefig('profiles_'+str(r_file)+'.png')






