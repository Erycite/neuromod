import neo
import pickle
import numpy as np
import matplotlib
matplotlib.use('Agg')

import matplotlib.pyplot as plt

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

rec_duration = 5000.#ms
py_data = pickle.load(open("py{'Populations.ext.cellparams.duration': 5000}.pkl", "rb"))
data_py = py_data.segments[0]
inh_data = pickle.load(open("inh{'Populations.ext.cellparams.duration': 5000}.pkl", "rb"))
data_inh = inh_data.segments[0]

thresh = 0.01 #less than 1% of cells firing in the specified time bin

binsize = 10
fr_py = rate(rec_duration, data_py.spiketrains, bin_size=binsize)
fr_inh = rate(rec_duration, data_inh.spiketrains, bin_size=binsize)

# portion_up_py = sum(fr_py >= thresh) / len(fr_py)
portion_down_py = sum(fr_py < thresh) / len(fr_py)
# portion_up_inh = sum(fr_inh >= thresh) / len(fr_inh)
portion_down_inh = sum(fr_inh < thresh) / len(fr_inh)

txt = ('pyra down state = ' +str(round(portion_down_py,3)*100) + '%  \ninhib down state = ' + str(round(portion_down_inh,3)*100) + ' %')

# plotting firing rate and stuffs
fig = plt.figure(56) # Why 56 ?? size of the fig ?
plt.plot(fr_py, 'b')
plt.plot(fr_inh,'r')
plt.title('firing thingy')
plt.ylim([.0,1.])
plt.ylabel('proportion of cell firing')
plt.xlabel('time bins with bin size = '+str(binsize) + '(ms)')
plt.legend(['py_neurons','inh_neurons'])
plt.hlines(thresh, 0, len(fr_py), color = 'k', linestyle='--')
plt.text(2,0.8, txt)
fig.savefig('firingrate_thingy'+'.svg')
fig.clf()
plt.close()
