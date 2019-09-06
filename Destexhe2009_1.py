{

    'run_time': 2000., # ms
    'dt': 0.01, # ms

    'Populations' : {
        'cell' : {
            'n': 1, # units
            'type': sim.EIF_cond_alpha_isfa_ista,
            'cellparams': {
                'tau_syn_E'  : 5.0,   # ms, time constant of exponential decay of conductance shift
                'tau_syn_I'  : 10.0,  # ms, as above
                'e_rev_E'    : 0.0,   # mV, reversal potential for excitatory inputs
                'e_rev_I'    : -80.,  # mV, reversal potential for inhibitory inputs
                'tau_refrac' : 2.5,   # ms, refractory period
                'v_rest'     : -60.0, # mV, resting potential
                'v_reset'    : -60.0, # mV, reset after spike
                'v_thresh'   : -50.0, # mV, spike threshold
                #'v_spike'    : 40.0,  # mV, spike detection (just to plot it nicely)
                'v_spike'    : -40.0, # mV, spike detection
                'delta_T'    : 2.5,   # mV, steepness of exponential approach to threshold
                'cm'         : 0.200, # nF, 1 uF/cm^2 with 20000 um^2 is the membrane area
                'tau_m'      : 20.0,  # ms, time constant of leak conductance
                'tau_w'      : 600.0, # ms, time constant of adaptation variable
                #'a'          : .001,  # uS, spike-frequency adaptation (NEST,NEURON figure 1ABC)
                #'a'          : 20.,   # uS, spike-frequency adaptation (NEST, NEURON figure 1D)
                #'a'          : 23.,   # uS, spike-frequency adaptation (NEST, NEURON figure 1E)
                'a'          : 50.,   # uS, spike-frequency adaptation (NEST, NEURON figure 1F)
                #'b'          : .08,   # nA, increment to the adaptation variable (figure1A) NEST:{'SC': 4, 'ISI': 1.0498, 'CV': 0.7785}, NEURON:{'SC': 4, 'ISI': 1.0498, 'CV': 0.7785}
                #'b'          : .028,  # nA, increment to the adaptation variable (figure1B) NEST:{'SC': 9, 'ISI': 0.4444, 'CV': 0.6491}, NEURON:{'SC': 9, 'ISI': 0.4445, 'CV': 0.6488}
                #'b'          : .007,  # nA, increment to the adaptation variable (figure1C) NEST:{'SC': 16, 'ISI': 0.2424, 'CV': 0.1198}, NEURON:{'SC': 16, 'ISI': 0.2424, 'CV': 0.1197}
                #'b'          : .007,  # nA, increment to the adaptation variable (figure1D) NEST:{'SC': 16, 'ISI': 0.4940, 'CV': 1.3792}, NEURON:{'SC': 16, 'ISI': 0.4940, 'CV': 1.3795}
                #'b'          : .0,    # nA, increment to the adaptation variable (figure1E) NEST:{'SC': 21, 'ISI': 0.3778, 'CV': 1.2797}, NEURON:{'SC': 28, 'ISI': 0.2835, 'CV': 1.9286}
                'b'          : .01,  # nA, increment to the adaptation variable (figure1E) NEST:{'SC': 9, 'ISI': 0.6796, 'CV': 1.3120}, NEURON:{'SC': 9, 'ISI': 0.6795, 'CV': 1.3126}

                #'i_offset'   : 0.25,  # nA, constant injected current
            }
        },
    },

    'Projections' : {
    },

    'Injections' : {
        'cell' : {
            'source' : sim.StepCurrentSource,
            #'amplitude' : [.25, .0], # figure 1ABC
            #'start' : [200., 600.], # figure 1ABC
            'amplitude' : [-.25, 0.0, .25, 0.0], # figure 1DEF
            'start' : [200., 600., 1000., 1400.], # figure 1DEF
            'stop' : 0.0
        },
    },

    'Recorders' : {
        'cell' : {
            'spikes' :  'all',
            'v' : 'all',
            #'v' : {
            #    'start':0,
            #    'end':1
            #}
        },
    },

    'Modifiers' :{
    }

}
