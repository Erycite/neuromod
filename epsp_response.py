{

    'run_time': 2000., # ms
    'dt': 0.01, # ms

    'Populations' : {
        'cell' : {
            'n': 1, # units
            'type': sim.EIF_cond_alpha_isfa_ista,
            'cellparams': {

                'tau_refrac' : 2.5,   # ms, refractory period (ReinagelReid2000)
                'delta_T'    : 2.5,   # mV, steepness of exponential approach to threshold (Destexhe2009)
                'v_thresh'   : -45.0, # mV, fixed spike threshold (https://www.neuroelectro.org/neuron/190/)
                'v_spike'    : -30.0, # mV, spike detection
                'v_rest'     : -63.0, # mV, resting potential (RE: https://www.neuroelectro.org/neuron/190/, ContrerasCurroSteriade1993; TC: McCormick1992, TurnerAndersonWilliamsCrunelli1997, Cerina et al. 2015); IN (PapeMcCormick1995)
                'tau_w'      : 270.0, # ms, time constant of adaptation variable (Cerina et al. 2015)
                'b'          : .02,  # nA, increment to the adaptation variable (Cerina et al. 2015)

                # # TC - DeschenesParadisRoySteriade1984 cat in-vivo anaesthetized
                # 'v_reset'    : -55.0, # mV, reset after spike (Destexhe2009)
                # 'a'          : 6,     # nS, spike-frequency adaptation (Cerina et al. 2015)(McCormickHuguenrad1992: IKleak fig13)
                # 'cm'         : 0.16,  # nF, tot membrane capacitance (Bloomfield Hamos Sherman 1987)
                # 'tau_m'      : 16.0,  # ms, time constant of leak conductance (cm/gl, with gl=0.01)

                # RE - ContrerasCurroSteriade1993 cat in-vivo awake
                'v_reset'    : -41.0, # mV, reset after spike (Toubul and Brette 2008: "the number of spikes per burst increases when Vr increases (since w∗ increases with Vr) and when b decreases")
                'a'          : 12,    # nS, spike-frequency adaptation (Cerina et al. 2015)(McCormickHuguenrad1992: IKleak fig13)
                'cm'         : 0.20, # nF, tot membrane capacitance (https://www.neuroelectro.org/neuron/190/, Uhlrich Cucchiaro Humphrey Sherman 1991)
                'tau_m'      : 20.,  # ms, time constant of leak conductance (cm/gl, with gl=0.01) (https://www.neuroelectro.org/neuron/190/)

                #'i_offset'   : 0.25,  # nA, constant injected current
            }
        },
    },
    # default_parameters = {'tau_refrac': 0.1, 'a': 4.0, 'tau_m': 9.3667, 'e_rev_E': 0.0, 'i_offset': 0.0, 'cm': 0.281, 'delta_T': 2.0, 'e_rev_I': -80.0, 'v_thresh': -50.4, 'b': 0.0805, 'tau_syn_E': 5.0, 'v_reset': -70.6, 'v_spike': -40.0, 'tau_syn_I': 5.0, 'tau_w': 144.0, 'v_rest': -70.6}
    # units = {'tau_refrac': 'ms', 'a': 'nS', 'tau_m': 'ms', 'e_rev_E': 'mV', 'v_spike': 'mV', 'cm': 'nF', 'i_offset': 'nA', 'gsyn_exc': 'uS', 'tau_w': 'ms', 'e_rev_I': 'mV', 'delta_T': 'mV', 'b': 'nA', 'tau_syn_E': 'ms', 'v_reset': 'mV', 'w': 'nA', 'v': 'mV', 'tau_syn_I': 'ms', 'gsyn_inh': 'uS', 'v_thresh': 'mV', 'v_rest': 'mV'}

    'Projections' : {
    },

    'Injections' : {
        'cell' : {
            'source' : sim.StepCurrentSource,

            # NOTES: 
            # - the model reproduces Avanzini et al. at -90, and it does the same with CurooDossi et al. at -63, more reactive in a smaller dynamic scale
            # - the amplitude of injection is encoded in the oscillatory freq

            # protocol for an awake resting -63.0, see above refs
            # 'amplitude' : [.2, .0], # no result
            # 'amplitude' : [.3, .0], # initial burst, then silent
            # 'amplitude' : [.37, .0], # postburst subthreshold oscillatory dynamic
            'amplitude' : [.5, .0], # chattering at ~5Hz
            # 'amplitude' : [.5, .0], # tonic
            # 'amplitude' : [.36, .0], # 

            # protocol as in figure 2C AvanziniCurtisPanzicaSpreafico1989
            # 'amplitude' : [.10, .0], # no result
            # 'amplitude' : [.55, .0], # initial burst, then silent
            # 'amplitude' : [.58, .0], # 
            # 'amplitude' : [.95, .0], # postburst subthreshold oscillatory dynamic
            # 'amplitude' : [1., .0], # chattering at ~1Hz
            # 'amplitude' : [1.1, .0], # tonic

            # 'amplitude' : [.25, .0], # default
            'start' : [200., 1600.], # long duration
            # 'start' : [200., 600.], # short duration
            'stop' : 0.0
        },
    },

    'Recorders' : {
        'cell' : {
            'spikes' :  'all',
            'v' : 'all',
            'w' : 'all',
            #'v' : {
            #    'start':0,
            #    'end':1
            #}
        },
    },

    'Modifiers' :{
    }

}
