{
    'DistanceDep': True,
    'run_time': 5000, # ms
    'dt': 0.1, # ms

    'Populations' : {
        'ext' : {
            'n' : 1,
            'type': sim.SpikeSourcePoisson,
            'cellparams' : {
                'start':0.0,
                'rate':50.,
                'duration':100.0
            }
        },

       'py' : {
            'n': 8000, # units
            'type': sim.EIF_cond_alpha_isfa_ista,
            'cellparams': {
                'tau_m'      : 20.0, #9.3 pynn default            # ms
                'tau_syn_E'  : 5.0, #0.2/0.006 = 33 ?
                'tau_syn_I'  : 10.0, #0.2/0.067 = 3 ?
                'tau_refrac' : 2.5, #D
                'v_rest'     : -60.0,#D
                'v_reset'    : -60.0,#D
                'v_thresh'   : -50.0,#D
                'delta_T'    : 2.5, #0.8 Naud et al.2008
                'tau_w'      : 600.0,#D
                'cm'         : 0.200,#nF D = Cm/cm2 * Surface
                'a'          : 0.001,#0.8e-3 Naud et al. 2008
                'b'          : .1 #.03 #0.065 Naud et al. 2008
            }
        },

        'inh' : {
            'n': {'ref':'py','ratio':0.25},
            'type': sim.EIF_cond_alpha_isfa_ista,
            'cellparams': {
                'tau_m'      : 20.0,             # ms
                'tau_syn_E'  : 5.0,
                'tau_syn_I'  : 10.0,
                'tau_refrac' : 2.5,
                'v_rest'     : -60.0,
                'v_reset'    : -60.0,
                'v_thresh'   : -50.0,
                'delta_T'    : 2.5,
                'tau_w'      : 600.0,#ms
                'cm'         : 0.200,#uS
                'a'          : 0.001,#uS
                'b'          : 0.015 #nA
            }
        },

        'tc' : {
            'n': {'ref':'py','ratio':0.0625},
            'type' :  sim.EIF_cond_alpha_isfa_ista,
            'cellparams' : {
                'tau_m'      : 30.0,             # ms
                'tau_syn_E'  : 5.0, #needs to change as ge/gi changes
                'tau_syn_I'  : 10.0,
                'tau_refrac' : 2.5,
                'v_rest'     : -60.0,
                'v_reset'    : -60.0,
                'v_thresh'   : -50.0,
                'delta_T'    : 2.5,
                'tau_w'      : 150.0,
                'cm'         : 0.15,#nF
                'a'          : 13.,#uS
                'b'          : 0.0 #nA
            }
                
        },

        're' : {
            'n': {'ref':'py','ratio':0.0625},
            'type' :  sim.EIF_cond_alpha_isfa_ista,
            'cellparams' : {
                'tau_m'      : 20.0,             # ms
                'tau_syn_E'  : 5.0,
                'tau_syn_I'  : 10.0,
                'tau_refrac' : 2.5,
                'v_rest'     : -60.0,
                'v_reset'    : -60.0,
                'v_thresh'   : -50.0,
                'delta_T'    : 2.5,
                'tau_w'      : 150.0,
                'cm'         : 0.15,#nF
                'a'          : 30.,#uS
                'b'          : 0.01 #nA
            }
        }
    },


    'Projections' : {
        'ext_py' : {
            'source' : 'ext',
            'target' : 'py',
            'connector' : sim.FixedProbabilityConnector(.02),
            'synapse_type' : sim.StaticSynapse,
            'weight' : 6e-3,
            'receptor_type' : 'excitatory'
        },
        'py_py' : {
            'source' : 'py',
            'target' : 'py',
            'connector' : sim.FixedProbabilityConnector(.02, allow_self_connections=False, rng=sim.NumpyRNG(1235342134, parallel_safe=False)),
            'synapse_type' : sim.StaticSynapse,
            'weight' : 6e-3,
            'receptor_type' : 'excitatory'
        },
        'py_inh' : {
            'source' : 'py',
            'target' : 'inh',
            'connector' : sim.FixedProbabilityConnector(.02, allow_self_connections=False, rng=sim.NumpyRNG(1235342134, parallel_safe=False)),
            'synapse_type' : sim.StaticSynapse,
            'weight' : 6e-3,
            'receptor_type' : 'excitatory'
        },
        'inh_py' : {
            'source' : 'inh',
            'target' : 'py',
            'connector' : sim.FixedProbabilityConnector(.02, allow_self_connections=False, rng=sim.NumpyRNG(1235342134, parallel_safe=False)),
            'synapse_type' : sim.StaticSynapse,
            'weight' : 67e-3,
            'receptor_type' : 'inhibitory'
        },
        'inh_inh' : {
            'source' : 'inh',
            'target' : 'inh',
            'connector' : sim.FixedProbabilityConnector(.02, allow_self_connections=False, rng=sim.NumpyRNG(1235342134, parallel_safe=False)),
            'synapse_type' : sim.StaticSynapse,
            'weight' : 67e-3,
            'receptor_type' : 'inhibitory'
        },

        'py_tc' : {
            'source' : 'py',
            'target' : 'tc',
            'connector' : sim.FixedProbabilityConnector(.02, allow_self_connections=False, rng=sim.NumpyRNG(1235342134, parallel_safe=False)),
            'synapse_type' : sim.StaticSynapse,
            'weight' : 6e-3,
            'receptor_type' : 'excitatory'
        },
        'py_re' : {
            'source' : 'py',
            'target' : 're',
            'connector' : sim.FixedProbabilityConnector(.02, allow_self_connections=False, rng=sim.NumpyRNG(1235342134, parallel_safe=False)),
            'synapse_type' : sim.StaticSynapse,
            'weight' : 6e-3,
            'receptor_type' : 'excitatory'
        },

        'tc_re' : {
            'source' : 'tc',
            'target' : 're',
            'connector' : sim.FixedProbabilityConnector(.02, allow_self_connections=False, rng=sim.NumpyRNG(1235342134, parallel_safe=False)),
            'synapse_type' : sim.StaticSynapse,
            'weight' : 30e-3,
            'receptor_type' : 'excitatory'
        },

        'tc_py' : {
            'source' : 'tc',
            'target' : 'py',
            'connector' : sim.FixedProbabilityConnector(.02, allow_self_connections=False, rng=sim.NumpyRNG(1235342134, parallel_safe=False)),
            'synapse_type' : sim.StaticSynapse,
            'weight' : 6e-3,
            'receptor_type' : 'excitatory'
        },

        'tc_inh' : {
            'source' : 'tc',
            'target' : 'inh',
            'connector' : sim.FixedProbabilityConnector(.02, allow_self_connections=False, rng=sim.NumpyRNG(1235342134, parallel_safe=False)),
            'synapse_type' : sim.StaticSynapse,
            'weight' : 67e-3,
            'receptor_type' : 'excitatory'
        },
        're_tc' : {
            'source' : 're',
            'target' : 'tc',
            'connector' : sim.FixedProbabilityConnector(.08, allow_self_connections=False, rng=sim.NumpyRNG(1235342134, parallel_safe=False)),
            'synapse_type' : sim.StaticSynapse,
            'weight' : 40e-3, #30e-3,
            'receptor_type' : 'inhibitory'
        },
        
        're_re' : {
            'source' : 're',
            'target' : 're',
            'connector' : sim.FixedProbabilityConnector(.08, allow_self_connections=False, rng=sim.NumpyRNG(1235342134, parallel_safe=False)),
            'synapse_type' : sim.StaticSynapse,
            'weight' : 30e-3,
            'receptor_type' : 'inhibitory'
        },

    
    },


    'Recorders' : {
        'py' : {
            'spikes' :  'all',
            'v' : {
                'start' :200,
                'end' : 210,
            }
        },
        'inh' : {
            'spikes' :  'all',
            'v' : {
                'start' : 0,
                'end' : 10,
            }
        },
        'tc' : {
            'spikes' :  'all',
            'v' : {
                'start' : 0,
                'end' : 10,
            }
        },
        're' : {
            'spikes' :  'all',
            'v' : {
                'start' : 0,
                'end' : 10,
            }
        }
    },

    'Modifiers' :{
    },


    'Injections' : {
    }

}