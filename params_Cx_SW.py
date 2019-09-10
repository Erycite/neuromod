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
                'duration':5000.0
            }
        },
       'py' : {
            'n': 1600, # units
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
                'tau_w'      : 600.0,
                'cm'         : 0.200,
                'a'          : 0.001,#0.8e-3 Naud et al. 2008
                'b'          : 0.1 #0.03
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
                'tau_w'      : 600.0,
                'cm'         : 0.200,
                'a'          : 0.001,#uS
                'b'          : 0.015 #nA
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
            'connector' : sim.FixedProbabilityConnector(.02, allow_self_connections=False, rng=sim.random.NumpyRNG(1235342134, parallel_safe=False)),
            'synapse_type' : sim.StaticSynapse,
            'weight' : 6e-3,
            'receptor_type' : 'excitatory'
        },
        'py_inh' : {
            'source' : 'py',
            'target' : 'inh',
            'connector' : sim.FixedProbabilityConnector(.02, allow_self_connections=False, rng=sim.random.NumpyRNG(1235342134, parallel_safe=False)),
            'synapse_type' : sim.StaticSynapse,
            'weight' : 6e-3,
            'receptor_type' : 'excitatory'
        },
        'inh_py' : {
            'source' : 'inh',
            'target' : 'py',
            'connector' : sim.FixedProbabilityConnector(.02, allow_self_connections=False, rng=sim.random.NumpyRNG(1235342134, parallel_safe=False)),
            'synapse_type' : sim.StaticSynapse,
            'weight' : 67e-3,
            'receptor_type' : 'inhibitory'
        },
        'inh_inh' : {
            'source' : 'inh',
            'target' : 'inh',
            'connector' : sim.FixedProbabilityConnector(.02, allow_self_connections=False, rng=sim.random.NumpyRNG(1235342134, parallel_safe=False)),
            'synapse_type' : sim.StaticSynapse,
            'weight' : 67e-3,
            'receptor_type' : 'inhibitory'
        }
    },

    'Recorders' : {
        'py' : {
            'spikes' : 'all',
            'v' : {
                'start' : 0,
                'end' : 100,
            }
        },
        'inh' : {
            'spikes' : 'all',
            'v' : {
                'start' : 0,
                'end' : 100,
            }
        },

    },

    'Modifiers' :{
    },

    'Injections' : {
    },

}
