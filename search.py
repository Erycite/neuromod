{
    # ------------------------------------------------------------------------------
    # Usage:
    # Examples

    # start the docker image:
    # $ docker run -v `pwd`:`pwd` -w `pwd` -i -t thalamus /bin/bash

    # Run simple code
    # python run.py --folder test --params epsp_response.py nest

    # Search Example:
    # python run.py --folder EPSPsearch --params epsp_response.py --search search.py --map yes nest
    # python run.py --folder IPSPsearch --params ipsp_response.py --search search.py --map yes nest
    # python plot_map.py

    # Analysis Example
    # python run.py --folder EPSPsearch --params epsp_response.py --search search.py --analysis true nest

    # ./execute.bash
    # ------------------------------------------------------------------------------

    # apply constraints from the literature to reduce the dimensionality of the parameter space

    #'run_time' : [5000],
    #'Populations.py.n' : [1600],
    #'Modifiers.py.cells.end' : [0.12,0.13,0.14],
    #'Modifiers.py.properties.a' : [0.01, .02, 0.03],

    # 'Populations.cell.cellparams.tau_m': np.arange(.0, 20., 1.), # 
    # 'Populations.cell.cellparams.v_rest': np.arange(-85., -55., 3), #
    # 'Populations.cell.cellparams.v_rest': np.arange(-65., -55., 0.5), # as in Cerina et al. 2015
    # 'Populations.cell.cellparams.v_rest': np.array([-58., -90.]), #np.arange(-70., -60., 2.),
    # 'Populations.cell.cellparams.v_rest': np.arange(-90., -58., 2.),
    # 'Populations.cell.cellparams.v_reset': np.arange(-90., -55., 7.),
    # 'Populations.cell.cellparams.v_thresh': np.arange(-90., -80., 2.), # mV: parameter search
    # 'Populations.cell.cellparams.v_thresh': np.arange(-60., -50., .5), # mV: parameter search
    # 'Populations.cell.cellparams.a': np.arange(.0, 36., 12.), # uS: Cerina et al. 2015
    # 'Populations.cell.cellparams.a': np.arange(.0, 30., 6), # uS: Cerina et al. 2015
    'Populations.py.cellparams.b': np.arange(.005, .02, .001), # nA
    # 'Populations.cell.cellparams.b': np.arange(.0, .00004, .000002), # nA
}
