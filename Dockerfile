##############################################################################
# A Docker image for running neuronal network simulations
#
# docker build (--no-chache) -t neuromod .
# docker ps
# docker run -e DISPLAY=$DISPLAY -v `pwd`:`pwd` -w `pwd` -i -t neuromod /bin/bash
# (in the image)# python run_size_closed.py nest 8 param/defaults_mea 'data_size'

FROM neuralensemble/simulationx

MAINTAINER domenico.guarino@cnrs.fr

##########################################################
# Xserver
#CMD export DISPLAY=:0
#CMD export DISPLAY=:0.0
#ENV DISPLAY :0
CMD export DISPLAY=0.0

#######################################################
# Additional prerequisite libraries

RUN apt-get autoremove -y && \
    apt-get clean


##########################################################
# Additions to run AdEx thalamus explorative study

WORKDIR $HOME
RUN git clone https://github.com/dguarino/neuromod.git

WORKDIR $HOME/neuromod

