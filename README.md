# Biological neuronal network simulation of the thalamus

## Docker Image

Directly derived from: https://hub.docker.com/r/neuralensemble/simulation/

With:

* shell environment with NEST 2.14, NEURON 7.5, and PyNN 0.9 installed.
* The Python 2.7 version provides Brian 1.4, the Python 3.4 version provides Brian 2.
* IPython, scipy, matplotlib and OpenMPI are also installed.

## Basic use

Start docker daemon

```
sudo systemctl restart docker
```

Enable the current user to launch docker images

```
sudo usermod -a -G docker $USER
```

Move to the folder "thalamus" checked out from github and build the image

```
docker build -t thalamus .
```

Check the existence of the image

```
docker images
```

Start a container with the "thalamus" image
```
docker run -i -t thalamus:py2 /bin/bash
```

And to allow for development bind-mount your local files in the container

```
docker run -v `pwd`:`pwd` -w `pwd` -i -t thalamus /bin/bash

```

Then check the container id (or name)

```
docker ps
```

