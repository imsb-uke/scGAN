FROM tensorflow/tensorflow:latest-gpu

# Install some dependencies

ENV LC_ALL C.UTF-8

RUN apt-get update

RUN apt-get install -y python3-pip

RUN apt-get -y install xml2 

RUN apt-get -y install ssh python3-tk

RUN pip3 install -U pip

RUN pip install matplotlib MulticoreTSNE h5py scipy pillow

RUN pip install seaborn scikit-learn statsmodels numba

RUN pip install scanpy==1.2.2

RUN pip install --upgrade numpy

RUN apt-get -y  install libxml2-dev

RUN pip install python-igraph

RUN pip install tables

RUN pip install louvain

MAINTAINER Mohamed Marouf <maroufmail@gmail.com>
