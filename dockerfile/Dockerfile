FROM tensorflow/tensorflow:1.8.0-gpu-py3

# Install some dependencies

ENV LC_ALL C.UTF-8

RUN apt-get update \
  && apt-get -y --no-install-recommends install ssh python3-tk python3-pip  libxml2-dev libhdf5-dev llvm \
  && pip3 install --upgrade 'pip<21' setuptools wheel

COPY requirements.txt /requirements.txt

RUN pip3 install cmake==3.18.2 && \
    pip3 install -r /requirements.txt

WORKDIR /scGAN

COPY . ./

CMD python main.py 
