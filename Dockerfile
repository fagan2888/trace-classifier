FROM ubuntu:18.04

SHELL ["/bin/bash", "-c"]

RUN apt-get clean

RUN apt-get update \
    && apt-get install -y \
    build-essential \
    python3.6 \
    python3-pip \
    openjdk-8-jdk


WORKDIR /code

RUN pip3 install virtualenv
RUN pip3 install pytest==5.0.1
RUN pip3 install pytest-cov==2.7.1
RUN pip3 install pyspark==2.4.4
RUN pip3 install numpy==1.14.5

COPY ./requirements* ./
COPY Makefile ./
RUN pip3 install -r requirements-dev.txt
RUN make install

COPY . ./
CMD make test-local
