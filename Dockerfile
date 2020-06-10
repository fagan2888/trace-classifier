FROM ubuntu:18.04

SHELL ["/bin/bash", "-c"]

RUN apt-get update \
    && apt-get install -y \
    build-essential \
    python3.4 \
    python3-pip \
    openjdk-8-jdk


WORKDIR /code

RUN pip3 install virtualenv
RUN pip3 install pytest==5.0.1
RUN pip3 install pytest-cov==2.7.1
COPY ./requirements* ./
COPY Makefile ./
RUN make install

COPY . ./
CMD make test-local
