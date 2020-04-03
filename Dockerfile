FROM ubuntu:18.04

SHELL ["/bin/bash", "-c"]

RUN apt-get update \
    && apt-get install -y \
    build-essential \
    python3.6 \
    python3-pip \
    openjdk-8-jdk


WORKDIR /code

RUN pip3 install virtualenv
COPY ./requirements* ./
COPY Makefile ./
RUN make install

COPY . ./
CMD make test-local
