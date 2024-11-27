# Copyright © 2024 INSERM U1219, Therapixel SA
# Contributors: Manel Rakez, Julien Guillaumin
# All rights reserved.
# This file is subject to the terms and conditions described in the
# LICENSE file distributed in this package.
FROM pytorch/pytorch:2.0.1-cuda11.7-cudnn8-runtime

# to silent warning from pip
ENV PIP_ROOT_USER_ACTION=ignore

RUN apt update -y && apt upgrade -y && \
    apt install -y --no-install-recommends \
      libsm6='2:1.2.3-1' \
      libxext6='2:1.3.4-0*' \
      libxrender-dev='1:0.9.10-1' \
      libgl1='1.3.2-1*' \
      libglib2.0-0='2.64.6-1*' && \
    apt clean -y && apt-get autoremove -y && \
    rm -rf /var/lib/apt/lists/*

COPY requirements/*.txt .
RUN pip3 install --upgrade pip && \
    pip3 install -r requirements.txt -r requirements-dev.txt && \
    rm *.txt

