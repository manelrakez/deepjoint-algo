FROM ubuntu:20.04

# to silent warning from pip
ENV PIP_ROOT_USER_ACTION=ignore
ENV LANG=C.UTF-8

# Install Python 3.10(.13) + latest R version + JAGS
RUN apt update -y && apt upgrade -y && \
    apt-get install -y --no-install-recommends software-properties-common curl wget && \
    add-apt-repository -y ppa:deadsnakes/ppa && \
    apt-get -y update && \
    apt-get install -y --no-install-recommends \
      python3.10='3.10.13-1+focal1' \
      python3.10-dev='3.10.13-1+focal1' \
      python3.10-distutils='3.10.13-1+focal1' && \
    echo "post-installation steps for python3.10" && \
    # remove python3.8 installed with software-properties-common
    apt-get -y purge python3.8 && apt-get -y autoremove && \
    ln -s -f $(which python3.10) /usr/bin/python3 && \
    ln -s -f $(which python3) /usr/bin/python && \
    curl -sS https://bootstrap.pypa.io/get-pip.py | python3.10 && \
    ln -s -f $(which pip3.10) /usr/bin/pip3 && \
    ln -s -f $(which pip3) /usr/bin/pip && \
    python3 -m pip install --upgrade pip && \
    echo "Finally install latest R (base & base-dev) and JAGS" && \
    echo "deb https://cloud.r-project.org/bin/linux/ubuntu focal-cran40/" >> /etc/apt/sources.list && \
    wget -qO- https://cloud.r-project.org/bin/linux/ubuntu/marutter_pubkey.asc | tee -a /etc/apt/trusted.gpg.d/cran_ubuntu_key.asc && \
    apt update -y && \
    apt-get install -y r-base r-base-dev jags && \
    # Clean apt, to reduce image size
    apt-get clean -y && apt-get autoremove -y && \
    rm -rf /var/lib/apt/lists/*

# Finally install Jupyter notebook with R support
COPY docker/deepjoint_r/r_pkg_install.R .
RUN pip install --upgrade 'jupyterhub==3.1' 'notebook==6' 'jupyterlab==3' \
    && Rscript r_pkg_install.R
