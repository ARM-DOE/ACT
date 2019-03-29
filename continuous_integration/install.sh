#!/bin/bash

set -e
# use next line to debug this script
set -x

# Install Miniconda
wget https://repo.continuum.io/miniconda/Miniconda3-latest-Linux-x86_64.sh \
    -O miniconda.sh
chmod +x miniconda.sh
./miniconda.sh -b
export PATH=/home/travis/miniconda3/bin:$PATH
conda config --set always_yes yes
conda config --set show_channel_urls true
conda update -q conda
conda info -a 

## Create a testenv with the correct Python version
conda env create -f continuous_integration/environment-$PYTHON_VERSION.yml
source activate testenv
conda install boto3
pip install -e .
pip install sphinx sphinx_rtd_theme


