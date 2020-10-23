============
Installation
============

In order to use ACT, you must have Python 3.6+ installed. We do not plan on 
having support for Python 2.x as it will be deprecated very soon.

In addition, in order to make Skew-T plots, metpy is needed. In order to install
MetPy, simply do::

    $ pip install metpy

Or, if you have Anaconda::

    $ conda install -c conda-forge metpy
    
You can build the Atmospheric data Community Toolkit from source and install it doing::


    $ git clone https://github.com/ARM-DOE/ACT
    $ cd ACT
    $ python setup.py install

We soon plan to implement pip install and conda install features. 

