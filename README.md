# craco-python

Python utilities for the CRACO project

# Installation

This will be on pyPI in future. For now do:

*Before you start*: Make sure you have no old virtualenvs enabled or any CRAFT or other
packages in  your $PYTHONPATH environment variable

```
# Create a virtual env if you don't have one already - note we need python3.6 for XRT compatibility - this will be improved in future
python3.6 -m venv venv --prompt craco
source venv/bin/activate

# Install craco-python directly from github
pip install https://github.com/askap-craco/craco-python.git

```

# To install for development

If you want to develop craco-python then it's best to clone the repository and install it as "editable" with PIP.

If you make any changes to the python files, it should be reflected immediately when you run another command.

If you change the package (e.g. add dependancies or change `setup.py`) you'll need to run `pip install -e .` again.

```

# clone the repository
git clone https://github.com/askap-craco/craco-python.git
cd craco-python

# Create a virtual env if you don't have one already
python3.6 -m venv venv --prompt craco

# Activate the virtualenv.
source venv/bin/activate

# Install craco-python as "editable" - this means you don't have to install every time you make a change
pip install -e .


```

# To run the test suite

It uses [pytest](http://www.pytest.org) to find and run tests. It uses the
`addopts = --doctest-modules` is in the `pytest.ini` file, so it will also find
and execute doctests if you have them.

This also creates a coverage report xml for use in glatb.ci

```
python setup.py test
```

# To view test coverage

We use [coverage.py](https://coverage.readthedocs.io)

```
python -m pytest --cov-report html --cov craco tests

# And open htmlcov/index.html
```


# Command-line Utilities

## corrsim - Simulates the ASKAP Correlator CRACO output

This script reads a simulated FRB UVFITS file (produced elsewhere), and sends UDP packets conforming
to the [Correlator packet specification](https://confluence.csiro.au/display/CRACO/Correlator+packet+format) to
the destination supplied.

Example Usage

```
$ corrsim  /data/craco/ban115/test_data/frb_d0_t0_a1_sninf_lm00/frb_d0_t0_a1_sninf_lm00.fits -d 127.0.0.1:1235 -b 3 -p 0,1 --nloops 1
Filename: /data/craco/ban115/test_data/frb_d0_t0_a1_sninf_lm00/frb_d0_t0_a1_sninf_lm00.fits
No.    Name      Ver    Type      Cards   Dimensions   Format
  0  PRIMARY       1 GroupsHDU      102   (3, 1, 256, 1, 1)   float32   3230 Groups  5 Parameters
  1  AIPS AN       1 BinTableHDU     63   20R x 12C   [8A, 3D, 0D, 1J, 1J, 1E, 1A, 1E, 3E, 1A, 1E, 3E]   
INFO:root:Sending data from /data/craco/ban115/test_data/frb_d0_t0_a1_sninf_lm00/frb_d0_t0_a1_sninf_lm00.fits to 127.0.0.1:1235 with beamID=3 and polID=[0, 1] tsamp=1728us
INFO:root:First time is UTC=2020-10-15T04:26:47.160285+00:00 utc_back=2020-10-15 04:26:49.160285+00:00 dutc=37 bat=0x12270531c5d51d jd=44117.685268 
Sent 8704 packets
```

Command line help:
```
$ corrsim -h
usage: corrsim [-h] -d DESTINATION [-b BEAMID] [-p POLID] [-t TSAMP] [-v]
               [-n NLOOPS]
               uvfits

Simulate the ASKAP Correlator in CRACO mode

positional arguments:
  uvfits

optional arguments:
  -h, --help            show this help message and exit
  -d DESTINATION, --destination DESTINATION
                        Desintation UDP adddress:port e.g. localhost:1234
                        (default: None)
  -b BEAMID, --beamid BEAMID
                        Beamid to put in headers, 0-35 (default: 0)
  -p POLID, --polid POLID
                        PolIDs. If input file has 1 polarisation, it
                        duplicates it with given pol IDs (default: [0, 1])
  -t TSAMP, --tsamp TSAMP
                        Sampling interval in microseconds (default: 1728)
  -v, --verbose         Be verbose (default: False)
  -n NLOOPS, --nloops NLOOPS
                        Number of times to loop through the file (default: 1)
```

# API Documentation
API Documentation can be found [https://craft-group.gitlab.io/craco-python/]

