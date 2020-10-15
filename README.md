# craco-python

Python utilities for the CRACO project

# Installation

This will be on pyPI in future. For now do:

*Before you start*: Make sure you have no old virtualenvs enabled or any CRAFT or other
packages in  your $PYTHONPATH environment variable

```
# Create a virtual env if you don't have one already
python3.8 -m venv venv --prompt craco
source venv/bin/activate

pip install git+https://bitbucket.csiro.au/scm/crac/craco-python.git

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

# Documentation

