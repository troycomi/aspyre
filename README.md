[![Travis Build Status](https://travis-ci.com/vineetbansal/aspyre.svg?branch=master)](https://travis-ci.com/vineetbansal/aspyre)
[![Appveyor Build status](https://ci.appveyor.com/api/projects/status/bdq2dn0bi14iy992?svg=true)](https://ci.appveyor.com/project/vineetbansal/aspyre-o47mb)
[![Coverage Status](https://coveralls.io/repos/github/vineetbansal/aspyre/badge.svg?branch=master&service=github)](https://coveralls.io/github/vineetbansal/aspyre?branch=master)
[![Documentation Status](https://readthedocs.org/projects/aspyre/badge/?version=latest)](https://cov3d.readthedocs.io/en/latest/?badge=latest)

# ASPyRE

Algorithms for Single Particle Reconstruction

## Installation Instructions

### Linux/Mac OS X/Windows

The simplest option is to use Anaconda 64-bit for your platform, and use the provided `environment.yml` file to build a Conda environment to run ASPyRE.

```
cd /path/to/git/clone/folder
conda env create -f environment.yml
conda activate aspyre
```

## Make sure everything works

Once ASPyRE is installed, make sure the unit tests run correctly on your platform by doing:
```
cd /path/to/git/clone/folder
python setup.py test
```

Tests currently take around 5 minutes to run. If some tests fail, you may realize that `python setup.py test` produces too much information. Re-running tests using `py.test tests` in `/path/to/git/clone/folder` may provide a cleaner output to analyze.

## Install

If the tests pass, install the ASPyRE package for the currently active Conda environment:
```
cd /path/to/git/clone/folder
python setup.py install
```
