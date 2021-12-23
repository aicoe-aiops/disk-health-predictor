# Disk Health Predictor

A python package that provides ready-to-use machine learning models to predict time-to-failure of a hard disk drive.


## Description

Disk failure prediction has been a topic of great interest in both academia and industry for a long time. A number of data science techniques have been explored, and several research papers have published, in this domain. However, a standard and ubiquitous way for users to easily access and actually use these models is yet to be established. This is the gap that Disk Health Predictor seeks to bridge.

Disk Health Predictor is Python module that packages several pre-trained disk failure prediction models into readily usable Python classes. Currently, it contains the publicly available machine learning models created by the following organizations

* ProphetStor
* Red Hat

These models take a hard disk drive's SMART data as input, and estimate the disk's health and its time-to-failure based on that.


## Installation

This package can be installed using `pip` as follows::
```bash
pip install git+https://github.com/aicoe-aiops/disk-health-predictor.git
```


## Usage

For examples of how to use this package, please see the demo jupyter notebook [here](examples/usage-demo.ipynb).


## Contributing

We welcome and encourage the open source community to contribute to this project!

If you have any suggestions, or run into any problems, please feel free to open issues [here](https://github.com/aicoe-aiops/disk-health-predictor/issues/new/choose).

If you want to work on an open issue, please let us know by commenting on that issue and then feel free to submit a pull request [here](https://github.com/aicoe-aiops/disk-health-predictor/compare).

While contributing to the project, please keep in mind that this project uses [pre-commit](http://pre-commit.com/), so make sure to install it before making any changes.
```bash
    pip install pre-commit
    cd disk-health-predictor
    pre-commit install
```


## Note

This project has been set up using PyScaffold 4.0.2. For details and usage
information on PyScaffold see https://pyscaffold.org/.
