# deepshape

** NB: This repository will undergo changes within the next few weeks, to make the code for  the coming publication as easily accessible as possible **

This repository contains source code for a deep reparametrization algorithm for reparametrization of parametric curves and surfaces. It was created as part of my masters thesis in Industrial Mathemetics at NTNU.

## Setup
This setup assumes anaconda-python. For `pip` it should (hopefully) suffice to install pytorch and matplotlib, as these in turn should install any additional dependencies.

To replicate environment
```
conda env create -f environment.yml
```

To install deepshape for development
```
pip install -e .
```
where pip may be installed through conda.


See the following notebooks for example usage.
1. [Curve Reparametrization](example-notebooks/curves-reparametrization.ipynb)
2. [Surface Reparametrization](example-notebooks/surfaces-reparametrization.ipynb)


## Datasets
ECG Heartbeats (Kaggle): https://www.kaggle.com/shayanfazeli/heartbeat
