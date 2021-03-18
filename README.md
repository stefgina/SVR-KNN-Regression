# Stock Prediction
```python
Author-> Stefanos Ginargyros
```
## Dependencies
In order to run the script, you will have to install the following dependencies:

```
pip install matplotlib
pip install pandas
pip install scikit-learn
pip install numpy
```

## Problem Description
The dataset for this particular problem is the stock values of FACEBOOK (NASDAQ:FB) for the month May in 2019, but you can choose whatever stock or sequential entity you want. The task is to predict the [open] stock price for the last day of May, based on the previous stock values of the same month.

<img src="https://github.com/stefgina/stock-prediction-svm-regression/blob/main/dataset.png">

## Algorithms
The following algorithms were used for this purpose :

- SVR with Linear Kernel
- SVR with Polyonomial Kernel
- SVR with RBF Kernel
- K-Neighbors

## Benchmarks
The regression benchmarks are depicted here:

<img src="https://github.com/stefgina/stock-prediction-svm-regression/blob/main/curves.png">






