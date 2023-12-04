# QuaternionPredictionAR
Prediction of qunaternions using the AR model 

## Introduction 
This repo provides a brief overview of selected available methods for predicting quaternions from time series. Propose another method base on the Yule-Walker algorithm.

## AR Model
AR model can be represent by following structures

![model_AR](https://github.com/BezdzietnyAntoni/QuatrnionPredictionAR/assets/84715548/9eed6ebf-579d-4bfc-894a-cdeb638ea8d6)

$\hat y(n) = [1-A(q^{-1})]y(n) + s(n)$

$\sum_{p=0} a_p y(n-P) = s(n)$

, where $a_0=1$.

Target minimalize prediction error $min(y(n) - \hat y(n))$.

## Quaternions
A quaternion is an expression of the form. where $q_0$, $q_1$, $q_2$, $q_3$, are real numbers, and i, j, k, are symbols that can be interpreted as unit-vectors pointing along the three spatial axes.
$$q = q_0 + \mathbf{q} = q_0 + \mathbf{i}q_1+\mathbf{j}q_2+\mathbf{k}q_3$$


## Bibliography
More information you can find in:

James Diebel - *Representing Attitude: Euler Angles, Unit Quaternions, and Rotation*

J. B. Kuipers - *Quaternions and Rotation Sequences*

C. C. Took and D. P. Mandic - *The Quaternion LMS Algorithm for Adaptive Filtering of Hypercomplex Processes*

T. Variddhisai and D. P. Mandic - *On an RLS-Like LMS Adaptive Filter*

M. Wang and W. Ma - *A structure-preserving algorithm for the quaternion Cholesky decomposition*

and more ...
