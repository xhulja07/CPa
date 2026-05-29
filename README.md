On the Use of Conformal Prediction to Assist the Identification of Correct Change-level Fault Predictions

Getting started:
This repository contains all the experimental components, data, and implementation details pertaining to the work “On the Use of Conformal Prediction to Assist the Identification of Correct Change-level Fault Predictions”

The repository contains 5 folders.
The first 4 folders, named EX<Nr>, contain the experimental data of each experiment, including: 
* the raw data of each experimental setting, for each objective 
* the summarized data across the experimental runs, for each experimental setting, and for each objective; and 
* the statistical significance results. 
The folder named “Implementation” contains the implementation details for each of the experiments, including the statistical tests.

Experimental Components
The conducted experiments leverage existing change-level fault prediction (CL-FP) models, whose implementation has been made publicly available by the corresponding authors.  To train the CL-FP models, we followed the instructions outlined in the corresponding repository.

CL-FP Models:

DeepJIT: https://github.com/soarsmu/DeepJIT/tree/master
LApredict: https://github.com/ZZR0/ISSTA21-JIT-DP
CodeBERT4JIT: https://github.com/Xin-Zhou-smu/Assessing-generalizability-of-CodeBERT

DATASETS:
We conducted our experiments with the QT and OPENSTACK datasets, which can are publicly available under: https://zenodo.org/records/3965246#.XyEDVnUzY5k

CP Models:
To carry out our experiments, we train CP models using the open-source implementation provided by Angelopoulos, Anastasios N & Bates, Stephen: https://github.com/aangelopoulos/conformal-prediction. 
In the first two experimental iterations, we apply inductive CP. In the final two iterations, we adopt class-conditional CP.
The implementation of the CP adaptations: CPa and CC-CPa, can be found in the Implementation folder, inside EX1 (see marginal_CPa.py) and EX3 (see Class_Conditional_CPa.py), respectively.

