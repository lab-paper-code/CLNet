# [Submission] osa_severity_classification (IEEE Access 2023)
A Deep Neural Network Based Wake-After-Sleep-Onset Time Aware Sleep Apnea Severity Estimation Scheme Using Single-lead ECG Data

## Dataset
[PhysioNet Apnea-ECG Database 1.0.0](https://physionet.org/content/apnea-ecg/1.0.0/)

## Usage
The trained model is provided as `CLNet.h5`.

'train_data', 'val_data', and 'test_data' are required to reproduce our results.

You must download the Apnea-ECG Database from the link. 

Create the pre_processing_data directory and then execute `preprocessing.py`.

Results are obtained by running `CLNet.py`.

*Note: Depending on the version of Keras and Tensorflow, the specification of the GPU, there may be some discrepancies with the actual result.*

## Requirements
Python==3.7.9 Keras==2.5.0 Tensorflow==2.5.0
