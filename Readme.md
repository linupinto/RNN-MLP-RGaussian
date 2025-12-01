# Vanilla RNN on UCI HAR Dataset

##  Description
This repository contains a **Vanilla RNN implementation** trained on the **UCI HAR Dataset** using **Mini-Batch Stochastic Gradient Descent (SGD)**. The model performs **activity recognition** with six output classes corresponding to:

- WALKING  
- WALKING_UPSTAIRS  
- WALKING_DOWNSTAIRS  
- SITTING  
- STANDING  
- LAYING

## Dataset
The details and download link for the dataset are provided in [Dataset_README.md](Dataset_README.md).

##  Key Features
- Vanilla RNN with **arbitrary number of hidden layers**  
- Six-class output  
- Implemented with custom activation functions:  
  - `R_Gaussian.py`  
  - `Tanh.py`  
- Training using **Mini-Batch SGD** for efficient learning  




