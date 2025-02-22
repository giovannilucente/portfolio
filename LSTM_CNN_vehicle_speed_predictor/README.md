# Vehicle speed predictor based on LSTM-CNN hybrid neural network architecture

This repository contains code for training a model that predicts vehicle speed using a LSTM-CNN hybrid network architecture. Follow the steps below to set up and run the training process.

## Installation and Setup

To install and run this project locally, follow these steps:

### 1. Clone the repository
First, clone the repository to your local machine:
```bash
git clone https://github.com/giovannilucente/portfolio/tree/main/LSTM_CNN_vehicle_speed_predictor.git
cd LSTM_CNN_vehicle_speed_predictor
```
### 2. Install the requirements
```bash
pip install -r requirements.txt
```

### 3. Download the NGSIM dataset
```bash
cd NGSIM
# Download here the trajectory dataset in .txt format
cd ..
```

### 4. Convert the dataset
```bash
python3 NGSIM_dataset_converter.py
```
A file called "NGSIM_data.pkl", containing the converted dataset, will be created.

### 5. Launch the training
```bash
python3 training.py
```
The best model will be saved in a folder called "model".
