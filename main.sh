#!/bin/bash
wget https://archive.ics.uci.edu/ml/machine-learning-databases/00226/OpportunityUCIDataset.zip
mv OpportunityUCIDataset.zip data/raw
# Download the raw data from the website
python preprocess.py -d opportunity_5imu -D opportunity/opportunity_5imu
# Preprocess the raw data
python graph.py -d opportunity_5imu -l opportunity_5imu
# Generate data pattern based graph
python main.py -w 64 -t 64
# Train and test our method.