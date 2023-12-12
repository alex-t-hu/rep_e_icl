#!/bin/bash
cd rethinking-demonstrations/preprocess
# Define the list of data items
data_list=("rotten_tomatoes")

# Loop through each item in the list
for data in "${data_list[@]}"; do
    # Call the Python script with the current item
    python3 ${data}.py --do_train
    python3 ${data}.py --do_test --test_k 64
done
