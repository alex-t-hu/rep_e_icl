#!/bin/bash
cd rethinking-demonstrations/preprocess
# Define the list of data items
data_list=("ade_classification" "poem_sentiment" "glue_rte" "sick" "glue_mrpc" "tweet_eval" "financial_phrasebank" "poem_sentiment" "glue_wnli" "climate_fever" "glue_rte" "superglue_cb" "sick" "medical_questions_pairs" "glue_mrpc" "hate_speech18" "ethos")

# Loop through each item in the list
for data in "${data_list[@]}"; do
    # Call the Python script with the current item
    python ${data}.py --do_train
    python ${data}.py --do_test --test_k 64
done
