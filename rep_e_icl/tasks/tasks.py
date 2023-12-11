import numpy as np
import json
import random
import pandas as pd

def load_dataset(data_path: str, positive_prompt: str, negative_prompt: str, user_tag: str = "", assistant_tag: str = "", seed: int = 0, num_examples: int = 2, len_dataset=None, shuffle: bool = False, overall_prompt="", is_test=False) -> (list, list):
    """
    Processes data to create training and testing datasets based on honesty.

    Args:
    - data_path (str): Path to the CSV containing the data.
    - tokenizer (PreTrainedTokenizer): Tokenizer to tokenize statements.
    - user_tag (str): Instruction template.
    - assistant_tag (str): Instruction template user tag.
    - seed (int): Random seed for reproducibility.
    - num_examples (int): the number of examples we show the model
    - len_dataset (int): upper bound on the number of training and test examples used, it's always half training half testing
    - positive_prompt (str): An instruction template used to stimulate the model to perform in-context learning
    - negative_prompt (str): An instruction template used to anti-stimulate the model to perform in-context learning
    
    Returns:
    - Tuple containing train and test data.
    - The labels are True for the positive prompt and False for the negative
    """

    # Setting the seed for reproducibility
    random.seed(seed)

    # Load the data
    data_str = open(data_path, 'r').read()
    data = [json.loads(line) for line in data_str.strip().split('\n')]
    df = pd.DataFrame(data)
    options = df.iloc[0]["options"]
    print(options)
    if len(options) > 2: 
        df = df[~df["output"].isin(["no_impact", "neutral", "none"])]
    if shuffle: 
        df['input'] = df['input'].sample(frac=1).reset_index(drop=True)

    positive_template_str = f"{overall_prompt}{user_tag} {positive_prompt} {assistant_tag} "
    negative_template_str = f"{overall_prompt}{user_tag} {negative_prompt} {assistant_tag} "
    
    processed_data = []
    processed_label = []
    ground_truth = []
    processed_str = "" 
    if len_dataset: 
        len_dataset = min(len_dataset, len(df) // num_examples)
    else: 
        len_dataset = len(df) // num_examples
    len_dataset -= len_dataset%2
    rand_choice = 0
    for i, row in df.iterrows():
            
        if len_dataset and i==num_examples*len_dataset:
            break
        if i%(2*num_examples) == 0:
            rand_choice = random.choice([0,1])
            if rand_choice == 0: 
                processed_str = positive_template_str
            else:
                processed_str = negative_template_str
        elif i%num_examples == 0:
            if rand_choice == 0:
                processed_str = negative_template_str
            else:
                processed_str = positive_template_str
        
        if is_test: 
            if (i+1)%num_examples != 0: 
                processed_str += row['input'] + "\n" + row['output'] + "\n"
            else: 
                processed_str += row['input'] + "\n"
                ground_truth.append(row['output'])
        else: 
            processed_str += row['input'] + "\n" + row['output'] + "\n"

        if (i+1)%num_examples == 0:
            processed_data.append(processed_str)
        if (i+1)%(2*num_examples) == 0:
            processed_label.append([bool(1-rand_choice), bool(rand_choice)])

    print(f"data len: {len(processed_data)}")

    return  {'data': processed_data, 'labels': processed_label}




def load_test_dataset(data_path: str, user_tag: str = "", assistant_tag: str = "", seed: int = 0, num_examples: int = 2, len_dataset=None, shuffle: bool = False, overall_prompt="") -> (list, list):
    """
    Processes data to create training and testing datasets based on honesty.

    Args:
    - data_path (str): Path to the CSV containing the data.
    - tokenizer (PreTrainedTokenizer): Tokenizer to tokenize statements.
    - user_tag (str): Instruction template.
    - assistant_tag (str): Instruction template user tag.
    - seed (int): Random seed for reproducibility.
    - num_examples (int): the number of examples we show the model
    - len_dataset (int): upper bound on the number of training and test examples used, it's always half training half testing
    - positive_prompt (str): An instruction template used to stimulate the model to perform in-context learning
    - negative_prompt (str): An instruction template used to anti-stimulate the model to perform in-context learning
    
    Returns:
    - Tuple containing train and test data.
    - The labels are True for the positive prompt and False for the negative
    """

    # Setting the seed for reproducibility
    random.seed(seed)
    template_str = f"{overall_prompt}{user_tag} {assistant_tag} "
    # Load the data
    data_str = open(data_path, 'r').read()
    data = [json.loads(line) for line in data_str.strip().split('\n')]
    df = pd.DataFrame(data)
    options = df.iloc[0]["options"]
    print(options)
    if len(options) > 2: 
        df = df[~df["output"].isin(["no_impact", "neutral", "none"])]
    if shuffle: 
        df['input'] = df['input'].sample(frac=1).reset_index(drop=True)
    
    processed_data = []
    labels = []
    processed_str = "" 
    if len_dataset: 
        len_dataset = min(len_dataset, len(df) // num_examples)
    else: 
        len_dataset = len(df) // num_examples
    for i, row in df.iterrows():
        if len_dataset and i==num_examples*len_dataset:
            break
        if i%num_examples == 0:
            processed_str = template_str
        
        if (i+1)%num_examples != 0: 
            processed_str += row['input'] + "\n" + row['output'] + "\n"
        else: 
            processed_str += row['input'] + "\n"
            labels.append(row['output'])

        if (i+1)%num_examples == 0:
            processed_data.append(processed_str)

    print(f"data len: {len(processed_data)}")
    return {'data': processed_data, "labels": labels}


def get_task_dataset(dataset_name, positive_prompt, negative_prompt, user_tag="[INST]", assistant_tag="[/INST]", ntrain=64, test_num_examples=2): 
    print(f"getting dataset for {dataset_name}")
    if ntrain <= 64: 
        train_data_path = f"data/{dataset_name}/{dataset_name}_64_100_train.jsonl"
    else: 
        train_data_path = f"data/{dataset_name}/{dataset_name}_16384_100_train.jsonl"
    dev_data_path = f"data/{dataset_name}/{dataset_name}_64_100_dev.jsonl"
    test_data_path = f"data/{dataset_name}/{dataset_name}_64_100_test.jsonl"
    train_data = load_dataset(train_data_path, positive_prompt, negative_prompt, user_tag, assistant_tag, len_dataset=ntrain)
    dev_data = load_dataset(dev_data_path, positive_prompt, negative_prompt, user_tag, assistant_tag)
    test_data = load_test_dataset(test_data_path, user_tag, assistant_tag, num_examples=test_num_examples)
    return {
        "train": train_data, 
        "val": dev_data, 
        "test": test_data
    }