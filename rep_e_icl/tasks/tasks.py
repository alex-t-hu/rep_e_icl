import numpy as np
import json
import random
import pandas as pd

TASKS_TO_PROMPTS = { 
                   
"ade_corpus_v2-classification": "Answer whether the drug and the adverse effect are related. ",
"rotten_tomatoes" : "Predict the sentiment of the following movie review. ",
"ethos-directed_vs_generalized": "Determine if the statement is directed at an individual or generalized.",
"ethos-disability": "Identify if the statement is about disability discrimination.",
"ethos-gender": "Detect gender bias in the given text.",
"ethos-national_origin": "Classify statements based on national origin bias.",
"ethos-race": "Identify racial bias in the provided statement.",
"ethos-religion": "Determine if the text contains religious discrimination.",
"ethos-sexual_orientation": "Detect bias based on sexual orientation.",
"financial_phrasebank": "Analyze the sentiment of the financial statement.",
"glue-mrpc": "Determine if two sentences are paraphrases of each other.",
"glue-rte": "Recognize textual entailment in the given sentences.",
"glue-wnli": "Understand if one sentence logically follows from another.",
"hate_speech18": "Identify hate speech in the text.",
"medical_questions_pairs": "Decide if two medical questions are semantically equivalent.",
"poem_sentiment": "Analyze the sentiment of the poem.",
"sick": "Determine the relatedness and entailment of sentence pairs.",
"superglue-cb": "Comprehension and entailment in broader contexts.",
"tweet_eval-hate": "Identify hateful content in tweets.",
"tweet_eval-stance_atheism": "Detect stance towards atheism in tweets.",
"tweet_eval-stance_feminist": "Identify feminist stance in tweets.",
"tweet_eval-offensive": "Detect offensive language in tweets."
}


def load_train_dataset_data_aug(data_path: str, tokenizer, positive_prompt: str, negative_prompt: str, user_tag: str = "", assistant_tag: str = "", seed: int = 0, num_examples: int = 2, len_dataset=None, shuffle: bool = False, overall_prompt="", is_test=False) -> (list, list):
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
    processed_str = ""
    for i, row in df.iterrows(): 
        if len_dataset and i == num_examples*len_dataset: 
            break 
        
        if i % num_examples == 0: 
            processed_str = ""
        processed_str += row['input'] + "\n" + row['output'] + "\n"
        if (i+1) % num_examples == 0: 
            processed_data.append(processed_str)
        
        
    
    combined_data = []
    
    for data in processed_data: 
        tokens = tokenizer.tokenize(data)
        if len(tokens) < 10: 
            combined_data.append(
                [positive_template_str+ data, negative_template_str + data]
            )
        else: 
            for idx in range(10, len(tokens)):
                truncated_tokens = tokens[:idx]
                truncated_statement = tokenizer.convert_tokens_to_string(truncated_tokens)
                combined_data.append(
                    [positive_template_str+ truncated_statement, negative_template_str + truncated_statement]
                )
            
    ntrain = 16 * len_dataset
    train_data= combined_data[:ntrain]
    train_labels = []
    for d in train_data:
        true_s = d[0]
        random.shuffle(d)
        train_labels.append([s == true_s for s in d])
        
    train_data = np.concatenate(train_data).tolist()
    print(f"data len: {len(train_data)}")

    return  {'data': train_data, 'labels': train_labels}

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
    template_str = f"{overall_prompt}"
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
            processed_str += f"{user_tag} {row['input']} {assistant_tag}" + "\n" + row['output'] + "\n"
        else: 
            processed_str += f"{user_tag} {row['input']} {assistant_tag}" + "\n"
            labels.append(row['output'])

        if (i+1)%num_examples == 0:
            processed_data.append(f"{processed_str}")

    print(f"data len: {len(processed_data)}")
    return {'data': processed_data, "labels": labels}


def get_baseline_dataset(data_path: str, user_tag: str = "", assistant_tag: str = "", seed: int = 0, num_examples: int = 2, len_dataset=None, shuffle: bool = False, overall_prompt="") -> (list, list):
    
    # Setting the seed for reproducibility
    random.seed(seed)
    template_str = f"{overall_prompt}"
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
        
        if (i+1)%num_examples == 0: 
            processed_str += f"{user_tag} {row['input']} {assistant_tag}" + "\n"
            labels.append(row['output'])

        if (i+1)%num_examples == 0:
            processed_data.append(f"{processed_str}")

    print(f"data len: {len(processed_data)}")
    return {'data': processed_data, "labels": labels}
    


def get_task_dataset(dataset_name, tokenizer, positive_prompt, negative_prompt, user_tag="[INST]", assistant_tag="[/INST]", ntrain=64, test_num_examples=2): 
    print(f"getting dataset for {dataset_name}")
    if ntrain <= 64: 
        train_data_path = f"data/{dataset_name}/{dataset_name}_64_100_train.jsonl"
    else: 
        train_data_path = f"data/{dataset_name}/{dataset_name}_16384_100_train.jsonl"
    dev_data_path = f"data/{dataset_name}/{dataset_name}_64_100_dev.jsonl"
    test_data_path = f"data/{dataset_name}/{dataset_name}_64_100_test.jsonl"
    train_data = load_dataset(train_data_path, positive_prompt, negative_prompt, user_tag=user_tag, assistant_tag=assistant_tag, len_dataset=ntrain)
    dev_data = load_dataset(dev_data_path, positive_prompt, negative_prompt, user_tag, assistant_tag)
    test_data = load_test_dataset(test_data_path, user_tag, assistant_tag, num_examples=test_num_examples)
    no_icl_baseline = get_baseline_dataset(test_data_path, user_tag, assistant_tag, num_examples=test_num_examples, overall_prompt=TASKS_TO_PROMPTS[dataset_name])
    return {
        "train": train_data, 
        "val": dev_data, 
        "test": test_data,
        "baseline": no_icl_baseline
    }