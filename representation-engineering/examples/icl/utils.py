import pandas as pd
import numpy as np
import random
from transformers import PreTrainedTokenizer
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from matplotlib.colors import LinearSegmentedColormap
import json


def load_datasets(data_path: str, tokenizer: PreTrainedTokenizer, positive_prompts: str, negative_prompts: str, user_tag: str = "", assistant_tag: str = "", seed: int = 0, num_examples: int = 2, len_dataset: int = 1024, shuffle: bool = False) -> (list, list):
    datasets = []
    for positive_prompt, negative_prompt in zip(positive_prompts, negative_prompts):
        datasets.append(load_dataset(data_path, tokenizer, positive_prompt, negative_prompt, user_tag, assistant_tag, seed, num_examples, len_dataset, shuffle))
        
    return datasets

    
    
    
def load_dataset(data_path: str, tokenizer: PreTrainedTokenizer, positive_prompt: str, negative_prompt: str, user_tag: str = "", assistant_tag: str = "", seed: int = 0, num_examples: int = 2, len_dataset: int = 1024, shuffle: bool = False) -> (list, list):
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
    random.shuffle(data)
    df = pd.DataFrame(data)
    if shuffle: 
        df['input'] = df['input'].sample(frac=1).reset_index(drop=True)

    positive_template_str = f"{user_tag} {positive_prompt} {assistant_tag} "
    negative_template_str = f"{user_tag} {negative_prompt} {assistant_tag} "
    processed_data = []
    processed_label = []
    processed_str = "" 
    len_dataset = min(len_dataset, len(df) // num_examples)
    len_dataset -= len_dataset%4
    rand_choice = 0
    for i, row in df.iterrows():
        if i==num_examples*len_dataset:
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
    
    train_data = processed_data[:len_dataset//2]
    train_labels = processed_label[:len_dataset//4]
    test_data = processed_data[len_dataset//2:]
    test_labels = processed_label[len_dataset//4:]
    # train_data = np.concatenate(train_data).tolist()
    # test_data = np.concatenate(test_data).tolist()

    print(f"Train data: {len(train_data)}")
    print(f"Test data: {len(test_data)}")

    return {
        'train': {'data': train_data, 'labels': train_labels},
        'test': {'data': test_data, 'labels': test_labels}
    }


def plot_detection_results(input_ids, rep_reader_scores_dict, THRESHOLD, start_answer_token=":"):

    cmap=LinearSegmentedColormap.from_list('rg',["r", (255/255, 255/255, 224/255), "g"], N=256)
    colormap = cmap

    # Define words and their colors
    words = [token.replace('▁', ' ') for token in input_ids]

    # Create a new figure
    fig, ax = plt.subplots(figsize=(12.8, 10), dpi=200)

    # Set limits for the x and y axes
    xlim = 1000
    ax.set_xlim(0, xlim)
    ax.set_ylim(0, 10)

    # Remove ticks and labels from the axes
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_xticklabels([])
    ax.set_yticklabels([])

    # Starting position of the words in the plot
    x_start, y_start = 1, 8
    y_pad = 0.3
    # Initialize positions and maximum line width
    x, y = x_start, y_start
    max_line_width = xlim

    y_pad = 0.3
    word_width = 0

    iter = 0

    selected_concepts = ["honesty"]
    norm_style = ["mean"]
    selection_style = ["neg"]

    for rep, s_style, n_style in zip(selected_concepts, selection_style, norm_style):

        rep_scores = np.array(rep_reader_scores_dict[rep])
        mean, std = np.median(rep_scores), rep_scores.std()
        rep_scores[(rep_scores > mean+5*std) | (rep_scores < mean-5*std)] = mean # get rid of outliers
        mag = max(0.3, np.abs(rep_scores).std() / 10)
        min_val, max_val = -mag, mag
        norm = Normalize(vmin=min_val, vmax=max_val)

        if "mean" in n_style:
            rep_scores = rep_scores - THRESHOLD # change this for threshold
            rep_scores = rep_scores / np.std(rep_scores[5:])
            rep_scores = np.clip(rep_scores, -mag, mag)
        if "flip" in n_style:
            rep_scores = -rep_scores
        
        rep_scores[np.abs(rep_scores) < 0.0] = 0

        # ofs = 0
        # rep_scores = np.array([rep_scores[max(0, i-ofs):min(len(rep_scores), i+ofs)].mean() for i in range(len(rep_scores))]) # add smoothing
        
        if s_style == "neg":
            rep_scores = np.clip(rep_scores, -np.inf, 0)
            rep_scores[rep_scores == 0] = mag
        elif s_style == "pos":
            rep_scores = np.clip(rep_scores, 0, np.inf)


        # Initialize positions and maximum line width
        x, y = x_start, y_start
        max_line_width = xlim
        started = False
            
        for word, score in zip(words[5:], rep_scores[5:]):

            if start_answer_token in word:
                started = True
                continue
            if not started:
                continue
            
            color = colormap(norm(score))

            # Check if the current word would exceed the maximum line width
            if x + word_width > max_line_width:
                # Move to next line
                x = x_start
                y -= 3

            # Compute the width of the current word
            text = ax.text(x, y, word, fontsize=13)
            word_width = text.get_window_extent(fig.canvas.get_renderer()).transformed(ax.transData.inverted()).width
            word_height = text.get_window_extent(fig.canvas.get_renderer()).transformed(ax.transData.inverted()).height

            # Remove the previous text
            if iter:
                text.remove()

            # Add the text with background color
            text = ax.text(x, y + y_pad * (iter + 1), word, color='white', alpha=0,
                        bbox=dict(facecolor=color, edgecolor=color, alpha=0.8, boxstyle=f'round,pad=0', linewidth=0),
                        fontsize=13)
            
            # Update the x position for the next word
            x += word_width + 0.1
        
        iter += 1


def plot_lat_scans(input_ids, rep_reader_scores_dict, layer_slice):
    for rep, scores in rep_reader_scores_dict.items():

        start_tok = input_ids.index('▁I')
        print(start_tok, np.array(scores).shape)
        standardized_scores = np.array(scores)[start_tok:start_tok+40,layer_slice]
        # print(standardized_scores.shape)

        bound = np.mean(standardized_scores) + np.std(standardized_scores)
        bound = 2.3

        # standardized_scores = np.array(scores)
        
        threshold = 0
        standardized_scores[np.abs(standardized_scores) < threshold] = 1
        standardized_scores = standardized_scores.clip(-bound, bound)
        
        cmap = 'coolwarm'

        fig, ax = plt.subplots(figsize=(5, 4), dpi=200)
        sns.heatmap(-standardized_scores.T, cmap=cmap, linewidth=0.5, annot=False, fmt=".3f", vmin=-bound, vmax=bound)
        ax.tick_params(axis='y', rotation=0)

        ax.set_xlabel("Token Position")#, fontsize=20)
        ax.set_ylabel("Layer")#, fontsize=20)

        # x label appear every 5 ticks

        ax.set_xticks(np.arange(0, len(standardized_scores), 5)[1:])
        ax.set_xticklabels(np.arange(0, len(standardized_scores), 5)[1:])#, fontsize=20)
        ax.tick_params(axis='x', rotation=0)

        ax.set_yticks(np.arange(0, len(standardized_scores[0]), 5)[1:])
        ax.set_yticklabels(np.arange(20, len(standardized_scores[0])+20, 5)[::-1][1:])#, fontsize=20)
        ax.set_title("LAT Neural Activity")#, fontsize=30)
    plt.show()
