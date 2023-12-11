import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline, PreTrainedTokenizer
import numpy as np
from itertools import islice
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from matplotlib.colors import LinearSegmentedColormap
from tqdm import tqdm
def get_rep_reader(
        model, 
        rep_pipeline,
        dataset,
        n_components = 1,
        rep_token = -1,
        max_length = 2048,
        n_difference = 1,
        direction_method = 'pca',
        batch_size = 32,
        seed =0, 
):
    hidden_layers = list(range(-1, -model.config.num_hidden_layers, -1))
    direction_finder_kwargs= {"n_components": n_components}

    rep_reader = rep_pipeline.get_directions(
        dataset['train']['data'], 
        rep_token=rep_token, 
        hidden_layers=hidden_layers, 
        n_difference=n_difference, 
        train_labels=dataset['train']['labels'], 
        direction_method=direction_method,
        direction_finder_kwargs=direction_finder_kwargs,
        batch_size=batch_size,
        max_length=max_length,
        # padding="longest",
    )

    return rep_reader 

def get_hidden_layers(model): 
    return list(range(-1, -model.config.num_hidden_layers, -1))

def get_h_test(model, rep_pipeline, rep_reader, dataset, rep_token = -1, batch_size=32): 
    hidden_layers = get_hidden_layers(model)
    return rep_pipeline(
        dataset['test']['data'], 
        rep_token=rep_token, 
        hidden_layers=hidden_layers, 
        rep_reader=rep_reader,
        batch_size=batch_size)

def plot_correlation(rep_reader, H_tests, hidden_layers): 
    results = {layer: {} for layer in hidden_layers}
    rep_readers_means = {layer: 0 for layer in hidden_layers}

    for layer in hidden_layers:
        H_test = [H[layer] for H in H_tests]
        rep_readers_means[layer] = np.mean(H_test)
        H_test = [H_test[i:i+2] for i in range(0, len(H_test), 2)]
        
        sign = rep_reader.direction_signs[layer]

        eval_func = min if sign == -1 else max
        cors = np.mean([eval_func(H) == H[0] for H in H_test])
        
        results[layer] = cors

    plt.plot(hidden_layers, [results[layer] for layer in hidden_layers])
    plt.show()
    
    
def get_rep_control(model, tokenizer, block_name="decoder_block", control_method="reading_vec"): 
    layer_id = list(range(-5, -18, -1))

    block_name="decoder_block"
    control_method="reading_vec"

    return pipeline(
        "rep-control", 
        model=model, 
        tokenizer=tokenizer, 
        layers=layer_id, 
        control_method=control_method)
    

def get_rep_controlled_results(rep_reader,rep_control_pipeline, inputs, layer_id, coeff=2.0, max_new_tokens=3, device="cuda"): 

    activations = {}
    for layer in layer_id:
        activations[layer] = torch.tensor(coeff * rep_reader.directions[layer] * rep_reader.direction_signs[layer]).to(device).half()
    neg_activations = {} 
    for layer in layer_id:
        neg_activations[layer] = torch.tensor(-coeff * rep_reader.directions[layer] * rep_reader.direction_signs[layer]).to(device).half()

    baseline_outputs = rep_control_pipeline(inputs, batch_size=4, max_new_tokens=max_new_tokens, do_sample=False)
    print("Done with baseline results!")
    control_outputs = rep_control_pipeline(inputs, activations=activations, batch_size=4, max_new_tokens=max_new_tokens, do_sample=False, repetition_penalty=1.1)
    print("Done with control outputs!")
    neg_control_outputs = rep_control_pipeline(inputs, activations=neg_activations, batch_size=4, max_new_tokens=max_new_tokens, do_sample=False, repetition_penalty=1.1)
    print("Done with neg control outputs")

    baseline_next_toks = [] 
    control_next_toks = []
    neg_control_next_toks = [] 
    for i, (input, b, c, n) in enumerate(zip(inputs, baseline_outputs, control_outputs, neg_control_outputs)): 
        baseline_next_toks.append(b[0]['generated_text'].replace(input, "").strip(' \n\t'))
        control_next_toks.append(c[0]['generated_text'].replace(input, "").strip(' \n\t'))
        neg_control_next_toks.append(n[0]['generated_text'].replace(input, "").strip(' \n\t'))
        

    return baseline_next_toks, control_next_toks, neg_control_next_toks
            
def get_acc_dict(baseline, pos_results, neg_results, ground_truth): 
    baseline_accs = 0 
    pos_accs = 0
    neg_accs = 0
    for b, p, n, g in zip(baseline, pos_results, neg_results, ground_truth): 
        if b.find(g) !=-1: 
            baseline_accs +=1 
        if p.find(g) != -1: 
            pos_accs+=1 
        if n.find(g) != -1: 
            neg_accs+= 1 
    return {
            "baseline_acc": round(baseline_accs / len(baseline), 3), 
            "pos_acc": round(pos_accs / len(baseline), 3), 
            "neg_acc": round(neg_accs / len(baseline), 3), 
            }
    
    
def get_test_data(model, tokenizer, test_input, prompt="", user_tag="", assistant_tag=""): 
        
    template_str = '{user_tag} {prompt} {assistant_tag}  {scenario} '
    test_input = [template_str.format(prompt=prompt, user_tag=user_tag, assistant_tag=assistant_tag, scenario=s) for s in test_input]

    test_data = []
    for t in test_input:
        with torch.no_grad():
            output = model.generate(**tokenizer(t, return_tensors='pt').to(model.device), max_new_tokens=30)
        completion = tokenizer.decode(output[0], skip_special_tokens=True)
        test_data.append(completion)
    return test_data
        
def get_rep_reader_scores_dict(model, tokenizer, rep_reading_pipeline, rep_reader, test_data, wanted_layers=None, chosen_idx=0, name="placeholder"): 
    hidden_layers = get_hidden_layers(model)
    chosen_str = test_data[chosen_idx]
    input_ids = tokenizer.tokenize(chosen_str)

    results = []

    for ice_pos in range(len(input_ids)):
        ice_pos = -len(input_ids) + ice_pos
        H_tests = rep_reading_pipeline([chosen_str],
                                    rep_reader=rep_reader,
                                    rep_token=ice_pos,
                                    hidden_layers=hidden_layers)
        results.append(H_tests)

    scores = []
    scores_means = []
    for pos in range(len(results)):
        tmp_scores = []
        tmp_scores_all = []
        for layer in hidden_layers:
            tmp_scores_all.append(results[pos][0][layer][0] * rep_reader.direction_signs[layer][0])
            if wanted_layers and layer in wanted_layers:
                tmp_scores.append(results[pos][0][layer][0] * rep_reader.direction_signs[layer][0])
        scores.append(tmp_scores_all)
        scores_means.append(np.mean(tmp_scores))
        
    rep_reader_scores_dict = {name: scores}
    rep_reader_scores_mean_dict = {name: scores_means} 
    return rep_reader_scores_dict, rep_reader_scores_mean_dict
        

def plot_detection_results(input_ids, rep_reader_scores_dict, THRESHOLD, start_answer_token=":", name="placeholder"):

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

    selected_concepts = [name]
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
