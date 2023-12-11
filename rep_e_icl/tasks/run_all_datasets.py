import numpy as np
import matplotlib.pyplot as plt
from tasks import get_task_dataset
from tasks_n_prompts import DATASET_NAMES

if __name__ == "__main__": 
    positive_prompt = "Pay attention to the following examples."
    negative_prompt = "Don't pay attention to the following exmaples."
    ntrain=64
    for task_name in DATASET_NAMES: 
        dataset = get_task_dataset(task_name, positive_prompt, negative_prompt, ntrain=ntrain)