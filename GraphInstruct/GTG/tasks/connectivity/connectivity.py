from GTG.utils.utils import NID, graph_generation, make_sample
from GTG.utils.evaluation import BoolEvaluator

import networkx as nx
import random
import numpy as np


TASK_NAME = 'connectivity'
_num_sample = 0
_num_yes = 0


def question_generation(config, g):
    ques = {}
    ques_str = "Determine whether the graph is connected."

    templates = [
        "Given the graph, answer this task: {}",
        "Based on the graph structure, solve the following request: {}",
        "Graph analysis prompt: {}",
        "Use graph reasoning to determine the result for: {}",
        "For the shown network, provide the answer to: {}",
    ]

    ques_str = random.choice(templates).format(ques_str)

    return ques, ques_str


def answer_and_inference_steps_generation(config, g, ques):
    steps_str = "Let's solve it step by step. We can check if all nodes are in the same connected component.\n"
    
    if g.is_directed():
        ans = nx.is_weakly_connected(g)
    else:
        ans = nx.is_connected(g)

    if ans:
        ans_str = 'Yes'
        steps_str += "The graph is connected, "
    else:
        ans_str = 'No'
        steps_str += "The graph is not connected, "
        
    steps_str += "so the answer is "
    
    reject = False
    if ans:
        global _num_sample
        global _num_yes
        if _num_yes / (_num_sample + 1) > 0.5:
            reject = True
        else:
            _num_yes += 1
            
    return steps_str, ans, ans_str, reject


def choices_generation(config, g, ques, ans):
    false_ans = []

    choi_str = "[Yes, No]"
    if ans:
        label_str = '0'
    else:
        label_str = '1'
    
    return choi_str, label_str


def generate_a_sample(config):
    g = graph_generation(config)
    ques, ques_str = question_generation(config, g)
    steps_str, ans, ans_str, reject = answer_and_inference_steps_generation(config, g, ques)
    
    while reject:
        g = graph_generation(config)
        ques, ques_str = question_generation(config, g)
        steps_str, ans, ans_str, reject = answer_and_inference_steps_generation(config, g, ques)
   
    choi_str, label_str = choices_generation(config, g, ques, ans)

    sample = make_sample(
        TASK_NAME, g, ques_str, ans_str, steps_str, choi_str, label_str
    )
    global _num_sample
    _num_sample += 1
    return sample


class Evaluator(BoolEvaluator):

    def __init__(self):
        super().__init__()
