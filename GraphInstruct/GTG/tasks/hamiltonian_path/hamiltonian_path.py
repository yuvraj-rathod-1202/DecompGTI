from GTG.utils.utils import NID, make_sample, generate_Hamiltonian_path_graph, edge_list_str_to_graph
from GTG.utils.evaluation import NodeListEvaluator

import networkx as nx
import random
import numpy as np


TASK_NAME = 'hamiltonian_path'


def generate_a_sample(config):
    g, path = generate_Hamiltonian_path_graph(config)
    ans_str = NID(path)

    templates = [
        "Find a Hamiltonian path in this graph. A Hamiltonian path in a graph is a path that visits each node exactly once, traversing along edges, and the starting node and ending node may be different. ",
        "Determine a Hamiltonian path in the given graph. A Hamiltonian path visits each node exactly once, following the edges, with the starting and ending nodes potentially being distinct. ",
        "Identify a Hamiltonian path within this network. A Hamiltonian path ensures every node is visited precisely once, moving along the connections, where the initial and final nodes can vary. ",
        "Locate a Hamiltonian path in the provided graph. A Hamiltonian path is defined as a route that visits each node exactly once, traversing through edges, and the starting and ending nodes may differ. ",
        "Discover a Hamiltonian path in this graph. A Hamiltonian path is one that visits"
    ]
    ques_str = random.choice(templates)

    sample = make_sample(
        TASK_NAME, g, ques_str, ans_str
    )
    return sample


class Evaluator(NodeListEvaluator):

    def __init__(self):
        super().__init__()

    def check_correctness(self, sample, output_ans):
        g = edge_list_str_to_graph(sample['graph'], directed=sample['directed'])
        if len(set(output_ans)) != g.number_of_nodes() or len(output_ans) != g.number_of_nodes():
            return False

        correct = True
        for i in range(1, len(output_ans)):
            if not g.has_edge(output_ans[i - 1], output_ans[i]):
                correct = False
                break
        return correct
