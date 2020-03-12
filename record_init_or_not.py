import warnings
import matplotlib

warnings.filterwarnings('ignore', category=matplotlib.MatplotlibDeprecationWarning)
warnings.filterwarnings('ignore', category=UserWarning)

import os
import acopy
import samepy
import tsplib95
import networkx as nx
import matplotlib.pyplot as plt
import copy
import json

from acopy.plugins import StatsRecorder, DrawGraph, Printout, InitialEdgePheromone, MaxMinPheromoneRestrict
from acopy.utils.plot import Plotter

# K = int(input())
K = 4

# -------------------------------------------------
# 初期グラフの作成
print("init-graph")
problem_path = os.path.join('tsp_model', 'bays29.tsp')
problem = tsplib95.load_problem(problem_path)
graph = problem.get_graph()
labels = {i: str(i) for i in graph.nodes()}

colony = acopy.Colony()
solver = acopy.Solver(top=1)
recorder = StatsRecorder('init_data')
printer = Printout()
restricter = MaxMinPheromoneRestrict(save_path='init_data')
solver.add_plugins(recorder, printer, restricter)

init_ans = solver.solve(graph, colony, limit=1000)
print(init_ans)
print("\n\n\n")


def draw_graph(G, path, title, is_save=False, save_path=""):
    plt.figure(dpi=400)
    _, ax = plt.subplots()
    pos = problem.display_data or problem.node_coords
    nx.draw_networkx_nodes(G, pos=pos, ax=ax)
    nx.draw_networkx_edges(G, pos=pos, edgelist=path, arrows=False)
    nx.draw_networkx_labels(G, pos=pos, labels=labels, font_color='w')
    ax.tick_params(left=True, bottom=True, labelleft=True, labelbottom=True)
    ax.set_title(title)
    if is_save:
        plt.savefig(save_path)
    plt.show()


res_data = {}
res_data['average'] = []
res_data['nothing'] = []
for __ in range(30):

    # -------------------------------------------------
    # make average graph

    print("make average graph")
    colony = samepy.Colony()
    solver = samepy.Solver()
    ave_graph = copy.deepcopy(graph)

    average_solutions = solver.solve(ave_graph, colony, limit=1000, gen_size=K, problem=problem)
    cnt = 0
    costs = []
    for sol in average_solutions:
        print(sol)
        path = sol.path
        title = f"average  cost {sol.cost}"
        print(path)
        costs.append(sol.cost)
        cnt += 1
    print("\n\n\n")
    res_data['average'].append(costs)

    # -------------------------------------------------
    # もし初期のフェロモンがないときのグラフ
    print("もしも初期フェロモンないとき")

    nothing_graph = problem.get_graph()
    for u, v in nothing_graph.edges:
        nothing_graph.edges[u, v].setdefault('pheromone', 1)
    colony = samepy.Colony()
    solver = samepy.Solver()

    init_nothing_solutions = solver.solve(nothing_graph, colony, limit=1000, gen_size=K, problem=problem)
    cnt = 0
    costs = []
    for sol in init_nothing_solutions:
        print(sol)
        path = sol.path
        title = f"nothing average  cost {sol.cost}"
        costs.append(sol.cost)
        print(path)
        cnt += 1
    print("\n\n\n")
    res_data['nothing'].append(costs)
    print(res_data)

with open('init_or_not.json', 'w') as f:
    json.dump(res_data, f)
