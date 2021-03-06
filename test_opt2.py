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

limit = 1000

init_ans = solver.solve(graph, colony, limit=limit // 4)
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


# -------------------------------------------------
# make average graph

print("make average graph")
colony = samepy.Colony()
solver = samepy.Solver()
ave_graph = copy.deepcopy(graph)

average_solutions = solver.solve(ave_graph, colony, limit=limit, gen_size=K, problem=problem, pheromone_update=True)
cnt = 0
for sol in average_solutions:
    print(sol)
    path = sol.path
    title = f"average  cost {sol.cost}"
    print(path)
    cnt += 1
print("\n\n\n")

# -------------------------------------------------
# pheromone update fix graph

#
print("test opt2 fix update")
colony = samepy.Colony()
solver = samepy.Solver()
update_graph = copy.deepcopy(graph)

average_solutions = solver.solve(update_graph, colony, limit=limit, gen_size=K, problem=problem, pheromone_update=True, is_opt2=True)
cnt = 0
for sol in average_solutions:
    print(sol)
    path = sol.path
    title = f"average  cost {sol.cost}"
    print(path)
    cnt += 1
print("\n\n\n")

# -------------------------------------------------
# pheromone update fix graph

#
print("only opt2")
colony = samepy.Colony()
solver = samepy.Solver()
update_graph = copy.deepcopy(graph)

average_solutions = solver.solve(update_graph, colony, limit=limit, gen_size=K, problem=problem, is_opt2=True)
cnt = 0
for sol in average_solutions:
    print(sol)
    path = sol.path
    title = f"average  cost {sol.cost}"
    print(path)
    cnt += 1
print("\n\n\n")


# -------------------------------------------------
# Greedy解の構築
print("greedy min k-path")
greedy_graph = copy.deepcopy(graph)
greedy_solutions = []
for k in range(K):
    print("k-greedy-path: ", k)
    colony = acopy.Colony()
    solver = acopy.Solver()
    greedy_ans = solver.exploit(greedy_graph, colony, limit=100)
    print(greedy_ans, "\n")
    for p in greedy_ans:
        x, y = p[0], p[1]
        greedy_graph.edges[x, y]['pheromone'] = 0
        greedy_graph.edges[y, x]['pheromone'] = 0
        greedy_graph.edges[x, y]['weight'] = 1e100
        greedy_graph.edges[y, x]['weight'] = 1e100
    greedy_solutions.append(greedy_ans)
print("\n\n\n")

# -------------------------------------------------
#  異なるパスの計算

print("異なるエッジの計算")

greedy_st = set()
same_st = set()
different_st = set()
for sol in greedy_solutions:
    for p in sol:
        x, y = min(p[0], p[1]), max(p[0], p[1])
        greedy_st.add((x, y))
for sol in average_solutions:
    for p in sol:
        x, y = min(p[0], p[1]), max(p[0], p[1])
        if (x, y) not in greedy_st:
            different_st.add((x, y))
        else:
            same_st.add((x, y))

print("共通エッジ", len(same_st), same_st)
print("異なるエッジ", len(different_st), different_st)
print("\n\n\n")

# -------------------------------------------------
# make average graph

print("make average graph with same or different")


def draw_graph_color_st(G, same_path, diff_path, title, is_save=False, save_path=""):
    plt.figure(dpi=400)
    _, ax = plt.subplots()
    pos = problem.display_data or problem.node_coords
    nx.draw_networkx_nodes(G, pos=pos, ax=ax)
    nx.draw_networkx_edges(G, pos=pos, edgelist=same_path, arrows=False, edge_color='blue')
    nx.draw_networkx_edges(G, pos=pos, edgelist=diff_path, arrows=False, edge_color='red')
    ax.tick_params(left=True, bottom=True, labelleft=True, labelbottom=True)
    ax.set_title(title)
    if is_save:
        plt.savefig(save_path)
    plt.show()


cnt = 0
for sol in average_solutions:
    print(sol)
    path = sol.path
    same_path = []
    diff_path = []
    for x, y in path:
        x, y = min(x, y), max(x, y)
        if (x, y) in greedy_st:
            same_path.append((x, y))
        else:
            diff_path.append((x, y))
    title = f"average  cost {sol.cost}"
    draw_graph_color_st(graph, same_path, diff_path, title, is_save=True, save_path=f"average_sample/color_{cnt}.png")
    cnt += 1
