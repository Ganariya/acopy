import os
import acopy
import samepy
import tsplib95
import matplotlib.pyplot as plt
import copy

from acopy.plugins import StatsRecorder, DrawGraph, Printout, InitialEdgePheromone, MaxMinPheromoneRestrict
from acopy.utils.plot import Plotter

# K = int(input())
K = 5

# -------------------------------------------------
# 初期グラフの作成
print("init-graph")
problem_path = os.path.join('tsp_model', 'bays29.tsp')
problem = tsplib95.load_problem(problem_path)
graph = problem.get_graph()

colony = acopy.Colony()
solver = acopy.Solver(top=1)
recorder = StatsRecorder('init_data')
drawer = DrawGraph(problem=problem, save_path='init_data', is_save=True)
printer = Printout()
restricter = MaxMinPheromoneRestrict(save_path='init_data')
solver.add_plugins(recorder, drawer, printer, restricter)

init_ans = solver.solve(graph, colony, limit=100)
print(init_ans)

# -------------------------------------------------
# Greedy解の構築
print("greedy min k-path")
greedy_graph = copy.deepcopy(graph)
for k in range(K):
    print("k-path")
    colony = acopy.Colony()
    solver = acopy.Solver()
    drawer = DrawGraph(problem=problem, save_path='greedy_data', is_save=True, leading=str(k) + "_greedy", )
    solver.add_plugins(drawer)
    greedy_ans = solver.exploit(greedy_graph, colony, limit=100)
    print(greedy_ans, "\n\n")
    for p in greedy_ans:
        x, y = p[0], p[1]
        greedy_graph.edges[x, y]['pheromone'] = 0
        greedy_graph.edges[y, x]['pheromone'] = 0
        greedy_graph.edges[x, y]['weight'] = 1e20
        greedy_graph.edges[y, x]['weight'] = 1e20

# -------------------------------------------------
# Greedy解の構築(フェロモン更新)
print("greedy min k-path")
greedy_graph = copy.deepcopy(graph)
for k in range(K):
    print("k-path")
    colony = acopy.Colony()
    solver = acopy.Solver()
    drawer = DrawGraph(problem=problem, save_path='greedy_data_update', is_save=True, leading=str(k) + "_greedy", )
    solver.add_plugins(drawer)
    greedy_ans = solver.solve(greedy_graph, colony, limit=100)
    print(greedy_ans, "\n\n")
    for p in greedy_ans:
        x, y = p[0], p[1]
        greedy_graph.edges[x, y]['pheromone'] = 0
        greedy_graph.edges[y, x]['pheromone'] = 0
        greedy_graph.edges[x, y]['weight'] = 1e20
        greedy_graph.edges[y, x]['weight'] = 1e20

# -------------------------------------------------
# グラフの比較

greedy_list = []
update_list = []
T = 10

for i in range(T):
    # Greedy解の構築
    greedy_graph = copy.deepcopy(graph)
    gl1 = []
    for k in range(K):
        colony = acopy.Colony()
        solver = acopy.Solver()
        greedy_ans = solver.exploit(greedy_graph, colony, limit=50)
        gl1.append(greedy_ans.cost)
        for p in greedy_ans:
            x, y = p[0], p[1]
            greedy_graph.edges[x, y]['pheromone'] = 0
            greedy_graph.edges[y, x]['pheromone'] = 0
            greedy_graph.edges[x, y]['weight'] = 1e20
            greedy_graph.edges[y, x]['weight'] = 1e20
    greedy_list.append(gl1)

    # -------------------------------------------------
    # Greedy解の構築(フェロモン更新)

    greedy_graph = copy.deepcopy(graph)
    ul1 = []
    for k in range(K):
        colony = acopy.Colony()
        solver = acopy.Solver()
        greedy_ans = solver.solve(greedy_graph, colony, limit=50)
        ul1.append(greedy_ans.cost)
        for p in greedy_ans:
            x, y = p[0], p[1]
            greedy_graph.edges[x, y]['pheromone'] = 0
            greedy_graph.edges[y, x]['pheromone'] = 0
            greedy_graph.edges[x, y]['weight'] = 1e20
            greedy_graph.edges[y, x]['weight'] = 1e20
    update_list.append(ul1)

greedy_list_ave = [0] * K
update_list_ave = [0] * K
for i in range(T):
    for j in range(K):
        greedy_list_ave[j] += greedy_list[i][j]
        update_list_ave[j] += update_list[i][j]
for i in range(K):
    greedy_list_ave[i] /= T
    update_list_ave[i] /= T

X = range(1, K + 1)

fig = plt.figure(dpi=200)
ax = fig.add_subplot(1, 1, 1)
ax.plot(X, greedy_list_ave, label='greedy')
ax.plot(X, update_list_ave, label='update')
ax.set_title("K min")
ax.legend()
fig.savefig('greedy_data/graph.png')
