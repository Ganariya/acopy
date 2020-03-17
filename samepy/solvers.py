# -*- coding: utf-8 -*-
import sys
import functools
import collections
import random
import copy
import os

import matplotlib.pyplot as plt
import networkx as nx

from . import utils


@functools.total_ordering
class Solution:
    """Tour for a graph.

    :param graph: a graph
    :type graph: :class:`networkx.Graph`
    :param start: starting node
    :param ant: ant responsible
    :type ant: :class:`~acopy.ant.Ant`
    """

    def __init__(self, graph, start, ant=None):
        self.graph = graph
        self.start = start
        self.ant = ant
        self.current = start
        self.cost = 0
        self.path = []
        self.nodes = [start]
        self.visited = set(self.nodes)

    def __iter__(self):
        return iter(self.path)

    def __eq__(self, other):
        return self.cost == other.cost

    def __lt__(self, other):
        return self.cost < other.cost

    def __contains__(self, node):
        return node in self.visited or node == self.current

    def __repr__(self):
        easy_id = self.get_easy_id(sep=',', monospace=False)
        return '{}\t{}'.format(self.cost, easy_id)

    def __hash__(self):
        return hash(self.get_id())

    def get_easy_id(self, sep=' ', monospace=True):
        nodes = [str(n) for n in self.get_id()]
        if monospace:
            size = max([len(n) for n in nodes])
            nodes = [n.rjust(size) for n in nodes]
        return sep.join(nodes)

    def get_id(self):
        """Return the ID of the solution.

        The default implementation is just each of the nodes in visited order.

        :return: solution ID
        :rtype: tuple
        """
        first = min(self.nodes)
        index = self.nodes.index(first)
        return tuple(self.nodes[index:] + self.nodes[:index])

    def add_node(self, node):
        """Record a node as visited.

        :param node: the node visited
        """
        self.nodes.append(node)
        self.visited.add(node)
        self._add_node(node)

    def close(self):
        """Close the tour so that the first and last nodes are the same."""
        self._add_node(self.start)

    def reconstruct(self):
        n = len(self.nodes)
        self.path = []
        for i in range(n):
            self.path.append((self.nodes[i], self.nodes[(i + 1) % n]))

    def _add_node(self, node):
        edge = self.current, node
        data = self.graph.edges[edge]
        self.path.append(edge)
        self.cost += data['weight']
        self.current = node

    def trace(self, q, rho=0):
        """Deposit pheromone on the edges.

        Note that by default no pheromone evaporates.

        :param float q: the amount of pheromone
        :param float rho: the percentage of pheromone to evaporate
        """
        amount = q / self.cost
        for edge in self.path:
            self.graph.edges[edge]['pheromone'] += amount
            self.graph.edges[edge]['pheromone'] *= 1 - rho
            if not self.graph.edges[edge]['pheromone']:
                self.graph.edges[edge]['pheromone'] = sys.float_info.min


class State:
    """Solver state.

    This class tracks the state of a solution in progress and is passed to each
    plugin hook. Specially it contains:

    ===================== ======================================
    Attribute             Description
    ===================== ======================================
    ``graph``             graph being solved
    ``colony``            colony that generated the ants
    ``ants``              ants being used to solve the graph
    ``limit``             maximum number of iterations
    ``gen_size``          number of ants being used
    ``solutions``         solutions found this iteration
    ``best``              best solution found this iteration
    ``is_new_record``     whether the best is a new record
    ``record``            best solution found so far
    ``previous_record``   previously best solution
    ===================== ======================================

    :param graph: a graph
    :type graph: :class:`networkx.Graph`
    :param list ants: the ants being used
    :param int limit: maximum number of iterations
    :param int gen_size: number of ants to use
    :param colony: source colony for the ants
    :type colony: :class:`~acopy.ant.Colony`
    """

    def __init__(self, graph, ants, limit, gen_size, colony, rho, q, top):
        self.graph = graph
        self.ants = ants
        self.limit = limit
        self.gen_size = gen_size
        self.colony = colony
        self.rho = rho
        self.q = q
        self.top = top
        self.solutions = None
        self.record = None
        self.previous_record = None
        self.is_new_record = False
        self._best = None

    @property
    def best(self):
        return self._best

    @best.setter
    def best(self, best):
        self.is_new_record = self.record is None or best < self.record
        if self.is_new_record:
            self.previous_record = self.record
            self.record = best
        self._best = best


class Solver:
    """ACO solver.

    Solvers control the parameters related to pheromone deposit and evaporation.
    If top is not specified, it defaults to the number of ants used to solve a
    graph.

    :param float rho: percentage of pheromone that evaporates each iteration
    :param float q: amount of pheromone each ant can deposit
    :param int top: number of ants that deposit pheromone
    :param list plugins: zero or more solver plugins
    """

    def __init__(self, rho=.03, q=1, top=None, plugins=None):
        self.rho = rho
        self.q = q
        self.top = top
        self.plugins = collections.OrderedDict()
        if plugins:
            self.add_plugins(*plugins)

    def __repr__(self):
        return (f'{self.__class__.__name__}(rho={self.rho}, q={self.q}, '
                f'top={self.top})')

    def solve(self, *args, **kwargs):
        """Find and return the best solution.

        Accepts exactly the same parameters as the :func:`~optimize` method.

        :return: best solution found
        :rtype: :class:`~Solution`
        """
        best = None
        for solution in self.optimize(*args, **kwargs):
            best = solution
        return best

    def optimize(self, graph, colony, gen_size=None, limit=None, problem=None, pheromone_update=False, is_opt2=False, is_maxmin=False):
        gen_size = gen_size
        ants = colony.get_ants(gen_size)

        state = State(graph=graph, ants=ants, limit=limit, gen_size=gen_size,
                      colony=colony, rho=self.rho, q=self.q, top=self.top)

        print("-----optimize begin-----")
        prev = 1e200
        cnt = 0
        success_list = []
        best_solutions = None

        for __ in utils.looper(limit):

            ng = copy.deepcopy(graph)
            solutions = self.find_solutions(ng, state.ants)

            costs = sum([s.cost for s in solutions])
            max_cost = max([s.cost for s in solutions])
            avg = costs / len(solutions)

            sd = sum([(s.cost - avg) ** 2 for s in solutions]) / len(solutions)

            theta = 3

            if sd < 1e30:
                costs += sd ** theta

            if max_cost > 1e30:
                cnt += 1

                # self.make_moving_image(solutions, graph, problem)
                #
                # if random.random() < 0.2:
                #     exit()

            if is_opt2 and max_cost > 1e30:
                self.opt2(ng, solutions, graph)

                costs = sum([s.cost for s in solutions])
                max_cost = max([s.cost for s in solutions])
                avg = costs / len(solutions)

                sd = sum([(s.cost - avg) ** 2 for s in solutions]) / len(solutions)

                theta = 3

                if sd < 1e30:
                    costs += sd ** theta

                if max_cost < 1e30:
                    cnt -= 1

            if pheromone_update:
                # success
                if sd < 1e30:
                    next_pheromones = collections.defaultdict(float)
                    for solution in solutions:
                        for edge in solution:
                            next_pheromones[edge] += self.q / solution.cost
                    for edge in state.graph.edges:
                        p = graph.edges[edge]['pheromone']
                        if is_opt2:
                            graph.edges[edge]['pheromone'] += next_pheromones[edge] / len(graph.nodes)
                        else:
                            graph.edges[edge]['pheromone'] = (1 - self.rho) * p + next_pheromones[edge]

                # 駄目な場合にフェロモンを修正する
                # else:
                #     edge_count = collections.defaultdict(int)
                #     for sol in solutions:
                #         for p in sol:
                #             x = min(p[0], p[1])
                #             y = max(p[0], p[1])
                #             edge_count[(x, y)] += 1
                #     for key in edge_count:
                #         if edge_count[key] > 1:
                #             x = key[0]
                #             y = key[1]
                #             d = edge_count[key]
                #             graph.edges[(x, y)]['pheromone'] *= (1 - self.rho) ** d
                #             graph.edges[(y, x)]['pheromone'] *= (1 - self.rho) ** d

                # else:
                #     next_pheromones = collections.defaultdict(float)
                #     for solution in solutions:
                #         for edge in solution:
                #             next_pheromones[edge] += self.q / solution.cost
                #     for edge in state.graph.edges:
                #         p = graph.edges[edge]['pheromone']
                #         graph.edges[edge]['pheromone'] = (1 - self.rho) * p + next_pheromones[edge]

            if costs < prev:
                prev = costs
                best_solutions = solutions
                print(__, [s.cost for s in solutions])
                yield solutions

            # 役に立たない
            # if is_maxmin and best_solutions:
            #     record_cost = prev
            #     rho = self.rho
            #     p_best = 0.05
            #     n = graph.number_of_nodes()
            #
            #     tau_max = (1 / rho) * (1 / record_cost)
            #     tau_min = (tau_max * (1 - p_best ** (1 / n))) / ((n / 2 - 1) * (p_best ** (1 / n)))
            #
            #     for edge in graph.edges():
            #         p = graph.edges[edge]['pheromone']
            #         p = min(tau_max, max(tau_min, p))
            #         graph.edges[edge]['pheromone'] = p

            if (__ + 1) % 100 == 0:
                print("-----", __ + 1, "times passed-----")

            success_list.append(cnt)

        print(f"構築に失敗した回数 {cnt}")
        # plt.plot([i for i in range(1000)], success_list)
        # plt.show()

    def exploit(self, *args, **kwargs):
        best = None
        for solution in self.exploitation(*args, **kwargs):
            best = solution
        return best

    def exploitation(self, graph, colony, gen_size=None, limit=None, k_visit=None, start=1):
        """ only Find and return increasingly better solutions.

        :param graph: graph to solve already optimized
        :type graph: :class:`networkx.Graph`
        :param colony: colony from which to source each :class:`~acopy.ant.Ant`
        :type colony: :class:`~acopy.ant.Colony`
        :param int gen_size: number of :class:`~acopy.ant.Ant` s to use
                             (default is one per graph node)
        :param int limit: maximum number of iterations to perform (default is
                          unlimited so it will run forever)
        :param int k_visit: number of cities to visit
        :return: better solutions as they are found
        :rtype: iter
        """
        gen_size = gen_size or len(graph.nodes)
        k_visit = k_visit or len(graph.nodes)
        ants = colony.get_ants(gen_size)
        for u, v in graph.edges:
            graph.edges[u, v].setdefault('pheromone', 0)

        state = State(graph=graph, ants=ants, limit=limit, gen_size=gen_size,
                      colony=colony, rho=self.rho, q=self.q, top=self.top)

        # call start hook for all plugins
        self._call_plugins('start', state=state)

        # find solutions
        for __ in utils.looper(limit):
            solutions = self.find_solution_k_visit(state.graph, state.ants, k_visit, start)

            # we want to ensure the ants are sorted with the solutions, but
            # since ants aren't directly comparable, so we interject a list of
            # unique numbers that satifies any two solutions that are equal
            data = list(zip(solutions, range(len(state.ants)), state.ants))
            data.sort()
            solutions, __, ants = zip(*data)

            state.solutions = solutions
            state.ants = ants

            self._call_plugins('before', state=state)

            # yield increasingly better solutions
            state.best = state.solutions[0]
            if state.is_new_record:
                yield state.record

            # call iteration hook for all plugins
            if self._call_plugins('iteration', state=state):
                break

        # call finish hook for all plugins
        self._call_plugins('finish', state=state)

    def find_solutions(self, graph, ants):
        """Return the solutions found for the given ants.

        :param graph: a graph
        :type graph: :class:`networkx.Graph`
        :param list ants: the ants to use
        :return: one solution per ant
        :rtype: list
        """
        for ant in ants:
            ant.init_solution(graph)
        for i in range(len(graph.nodes) - 1):
            for ant in ants:
                ant.move(graph)
            # random.shuffle(ants)
            ants.sort(key=lambda x: x.solution.cost, reverse=True)
        for ant in ants:
            ant.solution.close()
            ant.erase(graph, ant.solution.nodes[-1], ant.solution.nodes[0])
        solutions = [ant.solution for ant in ants]
        return solutions

    def find_solution_k_visit(self, graph, ants, k_visit, start=1):
        return [ant.k_tour(graph, k_visit, start) for ant in ants]

    def global_update(self, state):
        """Perform a global pheromone update.

        :param state: solver state
        :type state: :class:`~State`
        """
        next_pheromones = collections.defaultdict(float)
        solutions = state.solutions
        if self.top:
            solutions = solutions[:self.top]
        for solution in solutions:
            for edge in solution:
                next_pheromones[edge] += self.q / solution.cost
        for edge in state.graph.edges:
            p = state.graph.edges[edge]['pheromone']
            state.graph.edges[edge]['pheromone'] = (1 - self.rho) * p + next_pheromones[edge]

    def add_plugin(self, plugin):
        """Add a single solver plugin.

        If plugins have the same name, only the last one added is kept.

        :param plugin: the plugin to add
        :type plugin: :class:`acopy.plugins.SolverPlugin`
        """
        self.add_plugins(plugin)

    def add_plugins(self, *plugins):
        """Add one or more solver plugins."""
        for plugin in plugins:
            plugin.initialize(self)
            self.plugins[plugin.__class__.__qualname__] = plugin

    def get_plugins(self):
        """Return the added plugins.

        :return: plugins previously added
        :rtype: list
        """
        return self.plugins.values()

    def _call_plugins(self, hook, **kwargs):
        should_stop = False
        for plugin in self.get_plugins():
            try:
                plugin(hook, **kwargs)
            except StopIteration:
                should_stop = True
        return should_stop

    def make_moving_image(self, solutions, graph, problem):
        # 画像などの情報
        dic = collections.defaultdict(int)
        for s in solutions:
            for (x, y) in s:
                p1 = min(x, y)
                p2 = max(x, y)
                dic[(p1, p2)] += 1
        bads = set()
        for key in dic:
            if dic[key] > 1:
                bads.add(key)
        print(bads)
        print(solutions)

        labels = {i: str(i) for i in graph.nodes()}
        colors = ['red', 'blue', 'green', 'pink', 'gray']

        image_path = f"moving_images/{str(random.randint(0, 100003298932))}"

        if not os.path.isdir(image_path):
            os.mkdir(image_path)

        for i in range(len(graph.nodes())):
            index_color = 0
            plt.figure(dpi=400)
            _, ax = plt.subplots()
            pos = problem.display_data or problem.node_coords
            nx.draw_networkx_nodes(graph, pos)
            for sol in solutions:
                sol_path = sol.path
                path = []
                for j in range(0, i + 1):
                    p1 = min(sol_path[j][0], sol_path[j][1])
                    p2 = max(sol_path[j][0], sol_path[j][1])
                    p1 = sol_path[j][0]
                    p2 = sol_path[j][1]
                    path.append((p1, p2))
                nx.draw_networkx_edges(graph, edgelist=path, pos=pos, edge_color=colors[index_color])
                index_color += 1
            nx.draw_networkx_labels(graph, pos, labels=labels, font_color='white')
            ax.tick_params(left=True, bottom=True, labelleft=True, labelbottom=True)
            plt.savefig(image_path + f"/{i}.png")

    def opt2(self, graph, solutions, origin):
        edge_count = collections.defaultdict(int)
        for sol in solutions:
            for p in sol:
                x = min(p[0], p[1])
                y = max(p[0], p[1])
                edge_count[(x, y)] += 1
                edge_count[(y, x)] += 1

        n = len(graph.nodes)
        for sol in solutions:
            nodes = sol.nodes
            for i in range(0, n):
                best_cost = 1e100
                best_j = -1
                if edge_count[(nodes[i], nodes[(i + 1) % n])] > 1:
                    for j in range(0, n):
                        if i == j: continue

                        ii = min(i, j)
                        jj = max(i, j)
                        a = nodes[ii]
                        b = nodes[(ii + 1) % n]
                        c = nodes[jj]
                        d = nodes[(jj + 1) % n]

                        if edge_count[a, c] == 0 and edge_count[b, d] == 0:
                            dist = origin.edges[a, c]['weight'] + origin.edges[b, d]['weight'] - origin.edges[a, b]['weight'] - origin.edges[c, d]['weight']
                            if dist < best_cost:
                                best_cost = dist
                                best_j = j

                if best_j != -1:
                    ii = min(i, best_j)
                    jj = max(i, best_j)
                    a = nodes[ii]
                    b = nodes[(ii + 1) % n]
                    c = nodes[jj]
                    d = nodes[(jj + 1) % n]
                    if edge_count[a, c] == 0 and edge_count[b, d] == 0:
                        edge_count[a, b] -= 1
                        edge_count[b, a] -= 1
                        edge_count[c, d] -= 1
                        edge_count[d, c] -= 1
                        edge_count[a, c] += 1
                        edge_count[c, a] += 1
                        edge_count[b, d] += 1
                        edge_count[d, b] += 1
                        nodes[ii + 1: jj + 1] = reversed(nodes[ii + 1: jj + 1])
                        sol.path = []
                        sol.cost = 0
                        for k in range(n):
                            sol.path.append((nodes[k], nodes[(k + 1) % n]))
                            sol.cost += origin.edges[(nodes[k], nodes[(k + 1) % n])]['weight']

                        if edge_count[a, b] == 0:
                            graph.edges[a, b]['weight'] = origin.edges[a, b]['weight']
                            graph.edges[b, a]['weight'] = origin.edges[b, a]['weight']
                        if edge_count[c, d] == 0:
                            graph.edges[c, d]['weight'] = origin.edges[c, d]['weight']
                            graph.edges[d, c]['weight'] = origin.edges[d, c]['weight']

                        graph.edges[a, c]['weight'] = 1e100
                        graph.edges[c, a]['weight'] = 1e100
                        graph.edges[b, d]['weight'] = 1e100
                        graph.edges[d, b]['weight'] = 1e100


class SolverPlugin:
    """Solver plugin.

    Solver plugins can be added to any solver to customize its behavior.
    Plugins are initialized once when added, once before the first solver
    iteration, once after each solver iteration has completed, and once after
    all iterations have completed.

    Implementing each hook is optional.
    """

    #: unique name
    name = 'plugin'

    def __init__(self, **kwargs):
        self._params = kwargs

    def __repr__(self):
        params = ', '.join(f'{k}={v}' for k, v in self._params.items())
        return f'<{self.__class__.__qualname__}({params})>'

    def __call__(self, hook, **kwargs):
        return getattr(self, f'on_{hook}')(**kwargs)

    def initialize(self, solver):
        """Perform actions when being added to a solver.

        Though technically not required, this method should be probably be
        idempotent since the same plugin could be added to the same solver
        multiple times (perhaps even by mistake).

        :param solver: the solver to which the plugin is being added
        :type solver: :class:`acopy.solvers.Solver`
        """
        self.solver = solver

    def on_start(self, state):
        """Perform actions before the first iteration.

        :param state: solver state
        :type state: :class:`acopy.solvers.State`
        """
        pass

    def on_before(self, state):
        """Perform actions before the global update

        :param state: solver state
        :type state: :class:`acopy.solvers.State`
        """

    def on_iteration(self, state):
        """Perform actions after each iteration.

        :param state: solver state
        :type state: :class:`acopy.solvers.State`
        """
        pass

    def on_finish(self, state):
        """Perform actions once all iterations have completed.

        :param state: solver state
        :type state: :class:`acopy.solvers.State`
        """
        pass
