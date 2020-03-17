# -*- coding: utf-8 -*-
import sys
import itertools
import bisect
import random
import copy

from .utils import positive
from .solvers import Solution


class Ant:
    """An ant.

    Ants explore a graph, using alpha and beta to guide their decision making
    process when choosing which edge to travel next.

    :param float alpha: how much pheromone matters
    :param float beta: how much distance matters
    """

    def __init__(self, alpha=1, beta=3, **kwargs):
        self.alpha = alpha
        self.beta = beta
        self.solution = None
        self.unvisited = None

    @property
    def alpha(self):
        """How much pheromone matters. Always kept greater than zero."""
        return self._alpha

    @alpha.setter
    def alpha(self, value):
        self._alpha = positive(value)

    @property
    def beta(self):
        """How much distance matters. Always kept greater than zero."""
        return self._beta

    @beta.setter
    def beta(self, value):
        self._beta = positive(value)

    def __repr__(self):
        return f'Ant(alpha={self.alpha}, beta={self.beta})'

    def init_solution(self, graph, start=1):
        # start = random.randint(1, len(graph.nodes))
        self.solution = Solution(graph, start, ant=self)
        self.init_unvisited_nodes(graph)

    def init_unvisited_nodes(self, graph):
        self.unvisited = []
        for node in graph[self.solution.current]:
            if node not in self.solution:
                self.unvisited.append(node)

    def move(self, graph):
        node = self.choose_destination(graph)
        current = self.solution.current
        self.solution.add_node(node)
        self.unvisited.remove(node)
        self.erase(graph, current, node)

    def erase(self, graph, now, to):
        graph.edges[now, to]['pheromone'] = 0
        graph.edges[to, now]['pheromone'] = 0
        graph.edges[now, to]['weight'] = 1e100
        graph.edges[to, now]['weight'] = 1e100

    def choose_destination(self, graph):
        if len(self.unvisited) == 1:
            return self.unvisited[0]
        scores = self.get_scores(graph)
        return self.choose_node(scores)

    def get_scores(self, graph):
        scores = []
        for node in self.unvisited:
            edge = graph.edges[self.solution.current, node]
            score = self.score_edge(edge)
            # 残余度の計算
            # score = self.score_residual(graph, self.solution.current, node, score)
            scores.append(score)
        return scores

    def choose_node(self, scores, q_0=0.2):
        choices = self.unvisited
        total = sum(scores)
        cumdist = list(itertools.accumulate(scores)) + [total]
        index = bisect.bisect(cumdist, random.random() * total)
        q = random.random()
        # if q < q_0:
        #     cand = []
        #     for i in range(len(choices)):
        #         if scores[i] > 1e-30:
        #             cand.append(choices[i])
        #     if len(cand):
        #         return random.choice(cand)
        return choices[min(index, len(choices) - 1)]

    def score_edge(self, edge):
        weight = edge.get('weight', 1)
        if weight == 0:
            return sys.float_info.max
        pre = 1 / weight
        post = edge['pheromone']
        return post ** self.alpha * pre ** self.beta

    def score_residual(self, graph, now, to, score):
        cands = set(copy.deepcopy(self.unvisited))
        cands.remove(to)
        bad = []
        for cand in cands:
            if graph.edges[to, cand]['weight'] > 1e10:
                bad.append(cand)
        for x in bad:
            cands.remove(x)
        score = score / max(1, len(cands) ** 2)
        return score
