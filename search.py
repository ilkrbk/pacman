import util

class SearchProblem:
    def getStartState(self):
        util.raiseNotDefined()

    def isGoalState(self, state):
        util.raiseNotDefined()

    def getSuccessors(self, state):
        util.raiseNotDefined()

    def getCostOfActions(self, actions):
        util.raiseNotDefined()


def tinyMazeSearch(problem):
    from game import Directions
    s = Directions.SOUTH
    w = Directions.WEST
    return [s, s, w, s, w, w, s, w]

def buildActions(node, parent):
    action = node[1]
    if action is None: return []
    return buildActions(parent[node], parent) + [action]

def graphSearchWithoutCosts(problem, fringe):

    parent = dict()
    closed = set()

    start_state = problem.getStartState()
    start_action = None
    start_cost = 0
    start_node = (start_state, start_action, start_cost)
    fringe.push(start_node)

    while not fringe.isEmpty():
        node = fringe.pop()
        state = node[0]

        if problem.isGoalState(state): return buildActions(node, parent)

        if state not in closed:
            closed.add(state)

            for successor in problem.getSuccessors(state):
                successor_state = successor[0]

                if successor_state not in closed:
                    fringe.push(successor)
                    parent[successor] = node

    return []

def graphSearchWithCosts(problem, heuristic):

    parent = {}
    closed = set()
    cost = {}
    fringe = util.PriorityQueue()

    start_state = problem.getStartState()
    start_action = None
    start_cost = 0
    fringe.push(item=(start_state, start_action), priority=start_cost)
    cost[start_state] = start_cost

    while not fringe.isEmpty():
        node = fringe.pop()
        state = node[0]

        if problem.isGoalState(state): return buildActions(node, parent)

        if state not in closed:
            closed.add(state)

            for successor in problem.getSuccessors(state):
                successor_state = successor[0]
                successor_cost = successor[2]

                if successor_state not in closed:
                    g = cost[state] + successor_cost
                    h = heuristic(successor_state, problem)
                    # To understand why this works, check how update function acts
                    fringe.update(successor[:2], g+h)
                    cost[successor_state] = g
                    parent[successor[:2]] = node

    return []

def depthFirstSearch(problem):
    return graphSearchWithoutCosts(problem, util.Stack())

def breadthFirstSearch(problem):
    return graphSearchWithoutCosts(problem, util.Queue())

def uniformCostSearch(problem):
    return graphSearchWithCosts(problem, nullHeuristic)

def nullHeuristic(state, problem=None):
    return 0

def aStarSearch(problem, heuristic=nullHeuristic):
    return graphSearchWithCosts(problem, heuristic)

# Abbreviations
bfs = breadthFirstSearch
dfs = depthFirstSearch
astar = aStarSearch
ucs = uniformCostSearch
