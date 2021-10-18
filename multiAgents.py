from util import manhattanDistance
from game import Directions, Actions
import random
import util

from game import Agent


class ReflexAgent(Agent):
    def getAction(self, gameState):
        legalMoves = gameState.getLegalActions()

        scores = [self.evaluationFunction(
            gameState, action) for action in legalMoves]
        bestScore = max(scores)
        bestIndices = [index for index in range(
            len(scores)) if scores[index] == bestScore]
        chosenIndex = random.choice(bestIndices)

        return legalMoves[chosenIndex]

    def evaluationFunction(self, currentGameState, action):
        successorGameState = currentGameState.generatePacmanSuccessor(action)
        newPos = successorGameState.getPacmanPosition()
        newFood = successorGameState.getFood()
        newGhostStates = successorGameState.getGhostStates()
        newScaredTimes = [
            ghostState.scaredTimer for ghostState in newGhostStates]

        infinity = float('inf')
        ghostPositions = successorGameState.getGhostPositions()
        
        for ghostPosition in ghostPositions:
            if manhattanDistance(newPos, ghostPosition) < 2:
                return -infinity

        numFood = currentGameState.getNumFood()
        newNumFood = successorGameState.getNumFood()
        if newNumFood < numFood:
            return infinity

        min_distance = infinity
        for food in newFood.asList():
            distance = manhattanDistance(newPos, food)
            min_distance = min(min_distance, distance)
        return 1.0 / min_distance


def scoreEvaluationFunction(currentGameState):
    return currentGameState.getScore()


class MultiAgentSearchAgent(Agent):
    def __init__(self, evalFn='scoreEvaluationFunction', depth='2'):
        self.index = 0
        self.evaluationFunction = util.lookup(evalFn, globals())
        self.depth = int(depth)

    def isTerminalState(self, gameState):
        return gameState.isWin() or gameState.isLose()

    def isPacman(self, agent):
        return agent == 0


class MinimaxAgent(MultiAgentSearchAgent):
    def maxValue(self, gameState, agent, depth):
        bestValue = float("-inf")
        for action in gameState.getLegalActions(agent):
            successor = gameState.generateSuccessor(agent, action)
            v = self.minimax(successor, agent+1, depth)
            bestValue = max(bestValue, v)
            if depth == 1 and bestValue == v:
                self.action = action
        return bestValue

    def minValue(self, gameState, agent, depth):
        bestValue = float("inf")
        for action in gameState.getLegalActions(agent):
            successor = gameState.generateSuccessor(agent, action)
            v = self.minimax(successor, agent+1, depth)
            bestValue = min(bestValue, v)
        return bestValue

    def minimax(self, gameState, agent=0, depth=0):
        agent = agent % gameState.getNumAgents()

        if self.isTerminalState(gameState):
            return self.evaluationFunction(gameState)

        if self.isPacman(agent):
            if depth < self.depth:
                return self.maxValue(gameState, agent, depth+1)
            else:
                return self.evaluationFunction(gameState)
        else:
            return self.minValue(gameState, agent, depth)

    def getAction(self, gameState):
        self.minimax(gameState)
        return self.action


class AlphaBetaAgent(MultiAgentSearchAgent):
    def maxValue(self, gameState, agent, depth, alpha, beta):
        bestValue = float("-inf")
        for action in gameState.getLegalActions(agent):
            successor = gameState.generateSuccessor(agent, action)
            v = self.minimax(successor, agent + 1, depth, alpha, beta)
            bestValue = max(bestValue, v)
            if depth == 1 and bestValue == v:
                self.action = action
            if bestValue > beta:
                return bestValue
            alpha = max(alpha, bestValue)
        return bestValue

    def minValue(self, gameState, agent, depth, alpha, beta):
        bestValue = float("inf")
        for action in gameState.getLegalActions(agent):
            successor = gameState.generateSuccessor(agent, action)
            v = self.minimax(successor, agent + 1, depth, alpha, beta)
            bestValue = min(bestValue, v)
            if bestValue < alpha:
                return bestValue
            beta = min(beta, bestValue)
        return bestValue

    def minimax(self, gameState, agent=0, depth=0, alpha=float("-inf"), beta=float("inf")):
        agent = agent % gameState.getNumAgents()

        if self.isTerminalState(gameState):
            return self.evaluationFunction(gameState)

        if self.isPacman(agent):
            if depth < self.depth:
                return self.maxValue(gameState, agent, depth+1, alpha, beta)
            else:
                return self.evaluationFunction(gameState)
        else:
            return self.minValue(gameState, agent, depth, alpha, beta)

    def getAction(self, gameState):
        self.minimax(gameState)
        return self.action


class ExpectimaxAgent(MultiAgentSearchAgent):
    def maxValue(self, gameState, agent, depth):
        bestValue = float("-inf")
        for action in gameState.getLegalActions(agent):
            successor = gameState.generateSuccessor(agent, action)
            v = self.expectimax(successor, agent+1, depth)
            bestValue = max(bestValue, v)
            if depth == 1 and bestValue == v:
                self.action = action
        return bestValue

    def probability(self, legalActions):
        return 1.0 / len(legalActions)

    def expValue(self, gameState, agent, depth):
        legalActions = gameState.getLegalActions(agent)
        v = 0
        for action in legalActions:
            successor = gameState.generateSuccessor(agent, action)
            p = self.probability(legalActions)
            v += p * self.expectimax(successor, agent+1, depth)
        return v

    def expectimax(self, gameState, agent=0, depth=0):

        agent = agent % gameState.getNumAgents()

        if self.isTerminalState(gameState):
            return self.evaluationFunction(gameState)

        if self.isPacman(agent):
            if depth < self.depth:
                return self.maxValue(gameState, agent, depth+1)
            else:
                return self.evaluationFunction(gameState)
        else:
            return self.expValue(gameState, agent, depth)

    def getAction(self, gameState):
        self.expectimax(gameState)
        return self.action


def closestItemDistance(currentGameState, items):
    walls = currentGameState.getWalls()

    start = currentGameState.getPacmanPosition()

    distance = {start: 0}

    visited = {start}

    queue = util.Queue()
    queue.push(start)

    while not queue.isEmpty():

        position = x, y = queue.pop()

        if position in items:
            return distance[position]

        for action in [Directions.NORTH, Directions.SOUTH, Directions.EAST, Directions.WEST]:

            dx, dy = Actions.directionToVector(action)
            next_position = nextx, nexty = int(x + dx), int(y + dy)

            if not walls[nextx][nexty] and next_position not in visited:
                queue.push(next_position)
                visited.add(next_position)
                distance[next_position] = distance[position] + 1

    return None


def betterEvaluationFunction(currentGameState):
    infinity = float('inf')
    position = currentGameState.getPacmanPosition()
    score = currentGameState.getScore()
    ghostStates = currentGameState.getGhostStates()
    foodList = currentGameState.getFood().asList()
    capsuleList = currentGameState.getCapsules()

    if currentGameState.isWin():
        return infinity
    if currentGameState.isLose():
        return -infinity

    for ghost in ghostStates:
        d = manhattanDistance(position, ghost.getPosition())
        if ghost.scaredTimer > 6 and d < 2:
            return infinity
        elif ghost.scaredTimer < 5 and d < 2:
            return -infinity
    foodDistance = 1.0/closestItemDistance(currentGameState, foodList)

    capsuleDistance = closestItemDistance(currentGameState, capsuleList)
    capsuleDistance = 0.0 if capsuleDistance is None else 1.0/capsuleDistance

    return 10.0*foodDistance + 5.0*score + 0.5*capsuleDistance

better = betterEvaluationFunction
