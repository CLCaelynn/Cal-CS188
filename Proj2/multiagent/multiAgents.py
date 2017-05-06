# multiAgents.py
# --------------
# Licensing Information:  You are free to use or extend these projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to UC Berkeley, including a link to http://ai.berkeley.edu.
# 
# Attribution Information: The Pacman AI projects were developed at UC Berkeley.
# The core projects and autograders were primarily created by John DeNero
# (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# Student side autograding was added by Brad Miller, Nick Hay, and
# Pieter Abbeel (pabbeel@cs.berkeley.edu).


from util import manhattanDistance
from game import Directions
import random, util

from game import Actions
from game import Agent

# For finding the nearest salient point. Same code as I wrote for Proj1.
def nearestPoint(start, possibilities):
    if len(possibilities) == 0:
        return None
    else:
        filteredPossibilities = [point for point in possibilities if point != start]
        possibilityToDistance = {}
        for point in filteredPossibilities:
            possibilityToDistance[point] = manhattanDistance(start, point)
        return min(possibilityToDistance, key=possibilityToDistance.get) # Credit to Alex Martelli! So elegant.
                                                                         # http://stackoverflow.com/questions/3282823

class ReflexAgent(Agent):
    """
      A reflex agent chooses an action at each choice point by examining
      its alternatives via a state evaluation function.

      The code below is provided as a guide.  You are welcome to change
      it in any way you see fit, so long as you don't touch our method
      headers.
    """

    def getAction(self, gameState):
        """
        You do not need to change this method, but you're welcome to.

        getAction chooses among the best options according to the evaluation function.

        Just like in the previous project, getAction takes a GameState and returns
        some Directions.X for some X in the set {North, South, West, East, Stop}
        """
        # Collect legal moves and successor states
        legalMoves = gameState.getLegalActions()

        # Choose one of the best actions
        scores = [self.evaluationFunction(gameState, action) for action in legalMoves]
        bestScore = max(scores)
        bestIndices = [index for index in range(len(scores)) if scores[index] == bestScore]
        chosenIndex = random.choice(bestIndices) # Pick randomly among the best

        "Add more of your code here if you want to"

        return legalMoves[chosenIndex]

    def evaluationFunction(self, currentGameState, action):
        """
        Design a better evaluation function here.

        The evaluation function takes in the current and proposed successor
        GameStates (pacman.py) and returns a number, where higher numbers are better.

        The code below extracts some useful information from the state, like the
        remaining food (newFood) and Pacman position after moving (newPos).
        newScaredTimes holds the number of moves that each ghost will remain
        scared because of Pacman having eaten a power pellet.

        Print out these variables to see what you're getting, then combine them
        to create a masterful evaluation function.
        """
        "*** YOUR CODE HERE ***"
        successorGameState = currentGameState.generatePacmanSuccessor(action)
        currPos = currentGameState.getPacmanPosition()
        newPos = successorGameState.getPacmanPosition()
        newFood = successorGameState.getFood().asList()
        newGhostStates = successorGameState.getGhostStates()
        newScaredTimes = [ghostState.scaredTimer for ghostState in newGhostStates]
        finalScore = 0
        for ghostState in newGhostStates:
            if manhattanDistance(newPos, ghostState.getPosition()) <= 1:
                finalScore -= 1000
            if manhattanDistance(newPos, ghostState.getPosition()) == 2:
                finalScore -= 20
            if manhattanDistance(newPos, ghostState.getPosition()) == 3:
                finalScore -= 2
        nearestFood = nearestPoint(newPos, newFood)
        if not nearestFood is None:
            if manhattanDistance(newPos, nearestFood) < manhattanDistance(currPos, nearestFood):
                finalScore += 3
        return finalScore + successorGameState.getScore()

def scoreEvaluationFunction(currentGameState):
    """
      This default evaluation function just returns the score of the state.
      The score is the same one displayed in the Pacman GUI.

      This evaluation function is meant for use with adversarial search agents
      (not reflex agents).
    """
    return currentGameState.getScore()

class MultiAgentSearchAgent(Agent):
    """
      This class provides some common elements to all of your
      multi-agent searchers.  Any methods defined here will be available
      to the MinimaxPacmanAgent, AlphaBetaPacmanAgent & ExpectimaxPacmanAgent.

      You *do not* need to make any changes here, but you can if you want to
      add functionality to all your adversarial search agents.  Please do not
      remove anything, however.

      Note: this is an abstract class: one that should not be instantiated.  It's
      only partially specified, and designed to be extended.  Agent (game.py)
      is another abstract class.
    """

    def __init__(self, evalFn = 'scoreEvaluationFunction', depth = '2'):
        self.index = 0 # Pacman is always agent index 0
        self.evaluationFunction = util.lookup(evalFn, globals())
        self.depth = int(depth)

class MinimaxAgent(MultiAgentSearchAgent):
    """
      Your minimax agent (question 2)
    """

    def getAction(self, gameState):
        """
          Returns the minimax action from the current gameState using self.depth
          and self.evaluationFunction.

          Here are some method calls that might be useful when implementing minimax.

          gameState.getLegalActions(agentIndex):
            Returns a list of legal actions for an agent
            agentIndex=0 means Pacman, ghosts are >= 1

          gameState.generateSuccessor(agentIndex, action):
            Returns the successor game state after an agent takes an action

          gameState.getNumAgents():
            Returns the total number of agents in the game

          gameState.isWin():
            Returns whether or not the game state is a winning state

          gameState.isLose():
            Returns whether or not the game state is a losing state
        """
        "*** YOUR CODE HERE ***"
        finalAction = self.valueDecision(gameState, 0, 0) # Pacman goes first.
        return finalAction

    def valueDecision(self, gameState, depth, agent):
        bestChoice = 'none_yet_chosen'
        val = -999999999
        actions = gameState.getLegalActions(agent)
        for action in actions:
            prevVal = val # Keeping track of each state's utility.
            potentialState = gameState.generateSuccessor(0, action)
            val = max(val, self.minValue(potentialState, 0, 1))
            # If the previous potential state's utility is worse than this
            # potential state's utility, choose the action leading to this state.
            if prevVal < val:
                bestChoice = action
        return bestChoice

    def minValue(self, gameState, depth, agent):
        val = 999999999
        actions = gameState.getLegalActions(agent)
        # Check if state is terminal
        if gameState.isWin() or gameState.isLose() or depth == self.depth:
            return self.evaluationFunction(gameState)
        for action in actions:
            potentialState = gameState.generateSuccessor(agent, action)
            if agent < gameState.getNumAgents() - 1:
                val = min(val, self.minValue(potentialState, depth, agent + 1))
            else:
                val = min(val, self.maxValue(potentialState, depth + 1, 0))
        return val

    def maxValue(self, gameState, depth, agent):
        val = -999999999
        actions = gameState.getLegalActions(agent)
        # Check if state is terminal
        if gameState.isWin() or gameState.isLose() or depth == self.depth:
            return self.evaluationFunction(gameState)
        # Find the best potential state for Pacman based on how the ghost(s)
        # would react to Pacman's possible actions.
        for action in actions:
            potentialState = gameState.generateSuccessor(agent, action)
            val = max(val, self.minValue(potentialState, depth, agent + 1))
        return val


class AlphaBetaAgent(MultiAgentSearchAgent):
    """
      Your minimax agent with alpha-beta pruning (question 3)
    """

    def getAction(self, gameState):
        """
          Returns the minimax action using self.depth and self.evaluationFunction
        """
        # finalAction = self.alphaBetaDecision(gameState, 0, 0) # Pacman goes first.
        # return finalAction
        "*** YOUR CODE HERE ***"
        return self.alphaBetaMax(gameState, 0, 0, -999999999, 999999999)

    def alphaBetaMin(self, gameState, depth, agent, alpha, beta):
        val = 999999999
        actions = gameState.getLegalActions(agent)
        if gameState.isWin() or gameState.isLose() or depth == self.depth:
            return self.evaluationFunction(gameState)
        for action in actions:
            potentialState = gameState.generateSuccessor(agent, action)
            if agent < gameState.getNumAgents() - 1:
                val = min(val, self.alphaBetaMin(potentialState, depth, agent + 1, alpha, beta))
            if agent == gameState.getNumAgents() - 1:
                val = min(val, self.alphaBetaMax(potentialState, depth + 1, 0, alpha, beta))
            if val < alpha:
                return val
            beta = min(beta, val)
        return val

    def alphaBetaMax(self, gameState, depth, agent, alpha, beta):
        val = -999999999
        actions = gameState.getLegalActions(agent)
        if gameState.isWin() or gameState.isLose() or depth == self.depth:
            return self.evaluationFunction(gameState)
        for action in actions:
            potentialState = gameState.generateSuccessor(agent, action)
            minVal = max(val, self.alphaBetaMin(potentialState, depth, agent + 1, alpha, beta))
            if val < minVal:
                bestChoice = action
                val = minVal
            if val > beta:
                return val
            alpha = max(alpha, val)
        if depth == 0:
            return bestChoice
        return val

class ExpectimaxAgent(MultiAgentSearchAgent):
    """
      Your expectimax agent (question 4)
    """

    def getAction(self, gameState):
        """
          Returns the expectimax action using self.depth and self.evaluationFunction

          All ghosts should be modeled as choosing uniformly at random from their
          legal moves.
        """
        "*** YOUR CODE HERE ***"
        return self.exMax(gameState, 0, 0)

    def exRandom(self, gameState, depth, agent):
        finalVal = 0
        actions = gameState.getLegalActions(agent)
        if gameState.isWin() or gameState.isLose() or depth == self.depth:
            return self.evaluationFunction(gameState)
        for action in actions:
            potentialState = gameState.generateSuccessor(agent, action)
            if agent < gameState.getNumAgents() - 1:
                finalVal += self.exRandom(potentialState, depth, agent + 1)
            if agent == gameState.getNumAgents() - 1:
                finalVal += self.exMax(potentialState, depth + 1, 0)
        return finalVal/len(actions)

    def exMax(self, gameState, depth, agent):
        val = -999999999
        actions = gameState.getLegalActions(agent)
        if gameState.isWin() or gameState.isLose() or depth == self.depth:
            return self.evaluationFunction(gameState)
        for action in actions:
            potentialState = gameState.generateSuccessor(agent, action)
            randomVal = max(val, self.exRandom(potentialState, depth, agent + 1))
            if val < randomVal:
                bestChoice = action
                val = randomVal
        if depth == 0:
            return bestChoice
        return val

def betterEvaluationFunction(currentGameState):
    """
      Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable
      evaluation function (question 5).

      DESCRIPTION: Uses the score, distance to nearest food, distance to
                   nearest capsule, and distance to the ghost as heuristics.
                   Ignores distance to the ghost when the ghost is scared.
    """
    "*** YOUR CODE HERE ***"
    currPos = currentGameState.getPacmanPosition()
    food = currentGameState.getFood().asList()
    capsules = currentGameState.getCapsules()
    ghostStates = currentGameState.getGhostStates()
    ghostWeight, foodWeight = 0.0, 0.0 # Needed to account for different situations
    allGhostsScared = True # If false, that's noted three lines below.
    distToFood, distToCapsule = 0.1, 0.1
    for ghostState in ghostStates:
        ghostPos = int(ghostState.getPosition()[0]), int(ghostState.getPosition()[1])
        distToGhost = float(max(mazeDistance(currPos, ghostPos, currentGameState), 0.1))
        if ghostState.scaredTimer == 0:
            allGhostsScared = False
    if food:
         nearestFood= nearestPoint(currPos, food)
         distToFood = float(max(mazeDistance(currPos, nearestFood, currentGameState), 0.1))
    if capsules:
         nearestCapsule = nearestPoint(currPos, capsules)
         distToCapsule = float(max(mazeDistance(currPos, nearestCapsule, currentGameState), 0.1))
    if allGhostsScared == False:
        if distToGhost <= 1:
            ghostWeight = -2
        elif distToGhost == 2:
            ghostWeight = -1
        elif distToGhost == 3:
            ghostWeight = -0.5
        else:
            ghostWeight = 0
        return currentGameState.getScore() + (ghostWeight * (1.0 / distToGhost)) + (10.0 * (1.0 / distToFood)) + (9.0 * (1.0 / distToCapsule))
    else:
        return currentGameState.getScore() + (10.0 * (1.0 / distToFood)) + (5.0 * (1.0 / distToCapsule))


def mazeDistance(point1, point2, gameState):
    """
    Returns the maze distance between any two points, using the search functions
    you have already built. The gameState can be any game state -- Pacman's
    position in that state is ignored.

    Example usage: mazeDistance( (2,4), (5,6), gameState)

    This might be a useful helper function for your ApproximateSearchAgent.
    """
    x1, y1 = point1
    x2, y2 = point2
    walls = gameState.getWalls()
    assert not walls[x1][y1], 'point1 is a wall: ' + str(point1)
    assert not walls[x2][y2], 'point2 is a wall: ' + str(point2)
    prob = PositionSearchProblem(gameState, start=point1, goal=point2, warn=False, visualize=False)
    return len(breadthFirstSearch(prob))

def breadthFirstSearch(problem):
    """Search the shallowest nodes in the search tree first."""
    expanded = []
    fringe = util.Queue()
    fringe.push((problem.getStartState(), [], 0))
    while True:
        currentNode = fringe.pop()
        if problem.isGoalState(currentNode[0]):
            return currentNode[1]
        if currentNode[0] not in expanded:
            expanded.append(currentNode[0])
            for node in problem.getSuccessors(currentNode[0]):
                if node[0] not in expanded:
                    fringe.push((node[0], currentNode[1] + [node[1]], node[2]))
    return None

class SearchProblem:
    """
    This class outlines the structure of a search problem, but doesn't implement
    any of the methods (in object-oriented terminology: an abstract class).

    You do not need to change anything in this class, ever.
    """

    def getStartState(self):
        """
        Returns the start state for the search problem.
        """
        util.raiseNotDefined()

    def isGoalState(self, state):
        """
          state: Search state

        Returns True if and only if the state is a valid goal state.
        """
        util.raiseNotDefined()

    def getSuccessors(self, state):
        """
          state: Search state

        For a given state, this should return a list of triples, (successor,
        action, stepCost), where 'successor' is a successor to the current
        state, 'action' is the action required to get there, and 'stepCost' is
        the incremental cost of expanding to that successor.
        """
        util.raiseNotDefined()

    def getCostOfActions(self, actions):
        """
         actions: A list of actions to take

        This method returns the total cost of a particular sequence of actions.
        The sequence must be composed of legal moves.
        """
        util.raiseNotDefined()


class PositionSearchProblem(SearchProblem):
    """
    A search problem defines the state space, start state, goal test, successor
    function and cost function.  This search problem can be used to find paths
    to a particular point on the pacman board.

    The state space consists of (x,y) positions in a pacman game.

    Note: this search problem is fully specified; you should NOT change it.
    """

    def __init__(self, gameState, costFn = lambda x: 1, goal=(1,1), start=None, warn=True, visualize=True):
        """
        Stores the start and goal.

        gameState: A GameState object (pacman.py)
        costFn: A function from a search state (tuple) to a non-negative number
        goal: A position in the gameState
        """
        self.walls = gameState.getWalls()
        self.startState = gameState.getPacmanPosition()
        if start != None: self.startState = start
        self.goal = goal
        self.costFn = costFn
        self.visualize = visualize
        if warn and (gameState.getNumFood() != 1 or not gameState.hasFood(*goal)):
            print 'Warning: this does not look like a regular search maze'

        # For display purposes
        self._visited, self._visitedlist, self._expanded = {}, [], 0 # DO NOT CHANGE

    def getStartState(self):
        return self.startState

    def isGoalState(self, state):
        isGoal = state == self.goal

        # For display purposes only
        if isGoal and self.visualize:
            self._visitedlist.append(state)
            import __main__
            if '_display' in dir(__main__):
                if 'drawExpandedCells' in dir(__main__._display): #@UndefinedVariable
                    __main__._display.drawExpandedCells(self._visitedlist) #@UndefinedVariable

        return isGoal

    def getSuccessors(self, state):
        """
        Returns successor states, the actions they require, and a cost of 1.

         As noted in search.py:
             For a given state, this should return a list of triples,
         (successor, action, stepCost), where 'successor' is a
         successor to the current state, 'action' is the action
         required to get there, and 'stepCost' is the incremental
         cost of expanding to that successor
        """

        successors = []
        for action in [Directions.NORTH, Directions.SOUTH, Directions.EAST, Directions.WEST]:
            x,y = state
            dx, dy = Actions.directionToVector(action)
            nextx, nexty = int(x + dx), int(y + dy)
            if not self.walls[nextx][nexty]:
                nextState = (nextx, nexty)
                cost = self.costFn(nextState)
                successors.append( ( nextState, action, cost) )

        # Bookkeeping for display purposes
        self._expanded += 1 # DO NOT CHANGE
        if state not in self._visited:
            self._visited[state] = True
            self._visitedlist.append(state)

        return successors

    def getCostOfActions(self, actions):
        """
        Returns the cost of a particular sequence of actions. If those actions
        include an illegal move, return 999999.
        """
        if actions == None: return 999999
        x,y= self.getStartState()
        cost = 0
        for action in actions:
            # Check figure out the next state and see whether its' legal
            dx, dy = Actions.directionToVector(action)
            x, y = int(x + dx), int(y + dy)
            if self.walls[x][y]: return 999999
            cost += self.costFn((x,y))
        return cost

# Abbreviation
better = betterEvaluationFunction

