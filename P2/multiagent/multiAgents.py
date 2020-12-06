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

from game import Agent

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
        some Directions.X for some X in the set {NORTH, SOUTH, WEST, EAST, STOP}
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
        # Useful information you can extract from a GameState (pacman.py)
        successorGameState = currentGameState.generatePacmanSuccessor(action)
        newPos = successorGameState.getPacmanPosition()
        newFood = successorGameState.getFood()
        newGhostStates = successorGameState.getGhostStates()
        newScaredTimes = [ghostState.scaredTimer for ghostState in newGhostStates]

        "*** YOUR CODE HERE ***"
        totalScaredTime = newScaredTimes[0]

        distToGhosts = 99999

        for ghostState in newGhostStates:
            if distToGhosts > manhattanDistance(newPos, ghostState.getPosition()):
                distToGhosts = manhattanDistance(newPos, ghostState.getPosition())

        if distToGhosts > 5 or totalScaredTime > 0:
            distToGhosts = 9999999
        # find closest food
        tmp = 99999
        for x in range(newFood.width):
            for y in range(newFood.height):
                if (newFood[x][y] is True):
                    if tmp > manhattanDistance(newPos, (x, y)):
                        tmp = manhattanDistance(newPos, (x, y))

        if distToGhosts == 0:
            distToGhosts = 0.01

        if tmp == 0:
            tmp = 0.1

        return float(3.5 / tmp) + 10 * (currentGameState.getNumFood() - newFood.count()) - (0.6 / distToGhosts);

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
        return self.MiniMaxSearch(gameState, agentIndex=0, depth=self.depth)[1]

    def MiniMaxSearch(self, gameState, agentIndex, depth):
        if depth == 0 or gameState.isWin() or gameState.isLose():
            ret = self.evaluationFunction(gameState), Directions.STOP
        elif agentIndex == 0:
            ret = self.max_value(gameState, agentIndex, depth)
        else:
            ret = self.min_value(gameState, agentIndex, depth)
        return ret

    def max_value(self, gameState, agentIndex, depth):
        actions = gameState.getLegalActions(agentIndex)
        if agentIndex == gameState.getNumAgents() - 1:
            next_agent = 0
            next_depth = depth - 1
        else:
            next_agent = agentIndex + 1
            next_depth = depth
        max_score = -99999
        max_action = Directions.STOP
        for action in actions:
            successorGameState = gameState.generateSuccessor(agentIndex, action)
            new_score = self.MiniMaxSearch(successorGameState, next_agent, next_depth)[0]
            if new_score > max_score:
                max_score = new_score
                max_action = action
        return max_score, max_action

    def min_value(self, gameState, agentIndex, depth):
        actions = gameState.getLegalActions(agentIndex)
        if agentIndex == gameState.getNumAgents() - 1:
            next_agent = 0
            next_depth = depth - 1
        else:
            next_agent = agentIndex + 1
            next_depth = depth
        min_score = 99999
        min_action = Directions.STOP
        for action in actions:
            successorGameState = gameState.generateSuccessor(agentIndex, action)
            new_score = self.MiniMaxSearch(successorGameState, next_agent, next_depth)[0]
            if new_score < min_score:
                min_score = new_score
                min_action = action
        return min_score, min_action



class AlphaBetaAgent(MultiAgentSearchAgent):
    """
    Your minimax agent with alpha-beta pruning (question 3)
    """

    def getAction(self, gameState):
        """
        Returns the minimax action using self.depth and self.evaluationFunction
        """
        "*** YOUR CODE HERE ***"
        return self.AlphaBetaPruning(gameState, agentIndex=0, depth=self.depth, alpha=-99999, beta=99999)[1]

    def AlphaBetaPruning(self, gameState, agentIndex,depth, alpha, beta):
        if depth == 0 or gameState.isLose() or gameState.isWin():
            ret = self.evaluationFunction(gameState), Directions.STOP
        elif agentIndex == 0:
            ret = self.max_value(gameState, agentIndex, depth, alpha, beta)
        else:
            ret = self.min_value(gameState, agentIndex, depth, alpha, beta)
        return ret

    def max_value(self,gameState,agentIndex,depth, alpha, beta):
        actions = gameState.getLegalActions(agentIndex)
        if agentIndex == gameState.getNumAgents() - 1:
            next_agent = 0
            next_depth = depth - 1
        else:
            next_agent = agentIndex + 1
            next_depth = depth
        max_value = -99999
        max_action = Directions.STOP
        for action in actions:
            successorGameState = gameState.generateSuccessor(agentIndex, action)
            new_value = self.AlphaBetaPruning(successorGameState, next_agent, next_depth, alpha, beta)[0]
            if new_value > max_value:
                max_value = new_value
                max_action = action
            if max_value > beta:
                return max_value,max_action
            alpha = max(alpha, max_value)
        return max_value, max_action

    def min_value(self, gameState, agentIndex, depth, alpha, beta):
        actions = gameState.getLegalActions(agentIndex)
        if agentIndex == gameState.getNumAgents() - 1:
            next_agent = 0
            next_depth = depth - 1
        else:
            next_agent = agentIndex + 1
            next_depth = depth
        min_value = 99999
        min_action = Directions.STOP
        for action in actions:
            successorGameState = gameState.generateSuccessor(agentIndex, action)
            new_value = self.AlphaBetaPruning(successorGameState, next_agent, next_depth, alpha, beta)[0]
            if new_value < min_value:
                min_value = new_value
                min_action = action
            if min_value < alpha:
                return min_value, min_action
            beta = min(beta, min_value)
        return min_value, min_action




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
        return self.expectimaxSearch(gameState, agentIndex=0, depth=self.depth)[1]

    def expectimaxSearch(self,gameState, agentIndex, depth):
        if depth == 0 or gameState.isLose() or gameState.isWin():
            ret = self.evaluationFunction(gameState), Directions.STOP
        elif agentIndex == 0:
            ret = self.max_value(gameState, agentIndex, depth)
        else:
            ret = self.notOptimalGhost(gameState, agentIndex, depth)
        return ret

    def max_value(self,gameState, agentIndex, depth):
        actions = gameState.getLegalActions(agentIndex)
        if agentIndex == gameState.getNumAgents() - 1:
            next_agent = 0
            next_depth = depth - 1
        else:
            next_agent = agentIndex + 1
            next_depth = depth
        max_score = -99999
        max_action = Directions.STOP
        for action in actions:
            successorGameState = gameState.generateSuccessor(agentIndex, action)
            new_score = self.expectimaxSearch(successorGameState, next_agent, next_depth)[0]
            if new_score > max_score:
                max_score = new_score
                max_action = action
        return max_score, max_action

    def notOptimalGhost(self, gameState, agentIndex, depth):
        actions = gameState.getLegalActions(agentIndex)
        if agentIndex == gameState.getNumAgents() - 1:
            next_agent = 0
            next_depth = depth - 1
        else:
            next_agent = agentIndex + 1
            next_depth = depth
        new_score = 0
        min_action = Directions.STOP
        for action in actions:
            successorGameState = gameState.generateSuccessor(agentIndex, action)
            new_score += self.expectimaxSearch(successorGameState, next_agent, next_depth)[0]
        rand = random.randrange(len(actions)+1, len(actions) + 2)
        return new_score/rand, min_action

def betterEvaluationFunction(currentGameState):
    """
    Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable
    evaluation function (question 5).

    DESCRIPTION: <write something here so we know what you did>
    """
    "*** YOUR CODE HERE ***"

    util.raiseNotDefined()

# Abbreviation
better = betterEvaluationFunction
