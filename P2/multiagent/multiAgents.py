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
from game import Actions
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
        chosenIndex = random.choice(bestIndices)  # Pick randomly among the best

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
        nearest_ghost_dis = 1000000
        for ghost_state in newGhostStates:
            ghost_x, ghost_y = ghost_state.getPosition()
            ghost_x = int(ghost_x)
            ghost_y = int(ghost_y)
            if newScaredTimes == 0:
                nearest_ghost_dis = min(nearest_ghost_dis, manhattanDistance((ghost_x, ghost_y), newPos))
        foods = newFood.asList()
        nearest_food_dis = 100000
        for food in foods:
            nearest_food_dis = min(nearest_food_dis, manhattanDistance(food, newPos))
        if not foods:
            nearest_food_dis = 0
        return successorGameState.getScore() - 12 / nearest_ghost_dis - nearest_food_dis / 10


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

    def __init__(self, evalFn='scoreEvaluationFunction', depth='2'):
        self.index = 0  # Pacman is always agent index 0
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

    def AlphaBetaPruning(self, gameState, agentIndex, depth, alpha, beta):
        if depth == 0 or gameState.isLose() or gameState.isWin():
            ret = self.evaluationFunction(gameState), Directions.STOP
        elif agentIndex == 0:
            ret = self.max_value(gameState, agentIndex, depth, alpha, beta)
        else:
            ret = self.min_value(gameState, agentIndex, depth, alpha, beta)
        return ret

    def max_value(self, gameState, agentIndex, depth, alpha, beta):
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
                return max_value, max_action
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

    def expectimaxSearch(self, gameState, agentIndex, depth):
        if depth == 0 or gameState.isLose() or gameState.isWin():
            ret = self.evaluationFunction(gameState), Directions.STOP
        elif agentIndex == 0:
            ret = self.max_value(gameState, agentIndex, depth)
        else:
            ret = self.notOptimalGhost(gameState, agentIndex, depth)
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
        rand = random.randrange(len(actions) + 1, len(actions) + 2)
        return new_score / rand, min_action


class PositionSearching:
    """
    - Defining the state space, start state, goal test, successor function and cost function.
    - It should be used to find paths to a particular point on the pacman board.
    - The state space consists of (x,y) positions in a pacman game.
    """

    def __init__(self, gameState, costFn=lambda x: 1, goal=(1, 1), start=None, warn=True, visualize=True):

        self.visualize = visualize
        self.walls = gameState.getWalls()
        self.startState = gameState.getPacmanPosition()

        self.goal = goal  # goal - position in the gameState
        self.costFn = costFn  # costFn: A function from a search state (tuple) to a non-negative number

        if start is not None:
            self.startState = start
        if warn and (gameState.getNumFood() != 1 or not gameState.hasFood(*goal)):
            print(' ERROR!!! '
                  'Not a search maze')

        #  Display
        self._visited, self._visitedlist, self._expanded = {}, [], 0

    def checkGoal(self, state):
        return state == self.goal

    def getStartState(self):
        return self.startState

    def getSuccessors(self, state):
        arr = []
        for action in [Directions.NORTH, Directions.SOUTH, Directions.EAST, Directions.WEST]:
            x, y = state
            dx, dy = Actions.directionToVector(action)
            next_x, next_y = int(x + dx), int(y + dy)
            if not self.walls[next_x][next_y]:
                next_state = (next_x, next_y)
                cost = self.costFn(next_state)
                arr.append((next_state, action, cost))

        # Display
        self._expanded += 1
        if state not in self._visited:
            self._visited[state] = True
            self._visitedlist.append(state)

        return arr

    def getCostOfActions(self, actions):
        x, y = self.getStartState()
        cost = 0

        if actions is None:
            return float("inf")
        for action in actions:
            # Check figure out the next state and see whether its' legal
            dx, dy = Actions.directionToVector(action)
            x, y = int(x + dx), int(y + dy)
            if self.walls[x][y]:
                return float("inf")
            cost += self.costFn((x, y))

        # Returns the cost of a particular sequence of actions.
        return cost


class FoodSearching(PositionSearching):
    """
      - To finding a path to any food.
      - Inherits the methods of class PositionSearching.
    """

    def __init__(self, gameState):
        super().__init__(gameState)
        self.food = gameState.getFood()

    def checkGoal(self, state):
        x, y = state
        return self.food[x][y]


# DUC ANH CODE LAI A* O DAY NHE !!!
def nullHeuristic(state, problem=None):
    """
    A heuristic function estimates the cost from the current state to the nearest
    goal in the provided SearchProblem.  This heuristic is trivial.
    """
    return 0


def aStarSearch(problem, heuristic=nullHeuristic):
    fringe = PriorityQueue()
    closed_set = set()
    action_list = list()
    current_state_list = [problem.getStartState(), []]
    while True:
        if problem.isGoalState(current_state_list[0]):
            action_list = current_state_list[1]
            break
        if current_state_list[0] not in closed_set:
            for i in problem.getSuccessors(current_state_list[0]):
                if i[0] not in closed_set:
                    fringe.push([i[0], current_state_list[1] + [i[1]]],
                                problem.getCostOfActions(current_state_list[1] + [i[1]]) + heuristic(i[0], problem))
            closed_set.add(current_state_list[0])
        if fringe.isEmpty():
            break
        current_state_list = fringe.pop()
    return action_list


def nearestFoodHeuristic(pos, problem, info={}):
    food_distance = [
        manhattanDistance(pos, (x, y))
        for x, row in enumerate(problem.food)
        for y, check in enumerate(row)
        if check
    ]
    return min(food_distance) if food_distance else 0


def betterEvaluationFunction(currentGameState):
    """
    Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable
    evaluation function (question 5).

    DESCRIPTION: <write something here so we know what you did>

    - Using the reciprocal for having a limited value on the change in Score from grabbing food pellet.

    """
    "*** YOUR CODE HERE ***"
    score = scoreEvaluationFunction(currentGameState)
    if currentGameState.isWin():
        return float("inf")

    food = currentGameState.getFood()
    food_remaining = len(food.asList())

    problem = FoodSearching(currentGameState)

    nearest_food = aStarSearch(problem, heuristic=nearestFoodHeuristic)
    nearest_food = 1 / len(nearest_food) if nearest_food else 1000

    position = currentGameState.getPacmanPosition()
    ghosts = currentGameState.getGhostStates()
    ghosts = [ghost for ghost in ghosts if ghost.scaredTimer == 0]
    if ghosts:
        ghost_distances = [manhattanDistance(ghost.getPosition(), position)
                           for ghost in ghosts]
        nearest_ghost = min(ghost_distances)
        nearest_ghost = 9999 if (nearest_ghost == 0) else 1/nearest_ghost
    else:
        nearest_ghost = 0

    capsules = currentGameState.getCapsules()
    capsules_remaining = len(capsules)
    if capsules:
        capsule_distances = [manhattanDistance(capsule, position) for capsule in capsules]
        nearest_capsule = 1 / min(capsule_distances)
    else:
        nearest_capsule = 0

    scores = [nearest_food, nearest_capsule, nearest_ghost,food_remaining, capsules_remaining]
    S = [5, 10, -5, -50, -100]
    score = sum(a * b for a,b in zip(scores, S))
    return score


#util.raiseNotDefined()

# Abbreviation
better = betterEvaluationFunction
