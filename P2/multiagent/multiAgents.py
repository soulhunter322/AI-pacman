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
        myDepth = 0
        myAgentIndex = 0
        Action_Value = self.StateValue(gameState, myAgentIndex, myDepth)
        return Action_Value[0]

    def StateValue(self, gameState, myAgentIndex, myDepth):
        if myAgentIndex >= gameState.getNumAgents():
            myAgentIndex = 0
            myDepth += 1
        if myDepth == self.depth:
            return self.evaluationFunction(gameState)
        if myAgentIndex == 0:
            return self.get_MaxValue(gameState, myAgentIndex, myDepth)
        else:
            return self.get_MinValue(gameState, myAgentIndex, myDepth)

    def get_MaxValue(self, gameState, myAgentIndex, myDepth):
        Now_Action_Value = ("action", -float("inf"))
        if not gameState.getLegalActions(myAgentIndex):
            return self.evaluationFunction(gameState)
        for a in gameState.getLegalActions(myAgentIndex):
            if a == "Stop":
                continue
            Predict_action_value = self.StateValue(gameState.generateSuccessor(myAgentIndex, a), myAgentIndex + 1,
                                                   myDepth)
            try:
                Predict_value = Predict_action_value[1]
            except:
                Predict_value = Predict_action_value
            if Predict_value > Now_Action_Value[1]:
                Now_Action_Value = (a, Predict_value)
        return Now_Action_Value

    def get_MinValue(self, gameState, myAgentIndex, myDepth):
        Now_Action_Value = ("action", float("inf"))
        if not gameState.getLegalActions(myAgentIndex):
            return self.evaluationFunction(gameState)
        for a in gameState.getLegalActions(myAgentIndex):
            if a == "Stop":
                continue
            Predict_action_value = self.StateValue(gameState.generateSuccessor(myAgentIndex, a), myAgentIndex + 1,
                                                   myDepth)
            try:
                Predict_value = Predict_action_value[1]
            except:
                Predict_value = Predict_action_value
            if Predict_value < Now_Action_Value[1]:
                Now_Action_Value = (a, Predict_value)
        return Now_Action_Value

class AlphaBetaAgent(MultiAgentSearchAgent):
    """
    Your minimax agent with alpha-beta pruning (question 3)
    """

    def getAction(self, gameState):
        """
        Returns the minimax action using self.depth and self.evaluationFunction
        """
        "*** YOUR CODE HERE ***"
        myDepth = 0
        myAgentIndex = 0
        Alpha = -float("inf")
        Beta = float("inf")
        Action_Value = self.StateValue(gameState, myAgentIndex, myDepth, Alpha, Beta)
        return Action_Value[0]

    def StateValue(self, gameState, myAgentIndex, myDepth, Alpha, Beta):
        if myAgentIndex >= gameState.getNumAgents():
            myAgentIndex = 0
            myDepth += 1
        if myDepth == self.depth:
            return self.evaluationFunction(gameState)
        if myAgentIndex == 0:
            return self.get_MaxValue(gameState, myAgentIndex, myDepth, Alpha, Beta)
        else:
            return self.get_MinValue(gameState, myAgentIndex, myDepth, Alpha, Beta)

    def get_MaxValue(self, gameState, myAgentIndex, myDepth, Alpha, Beta):
        Now_Action_Value = ("action", -float("inf"))
        if not gameState.getLegalActions(myAgentIndex):
            return self.evaluationFunction(gameState)
        for a in gameState.getLegalActions(myAgentIndex):
            if a == "Stop":
                continue
            Predict_action_value = self.StateValue(gameState.generateSuccessor(myAgentIndex, a), myAgentIndex + 1,
                                                   myDepth, Alpha, Beta)
            try:
                Predict_value = Predict_action_value[1]
            except:
                Predict_value = Predict_action_value
            if Predict_value > Now_Action_Value[1]:
                Now_Action_Value = (a, Predict_value)
            if Predict_value > Beta:
                break
            if Predict_value > Alpha:
                Alpha = Predict_value
        return Now_Action_Value

    def get_MinValue(self, gameState, myAgentIndex, myDepth, Alpha, Beta):
        Now_Action_Value = ("action", float("inf"))
        if not gameState.getLegalActions(myAgentIndex):
            return self.evaluationFunction(gameState)
        for a in gameState.getLegalActions(myAgentIndex):
            if a == "Stop":
                continue
            Predict_action_value = self.StateValue(gameState.generateSuccessor(myAgentIndex, a), myAgentIndex + 1,
                                                   myDepth, Alpha, Beta)
            try:
                Predict_value = Predict_action_value[1]
            except:
                Predict_value = Predict_action_value
            if Predict_value < Now_Action_Value[1]:
                Now_Action_Value = (a, Predict_value)
            if Predict_value < Alpha:
                break
            if Predict_value < Beta:
                Beta = Predict_value
        return Now_Action_Value


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
        myDepth = 0
        myAgentIndex = 0
        Action_Value = self.StateValue(gameState, myAgentIndex, myDepth)
        return Action_Value[0]

    def StateValue(self, gameState, myAgentIndex, myDepth):
        if myAgentIndex >= gameState.getNumAgents():
            myAgentIndex = 0
            myDepth += 1
        if myDepth == self.depth:
            return self.evaluationFunction(gameState)
        if myAgentIndex == 0:
            return self.get_MaxValue(gameState, myAgentIndex, myDepth)
        else:
            return self.get_ExpValue(gameState, myAgentIndex, myDepth)

    def get_MaxValue(self, gameState, myAgentIndex, myDepth):
        Now_Action_Value = ("action", -float("inf"))
        if not gameState.getLegalActions(myAgentIndex):
            return self.evaluationFunction(gameState)
        for a in gameState.getLegalActions(myAgentIndex):
            if a == "Stop":
                continue
            Predict_action_value = self.StateValue(gameState.generateSuccessor(myAgentIndex, a), myAgentIndex + 1,
                                                   myDepth)
            try:
                Predict_value = Predict_action_value[1]
            except:
                Predict_value = Predict_action_value
            if Predict_value > Now_Action_Value[1]:
                Now_Action_Value = (a, Predict_value)
        return Now_Action_Value

    def get_ExpValue(self, gameState, myAgentIndex, myDepth):
        Expected_Value = 0
        if not gameState.getLegalActions(myAgentIndex):
            return self.evaluationFunction(gameState)

        Probability = 1 / len(gameState.getLegalActions(myAgentIndex))

        for a in gameState.getLegalActions(myAgentIndex):
            if a == "Stop":
                continue
            Predict_action_value = self.StateValue(gameState.generateSuccessor(myAgentIndex, a), myAgentIndex + 1,
                                                   myDepth)
            try:
                Predict_value = Predict_action_value[1]
            except:
                Predict_value = Predict_action_value
            Expected_Value += Predict_value * Probability
        Now_Action_Value = ("action", Expected_Value)
        return Now_Action_Value


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

from util import PriorityQueue
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
    Score = 0
    Pacman_Pos = currentGameState.getPacmanPosition()

    Current_food = currentGameState.getFood().asList()
    FoodNum = len(Current_food)
    Score -= FoodNum

    Fooddistance_list = list()
    for foodPos in Current_food:
        distance = manhattanDistance(foodPos, Pacman_Pos)
        Fooddistance_list.append(distance)
    if len(Fooddistance_list) > 0:
        Maxdistance_toFood = max(Fooddistance_list)
        Score -= Maxdistance_toFood

    GhostStates = currentGameState.getGhostStates()
    Ghostdistance_list = list()
    for state in GhostStates:
        distance = manhattanDistance(state.configuration.pos, Pacman_Pos)
        Ghostdistance_list.append(distance)
    if len(Ghostdistance_list) > 0:
        Mindistance_toGhost = min(Ghostdistance_list)
        Score += Mindistance_toGhost
    ScaredTimes = [ghostState.scaredTimer for ghostState in GhostStates]
    Score -= min(ScaredTimes)

    return Score + currentGameState.getScore()

#util.raiseNotDefined()

# Abbreviation
better = betterEvaluationFunction
