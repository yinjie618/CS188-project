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
from pacman import GameState

class ReflexAgent(Agent):
    """
    A reflex agent chooses an action at each choice point by examining
    its alternatives via a state evaluation function.

    The code below is provided as a guide.  You are welcome to change
    it in any way you see fit, so long as you don't touch our method
    headers.
    """


    def getAction(self, gameState: GameState):
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

    def evaluationFunction(self, currentGameState: GameState, action):
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
        score = 0
        pacmanX = newPos[0]
        pacmanY = newPos[1]

        foodDist = []
        for foodLoc in newFood.asList():
            foodX = foodLoc[0]
            foodY = foodLoc[1]
            distPacmanToFood = abs(pacmanX - foodX) + abs(pacmanY - foodY)
            foodDist.append(distPacmanToFood)
        if foodDist:
            closestDistToFood = min(foodDist)
            score = score + 1/closestDistToFood

        for ghost, scaredTime in zip(newGhostStates, newScaredTimes):
            ghostLoc = ghost.getPosition()
            ghostX = ghostLoc[0]
            ghostY = ghostLoc[1]
            distPacmanToGhost = abs(pacmanX - ghostX) + abs(pacmanY - ghostY)
    
        if scaredTime > 0:
            if distPacmanToGhost > 0:
                score = score + 5/distPacmanToGhost
        else:
            if distPacmanToGhost <= 1:
                score = score - 1000000   
            elif distPacmanToGhost <= 2:
                score = score - 1/distPacmanToGhost

        return successorGameState.getScore() + score

def scoreEvaluationFunction(currentGameState: GameState):
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

    def getAction(self, gameState: GameState):
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
        numAgent = gameState.getNumAgents()
        numGhost = numAgent - 1
        def maxMin(agentIndex,gameState: GameState,depth):
            if ((gameState.isWin())or(gameState.isLose())or(depth == self.depth)):
                return self.evaluationFunction(gameState), None
            
            if (agentIndex == 0):
                scoreAndAction = []
                for action in gameState.getLegalActions(agentIndex):
                    newSuccessor = gameState.generateSuccessor(agentIndex, action)
                    nextScore, _ = maxMin(1, newSuccessor, depth)
                    scoreAndAction.append((nextScore,action))
                maxScore, actionToMaxScore = max(scoreAndAction)
                return maxScore,actionToMaxScore

            if (agentIndex > 0):
                scoreAndAction = []
                for action in gameState.getLegalActions(agentIndex):
                    newSuccessor = gameState.generateSuccessor(agentIndex, action)
                    if (agentIndex + 1 <= numGhost):
                        nextScore, _ = maxMin(agentIndex + 1, newSuccessor, depth)
                    else:
                        nextScore, _ = maxMin(0, newSuccessor, depth + 1)
                    scoreAndAction.append((nextScore, action))
                minScore, actionToMinScore = min(scoreAndAction)
                return minScore,actionToMinScore
        _,actionToBestScore = maxMin(0,gameState,0)
        return actionToBestScore

class AlphaBetaAgent(MultiAgentSearchAgent):
    """
    Your minimax agent with alpha-beta pruning (question 3)
    """

    def getAction(self, gameState: GameState):
        """
        Returns the minimax action using self.depth and self.evaluationFunction
        """
        "*** YOUR CODE HERE ***"
        numAgent = gameState.getNumAgents()
        numGhost = numAgent - 1
        def alphaBetaPruning(agentIndex,gameState: GameState,depth,alpha,beta):
            if ((gameState.isWin())or(gameState.isLose())or(depth == self.depth)):
                return self.evaluationFunction(gameState), None
            
            if (agentIndex == 0):
                maxScore = -9999999999
                for action in gameState.getLegalActions(agentIndex):
                    newSuccessor = gameState.generateSuccessor(agentIndex, action)
                    score, _ = alphaBetaPruning(1, newSuccessor, depth,alpha, beta)
                    if score > maxScore:
                        maxScore,actionToMaxScore = score,action
                    if maxScore > beta:
                        return maxScore,actionToMaxScore
                    alpha = max(alpha,maxScore)
                return maxScore,actionToMaxScore

            if (agentIndex > 0):
                minScore = 9999999999
                for action in gameState.getLegalActions(agentIndex):
                    newSuccessor = gameState.generateSuccessor(agentIndex, action)
                    if agentIndex < numAgent - 1:
                        score, _ = alphaBetaPruning(agentIndex + 1, newSuccessor, depth, alpha, beta)
                    else:
                        score, _ = alphaBetaPruning(0, newSuccessor, depth + 1, alpha, beta)
                    if score < minScore:
                        minScore,actionToMinScore = score,action                           
                    if minScore < alpha:
                        return minScore,actionToMinScore
                    beta = min(beta,minScore)                    
                return minScore,actionToMinScore
        _,actionToBestScore = alphaBetaPruning(0,gameState,0,-999999999,9999999999)
        return actionToBestScore       

class ExpectimaxAgent(MultiAgentSearchAgent):
    """
      Your expectimax agent (question 4)
    """

    def getAction(self, gameState: GameState):
        """
        Returns the expectimax action using self.depth and self.evaluationFunction

        All ghosts should be modeled as choosing uniformly at random from their
        legal moves.
        """
        "*** YOUR CODE HERE ***"
        numAgent = gameState.getNumAgents()
        numGhost = numAgent - 1
        def maxExpect(agentIndex,gameState: GameState,depth):
            if ((gameState.isWin())or(gameState.isLose())or(depth == self.depth)):
                return self.evaluationFunction(gameState), None
            
            if (agentIndex == 0):
                scoreAndAction = []
                for action in gameState.getLegalActions(agentIndex):
                    newSuccessor = gameState.generateSuccessor(agentIndex, action)
                    nextScore, _ = maxExpect(1, newSuccessor, depth)
                    scoreAndAction.append((nextScore,action))
                maxScore, actionToMaxScore = max(scoreAndAction)
                return maxScore,actionToMaxScore

            if (agentIndex > 0):
                score = []
                for action in gameState.getLegalActions(agentIndex):
                    newSuccessor = gameState.generateSuccessor(agentIndex, action)
                    if (agentIndex + 1 <= numGhost):
                        nextScore, _ = maxExpect(agentIndex + 1, newSuccessor, depth)
                    else:
                        nextScore, _ = maxExpect(0, newSuccessor, depth + 1)
                    score.append(nextScore)
                avgScore = sum(score)/len(score)
                return avgScore,None
        _,actionToBestScore = maxExpect(0,gameState,0)
        return actionToBestScore
        

def betterEvaluationFunction(currentGameState: GameState):
    """
    Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable
    evaluation function (question 5).

    DESCRIPTION: <write something here so we know what you did>
    """
    "*** YOUR CODE HERE ***"
    newPos = currentGameState.getPacmanPosition()
    newFood = currentGameState.getFood()
    newGhostStates = currentGameState.getGhostStates()
    newScaredTimes = [g.scaredTimer for g in newGhostStates]

    score = 0
    pacmanX = newPos[0]
    pacmanY = newPos[1]

    foodDist = []
    for foodLoc in newFood.asList():
        foodX = foodLoc[0]
        foodY = foodLoc[1]
        distPacmanToFood = abs(pacmanX - foodX) + abs(pacmanY - foodY)
        foodDist.append(distPacmanToFood)
    if foodDist:
        closestDistToFood = min(foodDist)
        score = score + 1/closestDistToFood

    for ghost, scaredTime in zip(newGhostStates, newScaredTimes):
        ghostLoc = ghost.getPosition()
        ghostX = ghostLoc[0]
        ghostY = ghostLoc[1]
        distPacmanToGhost = abs(pacmanX - ghostX) + abs(pacmanY - ghostY)
    
    if scaredTime > 0:
        if distPacmanToGhost > 0:
            score = score + 5/distPacmanToGhost
    else:
        if distPacmanToGhost <= 1:
            score = score - 1000000   
        elif distPacmanToGhost <= 2:
            score = score - 1/distPacmanToGhost

    return currentGameState.getScore() + score


# Abbreviation
better = betterEvaluationFunction
