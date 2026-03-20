# valueIterationAgents.py
# -----------------------
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


# valueIterationAgents.py
# -----------------------
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


import mdp, util

from learningAgents import ValueEstimationAgent
import collections

class ValueIterationAgent(ValueEstimationAgent):
    """
        * Please read learningAgents.py before reading this.*

        A ValueIterationAgent takes a Markov decision process
        (see mdp.py) on initialization and runs value iteration
        for a given number of iterations using the supplied
        discount factor.
    """
    def __init__(self, mdp, discount = 0.9, iterations = 100):
        """
          Your value iteration agent should take an mdp on
          construction, run the indicated number of iterations
          and then act according to the resulting policy.

          Some useful mdp methods you will use:
              mdp.getStates()
              mdp.getPossibleActions(state)
              mdp.getTransitionStatesAndProbs(state, action)
              mdp.getReward(state, action, nextState)
              mdp.isTerminal(state)
        """
        self.mdp = mdp
        self.discount = discount
        self.iterations = iterations
        self.values = util.Counter() # A Counter is a dict with default 0
        self.runValueIteration()

    def runValueIteration(self):
        # Write value iteration code here
        "*** YOUR CODE HERE ***"

        iterationCounter = 0
        while(iterationCounter < self.iterations):

            oldValues = self.values.copy()
            newValues = {}
            states = self.mdp.getStates()

            for state in states:
                if(self.mdp.isTerminal(state) == True):
                    newValues[state] = 0
                    continue

                stateActions =self.mdp.getPossibleActions(state) #get each action in this state
                waitingValue = []
                for action in stateActions:
                    nextStatesAndProb = self.mdp.getTransitionStatesAndProbs(state, action)
                    actionValue = 0
                    for nextState,nextStateProb in nextStatesAndProb:
                        nextStateReward = self.mdp.getReward(state, action, nextState) #get R to next state
                        nextStateValue = oldValues[nextState] #get V of next state
                        actionValue = actionValue + nextStateProb * (nextStateReward + self.discount * nextStateValue)
                    waitingValue.append(actionValue)

                newValues[state] = max(waitingValue)

            self.values = newValues

            iterationCounter = iterationCounter + 1

    def getValue(self, state):
        """
          Return the value of the state (computed in __init__).
        """
        return self.values[state]


    def computeQValueFromValues(self, state, action):
        """
          Compute the Q-value of action in state from the
          value function stored in self.values.
        """
        "*** YOUR CODE HERE ***"

        nextStatesAndProb = self.mdp.getTransitionStatesAndProbs(state, action)
        actionValue = 0
        for nextState,nextStateProb in nextStatesAndProb:
            nextStateReward = self.mdp.getReward(state, action, nextState)
            nextStateValue = self.values[nextState]
            actionValue = actionValue + nextStateProb * (nextStateReward + self.discount * nextStateValue)

        return actionValue
        
    def computeActionFromValues(self, state):
        """
          The policy is the best action in the given state
          according to the values currently stored in self.values.

          You may break ties any way you see fit.  Note that if
          there are no legal actions, which is the case at the
          terminal state, you should return None.
        """
        "*** YOUR CODE HERE ***"
        if self.mdp.isTerminal(state):
            return None

        stateActions =self.mdp.getPossibleActions(state)
        waitingValuesAndActions = [] 
        for action in stateActions:
            valuenAndAction = self.computeQValueFromValues(state, action),action
            waitingValuesAndActions.append(valuenAndAction)
        _,action = max(waitingValuesAndActions)

        return action       

    def getPolicy(self, state):
        return self.computeActionFromValues(state)

    def getAction(self, state):
        "Returns the policy at the state (no exploration)."
        return self.computeActionFromValues(state)

    def getQValue(self, state, action):
        return self.computeQValueFromValues(state, action)


class PrioritizedSweepingValueIterationAgent(ValueIterationAgent):
    """
        * Please read learningAgents.py before reading this.*

        A PrioritizedSweepingValueIterationAgent takes a Markov decision process
        (see mdp.py) on initialization and runs prioritized sweeping value iteration
        for a given number of iterations using the supplied parameters.
    """
    def __init__(self, mdp, discount = 0.9, iterations = 100, theta = 1e-5):
        """
          Your prioritized sweeping value iteration agent should take an mdp on
          construction, run the indicated number of iterations,
          and then act according to the resulting policy.
        """
        self.theta = theta
        ValueIterationAgent.__init__(self, mdp, discount, iterations)

    def runValueIteration(self):
        "*** YOUR CODE HERE ***"
        predecessors = {}
        states = self.mdp.getStates()

        for state in states:
            predecessors[state] = set()

        for state in states:
            actions = self.mdp.getPossibleActions(state)
            for action in actions:
                for nextState, prob in self.mdp.getTransitionStatesAndProbs(state, action):
                    if prob > 0:
                        predecessors[nextState].add(state)

        priorityQueue = util.PriorityQueue()

        diff = {}
        for state in states:
            diff[state] = 0

        for state in states:
            waitingActionValue = []
            actions = self.mdp.getPossibleActions(state)
            if len(actions) > 0:
                for action in actions:
                    actionValue = self.computeQValueFromValues(state, action)
                    waitingActionValue.append(abs(self.getValue(state) - actionValue))
                diff[state] = max(waitingActionValue)
            else:
                diff[state] = 0

        for state in states:
            if self.mdp.isTerminal(state) == False:
                priorityQueue.push (state,(-1 * (diff[state])))

        iterationCounter = 0
        while(iterationCounter < self.iterations):
            if priorityQueue.isEmpty():
                break

            tempState = priorityQueue.pop()
            if self.mdp.isTerminal(tempState) == False:
                tempActions = self.mdp.getPossibleActions(tempState)
                waitingValue = []
                if len(tempActions) > 0:
                    for action in tempActions:
                        waitingValue.append(self.computeQValueFromValues(tempState,action))
                    self.values[tempState] = max(waitingValue)

            tempStatePredecessors = predecessors[tempState]
            for predecessor in tempStatePredecessors:
                predecessorActions = self.mdp.getPossibleActions(predecessor)
                waitingPredecessorValue = []
                if len(predecessorActions) > 0:
                    for action in predecessorActions:
                        waitingPredecessorValue.append(self.computeQValueFromValues(predecessor,action))
                    predecessorValue = max(waitingPredecessorValue)
                else:
                    predecessorValue = 0
                tempDiff = abs(self.getValue(predecessor) - predecessorValue)
                if tempDiff > self.theta:
                    priorityQueue.update(predecessor, -tempDiff)

            iterationCounter = iterationCounter + 1
