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
        iterationsRan = 0
        while iterationsRan < self.iterations:
            tempValues = util.Counter()
            for state in self.mdp.getStates():
                # "By convention, a terminal state has zero future rewards." per mdp.py
                if self.mdp.isTerminal(state):
                    tempValues[state] = 0
                else:
                    possibleValues = []
                    for action in self.mdp.getPossibleActions(state):
                        possibleValues.append(self.computeQValueFromValues(state, action))
                    tempValues[state] = max(possibleValues)
            self.values = tempValues
            iterationsRan += 1


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
        qValue = 0
        for pair in self.mdp.getTransitionStatesAndProbs(state, action):
            nextState, probability = pair[0], pair[1]
            reward = self.mdp.getReward(state, action, nextState)
            qValue += probability * (reward + (self.discount * self.values[nextState]))
        return qValue

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
        else:
            possiblePolicies = {}
            for action in self.mdp.getPossibleActions(state):
                possiblePolicies[action] = self.computeQValueFromValues(state, action)
            # Credit to Alex Martelli! Elegant. http://stackoverflow.com/questions/3282823
            return max(possiblePolicies, key=possiblePolicies.get)

    def getPolicy(self, state):
        return self.computeActionFromValues(state)

    def getAction(self, state):
        "Returns the policy at the state (no exploration)."
        return self.computeActionFromValues(state)

    def getQValue(self, state, action):
        return self.computeQValueFromValues(state, action)

class AsynchronousValueIterationAgent(ValueIterationAgent):
    """
        * Please read learningAgents.py before reading this.*

        An AsynchronousValueIterationAgent takes a Markov decision process
        (see mdp.py) on initialization and runs cyclic value iteration
        for a given number of iterations using the supplied
        discount factor.
    """
    def __init__(self, mdp, discount = 0.9, iterations = 1000):
        """
          Your cyclic value iteration agent should take an mdp on
          construction, run the indicated number of iterations,
          and then act according to the resulting policy. Each iteration
          updates the value of only one state, which cycles through
          the states list. If the chosen state is terminal, nothing
          happens in that iteration.

          Some useful mdp methods you will use:
              mdp.getStates()
              mdp.getPossibleActions(state)
              mdp.getTransitionStatesAndProbs(state, action)
              mdp.getReward(state)
              mdp.isTerminal(state)
        """
        ValueIterationAgent.__init__(self, mdp, discount, iterations)

    def runValueIteration(self):
        "*** YOUR CODE HERE ***"
        iterationsRan = 0
        stateIndex = 0
        states = self.mdp.getStates()
        while iterationsRan < self.iterations:
            currentState = states[stateIndex]
            if not self.mdp.isTerminal(currentState):
                possibleValues = []
                for action in self.mdp.getPossibleActions(currentState):
                    possibleValues.append(self.computeQValueFromValues(currentState, action))
                self.values[currentState] = max(possibleValues)
            stateIndex += 1
            if stateIndex >= len(states):
                stateIndex = 0
            iterationsRan += 1

class PrioritizedSweepingValueIterationAgent(AsynchronousValueIterationAgent):
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
        states = self.mdp.getStates()
        allPredecessors = {}
        for indexState in range(len(states)):
            currentState = states[indexState]
            statePredecessors = set()
            for indexComparisonState in range(len(states)):
                comparisonState = states[indexComparisonState]
                for action in self.mdp.getPossibleActions(comparisonState):
                    for nextState, prob in self.mdp.getTransitionStatesAndProbs(comparisonState, action):
                        if nextState == currentState:
                            statePredecessors.add(comparisonState)
                allPredecessors[currentState] = statePredecessors
        salientStates = util.PriorityQueue()
        for state in states:
            if not self.mdp.isTerminal(state):
                qValues = []
                for action in self.mdp.getPossibleActions(state):
                    qValues.append(self.computeQValueFromValues(state, action))
                diff = abs(self.values[state] - max(qValues))
                diff = -1 * diff
                salientStates.push(state, diff)
        iterationsRan = 0
        while iterationsRan < self.iterations and not salientStates.isEmpty():
            state = salientStates.pop()
            qValues = []
            if not self.mdp.isTerminal(state):
                for action in self.mdp.getPossibleActions(state):
                    qValues.append(self.computeQValueFromValues(state, action))
                self.values[state] = max(qValues)
            for predecessor in allPredecessors[state]:
                qValues = []
                for action in self.mdp.getPossibleActions(predecessor):
                    qValues.append(self.computeQValueFromValues(predecessor, action))
                diff = abs(self.values[predecessor] - max(qValues))
                if diff > self.theta:
                    diff = -1 * diff
                    salientStates.update(predecessor, diff)
            iterationsRan += 1


