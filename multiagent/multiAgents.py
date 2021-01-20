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
import random, util, math

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
        # Useful information you can extract from a GameState (pacman.py)
        successorGameState = currentGameState.generatePacmanSuccessor(action)
        newPos = successorGameState.getPacmanPosition()
        newFood = successorGameState.getFood()
        newGhostStates = successorGameState.getGhostStates()
        newScaredTimes = [ghostState.scaredTimer for ghostState in newGhostStates]
        foodlist=newFood.asList()
        value=successorGameState.getScore()
        def foodLeft(): 
          foodmin=float(1e6)
          for food in foodlist: #lower is better
            if(manhattanDistance(newPos, food)<foodmin):
              foodmin=manhattanDistance(newPos,food)
          return foodmin
        def ghostdist(): #higher is better
          ghostmin=float(1e6)
          for ghosts in newGhostStates:
            if (manhattanDistance(newPos,ghosts.getPosition())<ghostmin):
              ghostmin=manhattanDistance(newPos,ghosts.getPosition())
          return ghostmin
        value+= float(ghostdist()/(foodLeft()*5))


          

        
        

          
        return value

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
        """
        "*** YOUR CODE HERE ***"
        #Used only for pacman agent hence agentindex is always 0.
        def maximizingAgent(gameState,depth):
            if gameState.isWin():  
              return self.evaluationFunction(gameState)
            
            elif gameState.isLose():
              return self.evaluationFunction(gameState)
            
            elif depth+1==self.depth:
              return self.evaluationFunction(gameState)

            #All the above cases were terminal cases, in which we can't go any further and the search stops
        
            maxval = -1e6 #some arbritary really small number that is pretty much going to be reset
            for action in gameState.getLegalActions(0):
                successor= gameState.generateSuccessor(0,action)
                maxval = max (maxval,minimizingAgent(successor,depth+1,1))
            return maxval
        
        
        def minimizingAgent(gameState,depth, agentIndex):
            minval= 1e6 #some arbritary really big number that is going to be reset pretty much immediately
            if gameState.isWin(): 
              return self.evaluationFunction(gameState)
            
            elif gameState.isLose():
              return self.evaluationFunction(gameState)
            
            for action in gameState.getLegalActions(agentIndex):
                successor= gameState.generateSuccessor(agentIndex,action)
                if agentIndex == (gameState.getNumAgents() - 1):
                    minval = min (minval,maximizingAgent(successor,depth))
                else:
                    minval = min(minval,minimizingAgent(successor,depth,agentIndex+1))
            return minval
        
        #This is the minimax in action; we take the minimum of the next successors of the next level and then maximize those options
        current= -1e6 #some arbritary really small number that is going to be reset pretty much immediately once we get an actual score in
        move = ''
        for action in gameState.getLegalActions(0):
            nextState = gameState.generateSuccessor(0,action)
            # The min component of minimax
            value = minimizingAgent(nextState,0,1)
            # The max component of minimax
            if value > current:
                move = action
                current = value
        return move

        

        util.raiseNotDefined()

class AlphaBetaAgent(MultiAgentSearchAgent):
    """
      Your minimax agent with alpha-beta pruning (question 3)
    """

    def getAction(self, gameState):
        """
          Returns the minimax action using self.depth and self.evaluationFunction
        """
        "*** YOUR CODE HERE ***"
        def maximizingAgent(gameState,depth,alpha,beta):
          if gameState.isWin():  
            return self.evaluationFunction(gameState)
            
          elif gameState.isLose():
            return self.evaluationFunction(gameState)
            
          elif depth+1==self.depth:
            return self.evaluationFunction(gameState)

          #All the above cases were terminal cases, in which we can't go any further and the search stops
        
          maxval = -1e6 #some arbritary really small number that is pretty much going to be reset
          alphanew=alpha
          for action in gameState.getLegalActions(0):
            successors=gameState.generateSuccessor(0,action)
            maxval=max(maxval,minimizingAgent(successors,depth+1,1,alphanew, beta))
            if maxval > beta:
              return maxval
            alphanew = max(alphanew,maxval)
          return maxval
        
        
        
        def minimizingAgent(gameState,depth,agentIndex,alpha,beta):
          minval=1e6
          if gameState.isWin():  
            return self.evaluationFunction(gameState)
            
          elif gameState.isLose():
            return self.evaluationFunction(gameState)
            
          betanew=beta
          for action in gameState.getLegalActions(agentIndex):
            successors=gameState.generateSuccessor(agentIndex,action)
            if (agentIndex==gameState.getNumAgents()-1):
              minval=min(minval,maximizingAgent(successors,depth,alpha,betanew))
              if minval<alpha:
                return minval
              betanew=min(betanew,minval)
            else:
              minval=min(minval,minimizingAgent(successors,depth,agentIndex+1,alpha,betanew))
              if minval<alpha:
                return minval
              betanew=min(minval,betanew)
          return minval

        current=-1e6 #will be reset immediately
        move=''
        alpha=-1e6
        beta=1e6 
        for action in gameState.getLegalActions(0):
          successors=gameState.generateSuccessor(0,action)
          val=minimizingAgent(successors,0,1,alpha,beta)
          if val>current:
            move=action
            current=val
          if current>beta:
            return move
          alpha=max(current,alpha)
        return move   
                
          




        
        util.raiseNotDefined()

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
        def maximizingAgent(gameState,depth):
            if gameState.isWin():  # terminal cases
              return self.evaluationFunction(gameState)
            
            elif gameState.isLose():
              return self.evaluationFunction(gameState)
            
            elif depth+1==self.depth:
              return self.evaluationFunction(gameState)

            #All the above cases were terminal cases, in which we can't go any further and the search stops
        
            maxval = -1e6 #some arbritary really small number that is pretty much going to be reset immediately
            for action in gameState.getLegalActions(0):
                successor= gameState.generateSuccessor(0,action)
                maxval = max (maxval,expectLevel(successor,depth+1,1))
            return maxval
        
        
        def expectLevel(gameState,depth, agentIndex):
            if gameState.isWin() or gameState.isLose():   #Terminal Test 
                return self.evaluationFunction(gameState)
            actions = gameState.getLegalActions(agentIndex)
            total = []
            numberofactions = len(actions)
            for action in actions:
                successor= gameState.generateSuccessor(agentIndex,action)
                if agentIndex == (gameState.getNumAgents() - 1):
                    expectedvalue = maximizingAgent(successor,depth)
                else:
                    expectedvalue = expectLevel(successor,depth,agentIndex+1)
                total.append(expectedvalue)
            if numberofactions == 0:
                return  0
            return float(sum(total))/float(numberofactions) #expectimax functions by taking the average so that is what I did here
        
        #This is the minimax in action; we take the minimum of the next successors of the next level and then maximize those options
        current= -1e6 #some arbritary really small number that is going to be reset pretty much immediately once we get an actual score in
        move = ''
        for action in gameState.getLegalActions(0):
            nextState = gameState.generateSuccessor(0,action)
            # The min component of minimax
            value = expectLevel(nextState,0,1)
            # The max component of minimax
            if value > current:
                move = action
                current = value
        return move

                


        util.raiseNotDefined()

def betterEvaluationFunction(currentGameState):
    """
      Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable
      evaluation function (question 5).

      DESCRIPTION: <I basically copied what I did from Q1 and am messing with some of the weights and how I strucutre them in my magic formula>
    """
    "*** YOUR CODE HERE ***"
    newPos = currentGameState.getPacmanPosition()
    newFood = currentGameState.getFood()
    newGhostStates = currentGameState.getGhostStates()
    newScaredTimes = [ghostState.scaredTimer for ghostState in newGhostStates]
    foodlist=newFood.asList()
    value=currentGameState.getScore()
    def foodLeft(): 
      foods=0
      for food in foodlist: #lower is better
        foods+=manhattanDistance(newPos,food)
      if (foods==0):
        return 1e6

      return foods
    def ghostdist(): #higher is better
      ghost=0
      for ghosts in newGhostStates:
        ghost+=manhattanDistance(newPos,ghosts.getPosition())
      if (ghost<=0):
        return -1e6
      return ghost
    value+= float(math.sqrt(abs(ghostdist()))/(foodLeft())+sum(newScaredTimes))

          
       
    return value

    util.raiseNotDefined()

# Abbreviation
better = betterEvaluationFunction

