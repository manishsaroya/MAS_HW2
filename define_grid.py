from __future__ import division
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import animation
import sys
import random
import copy

def moveLeft(zvals_, pose):
    newPose = pose
    if pose[1] - 1 >= 0:
        t = zvals_[pose[0], pose[1]]
        zvals_[pose[0], pose[1]] = 0
        newPose[1] = pose[1] - 1
        #zvals_[newPose[0], newPose[1]] = t
    #im.set_data(zvals_)
    return zvals_, newPose
        
def moveRight(zvals_,pose):
    newPose = pose
    # compare with num of columns
    if pose[1] + 1 < len(zvals_[0]):
        t = zvals_[pose[0], pose[1]]
        zvals_[pose[0], pose[1]] = 0
        newPose[1] = pose[1] + 1
        #zvals_[newPose[0], newPose[1]] = t
    #else:
        #print "wall ahead"
    #im.set_data(zvals_)
    return zvals_, newPose

def moveUp(zvals_,pose):
    newPose = pose
    # compare with num of raws
    if pose[0] + 1 < len(zvals_):
        t = zvals_[pose[0], pose[1]]
        zvals_[pose[0], pose[1]] = 0
        newPose[0] = pose[0] + 1
        #zvals_[newPose[0], newPose[1]] = t
    #else:
        #print "wall ahead"
    #im.set_data(zvals_)
    return zvals_, newPose

def moveDown(zvals_,pose):
    newPose = pose
    if pose[0] - 1 >= 0:
        t = zvals_[pose[0], pose[1]]
        zvals_[pose[0], pose[1]] = 0
        newPose[0] = pose[0] - 1
        #zvals_[newPose[0], newPose[1]] = t
    #else:
        #print "wall ahead"
    #im.set_data(zvals_)
    return zvals_, newPose


# Some global learning parameters
numIterations = 10000 # these many times the robot will catch the target while in training mode.

# Set number of iterations to be used to evaluate Qlearning. Should be atleast 1
evaIterations = 10

#iterations = 0

# A transition reward of -1 is given to the agent at every time step.
transitionReward = -1

# the reward given to the agent at termination, i.e. when robot catches the target.
TerminateReward = 20

# learning rate between 0 and 1.
learningRate = 0.5

# exploration rate
epsilon = 0.3

# the map 
zvals = np.zeros((5,10))

# intialize the random robot pose and fixed target pose.
#robotPose = [np.random.randint(0,5), np.random.randint(0,10)]
targetPose = [1,9]
numAgents = 2
agentsPose = []

maxstates = (len(zvals[0]) * len(zvals)) ** (numAgents + 1) # plus one for target
Qvalues = np.zeros((maxstates,4)) #Qvalue is [States * Actions] matrix.
# the states are 50(target positions) * 50 ... (robot positions) 
# at every state the robot can take 4 actions.
agentsQvalues=[]
for i in range(numAgents):
    agentsQvalues.append(Qvalues)
    
def envReset():
    # the map
    global zvals
    zvals = np.zeros((5,10))

    

    # intialize the random robot pose and fixed target pose.
   # global robotPose
    robotPose = [np.random.randint(0,5), np.random.randint(0,10)]
    
    # Start all the agents at same random position
    global agentsPose
    agentsPose = []

    for i in range(numAgents):
        agentsPose.append(robotPose)
        
    global targetPose
    targetPose = [1,9]

    # assign color/ID values to robotpose and target pose in map
    for i in agentsPose:
        zvals[i[0],i[1]] = 1

    #zvals[robotPose[0],robotPose[1]] = 1
    zvals[targetPose[0],targetPose[1]] = 0.5

envReset()

# Visualization
fig = plt.figure()
im = plt.imshow(zvals, origin={'lower','left'})

def init():
    im.set_data(zvals)

def timestep(i):  
    # Before action mapping from robotPose & targetPose to Qvalues table.
    robotCell = len(zvals[0]) * robotPose[0] + robotPose[1]
    targetCell = len(zvals[0]) * targetPose[0] + targetPose[1]
    state_ = (len(zvals[0]) * len(zvals)) * robotCell + targetCell
    
    # move the target randomly.
    switcher = {0: moveLeft, 1: moveRight, 2: moveUp, 3: moveDown}
    func = switcher.get(np.random.randint(0,4), lambda: "Invalid input")
    newzvalstarget, newtargetPose = func(zvals,targetPose)
    
    # move the robot 
    switcher = {0: moveLeft, 1: moveRight, 2: moveUp, 3: moveDown}
    action = np.argmax(Qvalues[state_])
    func = switcher.get(action, lambda: "Invalid input")
    newzvals, newrobotPose = func(zvals, robotPose)

    # assign color/ID values to robotpose and target pose in map
    zvals[robotPose[0],robotPose[1]] = 1
    zvals[targetPose[0],targetPose[1]] = 0.5
    
    # After action mapping from robotPose & targetPose to Qvalues table. 
    robotCell = len(zvals[0]) * robotPose[0] + robotPose[1]
    targetCell = len(zvals[0]) * targetPose[0] + targetPose[1]
   
    print robotPose, targetPose, robotCell, targetCell
    im.set_data(zvals)
    if robotCell == targetCell:
        envReset()
        sys.exit("Demo done")

def QlearningEvaluation():
    i = 0
    totalReward = 0
    while i < evaIterations:
        # Before action mapping from robotPose & targetPose to Qvalues table.
        robotCell = len(zvals[0]) * robotPose[0] + robotPose[1]
        targetCell = len(zvals[0]) * targetPose[0] + targetPose[1]
        state_ = (len(zvals[0]) * len(zvals)) * robotCell + targetCell
    
        #move the target randomly.
        switcher = {0: moveLeft, 1: moveRight, 2: moveUp, 3: moveDown}
        func = switcher.get(np.random.randint(0,4), lambda: "Invalid input")
        newzvalstarget, newtargetPose = func(zvals,targetPose)
    
        #move the robot
        switcher = {0: moveLeft, 1: moveRight, 2: moveUp, 3: moveDown}
        action = np.argmax(Qvalues[state_])
        func = switcher.get(action, lambda: "Invalid input")
        newzvals, newrobotPose = func(zvals, robotPose)
    
        # assign color/ID values to robotpose and target pose in map
        zvals[robotPose[0],robotPose[1]] = 1
        zvals[targetPose[0],targetPose[1]] = 0.5
    
        # After action mapping from robotPose & targetPose to Qvalues table.
        robotCell = len(zvals[0]) * robotPose[0] + robotPose[1]
        targetCell = len(zvals[0]) * targetPose[0] + targetPose[1]
    
        #print robotPose, targetPose, robotCell, targetCell
        #im.set_data(zvals)
        totalReward += -1
        if robotCell == targetCell:
            totalReward += 20
            i = i + 1
            envReset()
    return totalReward / evaIterations

def computeState():
    targetCell = len(zvals[0]) * targetPose[0] + targetPose[1]
    #print "targetCell", targetCell
    agentsCell = []
    for robotPose in agentsPose:
        robotCell = len(zvals[0]) * robotPose[0] + robotPose[1]
        #print "robotCell", robotCell
        agentsCell.append(robotCell)

    # set the base value same as the size of zvals.
    base = (len(zvals[0]) * len(zvals))
    state_ = targetCell
    for robotCell in agentsCell:
        state_ += base * robotCell
        base *= (len(zvals[0]) * len(zvals))
    return state_

def moveRobot(robotPose,state_,index):
    switcher = {0: moveLeft, 1: moveRight, 2: moveUp, 3: moveDown}
    if random.uniform(0,1) < epsilon:
        action = np.random.randint(0,4)
    else:
        action = np.argmax(agentsQvalues[index][state_])
    func = switcher.get(action, lambda: "Invalid input")
    newzvals, newrobotPose = func(zvals, robotPose)
    return robotPose, action


def Qlearning():
    iterations = 0
    while iterations < numIterations:
        # Before action mapping from robotPose & targetPose to Qvalues table. 
        state_ = computeState()
    
        #move the target randomly.
        switcher = {0: moveLeft, 1: moveRight, 2: moveUp, 3: moveDown}
        func = switcher.get(np.random.randint(0,4), lambda: "Invalid input")
        newzvalstarget, newtargetPose = func(zvals,targetPose)
       
        agentsActions = []
        #move the robot
        for index in range(len(agentsPose)):
            #print "agentsPose[index]", agentsPose[index]
            agentsPose[index], action = moveRobot(copy.copy(agentsPose[index]),state_,index)
            agentsActions.append(action)
            #print "agentsActions", agentsActions
            #print "index", index
            #print "Post", agentsPose
    
        # After action mapping from robotPose & targetPose to Qvalues table. 
        state = computeState()
        #print state,"post motion state"

        
        #Change the Zvals;
        targetCell = len(zvals[0]) * targetPose[0] + targetPose[1]
        isTerminate = False 
        for index, robotPose in enumerate(agentsPose):
            robotCell = len(zvals[0]) * robotPose[0] + robotPose[1]
            isTerminate = (targetCell==robotCell) or isTerminate

            if targetCell!=robotCell:
                #update the Qtable
                sample = transitionReward + np.amax(agentsQvalues[index][state])
                agentsQvalues[index][state_,agentsActions[index]] = ((1 - learningRate) * agentsQvalues[index][state_,agentsActions[index]]) + (learningRate * sample)
            else:
                #update the Qtable
                sample = transitionReward + 20
                agentsQvalues[index][state_,agentsActions[index]] = ((1 - learningRate) * agentsQvalues[index][state_,agentsActions[index]]) + (learningRate * sample)
        if isTerminate:
            #global iterations
            iterations = iterations + 1
                
            envReset()
        
            if iterations == numIterations:
                print agentsQvalues
            
            #if targetCell!=robotCell:
                    
            #isTerminal = (robotCell == targetCell) or isTerminal 
       
    sys.exit("Testing")
#        if targetCell!=robotCell:
#            #update the Qtable
#            sample = transitionReward + np.amax(Qvalues[state])
#            Qvalues[state_,action] = ((1 - learningRate) * Qvalues[state_,action]) + (learningRate * sample)
#        else:
#            #update the Qtable
#            sample = transitionReward + 20
#            Qvalues[state_,action] = ((1 - learningRate) * Qvalues[state_,action]) + (learningRate * sample)
#    
#            #global iterations
#            iterations = iterations + 1
#            
#            envReset()
#    
#            if iterations == numIterations:
#                print Qvalues

Qlearning()
print QlearningEvaluation()
anim = animation.FuncAnimation(fig, timestep, init_func=init, frames=30,
                               interval=300)
plt.show()
