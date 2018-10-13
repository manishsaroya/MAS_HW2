import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from  matplotlib import animation
import sys

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
    else:
        print "wall ahead"
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
    else:
        print "wall ahead"
    #im.set_data(zvals_)
    return zvals_, newPose

def moveDown(zvals_,pose):
    newPose = pose
    if pose[0] - 1 >= 0:
        t = zvals_[pose[0], pose[1]]
        zvals_[pose[0], pose[1]] = 0
        newPose[0] = pose[0] - 1
        #zvals_[newPose[0], newPose[1]] = t
    else:
        print "wall ahead"
    #im.set_data(zvals_)
    return zvals_, newPose


# Some global learning parameters
numIterations = 100 # these many times the robot will catch the target while in training mode.
iterations = 0
Qvalues = np.zeros((2500,4)) #Qvalue is [States * Actions] matrix.
# the states are 50(target positions) * 50 (robot positions) 
# at every state the robot can take 4 actions.

# A transition reward of -1 is given to the agent at every time step.
transitionReward = -1

# the reward given to the agent at termination, i.e. when robot catches the target.
TerminateReward = 20

# learning rate between 0 and 1.
learningRate = 0.5

# the map 
zvals = np.zeros((5,10))

# intialize the random robot pose and fixed target pose.
robotPose = [np.random.randint(0,5), np.random.randint(0,10)]
targetPose = [1,9]

# assign color/ID values to robotpose and target pose in map
zvals[robotPose[0],robotPose[1]] = 1
zvals[targetPose[0],targetPose[1]] = 0.5

# Visualization
fig = plt.figure()
im = plt.imshow(zvals, origin={'lower','left'})

def init():
    im.set_data(zvals)

def timestep(i):
    
    # Before action mapping from robotPose & targetPose to Qvalues table.
    robotCell = len(zvals[0]) * robotPose[0] + robotPose[1]
    targetCell = len(zvals[0]) * targetPose[0] + robotPose[1]
    state_ = (len(zvals[0]) * len(zvals)) * robotCell + targetCell


    #move the target randomly.
    switcher = {0: moveLeft, 1: moveRight, 2: moveUp, 3: moveDown}
    func = switcher.get(np.random.randint(0,4), lambda: "Invalid input")
    newzvalstarget, newtargetPose = func(zvals,targetPose)
    
    #move the robot 
    switcher = {0: moveLeft, 1: moveRight, 2: moveUp, 3: moveDown}
    action = np.random.randint(0,4)
    func = switcher.get(action, lambda: "Invalid input")
    newzvals, newrobotPose = func(zvals, robotPose)
    

    # After action mapping from robotPose & targetPose to Qvalues table. 
    robotCell = len(zvals[0]) * robotPose[0] + robotPose[1]
    targetCell = len(zvals[0]) * targetPose[0] + robotPose[1]
    state = ((len(zvals[0]) * len(zvals)) * robotCell)+ targetCell 
    print robotCell , targetCell, state, (len(zvals[0]) * len(zvals[1]))

    #Change the Zvals;
    if targetCell!=robotCell:
        #update the Qtable
        sample = transitionReward + np.amax(Qvalues[state])
        Qvalues[state_,action] = ((1 - learningRate) * Qvalues[state_,action]) + (learningRate * sample)
    else:
        #update the Qtable
        sample = transitionReward + 20
        Qvalues[state_,action] = ((1 - learningRate) * Qvalues[state_,action]) + (learningRate * sample)

        global iterations
        iterations = iterations + 1
        # the map
        global zvals
        zvals = np.zeros((5,10))

        # intialize the random robot pose and fixed target pose.
        global robotPose
        robotPose = [np.random.randint(0,5), np.random.randint(0,10)]
        global targetPose 
        targetPose = [1,9]

        # assign color/ID values to robotpose and target pose in map
        zvals[robotPose[0],robotPose[1]] = 1
        zvals[targetPose[0],targetPose[1]] = 0.5

        if iterations == numIterations:
            for i in range(len(Qvalues)):
                if i%50 == 4:
                    print Qvalues[i]
            sys.exit("Target Found")
    #if cmp(targetPose, robotPose):
    #    zvals[robotPose[0],robotPose[1]] = 1
    #    zvals[targetPose[0],targetPose[1]] = 0.5
    #else:
    #    sys.exit("TARGET FOUND")
    
    # visualize
    print zvals
    im.set_data(zvals)
    
anim = animation.FuncAnimation(fig, timestep, init_func=init, frames=30,
                               interval=1)

plt.show()

