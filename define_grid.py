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
    
    #move the target randomly.
    switcher = {0: moveLeft, 1: moveRight, 2: moveUp, 3: moveDown}
    func = switcher.get(np.random.randint(0,4), lambda: "Invalid input")
    newzvalstarget, newtargetPose = func(zvals,targetPose)
    
    #move the robot 
    switcher = {0: moveLeft, 1: moveRight, 2: moveUp, 3: moveDown}
    func = switcher.get(np.random.randint(0,4), lambda: "Invalid input")
    newzvals, newrobotPose = func(zvals, robotPose)
    
    #Change the Zvals;
    if cmp(targetPose, robotPose):
        zvals[robotPose[0],robotPose[1]] = 1
        zvals[targetPose[0],targetPose[1]] = 0.5
    else:
        sys.exit("TARGET FOUND")
    
    # visualize
    print zvals
    im.set_data(zvals)
    
anim = animation.FuncAnimation(fig, timestep, init_func=init, frames=30,
                               interval=50)

plt.show()

