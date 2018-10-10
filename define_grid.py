import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from  matplotlib import animation

def moveLeft(i):
    newRobotPose = robotPose
    if robotPose[1] - 1 >= 0:
        zvals[robotPose[0], robotPose[1]] = 0
        newRobotPose[1] = robotPose[1] - 1
        zvals[newRobotPose[0], newRobotPose[1]] = 1
    im.set_data(zvals)
    return zvals, newRobotPose
        
def moveRight(i):
    newRobotPose = robotPose
    # compare with num of columns
    if robotPose[1] + 1 < len(zvals[0]):
        zvals[robotPose[0], robotPose[1]] = 0
        newRobotPose[1] = robotPose[1] + 1
        zvals[newRobotPose[0], newRobotPose[1]] = 1
    else:
        print "wall ahead"
    im.set_data(zvals)
    return zvals, newRobotPose

def moveUp(i):
    newRobotPose = robotPose
    # compare with num of raws
    if robotPose[0] + 1 < len(zvals):
        zvals[robotPose[0], robotPose[1]] = 0
        newRobotPose[0] = robotPose[0] + 1
        zvals[newRobotPose[0], newRobotPose[1]] = 1
    else:
        print "wall ahead"
    im.set_data(zvals)
    return zvals, newRobotPose

def moveDown(i):
    newRobotPose = robotPose
    if robotPose[0] - 1 >= 0:
        zvals[robotPose[0], robotPose[1]] = 0
        newRobotPose[0] = robotPose[0] - 1
        zvals[newRobotPose[0], newRobotPose[1]] = 1
    else:
        print "wall ahead"
    im.set_data(zvals)
    return zvals, newRobotPose


zvals = np.zeros((5,10))
robotPose = [3,4]
fig = plt.figure()
print robotPose[0]
zvals[robotPose[0],robotPose[1]] = 1
im = plt.imshow(zvals, origin={'lower','left'})

def init():
    im.set_data(zvals)

anim = animation.FuncAnimation(fig, moveDown, init_func=init, frames=5,
                               interval=1000)

plt.show()

