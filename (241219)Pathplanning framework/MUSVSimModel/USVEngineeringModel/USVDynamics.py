import math
import numpy as np

class USVDynamics:

    # Constructor
    def __init__(self, matInitialStates,dblFrictionSpeed):
        self.matState = matInitialStates
        self.dblFrictionSpeed = dblFrictionSpeed

    def getCurrentState(self):
        return self.matState

    def executeCommand(self, matCommand, dblTimeStep):

        dblCommandThrust = matCommand[0]
        dblCommandBank = matCommand[1]

        # Current state init.
        dblPositionX = self.matState[0]
        dblPositionY = self.matState[1]
        dblYaw = self.matState[2]
        dblSpeedX = self.matState[3]
        dblSpeedY = self.matState[4]
        dblAccelerationX = self.matState[5]
        dblAccelerationY = self.matState[6]

        dblNextPositionX = dblPositionX + dblSpeedX * dblTimeStep
        dblNextPositionY = dblPositionY + dblSpeedY * dblTimeStep

        dblSpeed = math.sqrt(math.pow(dblSpeedX,2)+math.pow(dblSpeedY,2))
        dblNextSpeed = dblSpeed*self.dblFrictionSpeed + dblTimeStep*dblCommandThrust
        dblNextSpeedX = dblNextSpeed * math.cos(dblCommandBank + dblYaw)
        dblNextSpeedY = dblNextSpeed * math.sin(dblCommandBank + dblYaw)

        dblNextAccelerationX = dblCommandThrust * math.cos(dblCommandBank + dblYaw)
        dblNextAccelerationY = dblCommandThrust * math.sin(dblCommandBank + dblYaw)


        if dblNextSpeedX == 0 and dblNextSpeedY == 0:
            dblNextYaw = 0
        else:
            dblNextYaw = math.atan2(dblNextSpeedY,dblNextSpeedX)

        self.matState[0] = dblNextPositionX
        self.matState[1] = dblNextPositionY
        self.matState[2] = dblNextYaw
        self.matState[3] = dblNextSpeedX
        self.matState[4] = dblNextSpeedY
        self.matState[5] = dblNextAccelerationX
        self.matState[6] = dblNextAccelerationY


