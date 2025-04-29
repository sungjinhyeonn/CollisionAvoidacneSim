import math
import numpy as np

class ManeuverState:

    def __init__(self, inputState):
        self.state = np.zeros(7)

        # state X[19]: Total Speed(0), Position(1: 3), DCM(4: 12), EulerAngle(13: 15), Command(16: 18)
        for i in range(7):
            self.state[i] = inputState[i]

        self.dblPositionX = inputState[0]
        self.dblPositionY = inputState[1]
        self.dblYaw = inputState[2]
        self.dblSpeedX = inputState[3]
        self.dblSpeedY = inputState[4]
        self.dblAccelerationX = inputState[5]
        self.dblAccelerationY = inputState[6]
        self.dblSpeed = math.sqrt(math.pow(self.dblSpeedX,2)+math.pow(self.dblSpeedY,2))

