import numpy as np

class CommandOutput:
    def __init__(self, inputTcmd, inputPcmd):
        self.dblCurrentThrust = inputTcmd
        self.dblCurrentBank = inputPcmd

    def getCommandMatrix(self):

        ret = np.zeros(2)

        ret[0] = self.dblCurrentThrust          # Thrust input (0.1~1.0)
        ret[1] = self.dblCurrentBank            # Bank angle rate input (rad/s)

        return ret
