

class MsgStopSimulation:

    def __init__(self, strID):
        self.strID = strID

    def __str__(self):
        ret = "STOP SIMULATION ("+self.strID+")"
        return ret
