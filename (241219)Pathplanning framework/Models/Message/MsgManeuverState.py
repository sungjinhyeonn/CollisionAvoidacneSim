

class MsgManeuverState:

    def __init__(self, strID, dblPositionN, dblPositionE, dblYaw=None, dblSpeed=None):
        self.strID = strID

        self.dblPositionN = dblPositionN
        self.dblPositionE = dblPositionE
        self.dblYaw = dblYaw
        self.dblSpeed = dblSpeed

    def __str__(self):
        ret = ""
        ret += "Manuever Message : " + str(self.strID) + " : ("+ \
            str(self.dblPositionN) + "," + str(self.dblPositionE) + ")"
        return ret

