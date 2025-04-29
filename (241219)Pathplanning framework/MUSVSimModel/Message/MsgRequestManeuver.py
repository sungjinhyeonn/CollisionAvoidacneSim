

class MsgRequestManeuver:

    def __init__(self, strID, strSide, strMode, dblPositionN, dblPositionE, dblYaw):
        self.strID = strID
        self.strSide = strSide
        self.strMode = strMode
        self.dblPositionN = dblPositionN
        self.dblPositionE = dblPositionE
        self.dblYaw = dblYaw

    def __str__(self):
        ret = ""
        ret += "Request Message : " + str(self.strID) + " : " + str(self.strSide) + " : (" + \
                str(self.dblPositionN) + "," + str(self.dblPositionE) + "," + \
                str(self.dblYaw) + ")"
        return ret