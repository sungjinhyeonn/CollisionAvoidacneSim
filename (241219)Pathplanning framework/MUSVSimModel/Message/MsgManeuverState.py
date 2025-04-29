

class MsgManeuverState:

    def __init__(self, strID, strSide, dblPositionN, dblPositionE, dblYaw=None, dblSpeed=None):
        self.strID = strID
        self.strSide = strSide
        self.dblPositionN = dblPositionN
        self.dblPositionE = dblPositionE
        self.dblYaw = dblYaw
        self.dblSpeed = dblSpeed

    def __str__(self):
        ret = ""
        ret += "Manuever Message : " + str(self.strID) + " : "+str(self.strSide)+" : ("+ \
            str(self.dblPositionN) + "," + str(self.dblPositionE) + ")"
        return ret

