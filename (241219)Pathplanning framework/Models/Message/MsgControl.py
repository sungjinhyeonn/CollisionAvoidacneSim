class MsgRequestManeuverControl:

    def __init__(self, strID, lin_vel, ang_vel):
        self.strID = strID
        self.lin_vel = lin_vel
        self.ang_vel = ang_vel

    # def __str__(self):
    #     ret = ""
    #     ret += "Request Message : " + str(self.strID) + " : " + f'x:{self.lin_vel}, y: {self.ang_vel}' 
                
    #     return ret