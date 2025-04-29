class MsgCurrentPose:

    def __init__(self, strID, x, y, yaw, lin_vel, ang_vel, time):
        self.strID = strID
        self.x = x
        self.y = y
        self.yaw = yaw
        self.lin_vel=lin_vel
        self.ang_vel=ang_vel
        self.time = time

    def __str__(self):
        ret = ""
        ret += f'{self.x}, {self.y}' 
                
        return ret