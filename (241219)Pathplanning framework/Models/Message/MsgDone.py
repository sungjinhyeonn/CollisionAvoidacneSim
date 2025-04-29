class MsgDone:

    def __init__(self, strID, goal_x, goal_y):
        self.strID = strID
        self.goal_x = goal_x
        self.goal_y = goal_y

    # def __str__(self):
    #     ret = ""
    #     ret += "Request Message : " + str(self.strID) + " : " + f'x:{self.goal_x}, y: {self.goal_y}' 
                
    #     return ret