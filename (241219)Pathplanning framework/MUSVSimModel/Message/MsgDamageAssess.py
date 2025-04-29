

class MsgDamageAssess:

    def __init__(self, strShooterID, strTargetID):
        self.strShooterID = strShooterID
        self.strTargetID = strTargetID

    def __str__(self):
        return "MsgDamageAssess : "+str(self.strShooterID)+"-->"+str(self.strTargetID)
