from SimulationEngine.ClassicDEVS.DEVSAtomicModel import DEVSAtomicModel
from MUSVSimModel.Message.MsgManeuverState import MsgManeuverState
import math

class USVSensor(DEVSAtomicModel):

    def __init__(self, strID, strSide, dblViewRadius):
        super().__init__(strID+"_Radar")

        self.addInputPort("ManeuverState_IN")
        self.addInputPort("StopSimulation")
        self.addInputPort("MyManeuverState")
        self.addOutputPort("ManeuverState_OUT")

        self.addStateVariable("RadarID",strID+"_Radar")

        self.addStateVariable("otherFighterID", None)
        self.addStateVariable("otherSide", None)
        self.addStateVariable("myPositionN", None)
        self.addStateVariable("myPositionE", None)
        self.addStateVariable("informationSent", True)
        self.addStateVariable("viewRadius", dblViewRadius)

        self.addStateVariable("Side", strSide)
        self.addStateVariable("mode", "ACTIVE")

    def funcExternalTransition(self, strPort, objEvent):
        if strPort == 'StopSimulation':
            self.setStateValue('mode','DEAD')
        if strPort == 'ManeuverState_IN':
            if self.getStateValue("myPositionN") != None:
                dblDistanceN = self.getStateValue("myPositionN") - objEvent.dblPositionN
                dblDistanceE = self.getStateValue("myPositionE") - objEvent.dblPositionE
                dblDistance = math.sqrt(math.pow(dblDistanceN,2)+math.pow(dblDistanceE,2))
                if dblDistance < self.getStateValue("viewRadius"):
                    self.setStateValue("informationSent",False)
                    self.setStateValue("otherFighterID", objEvent.strID)
                    self.setStateValue("otherSide", objEvent.strSide)
                    self.setStateValue("otherPositionN", objEvent.dblPositionN)
                    self.setStateValue("otherPositionE", objEvent.dblPositionE)

        if strPort == 'MyManeuverState':
            self.setStateValue("myPositionN", objEvent.dblPositionN)
            self.setStateValue("myPositionE", objEvent.dblPositionE)

    def funcOutput(self):
        if self.getStateValue("mode") == 'DEAD':
            return True
        if self.getStateValue("informationSent") == False:
            objManeuverMessage = MsgManeuverState(self.getStateValue("otherFighterID"),
                                                 self.getStateValue("otherSide"),
                                                 self.getStateValue("otherPositionN"),
                                                 self.getStateValue("otherPositionE"))
            self.addOutputEvent("ManeuverState_OUT", objManeuverMessage)
            self.setStateValue("informationSent", True)
        return True

    def funcInternalTransition(self):
        return True

    def funcTimeAdvance(self):
        if self.getStateValue("mode") == 'DEAD':
            return 99999999999999
        return 1

    def funcSelect(self):
        pass

