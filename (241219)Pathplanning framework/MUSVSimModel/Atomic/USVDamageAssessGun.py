from SimulationEngine.ClassicDEVS.DEVSAtomicModel import DEVSAtomicModel
from MUSVSimModel.Message.MsgDamageState import MsgDamageState

import numpy as np

class USVDamageAssessGun(DEVSAtomicModel):

    def __init__(self, strID, strSide, dblConeRad, dblConeDist):
        super().__init__(strID+"DAGun")

        self.addInputPort("DamageAssess")
        self.addInputPort("MyManeuverState")
        self.addInputPort("OtherManeuverState")
        self.addInputPort("StopSimulation")
        self.addOutputPort("DamageAssess")

        self.addStateVariable("GunID",strID+"Gun")

        self.addStateVariable("myPositionN", 0)
        self.addStateVariable("myPositionE", 0)

        self.addStateVariable("yaw", 0)

        self.addStateVariable("otherPositionN", 0)
        self.addStateVariable("otherPositionE", 0)

        self.addStateVariable("Damage", False)

        self.addStateVariable("Side", strSide)
        self.addStateVariable("mode", "ACTIVE")

        self.dblEffectRange = dblConeDist
        self.dblEffectAngle = dblConeRad

    def funcExternalTransition(self, strPort, objEvent):
        if strPort == 'StopSimulation':
            self.setStateValue('mode','DEAD')
        if strPort == "OtherManeuverState":
            self.setStateValue("otherPositionN", objEvent.dblPositionN)
            self.setStateValue("otherPositionE", objEvent.dblPositionE)
        if strPort == "MyManeuverState":
            self.setStateValue("myPositionN", objEvent.dblPositionN)
            self.setStateValue("myPositionE", objEvent.dblPositionE)
            self.setStateValue("yaw", objEvent.dblYaw)
        if strPort == "DamageAssess":
            self.setStateValue("Damage", True)

    def funcOutput(self):
        if self.getStateValue("mode") == 'DEAD':
            return True
        if self.getStateValue("Damage"):

            myPositionN = self.getStateValue("myPositionN")
            myPositionE = self.getStateValue("myPositionE")
            otherPositionN = self.getStateValue("otherPositionN")
            otherPositionE = self.getStateValue("otherPositionE")
            yaw = self.getStateValue("yaw")

            a = np.array([myPositionN, myPositionE])
            b = np.array([otherPositionN, otherPositionE])
            dblDistance = np.linalg.norm(a - b)

            dblAngle = 0
            # dblAngle = BearingAngle(myPositionN, myPositionE, myPositionD, otherPositionN, otherPositionE,
            #                         otherPositionD, roll, pitch, yaw)

            if dblDistance <= self.dblEffectRange and dblAngle < self.dblEffectAngle:
                objDAMessage = MsgDamageState(self.getStateValue("GunID"))
                self.addOutputEvent("DamageState", objDAMessage)
            self.setStateValue("Damage", False)
        return True

    def funcInternalTransition(self):
        return True

    def funcTimeAdvance(self):
        if self.getStateValue("mode") == 'DEAD':
            return 99999999999999
        return 1

    def funcSelect(self):
        pass
