from SimulationEngine.ClassicDEVS.DEVSAtomicModel import DEVSAtomicModel
from MUSVSimModel.Message.MsgDamageAssess import MsgDamageAssess

import numpy as np

class USVGun(DEVSAtomicModel):

    def __init__(self, strID, strSide, dblConeRad, dblConeDist):
        super().__init__(strID+"Gun")

        self.intBulletAmount = 100

        self.addInputPort("GunFire")
        self.addInputPort("StopSimulation")
        self.addOutputPort("DamageAssess")

        self.addStateVariable("USVID", strID)
        self.addStateVariable("Fire", False)
        self.addStateVariable("FireTargetID", False)
        self.addStateVariable("GunID",strID+"Gun")

        self.addStateVariable("Side", strSide)

        self.addStateVariable("mode","ACTIVE")

    def funcExternalTransition(self, strPort, objEvent):
        if strPort == 'GunFire':
            self.setStateValue("Fire", True)
            self.setStateValue("FireTargetID", objEvent.strTargetID)

        if strPort == 'StopSimulation':
            self.setStateValue('mode','DEAD')

        return True

    def funcOutput(self):
        if self.getStateValue("mode") == 'DEAD':
            return True
        if self.getStateValue("Fire"):
            self.intBulletAmount = self.intBulletAmount - 1
            objDAMessage = MsgDamageAssess(self.getStateValue("USVID"),\
                                           self.getStateValue("FireTargetID"))
            self.addOutputEvent("DamageAssess", objDAMessage)
            self.setStateValue("Fire", False)
        return True

    def funcInternalTransition(self):
        return True

    def funcTimeAdvance(self):
        if self.getStateValue("mode") == 'DEAD':
            return 99999999999999
        return 1

    def funcSelect(self):
        pass
