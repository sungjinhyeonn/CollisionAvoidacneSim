from SimulationEngine.ClassicDEVS.DEVSAtomicModel import DEVSAtomicModel

from MUSVSimModel.USVEngineeringModel.ManeuverState import ManeuverState
from MUSVSimModel.Message.MsgManeuverState import MsgManeuverState
from MUSVSimModel.USVEngineeringModel.CommandOutput import CommandOutput
from MUSVSimModel.USVEngineeringModel.USVDynamics import USVDynamics

from MUSVSimModel.Message.MsgStopSimulation import MsgStopSimulation
import sys
import math
import numpy as np

class USVManeuver(DEVSAtomicModel):

    def __init__(self, strID, strSide, lstPosition, dblBankLimit,dblThrustLimit,dblMaxSpeed,dblFrictionSpeed,\
                 dblYaw=0, dblSpeed=0, dblAcceleration=0):
        super().__init__(strID+"_Maneuver")

        # self.SL = SL
        self.strSide = strSide
        # SL.registerPlatform(strID, "aircraft", "f-16c_falcon")

        ManeuverMode = {'DEAD', 'POSITION', 'ANGLE'}

        self.addInputPort("RequestManeuver")
        self.addInputPort("StopSimulation")

        self.addOutputPort("ManeuverState")

        self.addStateVariable("mode", "POSITION")

        self.addStateVariable("USVID",strID)
        self.addStateVariable("Side", strSide)

        self.addStateVariable("dblMaxSpeed", dblMaxSpeed)
        self.addStateVariable("positionN", lstPosition[0])
        self.addStateVariable("positionE", lstPosition[1])
        if self.getStateValue("Side") == 'blue':
            self.addStateVariable("yaw", dblYaw)
            self.addStateVariable("speed", dblSpeed)
        else:
            self.addStateVariable("yaw", dblYaw)
            self.addStateVariable("speed", dblSpeed)

        self.addStateVariable("targetN", lstPosition[0])
        self.addStateVariable("targetE", lstPosition[1])
        self.addStateVariable("targetPitch", None)
        self.addStateVariable("targetYaw", None)
        self.addStateVariable("BankLimit",dblBankLimit)
        self.addStateVariable("ThrustLimit", dblThrustLimit)


        self.Lat = 0
        self.Lon = 0

        initial = self.generateInitialCondition(self.getStateValue("positionN"),
                                                self.getStateValue("positionE"),
                                                self.getStateValue("yaw"),
                                                dblSpeed,
                                                dblAcceleration)
        self.command = CommandOutput(0, 0)
        self.dynamics = USVDynamics(initial,dblFrictionSpeed)
        self.dynamics.executeCommand(self.command.getCommandMatrix(), 1.0)
        state = ManeuverState(self.dynamics.getCurrentState())

        self.addStateVariable("lastState", state)

    def funcExternalTransition(self, strPort, objEvent):
        if strPort == "StopSimulation":
            self.setStateValue("mode","DEAD")
            self.continueTimeAdvance()
            # self.SL.recordDead(self.getStateValue("FighterID"))
            return True

        if strPort == "RequestManeuver":
            if self.getStateValue("mode") != "DEAD":
                if objEvent.strMode == "EVADE":
                    self.setStateValue("mode",'ANGLE')
                    self.setStateValue("targetYaw", objEvent.dblYaw)
                else:
                    self.setStateValue("mode",'POSITION')
                    self.setStateValue("targetN", objEvent.dblPositionN)
                    self.setStateValue("targetE", objEvent.dblPositionE)

        return True

    def funcOutput(self):
        if self.getStateValue("mode") != "DEAD":
            objManeuverMessage = MsgManeuverState(self.getStateValue("USVID"),
                                                 self.getStateValue("Side"),
                                                 self.getStateValue("positionN"),
                                                 self.getStateValue("positionE"),
                                                 self.getStateValue("yaw"),
                                                 self.getStateValue("speed"))
            self.addOutputEvent("ManeuverState", objManeuverMessage)
        return True

    def funcInternalTransition(self):
        if self.getStateValue("mode") == 'DEAD':
            pass
        else:
            self.decideCommand()
            self.stateTransition()

        return True

    def funcTimeAdvance(self):
        if self.getStateValue("mode") == "DEAD":
            return 9999999999999
        else:
            return 1

    def funcSelect(self):
        pass

    def generateInitialCondition(self, positionN, positionE, yaw, speed, acceleration):
        originalSpeed = speed
        originalPosition1 = positionE
        originalPosition2 = positionN
        originalAngle1 = yaw

        # Initial State: Total Speed(0), Position(1:2), Angle(3)
        X_Init = np.zeros(7)
        X_Init[0] = originalPosition1 # Position X
        X_Init[1] = originalPosition2 # Position Y
        X_Init[2] = originalAngle1  # Position Y
        X_Init[3] = originalSpeed * math.cos(yaw) # Total Speed X
        X_Init[4] = originalSpeed * math.sin(yaw)  # Total Speed X
        X_Init[5] = acceleration * math.cos(yaw) # Total Speed X
        X_Init[6] = acceleration * math.sin(yaw)  # Total Speed X

        return X_Init

    def decideCommand(self):
        state = self.getStateValue("lastState")
        dblMaxSpeed = self.getStateValue("dblMaxSpeed")
        if self.getStateValue("mode") == "POSITION":
            targetN = self.getStateValue("targetN")
            targetE = self.getStateValue("targetE")
            self.command = self.controlUSVPositionPG(state, targetN, targetE, dblMaxSpeed)
            # input()


    def controlUSVPositionPG(self, objState, dblTargetN, dblTargetE, dblMaxSpeed):
        dblPositionN = objState.dblPositionY
        dblPositionE = objState.dblPositionX
        dblYaw = objState.dblYaw

        dblFutureSpeedX = 0.0 * objState.dblAccelerationX + objState.dblSpeedX
        dblFutureSpeedY = 0.0 * objState.dblAccelerationY + objState.dblSpeedY
        if dblFutureSpeedX == 0 and dblFutureSpeedY == 0:
            dblFutureSpeedYaw = 0
        else:
            dblFutureSpeedYaw = math.atan2(dblFutureSpeedY,dblFutureSpeedX)

        # if dblFutureSpeedX >= 0 and dblFutureSpeedY >= 0:
        #     if dblFutureSpeedX == 0 and dblFutureSpeedY == 0:
        #         dblFutureSpeedYaw = 0
        #     else:
        #         dblFutureSpeedYaw = math.atan(dblFutureSpeedY/dblFutureSpeedX)
        # if dblFutureSpeedX < 0 and dblFutureSpeedY >= 0:
        #     dblFutureSpeedYaw = 3.14159/2.0 + math.atan(-dblFutureSpeedY/dblFutureSpeedX)
        # if dblFutureSpeedX < 0 and dblFutureSpeedY < 0:
        #     dblFutureSpeedYaw = 3.14159 + math.atan(dblFutureSpeedY/dblFutureSpeedX)
        # if dblFutureSpeedX >= 0 and dblFutureSpeedY < 0:
        #     dblFutureSpeedYaw = 3.14159*3.0/2.0 + math.atan(-dblFutureSpeedY/dblFutureSpeedX)

        dblMoveN = dblTargetN - dblPositionN
        dblMoveE = dblTargetE - dblPositionE
        if dblMoveN == 0 and dblMoveE == 0:
            dblTargetYaw = 0
        else:
            dblTargetYaw = math.atan2(dblMoveN,dblMoveE)

        if math.isnan(dblTargetYaw):
            dblBank = 0
        else:
            dblBank = self.getBankFromYaw(dblTargetYaw,dblFutureSpeedYaw)

            if abs(dblBank) > self.getStateValue("BankLimit"):
                if dblBank < 0:
                    dblBank = -self.getStateValue("BankLimit")
                else:
                    dblBank = self.getStateValue("BankLimit")

        dblSpeed = math.sqrt(math.pow(objState.dblSpeedX,2) + math.pow(objState.dblSpeedY,2))
        dblFutureSpeed = math.sqrt(math.pow(objState.dblSpeedX, 2) + math.pow(objState.dblSpeedY, 2)) + \
                         math.sqrt(math.pow(objState.dblAccelerationX,2) + math.pow(objState.dblAccelerationY,2)) * \
                         0.0

        if (dblMaxSpeed*10.0) > math.sqrt(math.pow(dblMoveN,2) + math.pow(dblMoveE,2)):
            dblTargetSpeed = math.sqrt(math.pow(dblMoveN,2) + math.pow(dblMoveE,2)) / 10.0
        else:
            dblTargetSpeed = dblMaxSpeed

        dblThrust = dblTargetSpeed - dblFutureSpeed
        if abs(dblThrust) > self.getStateValue("ThrustLimit"):
            if dblThrust < 0:
                dblThrust = -self.getStateValue("ThrustLimit")
            else:
                dblThrust = self.getStateValue("ThrustLimit")

        objCommand = CommandOutput(dblThrust,dblBank)
        return objCommand


    def getBankFromYaw(self,dblTargetYaw,dblYaw):

        dblTagetYawAdjusted = dblTargetYaw - dblYaw

        if dblTagetYawAdjusted > 3.14159:
            dblTagetYawAdjusted = - ( 3.14159 * 2.0 - dblTagetYawAdjusted )
        if dblTagetYawAdjusted < -3.14159:
            dblTagetYawAdjusted = 3.14159 * 2.0 + dblTagetYawAdjusted

        return dblTagetYawAdjusted

        # dblBank1 = dblTargetYaw - dblYaw
        # dblBank2 = 3.14159 * 2.0 - ( dblTargetYaw - dblYaw )
        # if abs(dblBank1) > abs(dblBank2):
        #     return dblBank2
        # else:
        #     return dblBank1



    def stateTransition(self):
        self.dynamics.executeCommand(self.command.getCommandMatrix(), 1.0)
        nextState = ManeuverState(self.dynamics.getCurrentState())
        self.setStateValue("positionN", nextState.dblPositionY)
        self.setStateValue("positionE", nextState.dblPositionX)
        self.setStateValue("yaw", nextState.dblYaw)
        self.setStateValue("speed", nextState.dblSpeed)
        self.setStateValue("lastState", nextState)

        # LLA = convertNEDtoLLA([nextState.dblPositionX, nextState.dblPositionY, nextState.dblPositionZ], self.refLat,
        #                       self.refLon)
        # self.Lat = LLA[0]
        # self.Lon = LLA[1]

        # self.SL.recordMovement(self.getStateValue("FighterID"), \
        #                        self.getTime() / 100, LLA[0], LLA[1], LLA[2], nextState.dblYaw, \
        #                        nextState.dblPitch, nextState.dblRoll,nextState.dblPositionX, \
        #                        nextState.dblPositionY, nextState.dblPositionZ)
