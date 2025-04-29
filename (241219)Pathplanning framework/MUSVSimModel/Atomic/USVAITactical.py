from SimulationEngine.ClassicDEVS.DEVSAtomicModel import DEVSAtomicModel

from MUSVSimModel.Message.MsgStopSimulation import MsgStopSimulation
from MUSVSimModel.Message.MsgRequestManeuver import MsgRequestManeuver
from MUSVSimModel.Message.MsgGunFire import MsgGunFire
from MUSVSimModel.Utility.MUSVLogger import MUSVLogger
import math


class USVAITactical(DEVSAtomicModel):

    def __init__(self, objLogger, objGrid, objAIPlanner, \
                 strID, strSide, lstWaypoint, numBullet, dblConeRad, dblConeDist, intHP,\
                 dblViewRadius,dblWaypointPoximity):
        super().__init__(strID+"_Pilot")

        self.objLogger = objLogger
        self.objGrid = objGrid
        self.objAIPlanner = objAIPlanner

        self.addInputPort("OtherManeuverState")
        self.addInputPort("MyManeuverState")
        self.addInputPort("DamageState")

        self.addStateVariable("HP", intHP)

        self.addOutputPort("RequestManeuver")
        self.addOutputPort("GunFire")
        self.addInputPort("StopSimulation_IN")
        self.addOutputPort("StopSimulation")

        self.addStateVariable("WaypointPoximity",dblWaypointPoximity)
        self.addStateVariable("mode", "NAVIGATION")
        self.addStateVariable("PilotID",strID+"_Pilot")
        self.addStateVariable("USVID", strID)
        self.addStateVariable("Side", strSide)

        self.addStateVariable("Waypoint", lstWaypoint)
        self.addStateVariable("WaypointIdx", 1)

        self.addStateVariable("Fire", False)

        self.addStateVariable("myPositionN", lstWaypoint[0][0])
        self.addStateVariable("myPositionE", lstWaypoint[0][1])
        self.addStateVariable("myYaw", None)
        self.addStateVariable("myVelocityX", None)
        self.addStateVariable("myVelocityY", None)

        self.addStateVariable("otherUSVID", [])
        self.addStateVariable("otherDeadUSVID", [])
        self.addStateVariable("otherPositionN", [])
        self.addStateVariable("otherPositionE", [])

        self.addStateVariable("targetN", lstWaypoint[1][0])
        self.addStateVariable("targetE", lstWaypoint[1][1])
        self.addStateVariable("targetYaw", None)

        self.addStateVariable("distance", None)

        self.addStateVariable("numBullet", numBullet)

        self.addStateVariable("DeadBroadCast", False)

        self.addStateVariable("acceptRange", dblViewRadius)
        self.addStateVariable("dblEffectRange", dblConeDist)
        self.addStateVariable("dblEffectAngle", dblConeRad)
        self.addStateVariable("done", False)

        self.objLogger.addLogObject(0, self.getStateValue("USVID"), \
                                    MUSVLogger.lstLogType[4], lstWaypoint)
        dicDisplayInfo = {}
        dicDisplayInfo["strSide"] = strSide
        dicDisplayInfo["numBullet"] =numBullet
        dicDisplayInfo["dblConeRad"] =dblConeRad
        dicDisplayInfo["dblConeDist"] =dblConeDist
        dicDisplayInfo["intHP"] =intHP
        dicDisplayInfo["dblViewRadius"] = dblViewRadius
        dicDisplayInfo["dblWaypointPoximity"] =dblWaypointPoximity
        self.objLogger.addLogDictionary(0, self.getStateValue("USVID"), \
                                    MUSVLogger.lstLogType[5], dicDisplayInfo)

    def funcExternalTransition(self, strPort, objEvent):

        if self.getStateValue("DeadBroadCast") == True:
            self.continueTimeAdvance()
            return True

        if strPort == "DamageState":
            if objEvent.strTargetID == self.getStateValue("USVID"):
                self.setStateValue("HP",int(self.getStateValue("HP")-1))
                if int(self.getStateValue("HP")) == 0:
                    self.setStateValue("mode", 'DEAD')

        if strPort == "StopSimulation":
            self.setStateValue("HP",int(self.getStateValue("HP")-1))
            if int(self.getStateValue("HP")) <= 0:
                self.setStateValue("mode", 'DEAD')

        if strPort == "StopSimulation_IN":
            lstEnemies = self.getStateValue("otherUSVID")
            lstDeadEnemies = self.getStateValue("otherDeadUSVID")
            lstOtherPositionN = self.getStateValue("otherPositionN")
            lstOtherPositionE = self.getStateValue("otherPositionE")

            intIdxEnemy = -1
            for i in range(len(lstEnemies)):
                if lstEnemies[i] == objEvent.strID:
                    intIdxEnemy = i;
                    break
            if intIdxEnemy != -1:
                lstDeadEnemies.append(objEvent.strID)
                del lstEnemies[intIdxEnemy]
                del lstOtherPositionN[intIdxEnemy]
                del lstOtherPositionE[intIdxEnemy]

            self.setStateValue("otherUSVID", lstEnemies)
            self.setStateValue("otherDeadUSVID", lstDeadEnemies)
            self.setStateValue("otherPositionN", lstOtherPositionN)
            self.setStateValue("otherPositionE", lstOtherPositionE)

            if len(lstEnemies) == 0:
                self.setStateValue("mode", "NAVIGATION")

        if strPort == "MyManeuverState":
            self.setStateValue("myPositionN", objEvent.dblPositionN)
            self.setStateValue("myPositionE", objEvent.dblPositionE)

            self.setStateValue("myYaw", objEvent.dblYaw)

            myVectorX = math.cos(objEvent.dblYaw)
            myVectorY = math.sin(objEvent.dblYaw)

            self.setStateValue("myVelocityX", objEvent.dblSpeed * myVectorX)
            self.setStateValue("myVelocityY", objEvent.dblSpeed * myVectorY)

            dicManeuverLog = {}
            dicManeuverLog['PositionN'] = self.getStateValue("myPositionN")
            dicManeuverLog['PositionE'] = self.getStateValue("myPositionE")
            dicManeuverLog['VelocityE'] = self.getStateValue("myVelocityX")
            dicManeuverLog['VelocityN'] = self.getStateValue("myVelocityY")
            dicManeuverLog['Yaw'] = self.getStateValue("myYaw")
            self.objLogger.addLogDictionary(self.getTime(),self.getStateValue("USVID"),\
                                            MUSVLogger.lstLogType[0],dicManeuverLog)
            if self.objGrid.isValidPositionNE(objEvent.dblPositionN,objEvent.dblPositionE) == False:
                self.setStateValue("mode", 'DEAD')

        if strPort == "OtherManeuverState" and objEvent.strID != None:
            lstEnemies = self.getStateValue("otherUSVID")
            lstDeadEnemies = self.getStateValue("otherDeadUSVID")

            if objEvent.strID not in lstDeadEnemies:
                lstOtherPositionN = self.getStateValue("otherPositionN")
                lstOtherPositionE = self.getStateValue("otherPositionE")

                intIdxEnemy = -1
                for i in range(len(lstEnemies)):
                    if lstEnemies[i] == objEvent.strID:
                        intIdxEnemy = i
                        break
                if intIdxEnemy == -1:
                    lstEnemies.append(objEvent.strID)
                    lstOtherPositionN.append(objEvent.dblPositionN)
                    lstOtherPositionE.append(objEvent.dblPositionE)
                else:
                    lstOtherPositionN[intIdxEnemy] = objEvent.dblPositionN
                    lstOtherPositionE[intIdxEnemy] = objEvent.dblPositionE

                self.setStateValue("otherUSVID", lstEnemies)
                self.setStateValue("otherPositionN", lstOtherPositionN)
                self.setStateValue("otherPositionE", lstOtherPositionE)

        return True

    def funcOutput(self):
        if self.getStateValue("mode") != 'DEAD':
            objRequestMessage = MsgRequestManeuver(self.getStateValue("PilotID"),
                                                 self.getStateValue("Side"),
                                                 self.getStateValue("mode"),
                                                 self.getStateValue("targetN"),
                                                 self.getStateValue("targetE"),
                                                 self.getStateValue("targetYaw"))
            self.addOutputEvent("RequestManeuver", objRequestMessage)

        if self.getStateValue("Fire"):

            dblMyPositionN = self.getStateValue("myPositionN")
            dblMyPositionE = self.getStateValue("myPositionE")
            dblTargetPositionN = self.getStateValue("FireTargetPositionN")
            dblTargetPositionE = self.getStateValue("FireTargetPositionE")
            strTargetID = self.getStateValue("FireTargetID")
            # input()

            objFire = MsgGunFire(self.getStateValue("USVID"),self.getStateValue("FireTargetID"),\
                                 dblMyPositionN,dblMyPositionE,dblTargetPositionN,dblTargetPositionE)
            self.addOutputEvent("GunFire", objFire)
            self.setStateValue("Fire", False)

            dicFireLog = {}
            dicFireLog['strTargetID'] = strTargetID
            dicFireLog['dblMyPositionN'] = dblMyPositionN
            dicFireLog['dblMyPositionE'] = dblMyPositionE
            dicFireLog['dblTargetPositionN'] = dblTargetPositionN
            dicFireLog['dblTargetPositionE'] = dblTargetPositionE
            self.objLogger.addLogDictionary(self.getTime(),self.getStateValue("USVID"),\
                                            MUSVLogger.lstLogType[2],dicFireLog)

        if self.getStateValue("mode") == 'DEAD':
            objStopSimulation = MsgStopSimulation(self.getStateValue("USVID"))
            self.addOutputEvent("StopSimulation", objStopSimulation)
            self.setStateValue("DeadBroadCast", True)

            dicDeadLog = {}
            dicDeadLog['dblMyPositionN'] = self.getStateValue("myPositionN")
            dicDeadLog['dblMyPositionE'] = self.getStateValue("myPositionE")
            dicDeadLog['dblMyYaw'] = self.getStateValue("myYaw")
            dicDeadLog['intNumBullet'] = self.getStateValue("numBullet")
            dicDeadLog['HP'] = self.getStateValue("HP")
            self.objLogger.addLogDictionary(self.getTime(),self.getStateValue("USVID"),\
                                            MUSVLogger.lstLogType[3],dicDeadLog)

        if self.getStateValue("mode") == 'DONE':
            objStopSimulation = MsgStopSimulation(self.getStateValue("USVID"))
            self.addOutputEvent("StopSimulation", objStopSimulation)
            self.setStateValue("DeadBroadCast", True)

            dicDoneLog = {}
            dicDoneLog['dblMyPositionN'] = self.getStateValue("myPositionN")
            dicDoneLog['dblMyPositionE'] = self.getStateValue("myPositionE")
            dicDoneLog['dblMyYaw'] = self.getStateValue("myYaw")
            dicDoneLog['intNumBullet'] = self.getStateValue("numBullet")
            dicDoneLog['HP'] = self.getStateValue("HP")
            self.objLogger.addLogDictionary(self.getTime(),self.getStateValue("USVID"),\
                                            MUSVLogger.lstLogType[6],dicDoneLog)


        return True

    def funcInternalTransition(self):

        if self.getStateValue("DeadBroadCast") == True:
            return True

        dicState = {}
        for strKey in self.getStates().keys():
            dicState[strKey] = self.getStateValue(strKey)
        lstBehavior = self.objAIPlanner.produceBehavior(dicState,self.objGrid)
        if lstBehavior[0] != None:
            self.setStateValue("targetN",lstBehavior[0])
            self.setStateValue("targetE",lstBehavior[1])
        if lstBehavior[2] != None:

            myPositionN = self.getStateValue("myPositionN")
            myPositionE = self.getStateValue("myPositionE")

            dblOtherPositionN = lstBehavior[3]
            dblOtherPositionE = lstBehavior[4]

            deltaN = dblOtherPositionN - myPositionN
            deltaE = dblOtherPositionE - myPositionE
            dblDistance = math.sqrt(math.pow(deltaN, 2) + math.pow(deltaE, 2))

            if self.getStateValue("numBullet") > 0:
                if dblDistance <= self.getStateValue("dblEffectRange"):  # and dblAngle < self.dblEffectAngle:
                    self.setStateValue("numBullet", self.getStateValue("numBullet") - 1)
                    self.setStateValue("FireTargetID",lstBehavior[2])
                    self.setStateValue("FireTargetPositionN", lstBehavior[3])
                    self.setStateValue("FireTargetPositionE", lstBehavior[4])
                    self.setStateValue("Fire",True)

        if lstBehavior[5] != None:
            if lstBehavior[5] == True:
                self.setStateValue("mode","DONE")

        return True

    def funcTimeAdvance(self):
        if self.getStateValue("DeadBroadCast") == True:
            return 999999999999
        else:
            return 1

    def funcSelect(self):
        pass

