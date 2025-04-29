from SimulationEngine.ClassicDEVS.DEVSCoupledModel import DEVSCoupledModel
from MUSVSimModel.USV import USV
from MUSVSimModel.Utility.MUSVLogger import MUSVLogger
from MUSVSimModel.Environment.MapModel import GridEnvironment

class MUSVSimModel(DEVSCoupledModel):
    def __init__(self, objConfiguration):
        super().__init__("AirCombatModel")

        # Read Scenario
        self.objConfiguration = objConfiguration

        strLog = objConfiguration.getConfiguration("LogFile")
        strMapDataFile = objConfiguration.getConfiguration("MapDataFile")
        dblMeterPerMapPixel = objConfiguration.getConfiguration("MeterPerMapPixel")
        objLogger = MUSVLogger(strLog)
        objGrid = GridEnvironment(strMapDataFile, dblMeterPerMapPixel)

        intNumBlue = objConfiguration.getConfiguration("intNumBlue")
        intNumRed = objConfiguration.getConfiguration("intNumRed")

        lstBlueWaypoint = objConfiguration.getConfiguration("BlueWaypoint")
        lstRedWaypoint = objConfiguration.getConfiguration("RedWaypoint")
        refLat = objConfiguration.getConfiguration("refLat")
        refLon = objConfiguration.getConfiguration("refLon")

        intRedBullet = objConfiguration.getConfiguration("redBullet")
        intBlueBullet = objConfiguration.getConfiguration("blueBullet")
        dblRedConeRad = objConfiguration.getConfiguration("redConeRad")
        dblBlueConeRad = objConfiguration.getConfiguration("blueConeRad")
        dblRedConeDist = objConfiguration.getConfiguration("redConeDist")
        dblBlueConeDist = objConfiguration.getConfiguration("blueConeDist")
        intRedHP = objConfiguration.getConfiguration("redHP")
        intBlueHP = objConfiguration.getConfiguration("blueHP")
        dblRedViewRadius = objConfiguration.getConfiguration("redViewRadius")
        dblBlueViewRadius = objConfiguration.getConfiguration("blueViewRadius")
        dblRedBankLimit = objConfiguration.getConfiguration("redBankLimit")
        dblBlueBankLimit = objConfiguration.getConfiguration("blueBankLimit")
        dblRedThrustLimit = objConfiguration.getConfiguration("redThrustLimit")
        dblBlueThrustLimit = objConfiguration.getConfiguration("blueThrustLimit")
        dblWaypointPoximity = objConfiguration.getConfiguration("WaypointPoximity")
        dblRedMaxSpeed = objConfiguration.getConfiguration("redMaxSpeed")
        dblBlueMaxSpeed = objConfiguration.getConfiguration("blueMaxSpeed")
        dblRedFridtionSpeed = objConfiguration.getConfiguration("redFridtionSpeed")
        dblBlueFridtionSpeed = objConfiguration.getConfiguration("blueFridtionSpeed")

        lstBlueAI = objConfiguration.getConfiguration("blueAI")
        lstRedAI = objConfiguration.getConfiguration("redAI")

        self.SL = None
        # self.SL = SIMDISLogger(refLat, refLon, blnVisualize3D, intVisualize3DInterval,intHistPathLength,strShowAttitudeFighter, \
        #                        dblMinX,dblMinY,dblMinZ,dblMaxX,dblMaxY,dblMaxZ)

        #블루 모델 생성
        self.objBlueJets = []
        for i in range(intNumBlue):
            objBlueUSV = USV(objLogger,objGrid,lstBlueAI[i],\
                             'Blue'+str(i+1), 'blue', lstBlueWaypoint[i], self.SL, \
                             intBlueBullet,dblBlueConeRad,dblBlueConeDist,\
                             intBlueHP,dblBlueViewRadius,dblBlueBankLimit,dblBlueThrustLimit,\
                             dblWaypointPoximity,dblBlueMaxSpeed,dblBlueFridtionSpeed)
            self.objBlueJets.append(objBlueUSV)
            self.addModel(objBlueUSV)  # Simulation Engine registered

        self.objRedJets = []

        #레드 모델 생성
        for i in range(intNumRed):
            objRedUSV = USV(objLogger,objGrid,lstRedAI[i],\
                            'Red' + str(i + 1), 'red', lstRedWaypoint[i], self.SL, \
                            intRedBullet, dblRedConeRad,dblRedConeDist,\
                            intRedHP,dblRedViewRadius,dblRedBankLimit,dblRedThrustLimit,\
                            dblWaypointPoximity,dblRedMaxSpeed,dblRedFridtionSpeed)
            self.objRedJets.append(objRedUSV)
            self.addModel(objRedUSV)  # Simulation Engine registered

        for i in range(len(self.objBlueJets)):
            for j in range(len(self.objRedJets)):

                self.addCoupling(self.objBlueJets[i], "ManeuverState_OUT", self.objRedJets[j], "ManeuverState_IN")
                self.addCoupling(self.objRedJets[j], "ManeuverState_OUT", self.objBlueJets[i], "ManeuverState_IN")

                self.addCoupling(self.objBlueJets[i], "DamageState_OUT", self.objRedJets[j], "DamageState_IN")
                self.addCoupling(self.objRedJets[j], "DamageState_OUT", self.objBlueJets[i], "DamageState_IN")

                self.addCoupling(self.objBlueJets[i], "StopSimulation_OUT", self.objRedJets[j], "StopSimulation_IN")
                self.addCoupling(self.objRedJets[j], "StopSimulation_OUT", self.objBlueJets[i], "StopSimulation_IN")