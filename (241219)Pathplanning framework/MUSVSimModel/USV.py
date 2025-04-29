from SimulationEngine.ClassicDEVS.DEVSCoupledModel import DEVSCoupledModel
from MUSVSimModel.Atomic.USVManeuver import USVManeuver
from MUSVSimModel.Atomic.USVTactical import USVTactical
from MUSVSimModel.Atomic.USVAITactical import USVAITactical
from MUSVSimModel.Atomic.USVSensor import USVSensor
from MUSVSimModel.Atomic.USVGun import USVGun
from MUSVSimModel.Atomic.USVDamageAssessGun import USVDamageAssessGun

class USV(DEVSCoupledModel):
    def __init__(self, objLogger, objGrid, objAI,\
                 strName, strSide, lstWaypoint, SL, \
                 intNumBullet, dblConeRad, dblConeDist, intHP, dblViewRadius,\
                 dblBankLimit,dblThrustLimit,dblWaypointPoximity,dblMaxSpeed,dblFrictionSpeed):
        super().__init__(strName)

        self.addInputPort("ManeuverState_IN")
        self.addInputPort("DamageState_IN")
        self.addInputPort("StopSimulation_IN")
        self.addOutputPort("ManeuverState_OUT")
        self.addOutputPort("DamageState_OUT")
        self.addOutputPort("StopSimulation_OUT")

        self.SL = SL

        #USVManeuver 모델
        self.objManeuver = USVManeuver(strName, strSide, lstWaypoint[0],dblBankLimit,dblThrustLimit,dblMaxSpeed,\
                                       dblFrictionSpeed)
        if objAI == None:
            self.objPilot = USVTactical(objLogger, objGrid,\
                                        strName, strSide, lstWaypoint, intNumBullet, \
                                        dblConeRad, dblConeDist, intHP, dblViewRadius,dblWaypointPoximity)
        else:
            self.objPilot = USVAITactical(objLogger, objGrid, objAI,\
                                        strName, strSide, lstWaypoint, intNumBullet, \
                                        dblConeRad, dblConeDist, intHP, dblViewRadius,dblWaypointPoximity)
        self.objRadar = USVSensor(strName, strSide, dblViewRadius)
        self.objGun = USVGun(strName, strSide, dblConeRad, dblConeDist)
        # self.objDAGun = USVDamageAssessGun(strName, strSide, dblConeRad, dblConeDist)
        self.addModel(self.objManeuver)  # Simulation Engine registered
        self.addModel(self.objPilot)  # Simulation Engine registered
        self.addModel(self.objRadar)  # Simulation Engine registered
        self.addModel(self.objGun)  # Simulation Engine registered
        # self.addModel(self.objDAGun)  # Simulation Engine registered

        #port coupling
        self.addCoupling(self.objManeuver, "ManeuverState", self, "ManeuverState_OUT")
        self.addCoupling(self.objManeuver, "ManeuverState", self.objRadar, "MyManeuverState")
        self.addCoupling(self.objManeuver, "ManeuverState", self.objPilot, "MyManeuverState")
        # self.addCoupling(self.objManeuver, "ManeuverState", self.objDAGun, "MyManeuverState")

        # self.addCoupling(self, "ManeuverState_IN", self.objDAGun, "OtherManeuverState")
        self.addCoupling(self, "ManeuverState_IN", self.objRadar, "ManeuverState_IN")
        self.addCoupling(self.objRadar, "ManeuverState_OUT", self.objPilot, "OtherManeuverState")

        self.addCoupling(self.objPilot, "RequestManeuver", self.objManeuver, "RequestManeuver")

        self.addCoupling(self.objPilot, "GunFire", self.objGun, "GunFire")
        self.addCoupling(self.objGun, "DamageAssess", self, "DamageState_OUT")
        # self.addCoupling(self.objGun, "DamageAssess", self.objDAGun, "DamageAssess")
        # self.addCoupling(self.objDAGun, "DamageState", self, "DamageState_OUT")
        self.addCoupling(self, "DamageState_IN", self.objPilot, "DamageState")

        self.addCoupling(self.objPilot, "StopSimulation", self.objManeuver, "StopSimulation")
        self.addCoupling(self.objPilot, "StopSimulation", self.objRadar, "StopSimulation")
        self.addCoupling(self.objPilot, "StopSimulation", self.objGun, "StopSimulation")
        # self.addCoupling(self.objPilot, "StopSimulation", self.objDAGun, "StopSimulation")
        self.addCoupling(self.objPilot, "StopSimulation", self, "StopSimulation_OUT")
        self.addCoupling(self, "StopSimulation_IN", self.objPilot, "StopSimulation_IN")

