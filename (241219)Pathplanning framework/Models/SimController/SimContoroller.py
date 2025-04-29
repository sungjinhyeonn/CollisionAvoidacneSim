from SimulationEngine.ClassicDEVS.DEVSCoupledModel import DEVSCoupledModel
from Models.SimController.DataCollector import DataCollector

class SimController(DEVSCoupledModel):
    def __init__(self, ID, objConfiguration):
        super().__init__(ID)

        self.addInputPort("ManeuverState_IN")
        self.addInputPort("DoneReport")
        self.addInputPort("StopSimulation_IN")
        self.addOutputPort("StopSimulation_OUT")

        self.objCollecter = DataCollector(ID+'_Collector', objConfiguration)
        self.addModel(self.objCollecter)

        self.addCoupling(self, "ManeuverState_IN", self.objCollecter, "ManeuverState_IN")
        self.addCoupling(self, "DoneReport", self.objCollecter, "DoneReport")
        self.addCoupling(self.objCollecter, "StopSimulation_OUT", self, "StopSimulation_OUT")
        