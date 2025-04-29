from SimulationEngine.ClassicDEVS.DEVSAtomicModel import DEVSAtomicModel


class Buffer(DEVSAtomicModel):
    def __init__(self):
        super().__init__("Buffer")

        status = ['WAIT', 'SEND']

        #self.addOutputPort("gen_out")
        self.addStateVariable("msg", 0)
        self.addStateVariable("status", status[0])
        #self.cnt = 0
        self.msg = 0
        pass

    def funcExternalTransition(self, strPort, objEvent):
        print(objEvent)
        return True

    def funcOutput(self):
        return True 
    
    def funcInternalTransition(self):
        return True

    def funcTimeAdvance(self):
        if self.getStateValue('status') == 'WAIT':
            return 9999999999999
        elif self.getStateValue('status') == 'SEND':
            return 9999999999999
        pass
    
    def funcSelect(self):
        pass