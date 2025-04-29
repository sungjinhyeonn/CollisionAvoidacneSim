from SimulationEngine.ClassicDEVS.DEVSAtomicModel import DEVSAtomicModel


class Generator(DEVSAtomicModel):
    def __init__(self):
        super().__init__("Generator")

        self.status = ['GEN', 'END']

        self.addOutputPort("gen_out")
        self.addStateVariable("msg", 0)
        self.addStateVariable("status", self.status[0])
        self.cnt = 0
        self.msg = 0
        pass

    def funcExternalTransition(self, strPort, objEvent):
        return True

    def funcOutput(self):
        if self.getStateValue('status') == 'GEN':
            self.addOutputEvent('gen_out', self.msg)
            return True
        elif self.getStateValue('status') == 'END':
            return True
        else:
            return False 
    
    def funcInternalTransition(self):
        if self.getStateValue('status')=='GEN':
            self.cnt += 1
            
            if self.cnt > 5:
                self.setStateValue('status', self.status[1])
            else:
                pass
            return True
        elif self.getStateValue('status')== 'END':
            return True
        else:
            return False
        pass

    def funcTimeAdvance(self):
        if self.getStateValue('status') == 'GEN':
            return 1
        elif self.getStateValue('status') =='END':
            return 9999999999999
        pass

    def funcSelect(self):
        pass
    pass