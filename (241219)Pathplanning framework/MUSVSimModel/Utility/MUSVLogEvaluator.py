import csv


class MUSVLogEvaluator:

    def __init__(self,strLogFile):
        self.strLogFile = strLogFile

        objFile = open(strLogFile,'r',newline='')
        objReader = csv.reader(objFile)
        self.dblMaxTime = 0
        self.lstLog = []
        for lstLine in objReader:
            lstLine[0] = float(lstLine[0])
            if lstLine[0] > self.dblMaxTime:
                self.dblMaxTime = lstLine[0]
            self.lstLog.append(lstLine)
        objFile.close()

        self.lstLogDead = []
        for lstLogInstance in self.lstLog:
            if lstLogInstance[2] == 'Dead':
                self.lstLogDead.append(lstLogInstance)

        self.lstLogDone = []
        for lstLogInstance in self.lstLog:
            if lstLogInstance[2] == 'Done':
                self.lstLogDone.append(lstLogInstance)

    def getEvaluationState(self):
        return self.lstLogDead + self.lstLogDone
