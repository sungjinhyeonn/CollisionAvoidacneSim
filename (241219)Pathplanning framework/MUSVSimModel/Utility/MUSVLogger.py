import csv

class MUSVLogger:

    lstLogType = ['Maneuver','Sensor','Fire','Dead','Waypoint','Info','Done']

    def __init__(self,strFileName):
        self.objLog = open(strFileName,"w",newline='')
        self.objWriter = csv.writer(self.objLog)

    def __del__(self):
        self.objLog.close()

    def addLogDictionary(self,dblTimestep,strUSVID,strLogType,dicRecord):
        lstWrite = []
        lstWrite.append(dblTimestep)
        lstWrite.append(strUSVID)
        lstWrite.append(strLogType)
        for objKey in dicRecord.keys():
            lstWrite.append(objKey)
            lstWrite.append(dicRecord[objKey])
        self.objWriter.writerow(lstWrite)
        self.objLog.flush()

    def addLogObject(self,dblTimestep,strUSVID,strLogType,objRecord):
        lstWrite = []
        lstWrite.append(dblTimestep)
        lstWrite.append(strUSVID)
        lstWrite.append(strLogType)
        lstWrite.append(str(objRecord))
        self.objWriter.writerow(lstWrite)
        self.objLog.flush()