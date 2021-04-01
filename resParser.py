# -*- coding: utf-8 -*-
"""
Created on Thu Mar 25 12:50:34 2021

@author: Nikolina
"""
import sys
import os
from pathlib import Path
import matplotlib.pyplot as plt

class Result:
    def __init__(self, name):
        self.name = name
        split_string = name.split(".")
        split_string = split_string[0].split("_")
        self.datasetName = split_string[0]+"_"+split_string[1]
        self.leakageModel = split_string[2]
        self.method = split_string[3]
        self.componentNo = split_string[4]
        self.best_params = None
        
    def parseResult(self, content):
        if content.startswith("No Grid Search."):
            pass
        else:
            bp_string=content.split(";")[0].split(":")[3]
            self.best_params = int(bp_string[:-1])
        ge_start_index = content.find("ge:")+3
        ge_end_index = content.find("sr:")-1
        ge_res=content[ge_start_index:ge_end_index]
        self.geData= ge_res.split(";")
        self.geData.pop()
        self.geData = list(map(float, self.geData))
        sr_res=content[ge_end_index+5:]
        self.srData=sr_res.split(";")
        self.srData.pop()
        self.srData = list(map(float, self.srData))
        

results = []
rootFolderName = sys.argv[1]
subFolders = os.listdir(rootFolderName + '/')
for subFolder in subFolders:
    if os.path.isdir(rootFolderName + '/' + subFolder):  
        entries = os.listdir(rootFolderName + '/' + subFolder)
        for entry in entries:
            if ("log" not in entry) and ("output" not in entry):
                file = open(rootFolderName + '/' + subFolder + '/' + entry)
                line = file.read().replace("\n", " ")
                file.close()
                if line.startswith("!!!"):
                    continue
                else:
                    res = Result(entry)
                    res.parseResult(line)
                    results.append(res)

#best values
print("ches_ctf" +"-"+ "intermediate" +"-"+ "50")
for result in results:
    if result.datasetName=="ches_ctf" and result.leakageModel== "intermediate" and result.componentNo=="50":
        if result.best_params != None:
            print(result.method + ": " + str(result.best_params)) 
            


#GE
plt.figure()
for result in results:
    if result.datasetName=="ches_ctf" and result.leakageModel== "intermediate" and result.componentNo=="50":
        plt.plot(result.geData, label=result.method)

plt.title(result.datasetName +"-"+ result.leakageModel +"-"+ result.componentNo)  
plt.ylabel("GE vals")
plt.legend()
plt.show()


#SR
plt.figure()
for result in results:
    if result.datasetName=="ches_ctf" and result.leakageModel== "intermediate" and result.componentNo=="50":
        plt.plot(result.srData, label=result.method)
     
plt.title(result.datasetName +"-"+ result.leakageModel +"-"+ result.componentNo)           
plt.ylabel("SR vals")
plt.legend()
plt.show()

print("Finished")
                
    