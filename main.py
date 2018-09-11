import pandas as pd

filedir = "ml-1m/"
filename = "ratingsSmall.dat"

def main(filePath):
    matrix = preProcessing(filePath)
    print(matrix.head(10))
    print("\n")

    globalAvg = matrix["Rating"].mean()
    print("Global Average: "+str(globalAvg)+"\n")
    
    matrix = matrix.pivot(index="UserID",columns="MovieID")
    print(matrix.head(10))

def preProcessing(filepath):

    matrix = pd.read_csv(filepath,sep="::",engine="python",header=None,names=["UserID","MovieID","Rating","TS"])
    matrix = matrix.drop(columns=["TS"])

    return matrix

main(filedir+filename)
