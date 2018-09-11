import pandas as pd
from collections import OrderedDict as OD

filedir = "ml-1m/"
filename = "ratingsSmall.dat"

def main(filePath):
    matrix = preProcessing(filePath)
    
    print("Matrix: ")
    print(matrix.head(10))
    print("")

    globalAvg = matrix["Rating"].mean()
    print("Global Average: "+str(globalAvg)+"\n")
    
    print("Pivot: ")
    matrix = matrix.pivot(index="UserID",columns="MovieID")
    print(matrix.head(10))
    print()

    usersAvgs = usersAvg(matrix)
    movieAvgs = movieAvg(matrix)
    
    print("Users: ")
    printDic(usersAvgs)

    print("Movies: ")
    printDic(movieAvgs)

def usersAvg(matrix):
    usersAvgs = OD()
    for row in matrix.iterrows():
        userId = row[0]
        userData = row[1]

        userAvg = userData.sum() / userData.count()
        usersAvgs.update({userId:userAvg})
    return usersAvgs

def movieAvg(matrix):
    movieAvgs = OD()
    for column in matrix.iteritems():
        movieId = column[0][1]
        movieData = column[1]

        movieAvg = movieData.sum() / movieData.count()
        
        movieAvgs.update({movieId:movieAvg})
    return movieAvgs	

def preProcessing(filepath):

    matrix = pd.read_csv(filepath,sep="::",engine="python",header=None,names=["UserID","MovieID","Rating","TS"])
    matrix = matrix.drop(columns=["TS"])

    return matrix

def printDic(dic):
    for key in dic.keys():
       print("id: "+str(key)+" value: "+str(dic[key]))
    print()

main(filedir+filename)
