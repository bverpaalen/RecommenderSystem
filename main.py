import pandas as pd
import numpy as np
import Rating as rating
from collections import OrderedDict as OD
from sklearn import linear_model

filedir = "ml-1m/"
filename = "ratings1000.dat"
ratings = []
nfolds = 5

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

    rating.getTrainAndTestSets()

    trainAndTest(matrix, globalAvg, movieAvgs)

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

    f = open(filepath, 'r')
    for line in f:
        data = line.split('::')
        global ratings
        ratings.append([int(z) for z in data[:3]])
    f.close()
    ratings = np.array(ratings)

    return matrix

def printDic(dic):
    for key in dic.keys():
       print("id: "+str(key)+" value: "+str(dic[key]))
    print()

def trainAndTest(matrix, globalAvg, movieAvgs):

    errTrain = {"GlobalAvg": np.zeros(nfolds), "ItemAvg": np.zeros(nfolds)}
    errTest = {"GlobalAvg": np.zeros(nfolds), "ItemAvg": np.zeros(nfolds)}

    # to make sure you are able to repeat results, set the random seed to something:
    np.random.seed(17)

    seqs = [x % nfolds for x in range(len(ratings))]
    np.random.shuffle(seqs)

    # for each fold:
    for fold in range(nfolds):
        train_sel = np.array([x != fold for x in seqs])
        test_sel = np.array([x == fold for x in seqs])
        train = ratings[train_sel]
        test = ratings[test_sel]

        applyGlobalAvg(errTrain, errTest, fold, train, test, globalAvg)
        applyAvgPerItem(errTest, fold, train, test, movieAvgs)
        print()

    # print the final conclusion:
    print("\n")
    print("Mean error item AVG: " + str(np.mean(errTest["ItemAvg"])))

def applyGlobalAvg(errTrain, errTest, fold, train, test, globalAvg):
    print("The global average rating")
    errTrain["GlobalAvg"][fold] = np.sqrt(np.mean((train[:, 2] - globalAvg) ** 2))
    errTest["GlobalAvg"][fold] = np.sqrt(np.mean((test[:, 2] - globalAvg) ** 2))

    # print errors:
    print("Fold " + str(fold) + ": RMSE_train=" + str(errTrain["GlobalAvg"][fold]) + "; RMSE_test=" + str(errTest["GlobalAvg"][fold]))

def applyAvgPerItem(errTest, fold, train, test, movieAvgs):
    print("The average rating per item")

    # get all unique movies, and then their ratings, after that link them in a dictionary,
    # both for the training set as for the test set
    trainMovieIds = np.unique(train[:, 1])
    trainMovieRatings = np.array([list(train[train[:, 1] == i, 2]) for i in trainMovieIds])
    trainMovies = dict(zip(trainMovieIds, trainMovieRatings))

    testMovieIds = np.unique(test[:, 1])
    testMovieRatings = np.array([list(test[test[:, 1] == i, 2]) for i in testMovieIds])
    testMovies = dict(zip(testMovieIds, testMovieRatings))

    # to keep track of the RMSE's of all test sets
    rmses = []
    for id, ratings in trainMovies.items():
        movieAvg = np.mean(ratings)

        if id in testMovies:
            testMovieRatings = testMovies[id]
        else:
            # if movie does not exist in test set
            # use global avg of the movie
            testMovieRatings = movieAvgs[id]

        rmses.append(np.sqrt(np.mean((movieAvg - testMovieRatings) ** 2)))

    #avg err per fold
    errTest["ItemAvg"][fold] = np.mean(rmses)
    '''
    np.mean(ratings - movieAvg)
            for i, testMovieRating in enumerate(testMovieRatings):
                sse += movieAvg - testMovieRating


            som_err_train += np.mean(ratings - movieAvg)
            iterations += 1
    


    avg_err_train = som_err_train / iterations
    err_train[1][fold] = np.sqrt(np.mean((avg_err_train) ** 2))

    testMovieIds = np.unique(test[:, 1])
    testMovieRatings = np.array([list(test[test[:, 1] == i, 2]) for i in testMovieIds])
    som_err_test = 0
    iterations = 0
    for movieId in testMovieIds:
        for ratings in testMovieRatings:
            movieAvg = movieAvgs[movieId]
            som_err_test += np.mean(ratings - movieAvg)
            iterations += 1
    avg_err_test = som_err_test / iterations
    err_test[1][fold] = np.sqrt(np.mean((avg_err_test) ** 2))
    '''

    # print errors:
    print("Fold " + str(fold) + ": RMSE" + str(errTest["ItemAvg"][fold]))

main(filedir+filename)
