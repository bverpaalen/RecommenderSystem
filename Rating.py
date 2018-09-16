import numpy as np

nfolds = 5
ratings = []

def getTrainAndTestSets(ratings):
    # to make sure you are able to repeat results, set the random seed to something:
    np.random.seed(17)

    seqs = [x % nfolds for x in range(len(ratings))]
    np.random.shuffle(seqs)

    dataSets = []

    for fold in range(nfolds):
        train_sel = np.array([x != fold for x in seqs])
        test_sel = np.array([x == fold for x in seqs])
        train = ratings[train_sel]
        test = ratings[test_sel]
        dataSets.append([train, test])

    return dataSets

def printFirstKItems(ratings, k=10):
    print("Matrix first", k, "items:", sep= " ")
    print(ratings[:k])
    print("")

def calculateGlobalAvg(ratings):
    globalAvg = ratings[:, 2].mean()
    return globalAvg

def printGlobalAvg(globalAvg):
    print("Global Average: " + str(globalAvg) + "\n")

def aggregateByUser(trainSet, testSet):
    return aggregateById(trainSet, testSet, 0)

def aggregateByMovie(trainSet, testSet):
    return aggregateById(trainSet, testSet, 1)

def aggregateById(trainSet, testSet, column):
    # get all unique movies, and then their ratings, after that link them in a dictionary,
    # both for the training set as for the test set
    trainMovieIds = np.unique(trainSet[:, column])
    trainMovieRatings = np.array([list(trainSet[trainSet[:, column] == i, 2]) for i in trainMovieIds])
    trainMovies = dict(zip(trainMovieIds, trainMovieRatings))

    testMovieIds = np.unique(testSet[:, column])
    testMovieRatings = np.array([list(testSet[testSet[:, column] == i, 2]) for i in testMovieIds])
    testMovies = dict(zip(testMovieIds, testMovieRatings))

    return {"trainMovies": trainMovies, "testMovies": testMovies}


def userAvg(id):
    return ratingAvg(id, 0)

def movieAvg(id):
    return ratingAvg(id, 1)

def ratingAvg(id, column):
    movieRatings = ratings[np.where(ratings[:, column] == id)]
    ratingAvg = np.mean(movieRatings[:, 2])
    return ratingAvg