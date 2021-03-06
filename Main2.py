import numpy as np
import Data as data
import Rating as rating
import RatingPredictor as rp

filedir = "ml-1m/"
ratingsFilename = "ratings.dat"
#ratingsFilename = "ratings.csv"
nfolds = 5

def main():
    ratings = data.readFile(filedir+ratingsFilename, "::")
    rating.ratings = ratings

    rating.printFirstKItems(ratings)

    globalAvg = rating.calculateGlobalAvg(ratings)
    rating.printGlobalAvg(globalAvg)

    trainAndTestSets = rating.getTrainAndTestSets(ratings)

    rp.predictBasedOnGlobalAvg(trainAndTestSets, globalAvg)

    aggItemTrain = rp.predictRatingPerItem(trainAndTestSets, globalAvg)

    aggUserTrain = rp.predictRatingPerUser(trainAndTestSets, globalAvg)

    rp.predictByLinairRegression(trainAndTestSets, aggItemTrain, aggUserTrain, globalAvg)

    #userFactors, itemFactors = rating.getUserAndItemFactors(ratings)

    #rp.predictByMatrixFactorization(trainAndTestSets, aggUserTrain, aggItemTrain, globalAvg)


main()
