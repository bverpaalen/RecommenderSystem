import pandas as pd
import numpy as np
import Data as data
import Rating as rating
import RatingPredictor as rp
from collections import OrderedDict as OD
from sklearn import linear_model

filedir = "ml-1m/"
ratingsFilename = "ratings.dat"
nfolds = 5

def main():
    ratings = data.readFile(filedir+ratingsFilename)
    rating.ratings = ratings

    rating.printFirstKItems(ratings)

    globalAvg = rating.calculateGlobalAvg(ratings)
    rating.printGlobalAvg(globalAvg)

    trainAndTestSets = rating.getTrainAndTestSets(ratings)

    #rp.predictBasedOnGlobalAvg(trainAndTestSets, globalAvg)

    #predictedRatingsPerItem = rp.predictRatingPerItem(trainAndTestSets)

    #predictedRatingsPerUser = rp.predictRatingPerUser(trainAndTestSets)

    #rp.predictByLinairRegression(trainAndTestSets, predictedRatingsPerItem, predictedRatingsPerUser)

    #userFactors, itemFactors = rating.getUserAndItemFactors(ratings)

    rp.predictByMatrixFactorization(trainAndTestSets, 0, 0)

main()
