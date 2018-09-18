import numpy as np
import Rating as rating
import Styler as styler

def predictBasedOnGlobalAvg(trainAndTestSets, globalAvg):
    styler.printHeader("The global average rating")
    print("")

    rmses = []
    for fold, dataSet in enumerate(trainAndTestSets):
        trainSet = dataSet[0]
        testSet = dataSet[1]

        trainErr = np.sqrt(np.mean((globalAvg - trainSet[:, 2]) ** 2))
        testErr = np.sqrt(np.mean((globalAvg - testSet[:, 2]) ** 2))

        rmses.append([trainErr, testErr])

        # print errors:
        print("Fold " + str(fold) + ": RMSE_train=" + str(trainErr) + ";\n\t\tRMSE_test=" + str(testErr) + ".")

    rmses = np.array(rmses)
    # print the final conclusion for the global avg error as mean of the test sets:
    styler.printFooter("Mean global avg error: " + str(np.mean(rmses[:,1])))

def predictRatingPerUser(trainAndTestSets):
    styler.printHeader("The average rating per user")

    predictedRatings = predictRating(trainAndTestSets, 0)
    meanRmses = predictedRatings[0]

    # print the final conclusion for avg per user:
    styler.printFooter("Mean user avg error: " + str(np.mean(meanRmses)))

    return predictedRatings[1]

def predictRatingPerItem(trainAndTestSets):
    styler.printHeader("The average rating per item")

    predictedRatings = predictRating(trainAndTestSets, 1)
    meanRmses = predictedRatings[0]

    # print the final conclusion for avg per item:
    styler.printFooter("Mean item avg error: " + str(np.mean(meanRmses)))

    return predictedRatings[1]

# basedOn is column nr; user or movie
def predictRating(trainAndTestSets, basedOn):
    print("")
    meanRmses = []
    foldMovieRatings = [{} for i in range(len(trainAndTestSets))]
    for fold, dataSet in enumerate(trainAndTestSets):
        trainSet = dataSet[0]
        testSet = dataSet[1]

        movieRatings = []
        if(basedOn == 0):
            # aggregate ratings by user id
            movieRatings = rating.aggregateByUser(trainSet, testSet)
        else:
            # aggregate ratings by movie id
            movieRatings = rating.aggregateByMovie(trainSet, testSet)

        trainMovies = movieRatings["trainMovies"]
        testMovies = movieRatings["testMovies"]

        foldMovieRatings[fold].update(trainMovies)

        rmses = []
        for id, testRatings in testMovies.items():

            movieAvg = 0.0
            # if movie does not exist in train set, then use global avg of the movie
            if id in trainMovies:
                movieAvg = np.mean(trainMovies[id])
            else:
                movieAvg = rating.movieAvg(id)

            err = movieAvg - testRatings
            sqErr = (movieAvg - testRatings) ** 2
            # improve predictions by rounding values bigger than 5 to 5 and smaller than 1 to 1 (valid ratings are always between 1 and 5).
            #sqErr[sqErr < 1] = 1
            #sqErr[sqErr > 5] = 5
            rmses.append(np.sqrt(np.mean(sqErr)))

        meanRmses.append(np.mean(rmses))

        # print error:
        print("Fold " + str(fold) + ": RMSE: " + str(meanRmses[fold]))

    return [meanRmses, foldMovieRatings]

def predictByLinairRegression(trainAndTestSets, predictedRatingsPerItem, predictedRatingsPerUser):
    styler.printHeader("Linear combination of user and item averages")

    trainLoops = 100

    alpha = 2
    beta = 3
    delta = 0.5

    linearPrediction = []
    for fold, dataSet in enumerate(trainAndTestSets):
        trainSet = dataSet[0]

        for userId, itemId, actualRating in trainSet:
            predictedRatingPerUser = np.mean(predictedRatingsPerUser[fold][userId])
            predictedRatingPerItem = np.mean(predictedRatingsPerItem[fold][itemId])

            # Our task is to find coefficients A, B, C, such that the linear combination:
            # A*x + B*y + C  approximates y as good as possible.
            linearPrediction.append([predictedRatingPerUser, predictedRatingPerItem])

    linearPrediction = np.array(linearPrediction)

    predictionUserItem = alpha * linearPrediction[:, 0] + beta * linearPrediction[:, 1] + delta
    S = np.linalg.lstsq(linearPrediction, predictionUserItem, rcond=-1)

    SSEOld = S[1] 
    SSENew = 0

    for i in range(0,100,1):
        predictionUserItem = alpha * linearPrediction[:, 0] + beta * linearPrediction[:, 1] + delta + 0.1 * np.random.randn(len(linearPrediction[:,0]))

        # S[0] is what we need (coefficients, A, B, C):
        S = np.linalg.lstsq(linearPrediction, predictionUserItem,rcond=-1)

        SSENew = S[1]

        if(SSENew < SSEOld):
            alpha = S[0][0]
            beta = S[0][1]
            SSEOld = SSENew

    print("\nCoefficients: " + str(S[0][0:2]))

    #styler.printFooter("Intercept: " + str(S[0][2]))
