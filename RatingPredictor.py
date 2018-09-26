import numpy as np
import scipy.sparse as sparse
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
    styler.printFooter("Mean global avg error: " + "\n\t Train: "+ str(np.mean(rmses[:,0])) + "\n\t Test: " + str(np.mean(rmses[:,1])))

def predictRatingPerUser(trainAndTestSets, globalAvg):
    styler.printHeader("The average rating per user")

    predictedRatings = predictRating(trainAndTestSets, globalAvg, 0)
    meanRmses = predictedRatings[0]

    # print the final conclusion for avg per user:
    styler.printFooter("Mean train user RMSE: " + str(np.mean(meanRmses[:, 0])) + "\nMean test user RMSE: " + str(
        np.mean(meanRmses[:, 1])))

    return predictedRatings[1]

def predictRatingPerItem(trainAndTestSets, globalAvg):
    styler.printHeader("The average rating per item")

    predictedRatings = predictRating(trainAndTestSets, globalAvg, 1)
    meanRmses = predictedRatings[0]

    # print the final conclusion for avg per item:
    styler.printFooter("Mean train item RMSE: " + str(np.mean(meanRmses[:,0])) + "\nMean test item RMSE: " + str(np.mean(meanRmses[:,1])))

    return predictedRatings[1]

# basedOn is column nr; user or movie
def predictRating(trainAndTestSets, globalAvg, basedOn):
    meanRmses = []
    foldMovieRatings = [{} for i in range(len(trainAndTestSets))]
    for fold, dataSet in enumerate(trainAndTestSets):
        print("\nFold " + str(fold) + ":")

        trainSet = dataSet[0]
        testSet = dataSet[1]

        movieRatings = None
        if(basedOn == 0):
            # aggregate ratings by user id
            movieRatings = rating.aggregateByUser(trainSet, testSet)
        else:
            # aggregate ratings by movie id
            movieRatings = rating.aggregateByMovie(trainSet, testSet)

        trainMovies = movieRatings["trainMovies"]
        testMovies = movieRatings["testMovies"]

        foldMovieRatings[fold].update(trainMovies)

        #################
        # TRAIN
        #################
        trainRmses = []
        for id, trainRatings in trainMovies.items():

            # if movie does not exist in train set, then use global avg of the movie
            movieAvg = np.mean(trainMovies[id]) if id in trainMovies else globalAvg

            se = (movieAvg - trainRatings) ** 2
            rmse = np.sqrt(np.mean(se))

            trainRmses.append(rmse)

        # print train error:
        meanTrainRmse = np.mean(trainRmses)
        print("\tTrain RMSE: " + str(meanTrainRmse))

        #################
        # TEST
        #################
        testRmses = []
        for id, testRatings in testMovies.items():

            movieAvg = 0.0
            # if movie does not exist in train set, then use global avg of the movie
            if id in trainMovies:
                movieAvg = np.mean(trainMovies[id])
            else:
                movieAvg = globalAvg

            se = (movieAvg - testRatings) ** 2

            testRmses.append(np.sqrt(np.mean(se)))

        # print test error:
        meanTestRmse = np.mean(testRmses)
        print("\tTest RMSE: " + str(meanTestRmse))

        meanRmses.append([meanTrainRmse, meanTestRmse])

    return [np.array(meanRmses), foldMovieRatings]

def predictByLinairRegression(trainAndTestSets, predictedRatingsPerItem, predictedRatingsPerUser,globalAvg):
    styler.printHeader("Linear combination of user and item averages")

    linearPrediction = []
    RMSESum = 0
    for fold, dataSet in enumerate(trainAndTestSets):
        trainSet = dataSet[0]

        userMeans = []
        itemMeans = []

        users = np.unique(trainSet[:,0])
        items = np.unique(trainSet[:,1])

        usersMeanDic = {}
        itemsMeansDic = {}

        for userId, itemId, actualRating in trainSet:
            predictedRatingPerUser = np.mean(predictedRatingsPerUser[fold][userId])
            predictedRatingPerItem = np.mean(predictedRatingsPerItem[fold][itemId])

            # Our task is to find coefficients A, B, C, such that the linear combination:
            # A*x + B*y + C  approximates y as good as possible.
            userMeans.append(predictedRatingPerUser)
            itemMeans.append(predictedRatingPerItem)

            usersMeanDic.update({userId:predictedRatingPerUser})
            itemsMeansDic.update({itemId:predictedRatingPerItem})

        linearPrediction = np.vstack([userMeans,itemMeans,np.ones(len(userMeans))]).T
        
        S = np.linalg.lstsq(linearPrediction, trainSet[:,2],rcond=-1)

        SSE = str(S[1])

        alpha = S[0][0]
        beta = S[0][1]
        delta = S[0][2]

        sum = 0
        for userId,itemId, actualRating in trainSet:
            predictedRatingPerUser = usersMeanDic.get(userId)
            predictedRatingPerItem = itemsMeansDic.get(itemId)

            prediction = alpha * predictedRatingPerUser + beta * predictedRatingPerItem + delta

            if prediction > 5:
                prediction = 5
            elif prediction < 1:
                prediction = 1

            difference = prediction - actualRating
            se = difference**2
            sum += se
        meansum = sum/len(trainSet)
        RMSE = np.sqrt(meansum)

        print("Alpha: "+str(alpha)+" Beta: "+str(beta)+" Delta: "+str(delta))
        print("Fold: "+str(fold)+" RMSE Train: "+str(RMSE))
        testSet = dataSet[1]

        counter = 0
        sum = 0
        for userId,itemId,actualRating in testSet:
            if (userId in users) and (itemId in items):
                predictedRatingPerUser = usersMeanDic.get(userId)
                predictedRatingPerItem = itemsMeansDic.get(itemId)

                prediction = alpha * predictedRatingPerUser + beta * predictedRatingPerItem + delta

                if prediction > 5:
                    prediction = 5
                elif prediction < 1:
                    prediction = 1

            elif (userId in users and itemId not in items):
                prediction = usersMeanDic.get(userId)
                counter+= 1
            elif userId not in users and itemId in items:
                prediction = itemsMeansDic.get(itemId)
                counter+= 1
            else:
                counter+= 1
                prediction = globalAvg
            difference = prediction - actualRating
            se = difference**2
            sum += se
        meansum = sum/len(testSet)
        RMSE = np.sqrt(meansum)
        RMSESum += RMSE
        print("Fold: " + str(fold) + " RMSE Test: " + str(RMSE))
        print("Mismatch: "+str(counter))
    meanRMSE = RMSESum/len(trainAndTestSets)

    styler.printFooter("Mean LR avg error: " + str(meanRMSE))


def predictByMatrixFactorization(trainAndTestSets, aggUserTrain, aggItemTrain, globalAvg, numFactors=10, numIter=75, regularization=0.05, learnRate=0.005):

    styler.printHeader("Matrix factorization")
    print()

    # avg. rmse over all folds
    sumAvgRmse = 0
    sumTestRmse = 0
    uUsers = []
    vItems = []
    for fold, dataSet in enumerate(trainAndTestSets):
        print("Lengte train set: " + str(len(dataSet[0])))
        print("Lengte test set: " + str(len(dataSet[1])))
        trainSet = dataSet[0]

        # create 2D sparse ratings matrix, with user id's as rows and movie id's as columns
        trainUserIds = trainSet[:, 0]
        trainMovieIds = trainSet[:, 1]
        trainRatings = trainSet[:, 2]
        trainMatrix = sparse.coo_matrix((trainRatings,(trainUserIds, trainMovieIds)))

        # print the matrix
        #for row, col, value in zip(matrix.row, matrix.col, matrix.data):
            #print("({0}, {1}) {2}".format(row, col, value))

        # create user and item vectors of random digit factors
        userNrs, movieNrs = trainMatrix.shape
        np.random.seed(123)
        uniqueUserIds = set(trainUserIds)
        uniqueMovieIds = set(trainMovieIds)
        uUser = dict(zip(uniqueUserIds, [np.array(f) for f in zip(*np.random.rand(numFactors, len(uniqueUserIds)))]))
        uUsers.append(uUser)
        vItem = dict(zip(uniqueMovieIds, [np.array(f) for f in zip(*np.random.rand(numFactors, len(uniqueMovieIds)))]))
        vItems.append(vItem)
        #uUser = np.random.rand(userNrs, numFactors)
        #vItem = np.random.rand(numFactors, movieNrs)

        ratings = trainMatrix.data

        sumRmse = 0
        # iterate the matrix 'numIter' times
        for i in range(numIter):
            sse = 0
            for user, movie, rating in zip(trainMatrix.row, trainMatrix.col, ratings):

                # calculate the error
                error = rating - np.sum(uUser[user] * vItem[movie])

                # update parameters uUser and vItem
                uUser[user] = uUser[user] + learnRate * (error * vItem[movie] - regularization * uUser[user])
                vItem[movie] = vItem[movie] + learnRate * (error * uUser[user] - regularization * vItem[movie])

                sse += error ** 2

            rmse = np.sqrt(sse / len(ratings))
            sumRmse += rmse
            #print(rmse)

        avgRmse = sumRmse / numIter
        sumAvgRmse += avgRmse
        print("Avg train RMSE for fold {0}: {1}".format(fold, avgRmse))

        ###############################
        # TEST
        ###############################
        testSet = dataSet[1]
        testUserIds = testSet[:, 0]
        testMovieIds = testSet[:, 1]
        testRatings = testSet[:, 2]
        testMatrix = sparse.coo_matrix((testRatings, (testUserIds, testMovieIds)))

        testSse = 0
        movieMismatches = 0
        userMismatches = 0
        movieAndUserMismatches = 0
        for user, movie, rating in zip(testMatrix.row, testMatrix.col, testMatrix.data):
            # calculate the error
            prediction = 0
            if (user in uUser) and (movie in vItem):
                prediction = np.sum(uUser[user] * vItem[movie])
            elif user in uUser and not (movie in vItem):
                movieMismatches += 1
                prediction = np.mean(aggUserTrain[fold][user])
            elif movie in vItem and not (user in uUser):
                userMismatches += 1
                prediction = np.mean(aggItemTrain[fold][movie])
            else:
                movieAndUserMismatches += 1
                prediction = globalAvg

            testSse += (rating - prediction) ** 2

        print("\n- Number of user mismatches: " + str(userMismatches))
        print("- Number of item mismatches: " + str(movieMismatches))
        print("- Number of item and user mismatches: " + str(movieAndUserMismatches))

        testRmse = np.sqrt(testSse / len(testMatrix.data))
        sumTestRmse += testRmse
        print("\n- Avg test RMSE for fold {0}: {1}".format(fold, testRmse))
        print()

    styler.printFooter("Avg train RMSE: " + str(sumAvgRmse / len(trainAndTestSets)) + "\nAvg test RMSE: " + str(sumTestRmse / len(trainAndTestSets)))
