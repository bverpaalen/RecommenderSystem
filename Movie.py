def movieAvg(matrix):
    movieAvgs = OD()
    for column in matrix.iteritems():
        movieId = column[0][1]
        movieData = column[1]

        movieAvg = movieData.sum() / movieData.count()

        movieAvgs.update({movieId: movieAvg})
    return movieAvgs