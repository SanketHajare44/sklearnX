from ._KNeighborsBase import KNeighborsBase
import numpy as np 

class KNeighborsClassifierX(KNeighborsBase):

    ###################################################################
    #
    #   Function Name : predict
    #   Input         : Testing data
    #   Output        : Predicted data
    #   Description   : It is used to predict
    #   Author        : Sanket Sadashiv Hajare
    #   Date          : 02//04/2026
    #
    ###################################################################

    def predict(self, X):

        X = np.array(X)

        predictions = []

        for point in X:

            neighbors = self._get_n_neighbors(point)

            votes = {}

            for dist, label in neighbors:
                votes[label] = votes.get(label,0)+1
            
            prediction = max(votes, key = votes.get)

            predictions.append(prediction)

        return np.array(predictions) 
