from ._KNeighborsBase import KNeighborsBase

import numpy as np

class KNeighborsRegressorX(KNeighborsBase):

    ###################################################################
    #
    #   Function Name : predict
    #   Input         : Testing data
    #   Output        : Predicted data
    #   Description   : It is used to predict
    #   Author        : Sanket Sadashiv Hajare
    #   Date          : 03//04/2026
    #
    ###################################################################

    def predict(self,X):

        X = np.array(X)

        predictions = [] 

        for point in X:

            Value = []

            neighbors = self._get_n_neighbors(point)
            
            for dist, label in neighbors:
                Value.append(label)
            
            prediction_value = np.mean(Value)

            predictions.append(prediction_value)
        
        return np.array(predictions)

        
