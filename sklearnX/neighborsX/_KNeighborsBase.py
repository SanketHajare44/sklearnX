import numpy as np 

class KNeighborsBase:
    
    ###################################################################
    #
    #   Function Name : __init__
    #   Input         : Nothing
    #   Output        : Nothing
    #   Description   : It is constructor used to create instance variable
    #   Author        : Sanket Sadashiv Hajare
    #   Date          : 01//04/2026
    #
    ###################################################################

    def __init__(self, n_neighbors=5, metric='euclidean'):
        self.n_neighbors = n_neighbors
        self.metric = metric
        self._X_train = None
        self._Y_train = None

    ###################################################################
    #
    #   Function Name : fit
    #   Input         : X -> Independent variable , Y -> Dependent Variable
    #   Output        : Trained model instance
    #   Description   : Validates the training data and stores it inside the
    #                   model  for future predictions
    #   Author        : Sanket Sadashiv Hajare
    #   Date          : 01//04/2026
    #
    ###################################################################

    def fit(self, X ,Y):
        X = np.array(X, dtype=float)
        Y = np.array(Y)

        # checks the dimensions
        if(X.ndim != 2):
            raise ValueError("X must be a 2D array")
        
        # Check Rows of the feature and colmun
        if(len(X) != len(Y)):
            raise ValueError("rows of X and Y required same")

        if(self.n_neighbors > len(X)):
            raise ValueError(f"n_neighbors {self.n_neighbors} is greater than the number of training samples ")
        
        self._X_train = X 
        self._Y_train = Y 

        return self

    ###################################################################
    #
    #   Function Name : _check_is_fitted
    #   Input         : Nothing
    #   Output        : Nothing
    #   Description   : It is used to check training is completed
    #   Author        : Sanket Sadashiv Hajare
    #   Date          : 01//04/2026
    #
    ###################################################################

    def _check_is_fitted(self):
        if( self._X_train is None):
            raise ValueError("Model is not fit, call the first fit")
    
    ###################################################################
    #
    #   Function Name : _euclidean_distance
    #   Input         : P1 -> First data point
    #                   P2 -> Second data point
    #   Output        : float - Euclidean distance between P1 and P2
    #   Description   : Calculates the Euclidean distance between
    #                   two feature vectors.
    #   Author        : Sanket Sadashiv Hajare
    #   Date          : 01/04/2026
    #
    ###################################################################

    def _euclidean_distance(self,P1,P2):

        total = 0

        for i in range(len(P1)):

            total = total + ((P1[i] - P2[i])**2)
        
        return np.sqrt(total)

    ###################################################################
    #
    #   Function Name : _manhattan_distance
    #   Input         : P1 -> First data point
    #                   P2 -> Second data point
    #   Output        : float - Manhattan distance between P1 and P2
    #   Description   : Calculates the Manhattan distance between
    #                   two feature vectors.
    #   Author        : Sanket Sadashiv Hajare
    #   Date          : 01/04/2026
    #
    ###################################################################

    def _manhattan_distance(self,P1,P2):

        total = 0

        for i in range(len(P1)):

            total = total + abs(P1[i] - P2[i])
        
        return total

    ###################################################################
    #
    #   Function Name : _distance
    #   Input         : P2 -> Test data point.
    #   Output        : list of tuples - Each tuple contains
    #                   (distance, corresponding label/value).
    #
    #   Description   : Calculates the distance between the given
    #                   test sample and all training samples using
    #                   the selected distance metric (Euclidean or
    #                   Manhattan).
    #   Author        : Sanket Sadashiv Hajare
    #   Date          : 01/04/2026
    #
    ###################################################################

    def _distance(self,P2):
        
        distance = []

        for i in  range(len(self._X_train)):
 
            if(self.metric == 'euclidean'):
                dist = self._euclidean_distance(self._X_train[i],P2)
                distance.append((dist,self._Y_train[i]))
            
            elif(self.metric == 'manhattan'):
                dist = self._manhattan_distance(self._X_train[i],P2)
                distance.append((dist,self._Y_train[i]))
            
            else:
                raise ValueError("Unknown metric, use euclidean / manhattan")
            
            
        return distance
    
    ###################################################################
    #
    #   Function Name : _get_n_neighbors
    #   Input         : P2 -> Test data point
    #   Output        : list of tuples 
    #   Description   : Finds the k nearest neighbors of the given
    #                   test sample by computing distances with all
    #                   training samples and returning the closest
    #                   neighbors.
    #   Author        : Sanket Sadashiv Hajare
    #   Date          : 01/04/2026
    #
    ###################################################################

    def _get_n_neighbors(self,P2):
        
        self._check_is_fitted()

        distance = self._distance(P2)

        sorted_data = sorted(distance, key=lambda item : item[0])

        return sorted_data[:self.n_neighbors]
        
        