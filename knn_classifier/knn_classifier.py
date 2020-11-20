import numpy as np
from collections import Counter
from distances import get_distance

class KNN:
    """
    Self-built implementation for the K-nearest neighbor classifier algorithm in the field of Machine Learning
    """

    def __init__(self, k):
        assert k>=1
         
        self.k = k
        self.size = 0
        self.classes = []
        self.num_classes = 0
        self.absolute_classes_count = {}
        self.relative_classes_count =  {}
        self.classes_dict = {}
        self.data_dict = {}

    def __repr__(self):
        return f"<KNN-Model, k={self.k}>"

    def get_k(self):
        return self.k

    def get_classes(self):
        return self.classes if self.classes != [] else Exception("Cannot display classes, since the model hasn't yet been trained. Please train the model first")

    def get_num_classes(self):
        return self.num_classes if self.num_classes != 0 else Exception("Cannot diplay number of classes, since the model hasn't yet been trained. Please train the model first")

    def get_absolute_class_count(self):
        return self.absolute_classes_count if self.absolute_classes_count != {} else Exception("Cannot display absolute number of classes, since the model hasn't yet been trained. Please train the model first")

    def get_relative_class_count(self):
        return self.relative_classes_count if self.relative_classes_count != {} else Exception("Cannot display relative number of classes, since the model hasn't yet been trained. Please train the model first.")

    def get_size(self):
        return self.size if self.size != 0 else Exception("Cannot diplay number of trained datapoints, since the model hasn't yet been trained. Please train the model first.")

    def get_data(self, key='datapoint'):
        if key == 'datapoint':
            return self.data_dict if self.data_dict != {} else Exception("Cannot display observed datapoint, since the model hasn't yet been trained. Please train the model first.")
        elif key == 'classes':
            return self.classes_dict if self.classes_dict != {} else Exception("Cannot display observed datapoint, since the model hasn't yet been trained. Please train the model first.")
        else: Exception("Sorry, I don't know this key.")

    def fit(self, X, y):
        """Method to create the Model using K-nearest Neighbors Algorithm"""
        if X.shape[0] != y.shape[0]:
            raise ValueError

        self.size = len(y)
        self.classes = np.unique(y).tolist()
        self.num_classes = len(self.classes)
        self.absolute_classes_count = {self.classes[i]: np.unique(y, return_counts=True)[1][i] for i in range(len(self.classes))}
        self.relative_classes_count = {i: round(j/len(y),2) for i,j in self.absolute_classes_count.items()}
    
        # read in all datapoints into a dict with the key being associated class
        self.classes_dict = {y: [] for y in self.classes}
        for i, datapoint in enumerate(X):
            self.classes_dict[y[i]].append(datapoint.tolist())

        # reaad in all datapoint into a dict with each point being a key and the value being the target
        self.data_dict = {tuple(datapoint): None for datapoint in X}
        sorted_keys = list(self.data_dict.keys())
        #print(self.data_dict)
        #print(sorted_keys)
        for i, class_ in enumerate(y):
            self.data_dict[sorted_keys[i]] = class_

        # print(self.data_dict)

    def predict(self, X):

        if X.shape[1] != len(self.classes_dict[0][0]):
            raise ValueError # also raise error when dimension is wrong TODO

        self.pred = np.zeros(len(X))
        # print(self.pred)
        # print(self.pred.shape)
        # print("-----------------------------")

        for i, datapoint in enumerate(X):
            # print("-------------------")
            # print(i, datapoint)
            # print("\n")
            distances = [get_distance(datapoint, list(model_data)) for model_data in self.data_dict.keys()]
            # print(distances)
            # print("\n")
            k_smallest = np.argpartition(distances, self.k)[:self.k]
            # print(k_smallest)
            # print("\n")
            k_smallest_p = [list(self.data_dict.keys())[i] for i in k_smallest]
            # print(k_smallest_p)
            # print("\n")
            k_smallest_c = [self.data_dict[i] for i in k_smallest_p]
            # print(k_smallest_c)
            # print("\n")

            cnt = Counter(k_smallest_c)    
            self.pred[i] = cnt.most_common(1)[0][0]
        
        return self.pred