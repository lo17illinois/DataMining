# Submit this file to Gradescope
import math
from typing import Dict, List, Tuple
# You may use any built-in standard Python libraries
# You may NOT use any non-standard Python libraries such as numpy, scikit-learn, etc.

num_C = 7 # Represents the total number of classes

class Solution:
  
  def prior(self, X_train: List[List[int]], Y_train: List[int]) -> List[float]:
    """Calculate the prior probabilities of each class
    Args:
      X_train: Row i represents the i-th training datapoint
      Y_train: The i-th integer represents the class label for the i-th training datapoint
    Returns:
      A list of length num_C where num_C is the number of classes in the dataset
    """
    N = len(Y_train)
    class_counts = [Y_train.count(classlabel) for classlabel in range(1, 8)]
    priors = [(count + 0.1) / (N + 0.1 * 7) for count in class_counts]
    return priors

  def label(self, X_train: List[List[int]], Y_train: List[int], X_test: List[List[int]]) -> List[int]:
    """Calculate the classification labels for each test datapoint
    Args:
      X_train: Row i represents the i-th training datapoint
      Y_train: The i-th integer represents the class label for the i-th training datapoint
      X_test: Row i represents the i-th testing datapoint
    Returns:
      A list of length M where M is the number of datapoints in the test set
    """
    priors = self.prior(X_train, Y_train)
    possibleAttributeList = [2,2,2,2,2,2,2,2,2,2,2,2,6,2,2,2]
    class_counts = [Y_train.count(classlabel) for classlabel in range(1, num_C+1)]
    labels = []
    posterior = []
    for test_point in X_test:
        ALLattributelikelihoods = [1] * num_C
        for i in range(len(test_point)):  
            attr_value = test_point[i]  
            class_counts2 = [0] * num_C  
            uniqueval = possibleAttributeList[i]
            for classItr in range(1, 8): 
                count = 0
                for X_iter in range(len(X_train)):
                    y = Y_train[X_iter]
                    X_train_point = X_train[X_iter]
                    train_attr_value = X_train_point[i]
                    if train_attr_value == attr_value:
                        if y == classItr:
                            count += 1
                class_counts2[classItr - 1] += count
            attribute_likelihoods = [(class_counts2[classItr - 1] + 0.1) / (class_counts[classItr - 1] + 0.1 * uniqueval) for classItr in range(1, num_C+1)]
            ALLattributelikelihoods = [x * y for x, y in zip(ALLattributelikelihoods, attribute_likelihoods)]
        posterior.append([prior * likelihood for prior, likelihood in zip(priors, ALLattributelikelihoods)])
        max_index = posterior[-1].index(max(posterior[-1])) + 1
        labels.append(max_index)

    return labels
