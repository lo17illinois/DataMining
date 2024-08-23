from typing import List
import math

class Node:
  """
  This class, Node, represents a single node in a decision tree. It is designed to store information about the tree
  structure and the specific split criteria at each node. It is important to note that this class should NOT be
  modified as it is part of the assignment and will be used by the autograder.

  The attributes of the Node class are:
  - split_dim: The dimension/feature along which the node splits the data (-1 by default, indicating uninitialized)
  - split_point: The value used for splitting the data at this node (-1 by default, indicating uninitialized)
  - label: The class label assigned to this node, which is the majority label of the data at this node. If there is a tie,
    the numerically smaller label is assigned (-1 by default, indicating uninitialized)
  - left: The left child node of this node (None by default). Either None or a Node object.
  - right: The right child node of this node (None by default) Either None or a Node object.
  """
  def __init__(self):
    self.split_dim = -1
    self.split_point = -1
    self.label = -1
    self.left = None
    self.right = None
class Solution:
    def fit(self, train_data: List[List[float]], train_label: List[int]) -> None:
        self.root = Node()
        self.split_node(self.root, train_data, train_label, depth=2)

    def split_node(self, node: Node, data: List[List[float]], label: List[int], depth: int) -> None:
        # Check if the node is a leaf node
        if depth == 0 or len(set(label)) == 1 or len(set(label)) == 0:
            node.split_dim = -1
            node.split_point = -1.0
            node.left = None
            node.right = None
            if not label:
                node.label = -1  # Assign a default label
            else:
                # Assign the majority label as the node's label
                node.label = max(set(label), key=label.count)
            return
        else:
            entropy = self.calculate_entropy(label)
            num_variables = len(data[0])  # Number of variables
            max_info_gain = -float('inf')
            best_split_dim = -1
            best_split_point = -1.0
            for dim in range(num_variables):
                sorted_data = sorted(data, key=lambda x: x[dim])
                midpoints = [(sorted_data[i][dim] + sorted_data[i + 1][dim]) / 2 for i in range(len(sorted_data) - 1)]
                for split_point in midpoints:
                    info_gain = self.calculate_information_gain(data, label, dim, split_point, entropy)
                    if info_gain > max_info_gain:
                        max_info_gain = info_gain
                        best_split_dim = dim
                        best_split_point = split_point
            node.split_dim = best_split_dim
            node.split_point = best_split_point
            left_data, left_label, right_data, right_label = self.split_data(data, label, best_split_dim, best_split_point)
            node.label = max(set(label), key=label.count)
            node.left = Node()
            node.right = Node()
            self.split_node(node.left, left_data, left_label, depth - 1)
            self.split_node(node.right, right_data, right_label, depth - 1)

    def calculate_entropy(self, label: List[int]) -> float:
        total_size = len(label)
        entropy = 0.0
        for lbl in set(label):
            p_lbl = label.count(lbl) / total_size
            entropy -= p_lbl * math.log2(p_lbl)
        return entropy

    def calculate_information_gain(self, data: List[List[float]], label: List[int], split_dim: int, split_point: float, entropy: float) -> float:
        left_label = [label[i] for i in range(len(data)) if data[i][split_dim] <= split_point]
        right_label = [label[i] for i in range(len(data)) if data[i][split_dim] > split_point]
        left_size = len(left_label)
        right_size = len(right_label)
        total_size = len(label)
        p_left = left_size / total_size
        p_right = right_size / total_size
        entropy_left = self.calculate_entropy(left_label)
        entropy_right = self.calculate_entropy(right_label)
        split_info = p_left * entropy_left + p_right * entropy_right
        info_gain = entropy - split_info
        return info_gain

    def split_data(self, data: List[List[float]], label: List[int], split_dim: int, split_point: float) -> tuple[List[List[float]], List[int], List[List[float]], List[int]]:
        left_data = []
        left_label = []
        right_data = []
        right_label = []
        for i in range(len(data)):
            if data[i][split_dim] <= split_point:
                left_data.append(data[i])
                left_label.append(label[i])
            else:
                right_data.append(data[i])
                right_label.append(label[i])
        return left_data, left_label, right_data, right_label
    def split_info(self, data: List[List[float]], label: List[int], split_dim: int, split_point: float) -> float:
            """
            Compute the information needed to classify a dataset if it's split
            with the given splitting dimension and splitting point, i.e. Info_A in the slides.

            Parameters:
            data (List[List]): A nested list representing the dataset.
            label (List): A list containing the class labels for each data point.
            split_dim (int): The dimension/attribute index to split the data on.
            split_point (float): The value at which the data should be split along the given dimension.

            Returns:
            float: The calculated Info_A value for the given split. Do NOT round this value
            """
            left_label = [label[i] for i in range(len(data)) if data[i][split_dim] <= split_point]
            right_label = [label[i] for i in range(len(data)) if data[i][split_dim] > split_point]
            left_size = len(left_label)
            right_size = len(right_label)
            total_size = len(label)
            if left_size == 0 or right_size == 0:
                return 0  # No information needed to classify if one side is empty
            # Calculate proportions
            p_left = left_size / total_size
            p_right = right_size / total_size
            # Calculate entropy
            entropy_left = -sum((left_label.count(i) / left_size) * math.log2(left_label.count(i) / left_size) for i in set(left_label))
            entropy_right = -sum((right_label.count(i) / right_size) * math.log2(right_label.count(i) / right_size) for i in set(right_label))
            # Calculate split info
            split_info = p_left * entropy_left + p_right * entropy_right
            return split_info
    def classify(self, train_data: List[List[float]], train_label: List[int], test_data: List[List[float]]) -> List[int]:
        self.fit(train_data, train_label)  # Fit the decision tree using the training data
        predictions = []

        def traverse(node, data_point):
            if node.split_dim == -1:  # Check if the node is a leaf node
                return node.label  # Return the label assigned to the leaf node
            elif data_point[node.split_dim] <= node.split_point:
                return traverse(node.left, data_point)
            else:
                return traverse(node.right, data_point)

        for data_point in test_data:
            prediction = traverse(self.root, data_point)
            predictions.append(prediction)

        return predictions
