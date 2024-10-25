import adaboost
import decision_tree_regressor
import linear_regression
import neural_network
import random_forest
import svm_regressor

# Small ranker that ranks the models based on their accuracy
ranks = {
    'adaboost': adaboost.accuracy,
    'decision_tree_regressor': decision_tree_regressor.accuracy,
    'linear_regression': linear_regression.accuracy,
    'neural_network': neural_network.average_accuracy,
    'random_forest': random_forest.accuracy,
    'svm_regressor': svm_regressor.accuracy
}
print(ranks)