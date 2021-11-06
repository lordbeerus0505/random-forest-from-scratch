import numpy as np
import pandas as pd
import sys

def buildTree(train_data, test_data):
    # recursively constructing the tree
    tree = {}
    if len(train_data) == 0:
        return {}
    
    # if pure, no point calculating split up - find positive and negative first
    positive, negative = findDistribution()
def decisionTree(train_data, test_data):
    # first build the tree, store it and then use it for testing
    decision_tree = buildTree(train_data, test_data)
def treeMain(trainSetFile, testSetFile, operation):
    train_data = pd.read_csv(trainSetFile)
    test_data = pd.read_csv(testSetFile)
    if operation == 1:
        decisionTree(train_data, test_data)
    elif operation == 2:
        baggedTree(train_data, test_data)
    else:
        randomForest(train_data, test_data)

if __name__ == '__main__':
    # input are 
    treeConstruction(sys.argv[1], sys.argv[2], sys.argv[3])