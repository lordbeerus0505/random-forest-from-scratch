import numpy as np
import pandas as pd
import sys

def findDistribution(train_data):
    posCount, negCount = 0,0
    unique, counts = np.unique(train_data['decision'], return_counts=True)
    result = dict(zip(unique, counts))
    if result.has_key('1'):
        posCount = result['1']
    if result.has_key('0'):
        negCount = result['0']
    return posCount, negCount

def buildTree(train_data):
    # recursively constructing the tree
    tree = {}
    if len(train_data) == 0:
        return {}
    
    # if pure, no point calculating split up - find positive and negative first
    positive, negative = findDistribution(train_data)
    if positive == 0:
        return {'-'}
    elif negative == 0:
        return {'+'}
    
def decisionTree(train_data, test_data):
    # first build the tree, store it and then use it for testing
    decision_tree = buildTree(train_data)
def treeMain(trainSetFile, testSetFile, operation):
    train_data = pd.read_csv(trainSetFile)
    test_data = pd.read_csv(testSetFile)
    if operation == str(1):
        decisionTree(train_data, test_data)
    elif operation == str(2):
        baggedTree(train_data, test_data)
    else:
        randomForest(train_data, test_data)

if __name__ == '__main__':
    # input are 
    treeMain(sys.argv[1], sys.argv[2], sys.argv[3])