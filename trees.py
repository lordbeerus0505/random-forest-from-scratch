import numpy as np
import pandas as pd
import sys
import copy

def findDistribution(train_data):
    pos_count, neg_count = 0,0
    unique, counts = np.unique(train_data['decision'], return_counts=True)
    result = dict(zip(unique, counts))
    if 1 in result:
        pos_count = result[1]
    if 0 in result:
        neg_count = result[0]

    return pos_count, neg_count

def findMajority(train_data):
    pos_count, neg_count = 0,0
    unique, counts = np.unique(train_data['decision'], return_counts=True)
    result = dict(zip(unique, counts))
    if 1 in result:
        pos_count = result[1]
    if 0 in result:
        neg_count = result[0]
    if pos_count > neg_count:
        return '+'
    return '-'

def calculateGiniIndex(train_data):
    # Gini Index = 1- sum(p(x)**2), here everything is binary
    pos_count, neg_count = findDistribution(train_data)
    total = (pos_count + neg_count)
    gini_index = 1 - (pos_count/total)**2 - (neg_count/total)**2
    return gini_index


def bestAttribute(train_data):
    attributes = train_data.columns
    import pdb; pdb.set_trace()
    # find entropy and gain as per gini index for each attribute in attribute
    calculateGiniIndex(train_data)
    for attr in attributes:
        calculateGiniGain
def buildTree(train_data):
    # recursively constructing the tree
    # import pdb; pdb.set_trace()
    tree = {}
    if len(train_data) == 0:
        return {}
    if len(train_data.columns) == 2:
        # only decision and one other attribute is left, return majority
        return {findMajority(train_data)}
    # if pure, no point calculating split up - find positive and negative first
    positive, negative = findDistribution(train_data)
    if positive == 0:
        return {'-'}
    elif negative == 0:
        return {'+'}

    attr = bestAttribute(train_data)
    # now drop this attribute from train_data for recursive steps
    
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