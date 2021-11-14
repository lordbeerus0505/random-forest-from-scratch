import numpy as np
import pandas as pd
import sys
import copy
import math
import random
import pprint

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

def calculateGiniGain(attribute, train_data):
    # since all binary only 2 possibilities
    positive_data = train_data.loc[train_data[attribute]==1]
    negative_data = train_data.loc[train_data[attribute]==0]
    pos_count = len(positive_data)
    neg_count = len(negative_data)
    if pos_count == 0 or neg_count == 0:
        return 0
    total = pos_count + neg_count
    return pos_count/total * calculateGiniIndex(positive_data) + neg_count/total * calculateGiniIndex(negative_data)


def bestAttribute(train_data, random_forest = False):
    # Last column is decision and we dont want to test for that.
    attributes = train_data.columns[:-1]
    # print("Num of attributes is ", len(attributes))
    if random_forest == True:
        # take a subset of features, specifically, sqrt of the number
        subset_size = int(math.sqrt(len(attributes)))
        # Converting to int as sample doesnt allow float number of samples
        attributes = random.sample(set(attributes), subset_size)
    
    # find entropy and gain as per gini index for each attribute in attribute
    giniIndex = calculateGiniIndex(train_data)
    minReduction, bestAttr = 10000, ''
    for attr in attributes:
        loss = calculateGiniGain(attr, train_data)
        
        if minReduction >= loss:
            bestAttr = attr
            minReduction = loss

    # no need to subtract, the min here is the best there. 
    # If multiple share min, first one is used
    return bestAttr

def buildTree(train_data, max_depth, depth, random_forest = False):
    # recursively constructing the tree

    tree = {}
    if depth == max_depth:
        return findMajority(train_data)

    if len(train_data) < 50:
        # NOTE: need to handle for when less data there, return majority or something
        return findMajority(train_data)
    if len(train_data.columns) == 1:
        # only decision and one other attribute is left, if conflict return majority
        return findMajority(train_data)
    # if pure, no point calculating split up - find positive and negative first
    positive, negative = findDistribution(train_data)
    if positive == 0:
        return '-'
    elif negative == 0:
        return '+'

    attr = bestAttribute(train_data, random_forest)
    # now drop this attribute from train_data for recursive steps
    positive_data = train_data.loc[train_data[attr]==1]
    negative_data = train_data.loc[train_data[attr]==0]
    positive_data = positive_data.drop(attr, axis = 1)
    negative_data = negative_data.drop(attr, axis = 1)
    tree[attr] = {1: buildTree(positive_data, max_depth, depth+1), 0: buildTree(negative_data, max_depth, depth+1)}
    return tree

def calculateAccuracy(predicted_values, actual_values):
    # predicted_values = np.where(predicted_values == '+',1,0)
    diff = np.abs(np.subtract(predicted_values, actual_values))
    return 1 - np.sum(diff)/len(diff)
    
def recursiveParse(decision_tree, data_frame):
    if type(decision_tree) == type(""):
        return decision_tree
    key = [*decision_tree]
    values = decision_tree[key[0]]
    for v in values:
        if data_frame[1][key[0]] == v:
            return recursiveParse(decision_tree[key[0]][v], data_frame)

def predict(data_frame, decision_tree):
    result = []
    for data in data_frame.iterrows():
        key = [*decision_tree]
        values = decision_tree[key[0]]
        
        for v in values:
            if data[1][key[0]] == v:
                result.append(recursiveParse(decision_tree.copy()[key[0]][v], data))
    # print(result)
    return np.array(result)

def buildBaggedTree(train_data, max_depth, num_trees, random_forest = False):
    bagged_trees = []
    for i in range(num_trees):
        # Using seed value for consistent output
        new_train_data = train_data.sample(frac=1, random_state = 5*i, replace=True)
        decision_tree = buildTree(new_train_data.copy(), max_depth, 0, random_forest)
        bagged_trees.append(decision_tree)
    return bagged_trees

def randomForests(train_data, test_data, max_depth = 8, num_trees = 30):
    random_forest = buildBaggedTree(train_data, max_depth, num_trees, random_forest = True)

    # now predict on each of these bagged trees and find the average prediction (can use majority also).
    resultArr = []
    for tree in random_forest:
        predicted_values = predict(train_data, tree)
        predicted_values = np.where(predicted_values == '+',1,0)
        resultArr.append(predicted_values)
    # import pdb; pdb.set_trace()
    result_train = np.mean(resultArr, axis=0)
    normalized_result_train = np.where(result_train>0.5,1,0)

    resultArr = []
    for tree in random_forest:
        predicted_values = predict(test_data, tree)
        predicted_values = np.where(predicted_values == '+',1,0)
        resultArr.append(predicted_values)

    result_test = np.mean(resultArr, axis=0)
    normalized_result_test = np.where(result_test>0.5,1,0)

    return calculateAccuracy(normalized_result_train, train_data['decision']), calculateAccuracy(normalized_result_test, test_data['decision'])

def bagging(train_data, test_data, max_depth = 8, num_trees = 30):
    bagged_trees = buildBaggedTree(train_data, max_depth, num_trees, random_forest = False)

    # now predict on each of these bagged trees and find the average prediction (can use majority also).
    resultArr = []
    for tree in bagged_trees:
        predicted_values = predict(train_data, tree)
        predicted_values = np.where(predicted_values == '+',1,0)
        resultArr.append(predicted_values)
    # import pdb; pdb.set_trace()
    result_train = np.mean(resultArr, axis=0)
    normalized_result_train = np.where(result_train>0.5,1,0)

    resultArr = []
    for tree in bagged_trees:
        predicted_values = predict(test_data, tree)
        predicted_values = np.where(predicted_values == '+',1,0)
        resultArr.append(predicted_values)

    result_test = np.mean(resultArr, axis=0)
    normalized_result_test = np.where(result_test>0.5,1,0)

    return calculateAccuracy(normalized_result_train, train_data['decision']), calculateAccuracy(normalized_result_test, test_data['decision'])
     
def decisionTree(train_data, test_data, max_depth = 8):
    # first build the tree, store it and then use it for testing
    decision_tree = buildTree(train_data.copy(), max_depth, 0)
    # pprint.pprint(decision_tree)

    # Now predict the outcome
    # predict(test_data)
    results_train = predict(train_data, decision_tree)
    results_train = np.where(results_train == '+', 1, 0)
    results_test = predict(test_data, decision_tree)
    results_test = np.where(results_test == '+', 1, 0)

    # change to test data!
    # import pdb; pdb.set_trace()
    return calculateAccuracy(results_train, np.array(train_data['decision'])), calculateAccuracy(results_test, np.array(test_data['decision']))


def treeMain(trainSetFile, testSetFile, operation):
    train_data = pd.read_csv(trainSetFile) 
    test_data = pd.read_csv(testSetFile)
    if operation == str(1):
        train_acc, test_acc = decisionTree(train_data, test_data, max_depth = 8)
        print('Training Accuracy DT: %.2f'%train_acc)
        print('Test Accuracy DT: %.2f'%test_acc)
    elif operation == str(2):
        train_acc, test_acc = bagging(train_data, test_data, max_depth = 8, num_trees = 30)
        print('Training Accuracy BT: %.2f'%train_acc)
        print('Test Accuracy BT: %.2f'%test_acc)
    else:
        train_acc, test_acc = randomForests(train_data, test_data)
        print('Training Accuracy RF: %.2f'%train_acc)
        print('Test Accuracy RF: %.2f'%test_acc)

if __name__ == '__main__':
    # input are trainingSet, testSet and choice
    treeMain(sys.argv[1], sys.argv[2], sys.argv[3])