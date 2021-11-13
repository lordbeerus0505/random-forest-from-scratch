import pandas as pd
import numpy as np
import trees
from math import sqrt
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from scipy.stats import ttest_rel

def plot(num_trees_list, accuracy, standard_error):
    # import pdb; pdb.set_trace()
    fig = plt.figure()
    fig.set_figwidth(8)
    fig.set_figheight(7)
    plt.errorbar(num_trees_list, accuracy[0], fmt='-', yerr=standard_error[0], color='red', capsize=4, capthick=2)
    plt.errorbar(num_trees_list, accuracy[1], fmt='-', yerr=standard_error[1], color='blue', capsize=4, capthick=2)

    # fig.subplots_adjust(bottom=0.3)
    # plt.xticks([1, 2, 3, 4, 5, 6, 7, 8, 9, 10], [1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
    # plt.yticks(np.arange(0.5,0.8,0.05))
    plt.xlabel('Depth of Tree')
    plt.ylabel('Avg. Model Accuracy')
    red_patch = mpatches.Patch(color='red', label='Bagged Tree')
    blue_patch = mpatches.Patch(color='blue', label='Random Forest')

    plt.legend(handles=[red_patch, blue_patch])
    plt.savefig('learningCurves_num_tree.png')
    # plt.show()

def numTreesPerformance():
    train_data = pd.read_csv("trainingSet.csv")
    test_data = pd.read_csv("testSet.csv")

    # Using Random State of 18 and frac = 1, then taking 50% using random state 32
    train_data = train_data.sample(frac = 1, random_state=18)
    train_data = train_data.sample(frac = 0.5, random_state = 32)
    # Now for 10 fold cross validation, splitting using method from assignment 3
    size = len(train_data)//10
    train_data_split = []
    for i in range(10):
        train_data_split.append(train_data.iloc[size*i:size*(i+1)])

    # possible tree sizes are 10, 20, 40, 50
    num_trees_list = [10,20]

    accuracy = [[] for _ in range(2)]
    standard_error = [[] for _ in range(2)]
    t_stat_accuracy_BT = []
    t_stat_accuracy_RF = []
    for num in num_trees_list:
        accuracyDT, accuracyBT, accuracyRF = [], [], []
        for i in range(10):
            test_set = train_data_split[i]
            train_set = train_data.drop(test_set.index)

            trainAccBT, testAccBT = trees.bagging(train_set, test_set, max_depth = 8, num_trees = num)
            accuracyBT.append(testAccBT)

            trainAccRF, testAccRF = trees.randomForests(train_set, test_set, max_depth = 8, num_trees = num)
            accuracyRF.append(testAccRF)
            
            print('For Bagged Trees Train Acc %s Test Acc %s \nFor Random Forests Train Acc %s Test Acc %s \n'%(trainAccBT, testAccBT, trainAccRF, testAccRF))
        # import pdb; pdb.set_trace()
        # For each depth this is what you store
        accuracy[0].append(np.mean(accuracyBT))
        accuracy[1].append(np.mean(accuracyRF))
        standard_error[0].append(np.std(accuracyBT)/sqrt(10))
        standard_error[1].append(np.std(accuracyRF)/sqrt(10))
        t_stat_accuracy_BT.append(accuracyBT)
        t_stat_accuracy_RF.append(accuracyRF)

    # Plot the graphs
    plot(num_trees_list, accuracy, standard_error)

    for i,num in enumerate(num_trees_list):
        print("Null Hypothesis h0: Bagged Tree Accuracy = Random Forest Accuracy")
        print("Alternate Hypothesis h1: Bagged Tree Accuracy != Random Forest Accuracy")
        print('Running with max number of trees of %s'%num)
        print("Bagged Tree accuracies: %s"%t_stat_accuracy_BT[i])
        print("Random Forests accuracies: %s"%t_stat_accuracy_RF[i])
        
        pvalue = ttest_rel(t_stat_accuracy_BT[i], t_stat_accuracy_RF[i]).pvalue
        if pvalue < 0.05:
            print ("\nRejecting Null Hypothesis H0 since the pvalue is less than 0.05")
        else:
            print ("\nAccepting Null Hypothesis H0 since pvalue is greater than 0.05")

if __name__ == '__main__':
    numTreesPerformance()