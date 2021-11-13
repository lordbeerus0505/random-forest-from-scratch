import numpy as np
import pandas as pd
import trees
from math import sqrt
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from scipy.stats import ttest_rel

def plot(fraction_list, accuracy, standard_error):
    # import pdb; pdb.set_trace()

    fig = plt.figure()
    fig.set_figwidth(8)
    fig.set_figheight(7)
    plt.errorbar(fraction_list, accuracy[0], yerr=standard_error[0], fmt='-', color='red', capsize=4, capthick=2)
    plt.errorbar(fraction_list, accuracy[1], yerr=standard_error[1], fmt='-', color='blue', capsize=4, capthick=2)
    plt.errorbar(fraction_list, accuracy[2], yerr=standard_error[2], fmt='-', color='green', capsize=4, capthick=2)
    # fig.subplots_adjust(bottom=0.3)
    # plt.xticks([1, 2, 3, 4, 5, 6, 7, 8, 9, 10], [1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
    # plt.yticks(np.arange(0.5,0.8,0.05))
    plt.xlabel('Fraction of Input')
    plt.ylabel('Avg. Model Accuracy')
    red_patch = mpatches.Patch(color='red', label='Decision Tree')
    blue_patch = mpatches.Patch(color='blue', label='Bagged Tree')
    green_patch = mpatches.Patch(color='green', label='Random Forest')
    plt.legend(handles=[red_patch, blue_patch, green_patch])
    plt.savefig('learningCurves_fractions.png')
    # plt.show()

def fracPerformance():
    train_data = pd.read_csv("trainingSet.csv")
    test_data = pd.read_csv("testSet.csv")

    # Using Random State of 18 and frac = 1, then taking 50% using random state 32
    train_data = train_data.sample(frac = 1, random_state=18)
    

    # Now for 10 fold cross validation, splitting using method from assignment 3
    size = len(train_data)//10
    train_data_split = []
    for i in range(10):
        train_data_split.append(train_data.iloc[size*i:size*(i+1)])

    # possible fractions are [0.05, 0.075, 0.1, 0.15, 0.2]
    fraction_list = [0.05, 0.075, 0.1, 0.15, 0.2]

    accuracy = [[] for _ in range(3)]
    standard_error = [[] for _ in range(3)]
    t_stat_accuracy_DT = []
    t_stat_accuracy_RF = []
    for frac in fraction_list:
        accuracyDT, accuracyBT, accuracyRF = [], [], []
        for i in range(10):
            test_set = train_data_split[i]
            train_set = train_data.drop(test_set.index)
            # Only doing it for a fraction of the data.
            train_set = train_set.sample(frac = frac, random_state = 32)

            # Finding accuracy for decision tree
            trainAccDT, testAccDT = trees.decisionTree(train_set, test_set, max_depth = 8)
            accuracyDT.append(testAccDT)

            trainAccBT, testAccBT = trees.bagging(train_set, test_set, max_depth = 8, num_trees = 30)
            accuracyBT.append(testAccBT)

            trainAccRF, testAccRF = trees.randomForests(train_set, test_set, max_depth = 8, num_trees = 30)
            accuracyRF.append(testAccRF)
            print('For Decision Trees Train Acc %s Test Acc %s \nFor Bagged Trees Train Acc %s Test Acc %s \nFor Random Forests Train Acc %s Test Acc %s \n'%(trainAccDT, testAccDT, trainAccBT, testAccBT, trainAccRF, testAccRF))

        # For each depth this is what you store
        accuracy[0].append(np.mean(accuracyDT))
        accuracy[1].append(np.mean(accuracyBT))
        accuracy[2].append(np.mean(accuracyRF))
        standard_error[0].append(np.std(accuracyDT)/sqrt(10))
        standard_error[1].append(np.std(accuracyBT)/sqrt(10))
        standard_error[2].append(np.std(accuracyRF)/sqrt(10))
        t_stat_accuracy_DT.append(accuracyDT)
        t_stat_accuracy_RF.append(accuracyRF)

    # Plot the graphs
    plot(fraction_list, accuracy, standard_error)

    for i,frac in enumerate(fraction_list):
        print("Null Hypothesis h0: Decision Tree Accuracy = Random Forest Accuracy")
        print("Alternate Hypothesis h1: Decision Tree Accuracy != Random Forest Accuracy")
        print('Running with a fraction of %s'%frac)
        print("Decision Tree accuracies: %s"%t_stat_accuracy_DT[i])
        print("Random Forests accuracies: %s"%t_stat_accuracy_RF[i])
        
        pvalue = ttest_rel(t_stat_accuracy_DT[i], t_stat_accuracy_RF[i]).pvalue
        if pvalue < 0.05:
            print ("\nRejecting Null Hypothesis H0 since the pvalue is less than 0.05")
        else:
            print ("\nAccepting Null Hypothesis H0 since pvalue is greater than 0.05")

if __name__ == '__main__':
    fracPerformance()