# Implementing Decision Trees, Bagged Trees and Random Forests from Scratch

## Comparing Methods for Speed Dating Classication
### Submitted by Abhiram Natarajan PUID 0033250677
### Instructions to run code:
**Requirements**:
- pip install numpy
- pip install pandas
- pip install matplotplib
- pip install scipy
- python 3.6.3

#### Please add dating-full.csv to this list before running it.

# Question 1: Preprocessing 
As before, the single quotes, conversion to lower and everything else performed in Assignment 2 preprocess.py have been performed. Only the first 6500 entries have been considered. The data was then split into train and test that were stored in trainingSet.csv and testSet.csv (passed as arguments). Discretization was done to split into exactly 2 bins. Train and Test sets were built using random_state of 47 and fraction of 0.2 for the test set.
> Can be run using the command - `python preprocess-assg4.py trainingSet.csv testSet.csv`  

# Question 2: Implement Decision Trees, Bagging and Random Forests
All three models were implemented from scratch using Gini Gain scoring function instead of Information Gain as this produced smaller values. The code was written into `trees.py`. For the decision tree, pruning was performed when the depth reached 8 or the number of examples was below 50.  
For bagging, a function was written that created 30 such decision trees, all from the same dataset but with replacement set to true when sampling - bootstrapped sampling. This meant that there would be multiple copies of some data in the dataset and this added bias would produce better results.  
For random forests downsampling was done in terms of the number of features used. This was done by taking the square root of the number of features, which in this case was 8. From those 8, the `best` attribute was selected.
> Can be run using the command - `python trees.py trainingSet.csv testSet.csv x` where we replace x with 1 for Decision Trees, 2 for Bagged Trees and 3 for Random Forests

# Question 3: Influence of Tree Depth on Classifier Performance
Data was shuffled using a random state of 18 and to speed up testing, it was downsampled to 50%. On this new dataset, 10 fold cross validation was performed. Note that this was only on the trainingSet and the test set was left untouched. For cross validation, one of the 10 sets was marked as test while the other nine were used to train. The accuracy of the test set was noted and for all 10 iterations, an average was recorded. Depth was varied as 3,5,7,9 and for each, the average accuracy of the 3 models across 10 fold validation was recorded. The average accuracies were then plot on a graph with +-1 standard error.  
Hypothesis testing was done by comparing Decision Trees to Random Forests which for most cases crossed the 0.05 significance level.
> Can be run using the command - `python cv_depth.py` Note that there are print statements explicitly added to show the program is still running.

# Question 4: Compare Performance of Different models
This section dealt with changing the fraction of the dataset used and trainSet was then downsampled using a fractional value from the list. The test set for crossvalidation was still 10% of the original trainingSet though. A random state of 32 was used in addition to a depth of 8 and example limit of 50. 30 trees were learnt for both Bagging and Random Forests. Graphs were plot based off the average accuracies from the 10 folds as mentioned earlier. Hypothesis testing was performed between Decision Trees and Random Forests using a significance level of 0.05.
> Can be run using the command - `python cv_frac.py` Note that there are print statements explicitly added to show the program is still running.

# Question 5: The influence of number of trees on Classifier Performance
This section dealt with changing the number of trees in both the ensemble models and seeing if that affected the performance of the models. As in Question 4, only 50% of the dataset was used - thanks to sampling. The number of trees varied between 10, 20, 40, 50. The average accuracy was plot on a graph and hypothesis testing was performed with alph = 0.05
> Can be run using the command - `python cv_numtrees.py` Note that there are print statements explicitly added to show the program is still running.

# Question 6: Bonus
For the bonus, a Perceptron model for the same data was developed. As this too is a linear separator the accuracy was comparable to decision trees. Hyper Parameter tuning was performed to find the right number of epochs and learning rate.
> Can be run using the command - `python bonus.py`

#### Special conditions
Sometimes, there are warnings thrown. These are explictly ignored in the code but can be commented out if required.