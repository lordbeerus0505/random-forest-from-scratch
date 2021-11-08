import pandas as pd
import sys
import numpy


encoderDict = {}

def encoder(pd_series):
    # More information here https://pandas.pydata.org/docs/reference/api/pandas.Series.cat.categories.html easier that pandas.factorize
    encoding = {}
    name = pd_series.name
    count = 0
    for category in pd_series.cat.categories:
        encoding[category] = count
        count += 1
    
    encoderDict[name] = encoding
    return pd_series.cat.codes

def read_dataset(fromFile, toFile):

    data = pd.read_csv(fromFile)
    # Top 6500 elements
    data = data.head(6500)
    data = data.drop(columns=['race', 'race_o', 'field'])
    

    """ 
    Use label encoding to convert the categorical values in columns gender, race, race_o and
    field to numeric values start from 0. The process of label encoding works by mapping
    each categorical value of an attribute to an integer number between 0 and nvalues − 1 where
    nvalues is the number of distinct values for that attribute. Sort the values of each categorical
    attribute lexicographically/alphabetically before you start the encoding process. You
    are then asked to output the mapped numeric values for ‘male’ in the gender column, for
    ‘European/Caucasian-American’ in the race column, for ‘Latino/Hispanic American’ in the
    race o column, and for ‘law’ in the field column.
        Expected output lines:
            Value assigned for male in column gender: [value-for-male].  
            2 values for gender - assigning female as 0 male as 1

    """
    data['gender'] = encoder(data['gender'].astype('category'))

    # import pdb; pdb.set_trace()

    # print("Value assigned for male in column gender: %s"% encoderDict['gender']['male'])

    """ 
    Normalization: As the speed dating experiments are conducted in several different batches,
    the instructions participants received across different batches vary slightly. For example, in
    some batches of experiments participants are asked to allocate a total of 100 points among the
    six attributes (i.e., attractiveness, sincerity, intelligence, fun, ambition, shared interests) to
    indicate how much they value each of these attributes in their romantic partner—that is, the
    values in preference scores of participant columns of a row should sum up to 100 (similarly,
    values in preference scores of partner columns of a row should also sum up to 100)—while in
    some other batches of experiments, participants are not explicitly instructed to do so.
    To deal with this problem, let’s conduct one more pre-process step for values in preference scores of participant and preference scores of partner columns. For each row, let’s first
    sum up all the values in the six columns that belong to the set preference scores of participant
    (denote the sum value as total), and then transform the value for each column in the set preference scores of participant in that row as follows: new value=old value/total. We then conduct
    similar transformation for values in the set preference scores of partner.
    Finally, you are asked to output the mean values for each column in these two sets after
    the transformation.
        Expected output lines: (All 6 attrs of both kinds so total of 12)
            Mean of attractive important: [mean-rounded-to-2-digits].
            Mean of shared interests important: [mean-rounded-to-2-digits].
            Mean of pref o attractive: [mean-rounded-to-2-digits].
            ...
            Mean of pref o shared interests: [mean-rounded-to-2-digits].
    """
    importance = ['attractive_important', 'sincere_important', 'intelligence_important', 'funny_important', 'ambition_important', 'shared_interests_important']
    partner_metrics = ['pref_o_attractive','pref_o_sincere','pref_o_intelligence','pref_o_funny','pref_o_ambitious','pref_o_shared_interests']

    sum_importance = 0
    for i in importance:
        sum_importance += data[i]
    sum_partner_metrics = 0
    for i in partner_metrics:
        sum_partner_metrics += data[i]
    
    # most sum up to 100 but not all, divding all by the sum so % will be the same.

    for i in range(len(importance)):
        # both importance and partner_metrics are of length 6
        data[importance[i]] = data[importance[i]]/sum_importance
        data[partner_metrics[i]] = data[partner_metrics[i]]/sum_partner_metrics

    discretize(data, toFile)

def discretize(data_frame, toFile):
    otherCols = ['sports', 'tvsports', 'exercise', 'dining', 'museums', 'art', 'hiking', 
    'gaming', 'clubbing', 'reading', 'tv', 'theater', 'movies', 'concerts', 'music', 
    'shopping', 'yoga' , 'expected_happy_with_sd_people', 'like', 'interests_correlate'
    ]
    # Rest of the fields have already been handled for and seem fine. All of these have 
    # a range 0-10 (except interest corr) so check max and min

    """ 
    NOTE: SINCE ITS NOT ASKED TO PERFORM THIS STEP, NOT DOING IT AS IT ONLY REDUCES PERFORMANCE
    
    improper_data_list = []

    for col in otherCols[:-1]:
        maxi = data_frame[col].max()
        mini = data_frame[col].min()
        improper_data_list.append([mini, maxi])
    
    max_interests = data_frame[otherCols[len(otherCols)-1]].max()
    min_interests = data_frame[otherCols[len(otherCols)-1]].min()

    issue_data = []

    if (max_interests > 1.0 or min_interests < -1.0):
        issue_data.append(len(otherCols)-1)
    
    for i in range(len(improper_data_list)):
        if improper_data_list[i][0]<0 or improper_data_list[i][1]>10:
            issue_data.append(i)

    # issue_data has 7,9 so gaming and reading have max issues. The values correspond to 14 and 13 respectively.
    for i in [7,9]:
        col = otherCols[i]
        # for every row where data_frame[col]>10
        data_frame.loc[data_frame[col]>10, col] = 10
        
    """

    discrete_cols = ['gender', 'race', 'race_o', 'samerace', 'field', 'decision'] 
    attributes = ['pref_o_attractive','pref_o_sincere','pref_o_intelligence','pref_o_funny','pref_o_ambitious',
    'pref_o_shared_interests', 'attractive_important', 'sincere_important', 'intelligence_important', 'funny_important',
    'ambition_important', 'shared_interests_important'] 
    bins = 2
    for col in data_frame:
        if col not in discrete_cols:
            # Hasn't already been binned
            data_frame[col] = pd.cut(data_frame[col], bins, labels = [0,1], 
            include_lowest = True, retbins = False)

    test_data = data_frame.sample(frac=0.2, random_state = 47)
    train_data = data_frame.drop(test_data.index)
    train_data.to_csv('trainingSet.csv', index = False, mode = 'w')
    test_data.to_csv('testSet.csv', index = False, mode = 'w')
    # return data_frame

if __name__ == '__main__':
    read_dataset(sys.argv[1], sys.argv[2])
