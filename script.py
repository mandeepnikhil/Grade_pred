from flask import Flask
import jellyfish
from sklearn.feature_selection import SelectKBest, f_regression
from sklearn.linear_model import Ridge
import sklearn.linear_model
from sklearn.preprocessing import FunctionTransformer
from sklearn.pipeline import Pipeline
import numpy as np
import pandas as pd
from flask import render_template
from flask import request
import pickle
from sklearn.preprocessing import LabelEncoder
from sklearn.impute import SimpleImputer
imp = SimpleImputer(strategy='most_frequent')
le = LabelEncoder()

kbest = SelectKBest(score_func=f_regression, k=15)
rdg = Ridge(alpha=25)

app = Flask(__name__)

replace_dicts = {'Coaching City': {'D400': 1, 'M510': 2, 'N500': 3, 'P500': 4, 'O360': 5, 'K300': 6, 'P350': 7, 'H361': 8, 'I536': 9},
                     'Coaching Name': {'F320': 1, 'A450': 2, 'N500': 3, 'V165': 4, 'S623': 5, 'V355': 6, 'R255': 7, 'O360': 8, 'N650': 9},
                     'Home City': {'J160': 1, 'M510': 2, 'B140': 3, 'P500': 4, 'B524': 5, 'I536': 6, 'H361': 7, 'O360': 8, 'P350': 9},
                     "Dad's Job": {'A632': 1, 'L000': 2, 'M324': 3, 'M525': 4, 'T252': 5, 'B252': 6, 'G130': 7, 'E420': 8}}

top_values_dict = {'Coaching City': ['K300',
  'H361',
  'D400',
  'P350',
  'N500',
  'M510',
  'I536',
  'P500'],
 'Coaching Name': ['F320',
  'A450',
  'R255',
  'S623',
  'N500',
  'V165',
  'N650',
  'V355'],
 'Home City': ['H361', 'P350', 'M510', 'J160', 'B524', 'I536', 'P500', 'D400'],
 "Dad's Job": ['G130', 'B252', 'E420', 'T252', 'M525', 'M324', 'L000', 'A632']}


def top_or_not(value, top_values):
    if value in top_values:
        return value
    else:
        return jellyfish.soundex("other")

def string_col_preprocess(data, col, n):
    # Normal string preprocessing
    data[col] = data[col].str.lower().str.strip().fillna("none").replace({'-':"none", '--':'none', '---':'none'})

    # Creating a new column with soundex encoding
    data[col+"Sound"] = data[col].apply(lambda x: jellyfish.soundex(x))

    # Now taking only first n common values and making others as "others"
    top_values = top_values_dict[col]
    data[col+'Sound'] = data[col+'Sound'].apply(lambda x: top_or_not(x, top_values))

    return data


def preprocess_data(data):


    data = data.replace({'-':0, '--':0, '---':0, '<':0, '>':0, 'Null':0, '12:00':2, 'O':0})
    # Sex feature
    data['Sex'] = data['Sex'].replace({'Male':0, 'Female':1})

    # Branch feature
    data['Branch'] = data['Branch'].replace({'Design':0,
                                         'CSE':1,
                                         'MC':2,
                                         'ECE/EEE':3,
                                         'ME':4,
                                         'CL':5,
                                         'EP':6,
                                         'CE':7,
                                         'CST':8,
                                         'BSBE':9})

    # Dropper feature
    data['Dropper?'] = data['Dropper?'].replace({'Yes':1, 'No':0})

    # 10th Board & 12th Board Feature
    """data['10th Board'] = data['10th Board'].replace({'ICSE':2,
                                                 'CBSE':1,
                                                 'State':0})
    #data['12th Board'] = data['12th Board'].replace({'ICSE':2,
                                                 'CBSE':1,
                                                 'State':0})

    #Coaching Feature
    #data['Coaching'] = data['Coaching'].replace({'Yes':1, 'No':0})"""


    # Coaching City, Coaching Name, Home State, Home City
    #data = data.drop(['Coaching City', 'Coaching Name', 'Home State', 'Home City'], axis=1)
    for col in ['Coaching City', 'Coaching Name', 'Home City', 'Dad\'s Job']:
        #for col in ['Coaching City', 'Coaching Name', 'Home City']:
        data = string_col_preprocess(data, col, 8)
        data = data.drop(col, axis=1)

        # Now replacing these values by descending order of mean (Label encoding by mean)

        data[col+'Sound'] = data[col+'Sound'].replace(replace_dicts[col])

    # Mom Dad Education
    data['Mom\'s Education'] = data['Mom\'s Education'].fillna('Post Graduate')
    data['Mom\'s Education'] = data['Mom\'s Education'].replace({'<10th Pass':0,
                                                                 '< 10th Pass':0,
                                                                 '10th Pass':1,
                                                                 '12th Pass':2,
                                                                 'Graduate':3,
                                                                 'Post Graduate':4})
    data['Dad\'s Education'] = data['Dad\'s Education'].fillna('Post Graduate')
    data['Dad\'s Education'] = data['Dad\'s Education'].replace({'<10th Pass':0,
                                                                 '< 10th Pass':0,
                                                                 '10th Pass':1,
                                                                 '12th Pass':2,
                                                                 'Graduate':3,
                                                                 'Post Graduate':4})

    # Mom dad Job and Hostel
    #for col in ['Mom\'s Job', 'Dad\'s Job', 'Hostel?']:
    for col in ['Hostel?']:
        data[col] = le.fit_transform(data[col])

    # Study Time Feature
    data['Study Time?'] = data['Study Time?'].fillna('Irregular')
    data['Study Time?'] = data['Study Time?'].replace({'Irregular':0,
                                                        'Everyday upto 0-2 hours':1,
                                                        'Everyday upto 2-4 hours':2})

    # Technical Club Feature
    #technical_dummy = data['Which Technical Clubs are you part of ?'].str.get_dummies(sep=', ')
    #data = pd.concat([data.drop('Which Technical Clubs are you part of ?', axis=1), technical_dummy], axis=1)

    # Cultural Club Feature
    #cult_dummy = data['Which Cultural Clubs are you part of?'].str.get_dummies(sep=', ')
    #data = pd.concat([data.drop('Which Cultural Clubs are you part of?', axis=1), cult_dummy], axis=1)

    # Fest Feature
    #fest_dummy = data['Member of Fests\' organizing team?'].str.get_dummies(sep=', ')
    #data = pd.concat([data.drop('Member of Fests\' organizing team?', axis=1), fest_dummy], axis=1)

    # Education Loan Feature
    #data['Have you taken an educational loan?'] = data['Have you taken an educational loan?'].replace({'No':0, 'Yes':1})

    # Time Spent Outside Feature
    data['Time spent outside your room[except classes]? (daily average, in hours)'] = data['Time spent outside your room[except classes]? (daily average, in hours)'].fillna(4)


    #Attendance feature
    data['Attendance?'] = data['Attendance?'].replace({'Below 50?':50,
                                                       'Below 75?':62.5,
                                                       'Below 90?':87.5,
                                                       'Above 90?':95})

    # Relationship Feature
    data['Relationship status?'] = data['Relationship status?'].replace({'Committed':0,
                                                       'Complicated':0,
                                                       'Single':1})

    # Library Feature
    data['Library?'] = data['Library?'].replace({'Rarely':0,
                                                 'During Exams':1,
                                                 'Often':2})

    # Sleeping time
    """data['When do you sleep?'] = data['When do you sleep?'].replace({'Before 10 pm':0,
                                                                     'After 10 pm':1,
                                                                     'Around 12':2,
                                                                     'After 12 am':3,
                                                                     'Around 1':4,
                                                                     'After 2 am':5,
                                                                     '3':6,
                                                                     3:6,
                                                                     'Never':8,
                                                                     '6:30 am':8})"""
    #data = data.drop("When do you sleep?", axis=1)

    # Sleeping Duration
    data['Sleep Duration(Hrs)?'] = data['Sleep Duration(Hrs)?'].replace({'<=4':-2,
                                                                     '5':-1,
                                                                     5:-1,
                                                                     '6':0,
                                                                     6:0,
                                                                     '7':1,
                                                                     7:1,
                                                                     '>=8':2})

    # Sleep in Day
    #data['Do you sleep during the day?'] = data['Do you sleep during the day?'].replace({'Yes':1, 'No':0})

    # Addiction Feature
    #addiction_dummy = data['Addiction?'].str.get_dummies(sep=', ')
    #data = pd.concat([data.drop('Addiction?', axis=1), addiction_dummy], axis=1)

    # Group Study or Individual
    #data['Group Study/Individual'] = data['Group Study/Individual'].replace({'Group Study':1, 'Individual':0})

    # Study Material Preferred
    #data['Study Material Preferred'] = data['Study Material Preferred'].replace({'Online content':0, 'Books':1})

    # Core/NonCore
    #data['Core/Non-Core'] = data['Core/Non-Core'].replace({'Core':0, 'Non-Core':1})


    # Missing Values
    #data = pd.DataFrame(imp.fit_transform(data), columns=data.columns)
    return data.fillna(0)

@app.route('/')
def index():
    return render_template("index.html")

# prediction function
def ValuePredictor(to_predict_list):

    meaningful_features = ['Sex', 'Branch', 'Dropper?', 'Coaching City',
                       'Coaching Name', 'Home City', 'Mom\'s Education', 'Dad\'s Education', 'Dad\'s Job',
                       'Study Time?', 'Hostel?',
                       'At the time of Spardha/Kirti/Manthan, approx. time given per day? (hours)',
                       'Time spent outside your room[except classes]? (daily average, in hours)', 'Attendance?',
                       'Relationship status?', 'Library?', 'Sleep Duration(Hrs)?', 'Tut taken seriously',
                       'Quiz taken seriously', 'Midsem taken seriously', 'Endsem taken seriously', 'Aiming for branch change']

    # Final processing of String variables

    #df = pd.DataFrame(columns=meaningful_features)
    #df.loc[0] = to_predict_list
    #to_predict = np.array(to_predict_list).reshape(1, 12)
    #return to_predict_list
    X_test = pd.DataFrame(columns=meaningful_features)
    X_test.loc[0] = to_predict_list[:-2]
    #return X_test.values
    X_test = preprocess_data(X_test)

    X_train = pd.read_csv('transformed_data.csv')
    y_train = pd.read_csv('y_train.csv', header=None)

    kbest.fit(X_train, y_train)

    X_train = kbest.transform(X_train)
    X_test = kbest.transform(X_test)

    #return X_test
    rdg.fit(X_train, y_train)
    y_pred = rdg.predict(X_test)

    # Select 13 best features
    """df = df[['Branch', 'Dropper?', 'Mom\'s Education', 'Dad\'s Education',
       'Attendance?', 'Tut taken seriously', 'Quiz taken seriously',
       'Midsem taken seriously', 'Endsem taken seriously',
       'Aiming for branch change', 'Coaching CitySound', 'Coaching NameSound',
       'Home CitySound', 'Dad\'s JobSound']]"""

    #loaded_model = pickle.load(open("model.pkl", "rb"))
    #result = loaded_model.predict(df)
    return "Your predicted grade is {}".format(y_pred[0])


#app = Flask(__name__)


@app.route('/result', methods = ['POST'])
def result():
    if request.method == 'POST':
        to_predict_list = request.form.to_dict()
        to_predict_list = list(to_predict_list.values())
        #to_predict_list = list(map(int, to_predict_list))
        result = ValuePredictor(to_predict_list)
        return render_template("result.html", prediction = result)
