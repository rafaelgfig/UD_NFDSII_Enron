#!/usr/bin/python

""" 
    Starter code for exploring the Enron dataset (emails + finances);
    loads up the dataset (pickled dict of dicts).

    The dataset has the form:
    enron_data["LASTNAME FIRSTNAME MIDDLEINITIAL"] = { features_dict }

    {features_dict} is a dictionary of features associated with that person.
    You should explore features_dict as part of the mini-project,
    but here's an example to get you started:

    enron_data["SKILLING JEFFREY K"]["bonus"] = 5600000
    
"""

import pickle
import pandas as pd

enron_data = pickle.load(open("../final_project/final_project_dataset.pkl", "r"))
df = pd.DataFrame(enron_data)
#print(df.info())
#print(df.T.poi.sum())
#print df.columns.tolist()
#print df['PRENTICE JAMES']['total_stock_value']
#print df['COLWELL WESLEY']
print df['SKILLING JEFFREY K']
#print df.T['total_payments'].sort_values(ascending=False)
#print df.T['salary'].value_counts().head(3)
#print df.T['email_address'].value_counts().head(3)
#print df.T['total_payments'].value_counts().head(3)
#print df.T['total_payments'].count()
print df.T['total_payments'][df.T['poi'] == True]
