""" Simple prediction of house prices based on MLS Data Table
e.g. fields: Sq Ft, Lot Size, Total Bths, Bedrooms, Chg Date (sale date)
Removes columns MLS #, Street Address, Bths (which is full/half baths)
Needs some manual cleaning of data still""" 


from sklearn.ensemble import GradientBoostingRegressor
import pandas as pd
import datetime

today = datetime.datetime.today()
train = pd.read_csv('Downloads/Fremont Houses - Sheet1.csv')

def ChgDt_to_diff(x):
    return (today - datetime.datetime.strptime(x, "%Y-%m-%d")).days

def remove_comma(x):
    return float(x.replace(",",""))


train_Y = train["Price"]
train_Y = map(remove_comma, train_Y)

train = train.drop(["MLS #", "Price", "Bths","Street Address"], axis = 1)
print train.columns
train['Chg Dt'] = map(ChgDt_to_diff, train['Chg Dt'])
train['Lot Size'] = map(remove_comma, train['Lot Size'])
train = train.dropna(axis = 1)
train = train.astype(float)

# Input Test data has two entries, first for land + house, second for land only
test = pd.read_csv('Downloads/Fremont Houses - input-house.csv')
test = test.drop(["MLS #", "Bths","Street Address"], axis = 1)

test['Chg Dt'] = map(ChgDt_to_diff, test['Chg Dt'])
#test = test.fillna(0)
test = test.astype(float)

print train
print "Y", train_Y
print "predict on", test

gbm = GradientBoostingRegressor(subsample = .8, min_samples_split=1)
# Default parameters should work fine here
gbm.fit(train, train_Y)
predictions = gbm.predict(test)

print "Gradient Boosting"
print "Predicted House Value", round(predictions[0],0)
print "Predicted Land Value", round(predictions[1],0)
print "Score", gbm.score(train, train_Y)
imp = sorted(zip(train.columns, gbm.feature_importances_), key=lambda tup: tup[1], reverse=True)
for fea in imp:
    print fea[0], round(fea[1],4)
#print gbm.oob_score_
#print gbm.train_score_
print ""