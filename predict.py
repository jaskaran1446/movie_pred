import numpy as np
import pandas as pd
np.random.seed(10)

from sklearn.model_selection import train_test_split
from sklearn import linear_model
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error as mse, explained_variance_score as evs

X = pd.read_csv("movie_metadata.csv")
Y = X["imdb_score"]
X.drop(["imdb_score"], axis=1,inplace=True)
X.drop(["movie_title"], axis=1,inplace=True)

#clean and scale
X = X.fillna(0)
X["gross"] = X["gross"]/10000
X["num_voted_users"] = X["num_voted_users"]/10000
X["cast_total_facebook_likes"] = X["cast_total_facebook_likes"]/1000
X["movie_facebook_likes"] = X["movie_facebook_likes"]/1000
X["budget"] = X["budget"]/10000
X["actor_1_facebook_likes"] = X["actor_1_facebook_likes"]/1000
X["actor_2_facebook_likes"] = X["actor_2_facebook_likes"]/1000
X["actor_3_facebook_likes"] = X["actor_3_facebook_likes"]/1000

#split into train and test sets
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, train_size=0.8, test_size=0.2)

#create regressor
reg = Pipeline([('poly', PolynomialFeatures(degree=2)),('linear', linear_model.LinearRegression(fit_intercept=False))])

#train
reg.fit(X_train, Y_train)

#predict
Y_pred = reg.predict(X_test)

#metrics
print('Variance:' + str(evs(Y_test, Y_pred)))
print('Mean square error:' + str(mse(Y_test, Y_pred)))
