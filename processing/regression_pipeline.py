import pandas as pd
import json
from sqlalchemy import create_engine
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.pipeline import Pipeline


engine = create_engine("sqlite:///../data/sample_strategy/samples.db")
def regression(time):
    df = pd.read_sql(
        f'select x1, x2, x3, x4, x5, x6, x7, x8, x9, y from NearsestSample where TimeDim = {time}',
        engine
    )
    df = df.dropna()

    X = df[['x1', 'x2', 'x3', 'x4', 'x5', 'x6', 'x7', 'x8', 'x9']]
    y = df[['y']]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=131006)

    # pipeline = Pipeline([
    #     ('scaler', StandardScaler()),
    #     ('model', RandomForestRegressor(
    #         n_estimators=150,
    #         max_depth=None,
    #         random_state=131006,
    #         n_jobs=-1
    #     ))
    # ])

    scaler_X = StandardScaler()
    scaler_y = StandardScaler()

    X_train_scaled = scaler_X.fit_transform(X_train)
    X_test_scaled = scaler_X.transform(X_test)

    y_train_scaled = scaler_y.fit_transform(y_train)
    y_test_scaled = scaler_y.transform(y_test)

    # model = RandomForestRegressor(
    #         n_estimators=150,
    #         random_state=131006,
    #         n_jobs=-1,
    #         min_samples_leaf=2
    #     )
    model = LinearRegression()
    
    model.fit(X_train_scaled, y_train_scaled)
    y_pred = model.predict(X_test_scaled)

    mse = mean_squared_error(y_test_scaled, y_pred)
    r2 = r2_score(y_test_scaled, y_pred)

    print("MSE:", mse)
    print("R2:", r2)

    # print('Feature Importances:', model.feature_importances_)
    print('Coef:', model.coef_)
    print('Inter:', model.intercept_)
    # print("Train R2:", model.score(X_train_scaled, y_train_scaled))
    # print("Test R2:", model.score(X_test_scaled, y_test_scaled))
    print('-----------------------------------------------------')

regression(2011)
regression(2012)
regression(2013)
regression(2014)