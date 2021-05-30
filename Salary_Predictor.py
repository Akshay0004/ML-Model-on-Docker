import pandas
import joblib
import numpy
db = pandas.read_csv('Salary.csv')
y = db["Salary"]
x = db['YearsExperience']
x = x.values
x = x.values.reshape(-1,1)
from sklearn.linear_model import LinearRegression
model = LinearRegression()
model.fit( x, y)
years = int(input("Enter your Years Experience to predict the Salary"))
Salary_approx = model.predict([[ years ]])
print(Salary_approx)
