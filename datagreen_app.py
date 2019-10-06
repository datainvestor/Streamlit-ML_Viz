import streamlit as st
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
import plotly.express as px

st.title("ML visualization project")


#creat function that loads data:
df=pd.read_csv("Iris.csv")
df=df.drop(['Id'], axis=1)

#create ML model
X = np.array(df.ix[:, 0:4])
y = np.array(df['Species']) 
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)

nto_filter = st.slider('n', 1, 80, 59)
knn = KNeighborsClassifier(n_neighbors=nto_filter)
  
# fitting the model
knn.fit(X_train, y_train)
#  predict the response
pred = knn.predict(X_test)
dfp=pd.DataFrame(X_test)
dfp.columns = ['SepalLengthCm','SepalWidthCm', 'PetalLengthCm', 'PetalWidthCm']
dfp["PClass"]=pred

st.write(dfp)


#CREATE PLOTLY CHART

fig = px.scatter(dfp, x='SepalLengthCm', y='SepalWidthCm', color='PClass', hover_name='PClass')

st.plotly_chart(fig)