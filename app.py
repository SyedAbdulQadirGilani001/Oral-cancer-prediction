# # Only accuracy and error are not enough to evaluate the model. Generate a classification report and confusion matrices with ROC-AUC. If the number of samples is not equal you have to use SMOTE-like oversampling techniques to balance the dataset. use a loop approach to fit and create a prediction on it instead of separately using every model.
# # share the result of every algorithm like a Screen Shot of confusion matrices.  in streamlit
# # x=df[['diagnoses/0/ajcc_pathologic_stage', 'diagnoses/0/treatments/0/treatment_type','demographic/vital_status']]
# # y=df['demographic/days_to_death']
# # checking death rate, and survival rate
# # streamlit app ask user to upload a file, asks features, ask model
# # and predict the result
# ask user upload data, and select features, and give target to the user and apply different models given and do best accuracy and give predicted and actual dead alive vital_status patients
# # Only accuracy and error are not enough to evaluate the model. Generate a classification report and confusion matrices with ROC-AUC. If the number of samples is not equal you have to use SMOTE-like oversampling techniques to balance the dataset. use a loop approach to fit and create a prediction on it instead of separately using every model.
# # share the result of every algorithm like a Screen Shot of confusion matrices.  in streamlit
# # x=df[['diagnoses/0/ajcc_pathologic_stage', 'diagnoses/0/treatments/0/treatment_type','demographic/vital_status']]
# # y=df['demographic/days_to_death']
# # checking death rate, and survival rate
# # streamlit app ask user to upload a file, asks features, ask model
# # and predict the result
# ask user upload data, and select features, and give target to the user and apply different models given and do best accuracy and give predicted and actual dead alive vital_status patients

import streamlit as st # for web app
import pandas as pd # for data manipulation
import numpy as np # for numerical computation
import matplotlib.pyplot as plt # for plotting graphs
import seaborn as sns # for data visualization
import plotly.express as px # for data visualization, interactive graphs
import plotly.graph_objects as go # for data visualization, interactive graphs, and subplots

from sklearn.model_selection import train_test_split # for splitting the data into train and test
from sklearn.preprocessing import StandardScaler # for feature scaling
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report # for model evaluation
from sklearn.linear_model import LogisticRegression # for logistic regression
from sklearn.tree import DecisionTreeClassifier # for decision tree classifier
from sklearn.ensemble import RandomForestClassifier # for random forest classifier
from sklearn.svm import SVC # for support vector classifier
from sklearn.neighbors import KNeighborsClassifier # for K nearest neighbors classifier
from sklearn.naive_bayes import GaussianNB # for naive bayes classifier
from sklearn.model_selection import cross_val_score # for cross validation score
from sklearn.model_selection import GridSearchCV # for hyperparameter tuning
# smote
from imblearn.over_sampling import SMOTE



# title
st.title("Days to Death Oral Cancer Prediction")
# subtitle alive or death prediction
st.subheader("Alive or Death Prediction")
# description
st.markdown("This is a web app to predict if a patient will die or not based on the given features.")
# upload data from user
uploaded_file = st.file_uploader("Upload your input CSV file", type=["csv"])
# if user uploaded a file
if uploaded_file is not None:
    # read the file
    df = pd.read_csv(uploaded_file)
    # remove None values in the dataset
    df.dropna(axis=1, how='all', inplace=True)
    # remove None rows,column in the dataset
    df.dropna(axis=0, how='all', inplace=True)    
    # show the dataframe
    st.write('**First 5 rows of the dataset**')
    # show the dataframe
    st.write(df.head())
    X=df[['diagnoses/0/ajcc_pathologic_stage', 'demographic/days_to_death']]
    # pathologic stage replace to 1,2,3,4
    X.fillna(df.mode().iloc[0], inplace=True)
    X['diagnoses/0/ajcc_pathologic_stage'].replace({'Stage I':1,'Stage IA':1,'Stage IB':1,'Stage II':2,'Stage IIA':2,'Stage IIB':2,'Stage III':3,'Stage IIIA':3,'Stage IIIB':3,'Stage IIIC':3,'Stage IV':4,'Stage IVA':4,'Stage IVB':4,'Stage IVC':4,'Not Reported':0},inplace=True)
    # demographic/days_to_death replace to dummy variable
    y=df['demographic/vital_status']
    y.fillna(df.mode().iloc[0], inplace=True)
    # y.fillna(df.mode().iloc[0], inplace=True)
    # user asking model
    st.subheader("Select the model want to use for prediction")
    # user select model
    model = st.selectbox("Model", ["Logistic Regression", "Decision Tree", "Random Forest", "Support Vector Machine", "K Nearest Neighbors", "Naive Bayes"])
    # if user select logistic regression show accuracy
    if model == "Logistic Regression":
        # user select test size
        test_size = st.slider("Test size", 0.1, 0.9, 0.25, 0.05)
        # user select random state
        random_state = st.slider("Random state", 1, 100, 42, 1)
        # split the data into train and test
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)
        # feature scaling
        sc = StandardScaler()
        X_train = sc.fit_transform(X_train)
        X_test = sc.transform(X_test)
        # fit the model
        classifier = LogisticRegression()
        classifier.fit(X_train, y_train)
        # predict the test set
        y_pred = classifier.predict(X_test)
        # evaluate the model
        st.write("**Percentage Accuracy:**")
        st.write(str(accuracy_score(y_test, y_pred) * 100) + "%")
    # if user select decision tree show accuracy
    elif model == "Decision Tree":
        # user select test size
        test_size = st.slider("Test size", 0.1, 0.9, 0.25, 0.05)
        # user select random state
        random_state = st.slider("Random state", 1, 100, 42, 1)
        # split the data into train and test
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)
        # feature scaling
        sc = StandardScaler()
        X_train = sc.fit_transform(X_train)
        X_test = sc.transform(X_test)
        # fit the model
        classifier = DecisionTreeClassifier()
        classifier.fit(X_train, y_train)
        # predict the test set
        y_pred = classifier.predict(X_test)
        # evaluate the model
        st.write("**Percentage Accuracy:**")
        st.write(str(accuracy_score(y_test, y_pred) * 100) + "%")
    # if user select random forest show accuracy
    elif model == "Random Forest":
        # user select test size
        test_size = st.slider("Test size", 0.1, 0.9, 0.25, 0.05)
        # user select random state
        random_state = st.slider("Random state", 1, 100, 42, 1)
        # split the data into train and test
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)
        # feature scaling
        sc = StandardScaler()
        X_train = sc.fit_transform(X_train)
        X_test = sc.transform(X_test)
        # fit the model
        classifier = RandomForestClassifier()
        classifier.fit(X_train, y_train)
        # predict the test set
        y_pred = classifier.predict(X_test)
        # evaluate the model
        st.write("**Percentage Accuracy:**")
        st.write(str(accuracy_score(y_test, y_pred) * 100) + "%")
    # if user select support vector machine show accuracy
    elif model == "Support Vector Machine":
        # user select test size
        test_size = st.slider("Test size", 0.1, 0.9, 0.25, 0.05)
        # user select random state
        random_state = st.slider("Random state", 1, 100, 42, 1)
        # split the data into train and test
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)
        # feature scaling
        sc = StandardScaler()
        X_train = sc.fit_transform(X_train)
        X_test = sc.transform(X_test)
        # fit the model
        classifier = SVC()
        classifier.fit(X_train, y_train)
        # predict the test set
        y_pred = classifier.predict(X_test)
        # evaluate the model
        st.write("**Percentage Accuracy:**")
        st.write(str(accuracy_score(y_test, y_pred) * 100) + "%")
    # if user select k nearest neighbors show accuracy
    elif model == "K Nearest Neighbors":
        # user select test size
        test_size = st.slider("Test size", 0.1, 0.9, 0.25, 0.05)
        # user select random state
        random_state = st.slider("Random state", 1, 100, 42, 1)
        # split the data into train and test
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)
        # feature scaling
        sc = StandardScaler()
        X_train = sc.fit_transform(X_train)
        X_test = sc.transform(X_test)
        # fit the model
        classifier = KNeighborsClassifier()
        classifier.fit(X_train, y_train)
        # predict the test set
        y_pred = classifier.predict(X_test)
        # evaluate the model
        st.write("**Percentage Accuracy:**")
        st.write(str(accuracy_score(y_test, y_pred) * 100) + "%")
    # if user select naive bayes show accuracy
    elif model == "Naive Bayes":
        # user select test size
        test_size = st.slider("Test size", 0.1, 0.9, 0.25, 0.05)
        # user select random state
        random_state = st.slider("Random state", 1, 100, 42, 1)
        # split the data into train and test
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)
        # feature scaling
        sc = StandardScaler()
        X_train = sc.fit_transform(X_train)
        X_test = sc.transform(X_test)
        # fit the model
        classifier = GaussianNB()
        classifier.fit(X_train, y_train)
        # predict the test set
        y_pred = classifier.predict(X_test)
        # evaluate the model
        st.write("**Percentage Accuracy:**")
        st.write(str(accuracy_score(y_test, y_pred) * 100) + "%")
    # show actual and predicted
    st.write("**Actual and Predicted**")
    # show actual and predicted
    st.write(pd.DataFrame({"Actual": y_test, "Predicted": y_pred}))
    # plotly interactive graph histogram 
    fig = px.histogram(df, x="demographic/vital_status", color="demographic/vital_status", title="Alive or Death")
    st.plotly_chart(fig)
    # actual and predicted plotly factory bar graph
    fig = go.Figure(data=[
        go.Bar(name='Actual', x=y_test, y=y_test),
        go.Bar(name='Predicted', x=y_test, y=y_pred)
    ])
    # Change the bar mode
    fig.update_layout(barmode='group')
    st.plotly_chart(fig)
    # seaborn actual and predicted histogram
    fig, ax = plt.subplots()
    sns.histplot(data=y_test, ax=ax, label="Actual", color="green")
    sns.histplot(data=y_pred, ax=ax, label="Predicted", color="red")
    ax.legend()
    st.pyplot(fig)
    # roc curve plot
#     x=df[['diagnoses/0/ajcc_pathologic_stage', 'diagnoses/0/treatments/0/treatment_type','demographic/vital_status']]
# # # y=df['demographic/days_to_death']
# # # checking death rate, and survival rate
    death_rate = df[df['demographic/vital_status'] == 'Dead'].shape[0] / df.shape[0]
    survival_rate = df[df['demographic/vital_status'] == 'Alive'].shape[0] / df.shape[0]
    st.write("**Percentage of Dead Patients:**")
    st.write(str(death_rate * 100) + "%")
    # percentage of Not Reported
    st.write("**Percentage of Not Reported Patients:**")
    st.write(str((1 - death_rate - survival_rate) * 100) + "%")
    st.write("**Percentage of Alive Patients:**")
    st.write(str(survival_rate * 100) + "%")
    # sunburst plot of death rate and survival rate
    fig = px.sunburst(df, path=['demographic/vital_status'], title="Alive or Death", width=600, height=600, color_discrete_sequence=['#1E90FF', '#FF4500'])
    st.plotly_chart(fig)
