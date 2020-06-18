import streamlit as st
import pandas as pd
import numpy as np
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import plot_confusion_matrix, plot_roc_curve, plot_precision_recall_curve
from sklearn.metrics import precision_score, recall_score
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier



def main():   
    st.title("Social_Network_Ads_Purchas")
    st.sidebar.title("Binary Classification Web App")
    st.markdown("Are clinet purchased or not? ")
    st.sidebar.markdown("Are clinet purchased or not? ")

    @st.cache(persist=True)
    def load_data():
        filePath="Social_Network_Ads.csv" 
        data=pd.read_csv(filePath)
        data['Gender']=pd.get_dummies(data["Gender"])
        
        #print(data)
        return data

    @st.cache(persist=True)
    def split(df):
        y = df.Purchased

        x = df.iloc[:,1:-1]
        scaler = StandardScaler()
        x.EstimatedSalary=scaler.fit_transform(x[['EstimatedSalary']])
        x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=0)
        return x_train, x_test, y_train, y_test


    def plot_metrics(metrics_list):
        if 'Confusion Matrix' in metrics_list:
            st.subheader("Confusion Matrix")
            plot_confusion_matrix(model, x_test, y_test, display_labels=class_names)
            st.pyplot()

        if 'ROC Curve' in metrics_list:
            st.subheader("ROC Curve")
            plot_roc_curve(model, x_test, y_test)
            st.pyplot()
        
        if 'Precision-Recall Curve' in metrics_list:
            st.subheader('Precision-Recall Curve')
            plot_precision_recall_curve(model, x_test, y_test)
            st.pyplot()

    def cal_accuracy(pred):
        accuracy = model.score(x_test, y_test)
        st.write("Accuracy: ", accuracy.round(2))
        st.write("Precision: ", precision_score(y_test, pred, labels=class_names).round(2))
        st.write("Recall: ", recall_score(y_test, pred, labels=class_names).round(2))
        plot_metrics(metrics)    

    df = load_data()
    class_names = ['purchased', 'not purchused']

    x_train, x_test, y_train, y_test = split(df)

    st.sidebar.subheader("Choose Classifier")
    classifier = st.sidebar.selectbox("Classifier", ("Support Vector Machine (SVM)", "Logistic Regression", "Random Forest" , "Desicion Tree"))

    metrics = st.sidebar.multiselect("What metrics to plot?", ('Confusion Matrix', 'ROC Curve', 'Precision-Recall Curve'))

    if classifier == 'Support Vector Machine (SVM)':
        st.sidebar.subheader("Model Hyperparameters")
        C=st.sidebar.number_input(label="C = ",min_value=0.01,max_value=10.0,key='C_SVM')
        kernel=st.sidebar.radio(label='Kernel = ',options=('rbf','linear'),key='kernel')
        gamma = st.sidebar.radio("Gamma", ("scale", "auto"), key='gamma')   
        if st.sidebar.button("Classify", key='classify'): 
            st.subheader("Support Vector Machine (SVM) Results")   
            model=SVC(C=C,kernel=kernel,gamma=gamma)
            model=model.fit(x_train,y_train)
            y_pred = model.predict(x_test)
            cal_accuracy(y_pred)
        
    if classifier== "Desicion Tree":
        st.sidebar.subheader("Model Hyperparameters")
        max_depth=st.sidebar.number_input(label='max depth : ',min_value=1,max_value=15,step=1,key='depth')
        if st.sidebar.button("Classify", key='classify'):
            st.subheader("Desicion Tree Results")
            model=DecisionTreeClassifier(max_depth=max_depth,random_state=1)
            model=model.fit(x_train,y_train)
            y_pred=model.predict(x_test)
            cal_accuracy(y_pred)

        
    if classifier== "Random Forest":
        st.sidebar.subheader("Model Hyperparameters")
        max_depth=st.sidebar.number_input(label='max depth : ',min_value=1,max_value=15,step=1,key='depth')
        n_estimators=st.sidebar.number_input(label='n_estimators : ',min_value=1,max_value=15000,step=10,key='n_estimators')
        if st.sidebar.button("Classify", key='classify'):
            st.subheader("Random Forest Results")
            model=RandomForestClassifier(n_estimators= n_estimators, max_depth = max_depth,random_state=1)
            model=model.fit(x_train,y_train)
            y_pred=model.predict(x_test)
            cal_accuracy(y_pred)

    if classifier == 'Logistic Regression':
        st.sidebar.subheader("Model Hyperparameters")
        C = st.sidebar.number_input("C : ", 0.01, 10.0, step=0.01, key='C_LR')
        max_iter = st.sidebar.slider("Maximum number of iterations", 100, 500, key='max_iter')

        if st.sidebar.button("Classify", key='classify'):
            st.subheader("Logistic Regression Results")
            model = LogisticRegression(C=C, penalty='l2', max_iter=max_iter)
            model.fit(x_train, y_train)            
            y_pred = model.predict(x_test)
            cal_accuracy(y_pred)

    if st.sidebar.checkbox("Show raw data", False):
        st.subheader("Social_Network_Ads_Purchas Data Set (Classification)")
        st.write(df)
        st.markdown("This is a Social_Network_Ads_Purchas Data set")


if __name__ == '__main__':
    main()
   