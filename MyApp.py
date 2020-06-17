import streamlit as st
import pandas as pd
import numpy as np
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import plot_confusion_matrix, plot_roc_curve, plot_precision_recall_curve
from sklearn.metrics import precision_score, recall_score




def main():   
    st.title("Social_Network_Ads_Purchas")
    st.sidebar.title("Binary Classification Web App")
    st.markdown("Are clinet purchased or not? ")
    st.sidebar.markdown("Are clinet purchased or not? ")

    @st.cache(persist=True)
    def load_data():
        filePath="Social_Network_Ads.csv" 
        data=pd.read_csv(filePath)
        return data

    @st.cache(persist=True)
    def split(df):
        y = df.Purchased
        x = df.drop(columns=['Purchased'])
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


    df = load_data()
    class_names = ['purchased', 'not purchused']

    x_train, x_test, y_train, y_test = split(df)

    


if __name__ == '__main__':
    main()