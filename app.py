import pandas as pd
import numpy as np
from joblib import load
import streamlit as st
import pickle



    
def main():
    st.title("Arrhythmia classification with extracted features from ECG signal for arrhythmia detection")
    app_mode = st.sidebar.selectbox('Select Page',['Home','Prediction'])
    if app_mode == "Home":
        st.header("Home")
    elif app_mode == 'Prediction':
        st.header("Model Deployment")
    return app_mode

def model_prediction(X):
    
    
    st.write('<p style="font-family:sans-serif; color:Brown; font-size: 22px;">Our Best Model is GradientBoost and XGBoost</p>',unsafe_allow_html = True)
    
    choice = st.selectbox(

    'Select the model you want?',

    ('RandomForest','GradientBoost','AdaBoost','LogisticRegression',"XGBoost"))



    #displaying the selected option

    st.write('You have selected:', choice)
    if(choice == 'RandomForest'):
        model = st.file_uploader("CHOOSE RandomForest CLASSIFIER JOBLIB FILE")
        threshold = pickle.load(open("./Main_Data/rf_threshold.pkl","rb"))
    elif(choice =='GradientBoost'):
        model = st.file_uploader("CHOOSE GradientBoosting CLASSIFIER JOBLIB FILE")
        threshold = pickle.load(open("./Main_Data/gb_threshold.pkl","rb"))
    elif(choice == "AdaBoost"):
        model = st.file_uploader("CHOOSE AdaBoost CLASSIFIER JOBLIB FILE")
        threshold = pickle.load(open("./Main_Data/ab_threshold.pkl","rb"))
    elif(choice =="LogisticRegression"):
        model = st.file_uploader("CHOOSE LogisticRegression CLASSIFIER JOBLIB FILE")
        threshold = pickle.load(open("./Main_Data/log_threshold.pkl","rb"))
    else:
        model = st.file_uploader("CHOOSE XGBoost CLASSIFIER JOBLIB FILE")
        threshold = pickle.load(open("./Main_Data/xg_threshold.pkl","rb"))
   
    st.write("Threshold value:",threshold)
    if model is not None and X is not None:
        model = load(model)
        #st.write(model.n_features_)
        
        y_pred = model.predict_proba(X.values)[:, 1]
        
        y_pred_updated = np.array([1 if ele >= threshold else 0 for ele in y_pred])
        
        #st.write("PREDICTED values:",y_pred_updated)
        values = st.number_input(
            "Pick a number",
            0,X.shape[0])
        st.write("CHOSEN TEST CASE ENTRY:",values)
        if st.button('Predict Arrhythmia'):
            if(y_pred_updated[values]  == 0):
                t = '<p style="font-family:sans-serif; color:Green; font-size: 42px;">It\'s great that You don\'t have heart attack...</p>'
                st.write(t,unsafe_allow_html = True)
            else:
                l = '<p style="font-family:sans-serif; color:Red; font-size: 42px;">It\'s sad that you had heart attack...</p>'
                st.write(l,unsafe_allow_html  = True)
        


def load_data():
    data_file = st.file_uploader("CHOOSE CSV FILE CONTAINING VECTORS TO BE PREDICTED UPON")
    if data_file is not None:
        df = pd.read_csv(data_file)
        if st.checkbox("Show Data"):
            st.dataframe(df)
        return df
    

def load_mean_std_data():
    df_mean_std = pd.read_csv("./Main_Data/df_mean_std.csv")
    if st.checkbox("Show Mean and standard deviation of each features..."):
        st.write(df_mean_std)
    
    return df_mean_std
    

def standardization(data, mean_std):
    if data is not None:
        columns = data.columns
    
        for ind, row in mean_std.iterrows():
            data[columns[ind]] = (data[columns[ind]] - row['means'])/row['std']
        
        return data
    

def load_best_features():
    best_feats = pickle.load(open("./Main_Data/best_features.pkl","rb"))
    if st.checkbox("Show Best features..."):
        st.write(best_feats)
    
    return best_feats

def featurnselection(data, best_feats):
    if data is not None:
        y = data.loc[:,'class']
        X = data.loc[:,best_feats]
       
        return X
       
    
if __name__ == '__main__':
    choice = main()
    


    
if choice == "Home":
    """Arrhythmia is an abnormality of the heart’s rhythm. It may beat too
slowly, too quickly or irregularity. These abnormalities range from a minor
inconvenience or discomfort to a potentially fatal problem. Narrowed heart ar
teries, a heart attack, abnormal heart valves, prior heart surgery, heart failure,
cardiomyopathy and other heart damage are risk factors for almost any kind of
arrhythmia. High Blood Pressure: This condition increases the risk of develop
ing coronary artery disease.
The arrhythmia can be classified into two major categories. The first category consists of arrhythmia formed by a single irregular
heartbeat called morphological arrhythmia. The other category consists of arrhythmia formed by a set of irregular
heartbeats, are called rhythmic arrhythmia.
The Cardiovascular diseases are the leading cause of death. The current iden
tification method of diseases is analyzing the Electrocardiogram [ECG], which is
medical monitoring technology recording cardiac activity. Unfortunately, look
ing for experts to analyze a large amount of ECG data consumes too many
medical resources. Therefore, a method based on Machine Learning or Deep
Learning to accurately classify ECG characteristics."""

   

elif choice == 'Prediction':
    st.markdown("## Dataset :")
   
    data = load_data()
    
    st.markdown("## Mean and Variance of Each Features :")
    
    data_load_state = st.text("Loading mean and std data...")
    data_mean_std = load_mean_std_data()
    data_load_state.text("Loading mean and std data...done!")
    
    st.markdown("## Standardized Data :")
    
    input_data = standardization(data, data_mean_std)
    if st.checkbox("Show Standardized Data..."):
        st.write(input_data)
    
    st.markdown("## Feature Selection :")
    
    data_load_state = st.text("Loading best features...")
    best_feats = load_best_features()
    data_load_state.text("Loading best features...done!")
    
    
    
    X =  featurnselection(input_data,best_feats)
    
    if st.checkbox("Show Selected Features Data..."):
        st.write(X)
    
    
    
    
    st.markdown("## Model Predictions :")
    
    model_prediction(X)
    
    
    
    
  
    
    
    
    
    
    
    
