
import streamlit as st
import pandas as pd
import numpy as np
import pickle


st.title('Diabetes Mellitus Classification')

model = pickle.load(open('RF_class_model.pkl','rb'))
scaler = pickle.load(open('scal.pkl', 'rb'))
encoder = pickle.load(open ('enc.pkl', "rb"))



# Add CSS styles
st.markdown(
    """
    <style>
    body {
        background-color: #1f1f1f;
        color: #ffffff;
    }
    </style>
    """,
    unsafe_allow_html=True
)






#if(not df):
   # st.info('The prediction will begin, once you upload your data set')
   # st.stop()
    
#st.title('Diabetes Mellitus Classification')
    
def predict():
    c1,c2 = st.columns(2)
    with(c1):
        Age = st.number_input('Please enter your age')
        Gender = st.selectbox('Gender',['Female','Male'])
        Polyuria = st.selectbox('Do you experience excessive urination?',['No','Yes'])
        Polydipsia = st.selectbox('Do you experience excessive thirst?',['No','Yes'])
        sudden_weight_loss = st.selectbox('Do you experience sudden weight loss?',['No','Yes'])
        weakness = st.selectbox('Do you experience general body weakness?',['No','Yes'])
        Polyphagia = st.selectbox('Do you experience excessive hunger?',['No','Yes'])
        Genital_thrush = st.selectbox('Do you suffer genital infections?',['No','Yes'])
        
    with(c2):
        visual_blurring = st.selectbox('Do you experience blurred vision?',['No','Yes'])
        Itching = st.selectbox('Do you experience body itching?',['No','Yes'])
        Irritability = st.selectbox('Do you experience nausea?',['No','Yes'])
        delayed_healing = st.selectbox('Do you suffer from delayed healing?',['No','Yes'])
        partial_paresis = st.selectbox('Do you experience weakened muscle movement?',['No','Yes'])
        muscle_stiffness = st.selectbox('Do you suffer from muscle stiffness?',['No','Yes'])
        Alopecia = st.selectbox('Do you experience sudden hair loss?',['No','Yes'])
        Obesity = st.selectbox('Do you suffer from obesity?',['No','Yes'])
        
        feat = np.array([Age ,Gender, Polyuria, Polydipsia, sudden_weight_loss,
        weakness, Polyphagia, Genital_thrush, visual_blurring,
        Itching, Irritability, delayed_healing, partial_paresis,
        muscle_stiffness, Alopecia, Obesity]).reshape(1,-1)
        cols = ['Age', 'Gender', 'Polyuria', 'Polydipsia', 'sudden weight loss',
       'weakness', 'Polyphagia', 'Genital thrush', 'visual blurring',
       'Itching', 'Irritability', 'delayed healing', 'partial paresis',
       'muscle stiffness', 'Alopecia', 'Obesity']
        feat1 = pd.DataFrame(feat, columns=cols)
        
        return feat1
        
        


frame = predict()

#if st.button('Show Input Data'):
   # st.write(frame.head())
    
def prepare(df):
    #from sklearn.preprocessing import LabelEncoder

    #label_encoder = LabelEncoder()  # Create an instance of the LabelEncoder

    # Fit and transform the selected columns in the DataFrame 'df' using the label encoder
    #df['Age'] = label_encoder.fit_transform(df['Age'])
    #df['Age'] = df['Age'].map({'Yes': 1, 'No': 0})
    
    enc_data =pd.DataFrame(encoder.transform(df[['Gender', 'Polyuria', 'Polydipsia', 'sudden weight loss',
           'weakness', 'Polyphagia', 'Genital thrush', 'visual blurring',
           'Itching', 'Irritability', 'delayed healing', 'partial paresis',
           'muscle stiffness', 'Alopecia', 'Obesity']]).toarray())
    #enc_data.columns = encoder.get_feature_names_out()
    enc_data.columns = encoder.get_feature_names(['Gender', 'Polyuria', 'Polydipsia', 'sudden weight loss',
           'weakness', 'Polyphagia', 'Genital thrush', 'visual blurring',
           'Itching', 'Irritability', 'delayed healing', 'partial paresis',
           'muscle stiffness', 'Alopecia', 'Obesity'])
    df = df.join(enc_data)

    df.drop(['Gender', 'Polyuria', 'Polydipsia', 'sudden weight loss',
           'weakness', 'Polyphagia', 'Genital thrush', 'visual blurring',
           'Itching', 'Irritability', 'delayed healing', 'partial paresis',
           'muscle stiffness', 'Alopecia', 'Obesity'],axis=1,inplace=True)
    
    cols = df.columns
    #scaler =  MinMaxScaler()
    df = scaler.transform(df)
    df = pd.DataFrame(df,columns=cols)

    
    return df

frame2= prepare(frame)

#if st.button('Show process Data'):
 #   st.write(frame2.head())
    
    
if st.button('predict'):
    frame2= prepare(frame)
    pred = model.predict(frame2)
    if pred[0] == 'Negative':
        st.write('This individual does not have diabetes ')
    else:
        st.write('This individual has diabetes ')
        
        
    #st.write(pred[0])
    
