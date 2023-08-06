import streamlit as st
import pandas as pd
import numpy as np
import pickle
import sklearn
# import the model
pipe = pickle.load(open('pipe.pkl','rb'))
df = pd.read_csv('Processed_data.csv')
st.title("Laptop Price Prediction\n-An Initiative By Situ Electronics Limited")

company = st.selectbox('Brand',df['Company'].unique())
type = st.selectbox('Type',df['TypeName'].unique())
ram = st.selectbox('RAM in GB',[2,4,6,8,12,16,32,64])
weight = st.number_input('Weight of the Laptop in kg',1.8)
touchscreen = st.selectbox('Touch Screen',['Yes','No'])
ips = st.selectbox('IPS Screen',['Yes','No'])
screen_size = st.number_input("Screen Size in inch", 15.6)
resolution = st.selectbox('Screen Resolution',['1920x1080','1366x768','1600x900','3840x2160','3200x1800',
                                           '2888x1800','2560x1600','2560x1440','2304x1440'])

cpu = st.selectbox('CPU Brand',df['Cpu_brand'].unique())
hdd = st.selectbox('HDD in GB',[0,128,256,512,1024,2048])
ssd = st.selectbox('SSD in GB',[0,8,128,256,512,1024])
gpu = st.selectbox('GPU Brand',df['Gpu_brand'].unique())
os = st.selectbox('Operating System',df['os'].unique())


if st.button("Predict the Price of Laptop"):

    if touchscreen == 'Yes':
        touchscreen = 1
    else:
        touchscreen =0

    if ips == 'yes':
        ips = 1
    else:
        ips = 0

    X_res = int(resolution.split('x')[0])
    Y_res = int(resolution.split('x')[1])
    ppi = ((X_res**2)+ (Y_res**2))**0.5/screen_size
    query = np.array([company,type,ram,weight,touchscreen,ips,ppi,cpu,hdd,ssd,gpu,os])
    query = query.reshape(1,12)
    st.title(f"Predicted price of laptop is Rs. {int(np.exp(pipe.predict(query)[0]))}")


