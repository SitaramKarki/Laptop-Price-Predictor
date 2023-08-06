#!/usr/bin/env python
# coding: utf-8

# # Laptop Price Predictor | Machine Learning Project
# 

# ### Overview:
# #### The Laptop Price Predictor is an advanced Machine Learning project designed to predict the prices of laptops based on various features and specifications. This project aims to assist buyers, sellers, and manufacturers in estimating the appropriate price range for different laptops, making the purchasing process more informed and efficient. By leveraging the power of Machine Learning algorithms, the Laptop Price Predictor offers accurate and reliable price predictions, considering the complex relationships between laptops' attributes and their market values.
# 
# ### Project Objectives:
# 
# #### Predictive Accuracy: Develop a robust Machine Learning model that can accurately predict laptop prices based on various input features such as processor type, RAM, storage capacity, graphics card, display size, brand, and other relevant parameters.
# 
# #### Feature Selection: Identify the most influential features that significantly impact laptop prices and utilize them to create an efficient prediction model.
# 
# #### Data Collection and Preprocessing: Gather a large dataset of laptops from various sources, including e-commerce websites, manufacturer catalogs, and user reviews. Clean and preprocess the data to ensure its quality and relevance for training the predictive model.
# 
# #### Model Selection and Optimization: Explore different Machine Learning algorithms such as Regression, Random Forest, Gradient Boosting, or Neural Networks, and select the most suitable model for price prediction. Optimize the model's hyperparameters to enhance its performance.
# 
# #### User-Friendly Interface: Develop an intuitive and user-friendly interface where users can input laptop specifications and receive accurate price predictions in real-time.
# 
# ### Project Methodology:
# #### The Laptop Price Predictor project follows a structured approach, comprising several key steps:
# 
# #### Data Collection: Gather laptop data from reputable sources, including online retailers, manufacturers, and public datasets. The data should include specifications like CPU, RAM, storage, GPU, display size, brand, user ratings, and prices.
# 
# #### Data Preprocessing: Clean the dataset to handle missing values, remove duplicates, and address outliers. Perform feature engineering to extract valuable information from the raw data and prepare it for modeling.
# 
# #### Feature Selection: Analyze the dataset to identify the most relevant features that significantly affect laptop prices. Features with low importance will be excluded to simplify the model.
# 
# #### Model Training: Split the dataset into training and testing sets. Utilize various Machine Learning algorithms to train the prediction model on the training data. Compare the performance of different models using evaluation metrics like Mean Absolute Error (MAE), Mean Squared Error (MSE), and R-squared.
# 
# #### Model Evaluation: Assess the trained models' performance on the testing data to ensure their predictive accuracy and generalizability. Fine-tune the model parameters to improve its effectiveness.
# 
# #### Interface Development: Build a user-friendly interface that allows users to input laptop specifications easily and receive predicted prices based on the trained model.
# 
# ### Expected Outcomes:
# 
# #### The Laptop Price Predictor project aims to achieve the following outcomes:
# 
# #### Accurate Price Predictions: The developed model should provide reliable and precise laptop price predictions, enabling users to make well-informed decisions during laptop purchases or sales.
# 
# #### Enhanced User Experience: The user interface should be intuitive, responsive, and accessible to users of all technical backgrounds.
# 
# #### Real-world Applicability: The project's outcome should have practical applications in e-commerce, retail, and the laptop manufacturing industry, assisting in price setting and market analysis.
# 
# #### In conclusion, the Laptop Price Predictor is an ambitious Machine Learning project with the goal of delivering accurate and efficient laptop price predictions. By leveraging the power of data and advanced algorithms, this project will provide valuable insights into laptop pricing trends, benefiting both consumers and industry stakeholders alike.

# In[424]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# In[425]:


df = pd.read_csv('laptop_data.csv')


# In[426]:


df.head(5)


# In[427]:


df.shape


# #### The problem with the dataset is that there are very low datas

# #### Some contains more than one pice of information so need to do feature engineering

# In[428]:


df.info()


# In[429]:


df.duplicated().sum()


# In[430]:


df.isnull().sum()


# In[431]:


df.drop(columns = ['Unnamed: 0'],inplace = True)


# In[432]:


df.head()


# #### Now we will remove GB from Ram and Kg from the weight columns and change their type

# In[433]:


df['Ram'] = df['Ram'].str.replace('GB','')


# In[434]:


df.head()


# In[435]:


df['Weight'] = df['Weight'].str.replace('kg','')


# In[436]:


df.head()


# In[437]:


df['Ram'] = df['Ram'].astype('int32')
df['Weight'] = df['Weight'].astype('float32')


# In[438]:


df.info()


# #### Now let us do eda analysis to know more about the data

# In[439]:


import seaborn as sns


# In[440]:


sns.distplot(df['Price'])


# #### This shows that there are many laptops whose price are low and very less lsptop whose price is higher

# In[441]:


df.describe()


# #### The datas are skewed so there may be problem for some algoriths to coverge

# #### Now let see which company has how many laptops in our dataset

# In[442]:


df['Company'].value_counts().plot(kind = 'bar')


# #### Let us find our the average price for each of the company laptop

# In[443]:


sns.barplot(x = df['Company'], y = df['Price'])


# #### We are not able to see the name of the company so let's make company name vertical

# In[444]:


sns.barplot(x = df['Company'], y = df['Price'])
plt.xticks(rotation = 'vertical')
plt.show()


# #### This shows that price of laptop depends on the company 

# In[445]:


df['TypeName'].value_counts().plot(kind = 'bar')


# #### Let us find the average price range for each type of laptop

# In[446]:


sns.barplot(x = df['TypeName'], y = df['Price'])
plt.xticks(rotation = 'vertical')
plt.show()


# #### This shows that price depends on the type of laptop

# In[447]:


sns.distplot(df['Inches'])


# In[448]:


sns.scatterplot(x = df['Inches'], y = df['Price'])


# #### This shows that price depends abit on the inches too

# #### Let us move to the screen resolution column. This will be difficult for us as there are alot of information there and every columns have differently presented the information

# In[449]:


df['ScreenResolution'].value_counts()


# #### The only common in all is resolution so we will extract them. The another available we have is,is there touch screen or not and is it ips screen or not

# In[450]:


df['TouchScreen'] = df['ScreenResolution'].apply(lambda x:1 if 'Touchscreen' in x else 0)


# In[451]:


df.head()


# In[452]:


df.sample(5)


# In[453]:


df['TouchScreen'].value_counts().plot(kind = 'bar')


# In[454]:


sns.barplot(x = df['TouchScreen'], y = df['Price'])


# #### This shows that the touch screen laptops are expensive incomparison to the non touch screen

# In[455]:


df['Ips'] = df['ScreenResolution'].apply(lambda x : 1 if 'IPS' in x else 0)


# In[456]:


df.sample(5)


# In[457]:


df['Ips'].value_counts().plot(kind = 'bar')


# In[458]:


sns.barplot(x = df['Ips'], y = df['Price'])


# #### This shows that ips panel laptop cost more than non ips panel

# In[459]:


df['ScreenResolution'].str.split('x')


# In[460]:


df['ScreenResolution'].str.split('x', n = 1 , expand = True)


# #### In the given code snippet, df['ScreenResolution'].str.split('x', n=1, expand=True), we are working with a pandas DataFrame df, specifically accessing the column labeled 'ScreenResolution'. This operation involves splitting the values in the 'ScreenResolution' column into multiple columns using the delimiter 'x'.
# 
# #### Let's break down the individual components of the code:
# 
# #### df['ScreenResolution']: This part of the code accesses the 'ScreenResolution' column from the DataFrame df. It assumes that the 'ScreenResolution' column contains strings representing the resolution of the laptop screens in the format "width x height" (e.g., "1920x1080").
# 
# #### .str: This is a property of pandas Series that allows us to apply string methods to the elements in the Series. In this case, we are applying the split() method to split the strings.
# 
# #### .split('x', n=1, expand=True): This is the split() method being applied to each element in the 'ScreenResolution' column. It takes three parameters:
# 
# #### 'x': This is the delimiter that will be used to split the strings. In this case, we are using 'x' as the delimiter because the resolution strings are typically in the format "width x height".
# 
# #### n=1: This parameter specifies the maximum number of splits to perform. Here, we set it to 1, which means the split will happen at the first occurrence of 'x'. If there are additional 'x' characters in the string, they will not be split further. This helps in cases where there might be extra 'x' characters, such as "3840x2160x60Hz".
# 
# #### expand=True: This parameter is used to expand the split results into separate columns of the DataFrame. When set to True, the split values will be returned as a new DataFrame with multiple columns, one for each part of the split string. If set to False, the split values would be returned as a pandas Series.
# 
# #### After executing the code, the 'ScreenResolution' column will be transformed into multiple new columns, typically named '0' and '1', containing the width and height values of the screen resolutions, respectively. For example, if the original 'ScreenResolution' column contained the string "1920x1080", the resulting DataFrame would have two new columns with the values 1920 and 1080, respectively.

# In[461]:


new = df['ScreenResolution'].str.split('x', n = 1 , expand = True)


# In[462]:


df['X_res'] = new[0]
df['Y_res'] = new[1]


# In[463]:


df.sample(5)


# #### The Y_res is as we expect but the X_res has more information than we want

# In[464]:


df['X_res'].str.replace(',','').str.findall(r'(\d+\.?\d+)')


# #### The code df['X_res'].str.replace(',', '').str.extract(r'(\d+)') is a series of pandas string operations applied to the 'X_res' column of the DataFrame df. It aims to extract numeric values from the strings in the 'X_res' column while handling any possible commas in the numbers.
# 
# #### Let's break down the code step by step:
# 
# #### df['X_res']: This part accesses the 'X_res' column from the DataFrame df. It assumes that the 'X_res' column contains strings with some numeric values, possibly with commas as thousands separators (e.g., "1,920" or "3,600").
# 
# #### .str.replace(',', ''): This is the first string operation applied to each element in the 'X_res' column. It uses the replace() method to remove commas (,) from the strings. For example, "1,920" would be transformed into "1920", and "3,600" into "3600".
# 
# #### .str.extract(r'(\d+)'): After removing commas, this is the second string operation applied to each element in the 'X_res' column. It uses the extract() method with a regular expression r'(\d+)' to extract numeric values from the strings.
# 
# #### \d+ in the regular expression matches one or more digits. So, the extract() method will capture all consecutive digits in the string as a group.
# 
# #### Since the regular expression contains parentheses (), the extract() method will return the captured group as a new pandas DataFrame with a single column.
# 
# #### The final output of this code will be a pandas DataFrame with a single column containing the extracted numeric values from the 'X_res' column, with any commas removed. For example, if the 'X_res' column originally contained the strings "1,920" and "3,600", the resulting DataFrame would have a single column with the values 1920 and 3600, respectively.
# 
# #### In summary, the code effectively removes commas from the numeric values in the 'X_res' column and then extracts those numeric values as a separate DataFrame, making it easier to work with numeric data in further analyses or computations.
# 
# 
# 
# 
# 
# 
# 

# In[465]:


df['X_res'] = df['X_res'].str.replace(',','').str.extract(r'(\d+\.?\d+)')


# In[466]:


df.sample(5)


# In[467]:


df.info()


# #### As we can see that the data type of X_res and Y_res are object and i need to convert it to the integer

# In[468]:


df['X_res'] = df['X_res'].astype(int)
df['Y_res'] = df['Y_res'].astype(int)


# In[469]:


df.info()


# In[470]:


df.corr()['Price']


# #### This shows that X_res and Y_res has higher correlation with price

# #### Now we will make a new column called Ppi pixel per inches. As we have heard that higher pixel means  higher price for the laptop.

# #### We will not use X_res and Y_res as they have multi coliranity. We can do it with formula
# #### (df['X_res']**2 + df['Y_res']**2)**0.5/df['Inches']

# In[471]:


(df['X_res']**2 + df['Y_res']**2)**0.5/df['Inches']


# In[472]:


df['ppi'] = ((df['X_res']**2 + df['Y_res']**2)**0.5/df['Inches']).astype('float')


# In[473]:


df.corr()['Price']


# #### Now we can see that ppi has strong correlation with price. We won't use column X_res , Y_res and Inches for the prediction as we have ppi for all of   them

# #### Now we will drop screen resolution column

# In[474]:


df.drop(columns = ['ScreenResolution'], inplace = True)


# In[475]:


df.head()


# In[476]:


df.drop(columns = ['X_res', 'Y_res','Inches'], inplace = True)


# In[477]:


df.head()


# #### Now let us go with the cpu column

# In[478]:


df['Cpu'].value_counts()


# #### Let's use feature engineering to go deep into it. We will make 5 columns from the Cpu column

# #### In the beginning we will extract first three words from the Cpu column

# In[479]:


df['Cpu'].apply(lambda x :x.split())


# In[482]:


df['Cpu_Name']= df['Cpu'].apply(lambda x :" ".join(x.split()[0:3]))


# In[484]:


df['Cpu_Name']


# #### Let us make function to find out other processor other than intel 

# In[485]:


def fetch_processor(text):
    if text == 'Intel Core i7' or text == 'Intel Core i5' or text == 'Intel Core i3':
        return text
    else:
        if text.split()[0] == 'Intel':
            return 'Other Intel Processor'
        else:
            return 'AMD Processor'


# In[486]:


df['Cpu_brand'] = df['Cpu_Name'].apply(fetch_processor)


# In[487]:


df['Cpu_brand']


# In[488]:


df.head()


# In[489]:


df['Cpu_brand'].value_counts().plot(kind= 'bar')


# In[490]:


sns.barplot(x = df['Cpu_brand'], y = df['Price'])
plt.xticks(rotation = 'vertical')
plt.show()


# #### This shows that price depends on the processor

# In[494]:


df.drop(columns = ['Cpu_ame', 'Cpu'bb], inplace = True)


# In[495]:


df.drop(columns = ['Cpu_ame'], inplace = True)


# In[496]:


df.head()


# In[497]:


df['Ram'].value_counts().plot(kind = 'bar')


# In[498]:


sns.barplot(x = df['Ram'], y = df['Price'])


# #### Above graph shows that the linear relationship between Ran and Price

# #### Now let's work with the memory column. It contains different information. We will extract the important one

# In[499]:


df['Memory'].value_counts()


# #### Now we will make 4 columns from this. HDD, SDD, flash storage and hybrid

# In[500]:


df['Memory']= df['Memory'].astype(str).replace('\.0','',regex = True)
df['Memory']= df['Memory'].str.replace('GB','')
df['Memory']= df['Memory'].str.replace('TB','000')
new = df['Memory'].str.split("+",n = 1,expand= True)

df['first'] = new[0]
df['first'] = df['first'].str.strip()
df['second']= new[1]
df['Layer1HDD'] = df['first'].apply(lambda x : 1 if 'HDD' in x else 0)
df['Layer1SSD'] = df['first'].apply(lambda x : 1 if 'SSD' in x else 0)
df['Layer1Flash_Storage'] = df['first'].apply(lambda x : 1 if 'Flash Storage' in x else 0)
df['Layer1Hybrid'] = df['first'].apply(lambda x : 1 if 'Hybrid' in x else 0)

df['first'] = df['first'].str.replace(r'\D' , '')
df['second'].fillna("0", inplace = True)    
df['Layer2HDD'] = df['second'].apply(lambda x :1 if 'HDD'in x else 0)
df['Layer2SSD'] = df['second'].apply(lambda x :1 if 'SSD'in x else 0)       
df['Layer2Flash_Storage'] = df['second'].apply(lambda x : 1 if 'Flash Storage' in x else 0)
df['Layer2Hybrid']=df['first'].apply(lambda x : 1 if 'Hybrid' in x else 0)
                                                                
df['second'] = df['second'].str.replace(r'\D' , '')

df['first'] = df['first'].astype(int)
df['second'] = df['second'].astype(int)

df['HDD'] = (df['first']*df['Layer1HDD'] + df['second']*df['Layer2HDD'])
df['SSD'] = (df['first']*df['Layer1SSD'] + df['second']*df['Layer2SSD'])
df['Hybrid'] = (df['first']*df['Layer1Hybrid'] + df['second']*df['Layer2Hybrid'])
df['Flash Storage'] = (df['first']*df['Layer1Flash_Storage'] + df['second']*df['Layer2Flash_Storage'])

df.drop(columns = ['first', 'second','Layer1HDD','Layer2HDD','Layer1SSD','Layer2SSD', 'Layer1Hybrid','Layer2Flash_Storage'
                ,'Layer2Hybrid','Layer1Flash_Storage'],inplace = True)
                      


# In[502]:


df.head()


# #### Now let us drop the memory column

# In[503]:


df.drop(columns = ['Memory'], inplace = True)


# In[504]:


df.sample(5)


# In[505]:


df.corr()['Price']


# #### See the correlation between the price and ssd> Strong correlation. But with hdd it has weak correlation

# #### Since the hybrid and flash storage has no much impact on price so i want to drop these columns 

# In[506]:


df.drop(columns = ['Hybrid', 'Flash Storage'], inplace = True)


# In[507]:


df.sample()


# In[508]:


df['Gpu'].value_counts()


# #### Now let us focus on Gpu column. Here are so  many categories. Since much informaiton is not given so i will extract brand name from the Gpu column.

# In[509]:


df['Gpu'].apply(lambda x: x.split()[0])


# In[510]:


df['Gpu_brand'] = df['Gpu'].apply(lambda x: x.split()[0])


# In[511]:


df.head()


# In[512]:


df['Gpu_brand'].value_counts()


# #### There is one laptop with ARM's gpu. Let us drop that row

# In[513]:


df[df['Gpu_brand'] == 'ARM']


# In[514]:


df = df[df['Gpu_brand'] != 'ARM']


# In[515]:


df['Gpu_brand'].value_counts()


# In[516]:


sns.barplot(x = df['Gpu_brand'], y = df['Price'])
plt.xticks(rotation = 'vertical')
plt.show()


# In[517]:


sns.barplot(x = df['Gpu_brand'], y = df['Price'], estimator = np.median)
plt.xticks(rotation = 'vertical')
plt.show()


# #### This shows that price in influenced ny gpu brand
# 

# #### Let us drop the Gpu column

# In[518]:


df.drop(columns = ['Gpu'], inplace = True)


# In[519]:


df.head()


# #### Let's work with OpSys and Weight

# In[520]:


df['OpSys'].value_counts()


# In[521]:


sns.barplot(x = df['OpSys'],y = df['Price'])
plt.xticks(rotation = 'vertical')
plt.show()


# #### Let us merge the operating system as window , os and others 

# In[522]:


def cat_os(inp):
    if inp =='Windows 10'or inp == 'Windows 10 S' or inp == 'Window 7' :
        return 'Windows'
    elif inp == 'Mac OS X' or inp == 'macOS':
        return 'Mac'
    else:
        return "Others/NO/No OS/Linux"


# In[523]:


df['os'] = df['OpSys'].apply(cat_os)


# In[524]:


df.head()


# #### Now let us drop the OpSys column as it is not needed for us

# In[525]:


df.drop(columns = ['OpSys'], inplace = True)


# In[526]:


df.head()


# In[527]:


sns.barplot(x = df['os'], y = df['Price'])
plt.xticks(rotation = 'vertical')
plt.show()


# In[528]:


sns.distplot(df['Weight'])


# In[529]:


sns.scatterplot(x = df['Weight'],y = df['Price'])


# In[530]:


backup = df.copy()


# In[531]:


df.to_csv('Processed_data.csv',index = False)


# In[532]:


df.corr()['Weight']


# In[533]:


df.corr()


# In[534]:


#### Let's plot the heatmap to see in details


# In[535]:


sns.heatmap(df.corr())


# #### Since there is no any columns which have highest correlation with one another except price so we will use all of them

# #### Since our target column price is skewed so may be it will create probelem to our machine learning algorithms

# In[536]:


sns.distplot(df['Price'])


# #### We can apply log transformation to it to make the price normally distributed.

# In[537]:


sns.distplot(np.log(df['Price']))


# #### When you do log transformationm, during predction you have to use exponential to cancel it during prediction

# In[538]:


X = df.drop(columns = ['Price'])
y = np.log(df['Price'])


# In[539]:


X


# In[540]:


y


# In[541]:


from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size = 0.2,random_state = 2)


# In[542]:


X_train


# #### Now we will use one hot encoding for the categorical columns

# In[543]:


from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import r2_score,mean_absolute_error


# In[544]:


from sklearn.linear_model import LinearRegression,Ridge,Lasso
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor,GradientBoostingRegressor,AdaBoostRegressor,ExtraTreesRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.svm import SVR
from xgboost import XGBRegressor


# In[545]:


step1 = ColumnTransformer(transformers = [
('col_tnf',OneHotEncoder(sparse = False,drop = 'first'),[0,1,7,10,11])],remainder = 'passthrough')
    


# #### The code you provided is creating a ColumnTransformer object named step1. A ColumnTransformer is a powerful preprocessing tool in scikit-learn that allows you to apply different transformations to different subsets of columns in your dataset. This is particularly useful when you have a mix of numerical and categorical features and you want to preprocess them differently before feeding them into a machine learning model.
# 
# #### Let's break down the code step by step:
# 
# #### ColumnTransformer(transformers=[...], remainder='passthrough'):
# 
# #### ColumnTransformer: This is a class from scikit-learn that allows you to specify different transformations for different subsets of columns in your dataset.
# 
# #### transformers: This parameter takes a list of tuples, where each tuple defines a transformation to be applied to a subset of columns. In your code, you have one tuple in the list.
# 
# #### ('col_tnf', OneHotEncoder(sparse=False, drop='first'), [0, 1, 7, 10, 11]): This is the tuple defining the transformation for a subset of columns.
# 
# #### 'col_tnf': This is a name given to this transformation. It can be any string and is used as an identifier for this transformation.
# 
# #### OneHotEncoder(sparse=False, drop='first'): This is the transformation to be applied. In this case, it is the OneHotEncoder from scikit-learn, which is used to convert categorical variables into a one-hot encoded representation. The sparse=False parameter means that the resulting one-hot encoded matrix will be a dense array (not a sparse matrix), and drop='first' means that the first category of each categorical feature will be dropped to avoid multicollinearity.
# 
# #### [0, 1, 7, 10, 11]: This is the list of column indices (integer positions) that will be transformed using the specified OneHotEncoder. In this example, columns at positions 0, 1, 7, 10, and 11 will be one-hot encoded, assuming the input data is a DataFrame or a 2D array.
# 
# #### remainder='passthrough': This parameter specifies how to handle the remaining columns in the dataset that are not explicitly transformed. 'passthrough' means that these columns will be kept as they are, without any transformation. In other words, they will be included in the final output without any preprocessing.
# 
# #### In summary, the step1 ColumnTransformer applies the OneHotEncoder transformation to the columns at positions 0, 1, 7, 10, and 11 of the input data (assuming the input data is a DataFrame or a 2D array). The rest of the columns will be kept unchanged in the final output. This can be a useful preprocessing step in machine learning pipelines when dealing with mixed data types and wanting to apply different transformations to different subsets of features.
# 

# In[546]:


step2 = LinearRegression()


# In[547]:


pipe = Pipeline([('step1' ,step1),
                 ('step2', step2)] )


# In[548]:


pipe.fit(X_train,y_train)


# In[549]:


y_pred =pipe.predict(X_test)


# In[550]:


print("r2_score",r2_score(y_test,y_pred))
print("mean_absolute_error",mean_absolute_error(y_test,y_pred))


# # Ridge Regression

# In[551]:


step1 = ColumnTransformer(transformers = [('col_tnf',OneHotEncoder(sparse = False,drop = 'first'),[0,1,7,10,11])],remainder = 'passthrough')
step2 = Ridge(alpha = 10)  
pipe = Pipeline([('step1' ,step1),
                 ('step2', step2)] )
pipe.fit(X_train,y_train)
y_pred =pipe.predict(X_test)
print("r2_score",r2_score(y_test,y_pred))
print("mean_absolute_error",mean_absolute_error(y_test,y_pred))


# # Linear Regression

# In[552]:


step1 = ColumnTransformer(transformers = [('col_tnf',OneHotEncoder(sparse = False,drop = 'first'),[0,1,7,10,11])],remainder = 'passthrough')
step2 = LinearRegression()  
pipe = Pipeline([('step1' ,step1),
                 ('step2', step2)] )
pipe.fit(X_train,y_train)
y_pred =pipe.predict(X_test)
print("r2_score:",r2_score(y_test,y_pred))
print("mean_absolute_error:",mean_absolute_error(y_test,y_pred))


# # Lasso Regresssion

# In[553]:


step1 = ColumnTransformer(transformers = [('col_tnf',OneHotEncoder(sparse = False,drop = 'first'),[0,1,7,10,11])],remainder = 'passthrough')
step2 = Lasso(alpha = 0.001)  
pipe = Pipeline([('step1' ,step1),
                 ('step2', step2)] )
pipe.fit(X_train,y_train)
y_pred =pipe.predict(X_test)
print("r2_score:",r2_score(y_test,y_pred))
print("mean_absolute_error:",mean_absolute_error(y_test,y_pred))


# # KNN

# In[554]:


step1 = ColumnTransformer(transformers = [('col_tnf',OneHotEncoder(sparse = False,drop = 'first'),[0,1,7,10,11])],remainder = 'passthrough')
step2 = KNeighborsRegressor(n_neighbors = 3) 
pipe = Pipeline([('step1' ,step1),
                 ('step2', step2)] )
pipe.fit(X_train,y_train)
y_pred =pipe.predict(X_test)
print("r2_score:",r2_score(y_test,y_pred))
print("mean_absolute_error:",mean_absolute_error(y_test,y_pred))


# # Decision Tree

# In[555]:


step1 = ColumnTransformer(transformers = [('col_tnf',OneHotEncoder(sparse = False,drop = 'first'),[0,1,7,10,11])],remainder = 'passthrough')
step2 = DecisionTreeRegressor(max_depth = 8)  
pipe = Pipeline([('step1' ,step1),
                 ('step2', step2)] )
pipe.fit(X_train,y_train)
y_pred =pipe.predict(X_test)
print("r2_score:",r2_score(y_test,y_pred))
print("mean_absolute_error:",mean_absolute_error(y_test,y_pred))


# # SVM

# In[556]:


step1 = ColumnTransformer(transformers = [('col_tnf',OneHotEncoder(sparse = False,drop = 'first'),[0,1,7,10,11])],remainder = 'passthrough')
step2 = SVR(kernel = 'rbf',C = 10000, epsilon = 0.1) 
pipe = Pipeline([('step1' ,step1),
                 ('step2', step2)] )
pipe.fit(X_train,y_train)
y_pred =pipe.predict(X_test)
print("r2_score:",r2_score(y_test,y_pred))
print("mean_absolute_error:",mean_absolute_error(y_test,y_pred))


# # Random Forest 

# In[572]:


step1 = ColumnTransformer(transformers = [('col_tnf',OneHotEncoder(sparse = False,drop = 'first'),[0,1,7,10,11])],remainder = 'passthrough')
step2 = RandomForestRegressor(n_estimators =100,
                             random_state = 3,
                             max_samples = 0.5,
                             max_features = 0.75,
                             max_depth = 15)  
pipe = Pipeline([('step1' ,step1),
                 ('step2', step2)] )
pipe.fit(X_train,y_train)
y_pred =pipe.predict(X_test)
print("r2_score:",r2_score(y_test,y_pred))
print("mean_absolute_error:",mean_absolute_error(y_test,y_pred))


# # ExtraTrees

# In[558]:


step1 = ColumnTransformer(transformers = [('col_tnf',OneHotEncoder(sparse = False,drop = 'first'),[0,1,7,10,11])],remainder = 'passthrough')
step2 = ExtraTreesRegressor(n_estimators =100,
                             random_state = 3,
                             max_features = 0.75,
                             max_depth = 15)  
pipe = Pipeline([('step1' ,step1),
                 ('step2', step2)] )
pipe.fit(X_train,y_train)
y_pred =pipe.predict(X_test)
print("r2_score:",r2_score(y_test,y_pred))
print("mean_absolute_error:",mean_absolute_error(y_test,y_pred))


# # AdaBoost

# In[559]:


step1 = ColumnTransformer(transformers = [('col_tnf',OneHotEncoder(sparse = False,drop = 'first'),[0,1,7,10,11])],remainder = 'passthrough')
step2 = AdaBoostRegressor(n_estimators = 15, learning_rate = 1.0 )  
pipe = Pipeline([('step1' ,step1),
                 ('step2', step2)] )
pipe.fit(X_train,y_train)
y_pred =pipe.predict(X_test)
print("r2_score:",r2_score(y_test,y_pred))
print("mean_absolute_error:",mean_absolute_error(y_test,y_pred))


# # Gradient Boost

# In[560]:


step1 = ColumnTransformer(transformers = [('col_tnf',OneHotEncoder(sparse = False,drop = 'first'),[0,1,7,10,11])],remainder = 'passthrough')
step2 = GradientBoostingRegressor(n_estimators = 500 , max_features = 0.5)  
pipe = Pipeline([('step1' ,step1),
                 ('step2', step2)] )
pipe.fit(X_train,y_train)
y_pred =pipe.predict(X_test)
print("r2_score:",r2_score(y_test,y_pred))
print("mean_absolute_error:",mean_absolute_error(y_test,y_pred))


# # XgBoost

# In[561]:


step1 = ColumnTransformer(transformers = [('col_tnf',OneHotEncoder(sparse = False,drop = 'first'),[0,1,7,10,11])],remainder = 'passthrough')
step2 = XGBRegressor(n_estimators = 45,max_depth = 5,learning_rate = 0.5)  
pipe = Pipeline([('step1' ,step1),
                 ('step2', step2)] )
pipe.fit(X_train,y_train)
y_pred =pipe.predict(X_test)
print("r2_score:",r2_score(y_test,y_pred))
print("mean_absolute_error:",mean_absolute_error(y_test,y_pred))


# # Voting Regressor

# In[562]:


from sklearn.ensemble import VotingRegressor,StackingRegressor


# In[563]:


step1 = ColumnTransformer(transformers = [('col_tnf',OneHotEncoder(sparse = False,
        drop = 'first'),[0,1,7,10,11])],remainder = 'passthrough')

rf = RandomForestRegressor(n_estimators =350,
                             random_state = 3,
                             max_samples = 0.5,
                             max_features = 0.75,
                             max_depth = 15) 
gbdt = GradientBoostingRegressor(n_estimators = 500 , max_features = 0.5)
xgb = XGBRegressor(n_estimators = 25 ,max_depth = 5,learning_rate = 0.3) 
et = ExtraTreesRegressor(n_estimators =100,
                             random_state = 3,
                             max_features = 0.75,
                             max_depth = 16)  
step2 = VotingRegressor([('rf',rf),('gbdt',gbdt),('xgb',xgb),('et',et)],weights = [5,1,1,1]) 
pipe = Pipeline([('step1' ,step1),
                 ('step2', step2)] )
pipe.fit(X_train,y_train)
y_pred =pipe.predict(X_test)
print("r2_score:",r2_score(y_test,y_pred))
print("mean_absolute_error:",mean_absolute_error(y_test,y_pred))


# # Stacking

# In[570]:


step1 = ColumnTransformer(transformers = [('col_tnf',OneHotEncoder(sparse = False,
        drop = 'first'),[0,1,7,10,11])],remainder = 'passthrough')

estimators = [ ('rf' ,RandomForestRegressor(n_estimators =350,
                             random_state = 3,
                             max_samples = 0.5,
                             max_features = 0.75,
                             max_depth = 15)),
('gbdt' , GradientBoostingRegressor(n_estimators = 500 , max_features = 0.5)),
('xgb' , XGBRegressor(n_estimators = 25 ,max_depth = 5,learning_rate = 0.3))]

step2 = StackingRegressor(estimators = estimators,final_estimator = Ridge(alpha = 100))
pipe = Pipeline([('step1' ,step1),
                 ('step2', step2)] )
pipe.fit(X_train,y_train)
y_pred =pipe.predict(X_test)
print("r2_score:",r2_score(y_test,y_pred))
print("mean_absolute_error:",mean_absolute_error(y_test,y_pred))


# # Exporting the model

# In[565]:


import pickle
pickle.dump(df,open('df.pkl','wb'))


# In[566]:


df.head()


# In[574]:


pickle.dump(pipe,open('pipe.pkl','wb'))


# In[569]:


df['Cpu_brand']


# In[ ]:




