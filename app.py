# Load Core Pkgs
import streamlit as st
import webbrowser

# EDA Pkgs
import pandas as pd
import numpy as np

# Data Viz Pkgs
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns



# ML Pkgs
import os
import joblib
from PIL import Image
# Storing Data
import sqlite3

conn = sqlite3.connect('data.db')
c = conn.cursor()

def create_table():
	c.execute('CREATE TABLE IF NOT EXISTS notestable(author TEXT,title TEXT,message TEXT)')


def add_data(author,title,message):
	c.execute('INSERT INTO notestable(author,title,message) VALUES (?,?,?)',(author,title,message))
	conn.commit()


def view_all_notes():
	c.execute('SELECT * FROM notestable')
	data = c.fetchall()
	# for row in data:
	# 	print(row)
	return data

class Monitor(object):
	"""docstring for Monitor"""

	conn = sqlite3.connect('data.db')
	c = conn.cursor()

	def __init__(self,age=None ,workclass=None ,fnlwgt=None ,education=None ,education_num=None ,marital_status=None ,occupation=None ,relationship=None ,race=None ,sex=None ,capital_gain=None ,capital_loss=None ,hours_per_week=None ,native_country=None,predicted_class=None,model_class=None):
		super(Monitor, self).__init__()
		self.age = age
		self.workclass = workclass
		self.fnlwgt = fnlwgt
		self.education = education
		self.education_num = education_num
		self.marital_status = marital_status
		self.occupation = occupation
		self.relationship = relationship
		self.race = race
		self.sex = sex
		self.capital_gain = capital_gain
		self.capital_loss = capital_loss
		self.hours_per_week = hours_per_week
		self.native_country = native_country
		self.predicted_class = predicted_class
		self.model_class = model_class

	def __repr__(self):
		# return "Monitor(age ={self.age},workclass ={self.workclass},fnlwgt ={self.fnlwgt},education ={self.education},education_num ={self.education_num},marital_status ={self.marital_status},occupation ={self.occupation},relationship ={self.relationship},race ={self.race},sex ={self.sex},capital_gain ={self.capital_gain},capital_loss ={self.capital_loss},hours_per_week ={self.hours_per_week},native_country ={self.native_country},predicted_class ={self.predicted_class},model_class ={self.model_class})".format(self.age,self.workclass,self.fnlwgt,self.education,self.education_num,self.marital_status,self.occupation,self.relationship,self.race,self.sex,self.capital_gain,self.capital_loss,self.hours_per_week,self.native_country,self.predicted_class,self.model_class)
		"Monitor(age = {self.age},workclass = {self.workclass},fnlwgt = {self.fnlwgt},education = {self.education},education_num = {self.education_num},marital_status = {self.marital_status},occupation = {self.occupation},relationship = {self.relationship},race = {self.race},sex = {self.sex},capital_gain = {self.capital_gain},capital_loss = {self.capital_loss},hours_per_week = {self.hours_per_week},native_country = {self.native_country},predicted_class = {self.predicted_class},model_class = {self.model_class})".format(self=self)

	def create_table(self):
		self.c.execute('CREATE TABLE IF NOT EXISTS predictiontable(age NUMERIC,workclass NUMERIC,fnlwgt NUMERIC,education NUMERIC,education_num NUMERIC,marital_status NUMERIC,occupation NUMERIC,relationship NUMERIC,race NUMERIC,sex NUMERIC,capital_gain NUMERIC,capital_loss NUMERIC,hours_per_week NUMERIC,native_country NUMERIC,predicted_class NUMERIC,model_class TEXT)')

	def add_data(self):
		self.c.execute('INSERT INTO predictiontable(age,workclass,fnlwgt,education,education_num,marital_status,occupation,relationship,race,sex,capital_gain,capital_loss,hours_per_week,native_country,predicted_class,model_class) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)',(self.age,self.workclass,self.fnlwgt,self.education,self.education_num,self.marital_status,self.occupation,self.relationship,self.race,self.sex,self.capital_gain,self.capital_loss,self.hours_per_week,self.native_country,self.predicted_class,self.model_class))
		self.conn.commit()

	def view_all_data(self):
		self.c.execute('SELECT * FROM predictiontable')
		data = self.c.fetchall()
		# for row in data:
		# 	print(row)
		return data




def get_value(val,my_dict):
	for key ,value in my_dict.items():
		if val == key:
			return value

# Find the Key From Dictionary
def get_key(val,my_dict):
	for key ,value in my_dict.items():
		if val == value:
			return key

# Load Models
def load_prediction_models(model_file):
	loaded_model = joblib.load(open(os.path.join(model_file),"rb"))
	return loaded_model

def main():
	"""Salary Predictor with ML"""

	st.title("Salary Prediction")
	activity = ["Exploratory Data Analysis","Prediction","Country-Specific Information","Metrics","About"]
	choice = st.sidebar.selectbox("Choose An Activity",activity)
	
	# Load File
	df = pd.read_csv("data/Adult_Salary_Dataset_2.csv")
	df2 = pd.read_csv("data/Adult_Salary_Data.csv")

    #EDA
	if choice == 'Exploratory Data Analysis':
		st.markdown("""
			## EDA Section
			### Quick Facts about Data 
			""")
		st.text("")
		st.text("")
		# st.subheader("EDA Section")
		# st.text("Exploratory Data Analysis")
		# Preview Our Data
		
		if st.button('Jupyter Notebook'):
			webbrowser.open_new_tab(url)

		st.text("Find the streamlit code and associated files below:")

		url = 'https://github.com/Dubeyrock'
		if st.button('app.py'):
			webbrowser.open_new_tab(url)

		st.text("")
		st.text("")


		if st.checkbox("Preview Dataset"):
			number = st.number_input("Number to Show")
			st.dataframe(df.head(int(number)))

		

		
		# Show Columns/Rows
		if st.button("Column Names"):
			st.write(df.columns)

		

		# Description 
		if st.checkbox("Description"):
			st.write(df.describe())
		

		# Shape
		if st.checkbox("Show Shape of Dataset"):
			st.write(df.shape)
			data_dim = st.radio("Show Dimensions by",("Rows","Columns"))

			if data_dim == 'Rows':
				st.text("Number of Rows")
				st.write(df.shape[0])

			elif data_dim == 'Columns':
				st.text('Number of Columns')
				st.write(df.shape[1])
			else: 
				st.write(df.shape)

		# Selections
		if st.checkbox("Select Columns to Show"):
			all_columns = df.columns.tolist()
			selected_columns = st.multiselect("Select Columns",all_columns)
			new_df = df[selected_columns]
			st.dataframe(new_df)

		if st.checkbox("Select Rows to Show"):
			selected_index = st.multiselect("Select Rows", df.head(10).index)
			selected_rows = df.loc[selected_index]
			st.dataframe(selected_rows)


		# Value Counts
		if st.button("Value Counts"):
			st.text("Value Counts By Class")
			st.write(df.iloc[:,-1].value_counts())
			df['class'].value_counts().plot(kind='bar')
			st.pyplot()

		# Plot
		if st.checkbox("Show Correlation Plot [Matplotlib]"):
			plt.matshow(df.corr())
			st.pyplot()
			st.set_option('deprecation.showPyplotGlobalUse', False)

		if st.checkbox("Show Correlation Plot [Seaborn]"):
			fig, ax = plt.subplots(figsize=(14,10))
			st.write(sns.heatmap(df.corr(),annot=True,linewidths=.5, ax=ax))
			st.pyplot()


    # PREDICTION
	elif choice == 'Prediction':
		st.subheader("Prediction Section")

		d_workclass = {"Never-worked": 0, "Private": 1, "Federal-gov": 2, "?": 3, "Self-emp-inc": 4, "State-gov": 5, "Local-gov": 6, "Without-pay": 7, "Self-emp-not-inc": 8}

		d_education = {"Some-college": 0, "10th": 1, "Doctorate": 2, "1st-4th": 3, "12th": 4, "Masters": 5, "5th-6th": 6, "9th": 7, "Preschool": 8, "HS-grad": 9, "Assoc-acdm": 10, "Bachelors": 11, "Prof-school": 12, "Assoc-voc": 13, "11th": 14, "7th-8th": 15}

		d_marital_status = {"Separated": 0, "Married-spouse-absent": 1, "Married-AF-spouse": 2, "Married-civ-spouse": 3, "Never-married": 4, "Widowed": 5, "Divorced": 6}

		d_occupation = {"Tech-support": 0, "Farming-fishing": 1, "Prof-specialty": 2, "Sales": 3, "?": 4, "Transport-moving": 5, "Armed-Forces": 6, "Other-service": 7, "Handlers-cleaners": 8, "Exec-managerial": 9, "Adm-clerical": 10, "Craft-repair": 11, "Machine-op-inspct": 12, "Protective-serv": 13, "Priv-house-serv": 14}

		d_relationship = {"Other-relative": 0, "Not-in-family": 1, "Own-child": 2, "Wife": 3, "Husband": 4, "Unmarried": 5}

		d_race = {"Amer-Indian-Eskimo": 0, "Black": 1, "White": 2, "Asian-Pac-Islander": 3, "Other": 4}

		d_sex = {"Female": 0, "Male": 1}

		d_native_country = {"Canada": 0, "Philippines": 1, "Thailand": 2, "Scotland": 3, "Germany": 4, "Portugal": 5, "India": 6, "China": 7, "Japan": 8, "Peru": 9, "France": 10, "Greece": 11, "Taiwan": 12, "Laos": 13, "Hong": 14, "El-Salvador": 15, "Outlying-US(Guam-USVI-etc)": 16, "Yugoslavia": 17, "Cambodia": 18, "Italy": 19, "Honduras": 20, "Puerto-Rico": 21, "Dominican-Republic": 22, "Vietnam": 23, "Poland": 24, "Hungary": 25, "Holand-Netherlands": 26, "Ecuador": 27, "South": 28, "Guatemala": 29, "United-States": 30, "Nicaragua": 31, "Trinadad&Tobago": 32, "Cuba": 33, "Jamaica": 34, "Iran": 35, "?": 36, "Haiti": 37, "Columbia": 38, "Mexico": 39, "England": 40, "Ireland": 41}

		d_class = {">50K": 0, "<=50K": 1}

		# ML Aspect User Input
		age = st.slider("Select Age", 17,90)
		workclass = st.selectbox("Select Work Class",tuple(d_workclass.keys()))
		fnlwgt = st.number_input("Enter FNLWGT",1.228500e+04,1.484705e+06)
		education = st.selectbox("Select Education",tuple(d_education.keys()))
		education_num = st.slider("Select Your Level",1,16)
		marital_status = st.selectbox("Select Marital Status",tuple(d_marital_status.keys()))
		occupation = st.selectbox("Select Occupation",tuple(d_occupation.keys()))
		relationship = st.selectbox("Select Relationship",tuple(d_relationship.keys()))
		race = st.selectbox("Select Race",tuple(d_race.keys()))
		# load_icon('people')
		sex = st.radio("Select Sex",tuple(d_sex.keys()))
		# load_icon('attach_money')
		# load_icon('mood')
		capital_gain = st.number_input("Capital Gain",0,99999)
		# load_icon('mood_bad')
		capital_loss = st.number_input("Capital Loss",0,4356)
		# load_icon('access_time')
		hours_per_week = st.number_input("Hours Per Week ",0,99)
		native_country = st.selectbox("Select Native Country",tuple(d_native_country.keys()))

		# User Input
		k_workclass = get_value(workclass,d_workclass)
		k_education = get_value(education,d_education)
		k_marital_status = get_value(marital_status,d_marital_status)
		k_occupation = get_value(occupation,d_occupation)
		k_relationship = get_value(relationship,d_relationship)
		k_race = get_value(race,d_race)
		k_sex = get_value(sex,d_sex)
		k_native_country = get_value(native_country,d_native_country)

		# RESULT OF USER INPUT
		selected_options = [age ,workclass ,fnlwgt ,education ,education_num ,marital_status ,occupation ,relationship ,race ,sex ,capital_gain ,capital_loss ,hours_per_week ,native_country]
		vectorized_result = [age ,k_workclass ,fnlwgt ,k_education ,education_num ,k_marital_status ,k_occupation ,k_relationship ,k_race ,k_sex ,capital_gain ,capital_loss ,hours_per_week ,k_native_country]
		sample_data = np.array(vectorized_result).reshape(1, -1)
		st.info(selected_options)
		st.text("Using this encoding for prediction")
		st.success(vectorized_result)

		prettified_result = {"age":age,
		"workclass":workclass,
		"fnlwgt":fnlwgt,
		"education":education,
		"education_num":education_num,
		"marital_status":marital_status,
		"occupation":occupation,
		"relationship":relationship,
		"race":race,
		"sex":sex,
		"capital_gain":capital_gain,
		"capital_loss":capital_loss,
		"hours_per_week":hours_per_week,
		"native_country":native_country}
		
		st.subheader("Prettify JSON")
		st.json(prettified_result)

		st.write(vectorized_result)

		# MAKE PREDICTIONS
		st.subheader("Prediction")
		if st.checkbox("Make Prediction"):
			all_ml_dict = {'LR':"LogisticRegression",
				'RForest':"RandomForestClassifier",
				'NB':"MultinomialNB"}

			# Model Selection
			model_choice = st.selectbox('Model Choice',list(all_ml_dict.keys()))
			prediction_label = {">50K": 0, "<=50K": 1}
			if st.button("Predict"):
				if model_choice == 'RForest':
					model_predictor = load_prediction_models("models/salary_rf_model.pkl")
					prediction = model_predictor.predict(sample_data)
					# st.write(prediction)
				elif model_choice == 'LR':
					model_predictor = load_prediction_models("models/salary_logit_model.pkl")
					prediction = model_predictor.predict(sample_data)
					# st.write(prediction)
				elif model_choice == 'NB':
					model_predictor = load_prediction_models("models/salary_nv_model.pkl")
					prediction = model_predictor.predict(sample_data)
					# st.write(prediction)

				final_result = get_key(prediction,prediction_label)
				monitor = Monitor(age ,workclass ,fnlwgt ,education ,education_num ,marital_status ,occupation ,relationship ,race ,sex ,capital_gain ,capital_loss ,hours_per_week ,native_country,final_result,model_choice)
				monitor.create_table()
				monitor.add_data()
				st.success("Predicted Salary as :: {}".format(final_result))

	elif choice == 'Country-Specific Information':
		st.text("Demographics")
		d_native_country = {"Canada": 0, "Philippines": 1, "Thailand": 2, "Scotland": 3, "Germany": 4, "Portugal": 5, "India": 6, "China": 7, "Japan": 8, "Peru": 9, "France": 10, "Greece": 11, "Taiwan": 12, "Laos": 13, "Hong": 14, "El-Salvador": 15, "Outlying-US(Guam-USVI-etc)": 16, "Yugoslavia": 17, "Cambodia": 18, "Italy": 19, "Honduras": 20, "Puerto-Rico": 21, "Dominican-Republic": 22, "Vietnam": 23, "Poland": 24, "Hungary": 25, "Holand-Netherlands": 26, "Ecuador": 27, "South": 28, "Guatemala": 29, "United-States": 30, "Nicaragua": 31, "Trinadad&Tobago": 32, "Cuba": 33, "Jamaica": 34, "Iran": 35, "?": 36, "Haiti": 37, "Columbia": 38, "Mexico": 39, "England": 40, "Ireland": 41}
		selected_countries = st.selectbox("Select Country",tuple(d_native_country.keys()))
		st.text(selected_countries)

		result_df = df2[df2['native-country'].str.contains(selected_countries)]
		st.dataframe(result_df)

		# if st.checkbox('Pair Plot'):
		#			st.title("Hello")
		#			fig = sns.pairplot(df)
		#			st.pyplot(fig)
	

		if st.checkbox("Select Columns To Show"):
				result_df_columns = result_df.columns.tolist()
				selected_columns = st.multiselect('Select',result_df_columns)
				new_df = df2[selected_columns]
				st.dataframe(new_df)

				if st.checkbox("Plot"):
					st.area_chart(df[selected_columns])
					st.pyplot()



	elif choice == 'Metrics':
		st.subheader("Metrics Section")
		# Create your connection.
		cnx = sqlite3.connect('data.db')

		mdf = pd.read_sql_query("SELECT * FROM predictiontable", cnx)
		st.dataframe(mdf)

	if choice == 'About':
		# st.subheader("About")
		st.markdown("""
			## Productionizing Machine Learning Models
			### Adult Census Income Prediction Machine Learning Web Application 
			#### Built with Streamlit

			### By
			+ Shivam Dubey
			+ shivvam2002@gmail.com

			""")

		url = 'https://github.com/Dubeyrock'
		if st.button('Github'):
			webbrowser.open_new_tab(url)
		url = 'https://www.linkedin.com/in/shivam-dubey-371a591a8/'
		if st.button('Linkedln'):
			webbrowser.open_new_tab(url)
		st.markdown("![Adult Census Income Prediction Machine Learning Web Application](https://media.giphy.com/media/3o8dFzIXb0qaE3pYWs/giphy.gif)")



  




if __name__ == '__main__':
	main()