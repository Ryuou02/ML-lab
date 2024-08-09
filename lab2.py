import pandas as pd
import seaborn as sns
from math import sqrt
import xlrd
import numpy as nm
import statistics
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler


def priceSheetAnalysis():
	data = pd.read_excel("Lab Session Data.xlsx", sheet_name = 0)

	#preprocessing of data

	df = pd.DataFrame(data)

	del df["Candy"]
	del df["Mango"]
	del df["Milk"]
	for i in range(5,19):
		del df["Unnamed: " + str(i)]

	#creating temporary matrix for numeric data only
	df2 = df.copy()
	del df2["Customer"]

	print(df2)


	print(f"dimensionalty of data is -> {df.shape}")
	print(f"number of vectors in {df2.size}")
	print(f"rank of matrix is -> {nm.linalg.matrix_rank(df2)}")

	matrixA = df2.copy();
	del matrixA["Payment (Rs)"]
	matrixC = df[["Payment (Rs)"]].copy()

	matrixAinverse = nm.linalg.pinv(matrixA)
	print(matrixA)

	matrixX =nm.matmul(matrixAinverse,matrixC).to_numpy()

	print(matrixX)
	print(nm.matmul(matrixA, matrixX).to_numpy)
	print(matrixC)

	tempList = []
	#print("---------------------")
	for d in df.index:
	#	print(d,type(d))
		if(df['Payment (Rs)'][d] < 200):
			tempList.append("POOR")
		else:
			tempList.append("RICH")

	#print("---------------------")
	df['Status'] = tempList
	print(df)

	trainingData = df[:9].copy()
	print(trainingData)
	print("\n\nindexes - ")
	for i in trainingData.index:
		print(i)
	testingData = df2[9:10]
	print(testingData)
	distances = []

	for record in trainingData.index:
		print(record)
		distance = 0
		for colName in testingData:
			distance += (trainingData[colName][record] - testingData[colName].iloc[0]) ** 2
		distances.append(distance)

	trainingData['distances'] = distances

	'''
	example for sorting based on distance -
	df = pd.DataFrame({'Name': ['John', 'Alice', 'Bob'],
	                   'Age': [25, 30, 20],
	                   'Salary': [50000, 60000, 45000]})
	sorted_df = df.sort_values(by='Salary', ascending=False)
	print(sorted_df)
	'''
	trainingData = trainingData.sort_values(by='distances',ascending = True)
	print(trainingData)

	testingData['Status'] = trainingData['Status'].iloc[0]
	print(testingData)


def IRCTCanalysis():
	data = pd.read_excel("Lab Session Data.xlsx", sheet_name = 1)
	df = pd.DataFrame(data)
	print("mean price :", statistics.mean(df['Price']))
	print("variance of price : ", statistics.variance(df['Price']))
	WednesdayPrices = []
	for row in df.index:
		if(df["Day"][row] == "Wed"):
			WednesdayPrices.append(df['Price'][row])
	SampleMean = statistics.mean(WednesdayPrices)
	print("sample mean for wedneseday only data : ", SampleMean)

	# Observation of difference between sample mean and population mean -> sample mean is 10 less than population mean => the prices are lesser on wednesdays
	AprilPrices = []
	for row in df.index:
		if(df["Month"][row] == "Apr"):
			AprilPrices.append(df['Price'][row])
	SampleMean = statistics.mean(AprilPrices)
	print("sample mean for april only data : ", SampleMean)

	# Observation of difference between sample mean and population mean -> sample mean is 100 more than population mean => prices during april were higher than usual

	negatives = 0
	for row in df.index:
		if(df["Chg%"][row] < 0):
			negatives += 1
	print(f"probability of loss in stock prices is {negatives / df['Chg%'].count() * 100} %")
	pos = 0
	for row in df.index:
		if(df["Chg%"][row] > 0 and df["Day"][row] == "Wed"):
			pos += 1
	print(f"probability of gain in stock prices on a wednesday is {pos / len(WednesdayPrices)} %")
	print(f"probability of making profit given that it is a wednesday is { ( pos/ df['Chg%'].shape[0]  ) / ( len(WednesdayPrices) / df['Chg%'].shape[0] )}")
	plt.scatter(df["Chg%"],df["Day"])
	plt.show()

def Thyroid():
	data = pd.read_excel("Lab Session Data.xlsx", sheet_name = 2)
	df = pd.DataFrame(data)

	
	# attributes and their datatypes - 
	'''
	Record ID	 - Ratio
	age	- Ratio
	sex	- catagorical
	on thyroxine - nominal
	query on thyroxine - nominal
	on antithyroid medication - nominal
	sick - nominal
	pregnant	- nominal
	thyroid - nominal
	surgery	- nominal
	I131 treatment - nominal	
	query hypothyroid - nominal	
	query hyperthyroid - nominal	
	lithium - nominal	
	goitre - nominal	
	tumor - nominal	
	hypopituitary - nominal	
	psych	- nominal
	TSH measured - nominal 	
	TSH	- ratio 
	T3 measured	- nominal
	T3 - ratio 
	TT4 measured - nominal
	TT4	- ratio 
	T4U measured - nominal	
	T4U- ratio 
	FTI measured - nominal	
	FTI	- ratio 
	TBG measured	- nominal
	TBG	- ratio 
	referral - ordinal 
	source - ordinal	
	Condition - ordinal

	'''
	NumericData = ["Record ID", "age", "TSH", "T3", "TT4", "T4U", "FTI", "TBG"]
	for record in df:
		print(f"datatype of field '{record}'=>\t\t ", type(df[record].iloc[0]))
		missingValues = 0
		for ind in df.index:			
			if(df[record][ind] == '?'):
				df[record].at[ind] = nm.nan
				missingValues += 1
		print(f"{record} attribute has {missingValues} missing values")

	# identfying data range
	for record in df:
		if(record in NumericData):
			min = 1000000000000000000000000000000
			max = 0
			for ind in df.index:
				try:
					if(df[record].iloc[ind] != nm.nan):
						if(df[record].iloc[ind] < min):
							min = df[record][ind]
						if(df[record].iloc[ind] < max):
							max = df[record][ind]
				except TypeError:
					print("exception -> ", type(df[record].iloc[0]))
			print(f" values in {record} vary from {min} to {max}")

	ordinalAttr = {}
	#finding outliers of each attribute - 
	for record in df:
		if(record in NumericData):
			Quartile1 = nm.percentile(df[record], 25)
			Quartile3 = nm.percentile(df[record], 75)
			interQuartileRange = Quartile3 - Quartile1
			outliers =[]
			for ind in df.index:
				if(df[record].at[ind] != nm.nan and (df[record].at[ind] > Quartile3 + interQuartileRange * 1.5 or df[record].at[ind] < Quartile1 - interQuartileRange * 1.5)):
					outliers.append(df[record].at[ind])
			temp = df[record].dropna()
			print(f"outliers of {record} field => {outliers}")
			if len(outliers) == 0:
				df[record] = df[record].fillna(statistics.mean(temp))
			else:
				df[record] = df[record].fillna(statistics.median(temp))
			print(f"mean of {record} is {statistics.mean(temp)}")
			print(f"variance of {record} is {statistics	.variance(temp)}")
		else:
			catTypeCount = {}
			for ind in df.index:
				if(df[record].at[ind] != nm.nan):
					try:
						catTypeCount[df[record].at[ind]] += 1
					except KeyError:
						catTypeCount[df[record].at[ind]] = 1
			max = 0
			# finding mode in each attribute
			for category in catTypeCount:
				if catTypeCount[category] > max:
					modeCat = category
			df[record] = df[record].fillna(modeCat)
			ordinalAttr[record] = catTypeCount
	print(df)

	print(ordinalAttr)
	for record in NumericData:
		min = df[record].min()
		max = df[record].max()
		for i in df.index:
			df[record].at[i] = (df[record].at[i] - min) / (max - min)
	print(df)

	observations = df[0:2].copy()
	print(observations)
	f11 = 0
	f10 = 0
	f01 = 0
	f00 = 0
	for record in observations:
		if(observations[record].at[0] == observations[record].at[1]):
			if type(observations[record].at[0]) == 'f':
				f00 += 1
			else:
				f11 += 1
		else:
			if type(observations[record].at[0]) == 'f':
				f01 += 1
			else:
				f10 += 1
	
	axb = 0
	normA = 0
	normB = 0
	for record in observations:
		if record not in NumericData:
			count = 0
			for cat in ordinalAttr[record]:
				ordinalAttr[record][cat] = count
				count += 1
			for i in observations.index:
				if(len(ordinalAttr[record]) > 2):
					observations[record].at[i] = ordinalAttr[record][observations[record].at[i]]
				else:
					if(observations[record].iloc[i].lower() == "f"):
						observations[record].at[i] = 0
					else:
						observations[record].at[i] = 1
		axb += observations[record].at[1] * observations[record].at[0]
		normA += observations[record].at[0] * observations[record].at[0]
		normB += observations[record].at[1] * observations[record].at[1]

	print(ordinalAttr)
	print(observations)
	#axb = sqrt(axb)
	normA = sqrt(normA)
	normB = sqrt(normB)

	print("JC similarity =>", f11 / (f10 + f01))
	print("SMC similarity ->", (f11 + f00) / (f10 + f01))
	print("cosine similarity =>", axb/(normA * normB))

	# before making heatmap, we need to convert all data to numeric data
	df = df.dropna()
	for record in df:
		if record not in NumericData:
			for i in df.index:
				try:
					df[record].at[i] = ordinalAttr[record][df[record].iloc[i]]
				except KeyError:
					print("key error -> ", df[record].iloc[i])
				except IndexError:
					pass
	print(df)
	
	
	sns.heatmap(df,annot = True)
if __name__ == "__main__":
#	priceSheetAnalysis()
#	IRCTCanalysis()
	Thyroid()
