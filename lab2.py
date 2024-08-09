import pandas as pd
import xlrd
import numpy as nm
import statistics

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
	print(f"{len(WednesdayPrices)}")
	print(f"probability of making profit given that it is a wednesday is { ( pos/len(WednesdayPrices) ) / ( len(WednesdayPrices) / df['Chg%'].shape[0] )}")


if __name__ == "__main__":
#	priceSheetAnalysis()
	IRCTCanalysis()
