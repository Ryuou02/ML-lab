import pandas as pd
import xlrd
import numpy as nm

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

matrixA = df2;
del matrixA["Payment (Rs)"]
matrixC = df[["Payment (Rs)"]].copy()

matrixAinverse = nm.linalg.pinv(matrixA)
print(matrixA)

matrixX =nm.matmul(matrixAinverse,matrixC).to_numpy()

print(matrixX)
print(nm.matmul(matrixA, matrixX).to_numpy)
#print(matrixA)
print(matrixC)

tempList = []
for d in df.index:
	if(df['Payment (Rs)'][d] < 200):
		tempList.append("POOR")
	else:
		tempList.append("RICH")


df['Status'] = tempList
print(df)
