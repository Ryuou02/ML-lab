# Write a program that accepts a matrix as input and returns its transpose

def takeMatrixInput():
	matrix = []
	rows = int(input("enter rows : "))
	cols = int(input("enter cols : "))
	for i in range(rows):
		temp = []
		for j in range(cols):
			temp.append(int(input(f"matrix[{i}][{j}] = ")))
		matrix.append(temp)
	return matrix

def returnTranspose(matrix):
	mat = []
	for i in range(len(matrix[0])):
		temp = []
		for j in range(len(matrix)):
			temp.append(matrix[j][i])
		mat.append(temp)
	return mat

m1 = takeMatrixInput()
print(m1)
print(returnTranspose(m1))
