#Write a program that accepts two matrices A and B as input and returns their product AB. Check if A & B are multipliable; if not, return error message.
def matrixMultiply(a,b):
	# number of columns in a should be equal to number of rows in b
	if(len(a[0]) != len(b)):
		raise("the given matrices cannot be multiplied")
	resultRows = len(a)
	resultCols = len(b[0])
	productMatrix = [[0 for i in range(resultCols)] for i in range(resultRows)]
	print(productMatrix)
	for x in range(resultRows):
		for y in range(resultCols):
			for i in range(len(b)):
				productMatrix[y][x] += a[y][i] * b[i][x]
	return productMatrix


a = [[1,2,3],[4,5,6],[7,8,9]]
print(matrixMultiply(a,a))
print(numpy.matmul(a,a))
