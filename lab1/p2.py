#Write a program that accepts two matrices A and B as input and returns their product AB. Check if A & B are multipliable; if not, return error message.
import numpy
def matrixMultiply(a,b):
	# number of columns in a should be equal to number of rows in b
	if(len(a[0]) != len(b)):
		raise("the given matrices cannot be multiplied")
	productMatrix = [[0 for i in range(len(b))] for i in range(len(a[0]))]
	print(productMatrix)
	for x in range(len(b)):
		for y in range(len(a[0])):
			productMatrix[x][y] = 0
			for i in range(len(b)):
				productMatrix[x][y] += a[y][i] * b[x][i]
	return productMatrix

a = [[1,2,3],[4,5,6],[7,8,9]]
print(matrixMultiply(a,a))
print(numpy.matmul(a,a))
