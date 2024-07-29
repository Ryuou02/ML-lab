#Write a program to find the number of common elements between two lists. The lists contain integers.

def findCommonElements(list1, list2):
	commonElements = []
	for i in range(len(list1)):
		for j in range(len(list2)):
			if(list1[i] == list2[j]):
				commonElements.append(list1[i])
	return commonElements

l1 = [1,2,3]
l2 = [2,3,5]
print(findCommonElements(l1,l2))
