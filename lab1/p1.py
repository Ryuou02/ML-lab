# Write a program to count the number of vowels and consonants present in an input string.

def getVowelConsonantCount(sentence):
	vowels = "aeiouAEIOU"
	count_vowels = 0
	consonants = 0
	for i in range(len(sentence)):
		if sentence[i] in vowels:
			count_vowels += 1
		elif ( sentence[i] >= 'a' and  sentence[i] <='z' or  sentence[i] >= 'A' and  sentence[i] <='Z'):
			consonants += 1
	return count_vowels,consonants

sentence = input("enter a sentence:")
vowels,consonants = getVowelConsonantCount(sentence)
print(f"there are {vowels} vowels and {consonants} consonants in the given sentence")
