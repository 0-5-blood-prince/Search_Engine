from util import *
from spellchecker import SpellChecker

spell = SpellChecker()



class Tokenization():

	def naive(self, text):
		"""
		Tokenization using a Naive Approach

		Parameters
		----------
		arg1 : list
			A list of strings where each string is a single sentence

		Returns
		-------
		list
			A list of lists where each sub-list is a sequence of tokens
		"""

		tokenizedText = [list(filter(bool,re.split('[^a-zA-Z_]', sentence))) for sentence in text]

		#Fill in code here
		# for i in range(len(tokenizedText)):
		# 	for j in range(len(tokenizedText[i])):
		# 		token = tokenizedText[i][j]
		# 		if token[0].isalpha() or token[0].isdigit():
		# 			tokenizedText[i][j] = token
		# 		else:
		# 			tokenizedText[i][j] = token[1:]
				
		# 		token = tokenizedText[i][j]
		# 		if len(token)==0:
		# 			continue
		# 		if token[-1].isalpha() or token[-1].isdigit():
		# 			tokenizedText[i][j] = token
		# 		else:
		# 			print(token)
		# 			tokenizedText[i][j] = token[:-1]
		# 		token = tokenizedText[i][j]
				# tokenizedText[i][j] = spell.correction(token)
		return tokenizedText



	def pennTreeBank(self, text):
		"""
		Tokenization using the Penn Tree Bank Tokenizer

		Parameters
		----------
		arg1 : list
			A list of strings where each string is a single sentence

		Returns
		-------
		list
			A list of lists where each sub-list is a sequence of tokens
		"""

		tokenizedText = [TreebankWordTokenizer().tokenize(sentence) for sentence in text]
		# tokenizedText = [word_tokenizer(sentence) for sentence in text]
		# for i in range(len(tokenizedText)):
		# 	for j in range(len(tokenizedText[i])):
		# 		token = tokenizedText[i][j]
		# 		if token[0].isalpha() or token[0].isdigit():
		# 			tokenizedText[i][j] = token
		# 		else:
		# 			tokenizedText[i][j] = token[1:]
				
		# 		token = tokenizedText[i][j]
		# 		if len(token)==0:
		# 			continue
		# 		if token[-1].isalpha() or token[-1].isdigit():
		# 			tokenizedText[i][j] = token
		# 		else:
		# 			print(token)
		# 			tokenizedText[i][j] = token[:-1]
		# 		token = tokenizedText[i][j]
				# tokenizedText[i][j] = spell.correction(token)
		#Fill in code here

		return tokenizedText