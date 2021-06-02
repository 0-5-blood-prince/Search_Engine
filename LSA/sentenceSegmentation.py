from util import *



class SentenceSegmentation():

	def naive(self, text):
		"""
		Sentence Segmentation using a Naive Approach

		Parameters
		----------
		arg1 : str
			A string (a bunch of sentences)

		Returns
		-------
		list
			A list of strings where each string is a single sentence
		"""
		### Punctuation and Spaces
		lines = text.splitlines()
		segmentedText = []
		for line in lines:
			l = list(filter(bool,re.split('[?!:.]',line)))
			segmentedText.extend([i.strip() for i in l])

		#Fill in code here

		return segmentedText





	def punkt(self, text):
		"""
		Sentence Segmentation using the Punkt Tokenizer

		Parameters
		----------
		arg1 : str
			A string (a bunch of sentences)

		Returns
		-------
		list
			A list of strings where each strin is a single sentence
		"""

		segmentedText = tokenizer(text)

		#Fill in code here
		
		return segmentedText