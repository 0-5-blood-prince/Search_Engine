# Add your import statements here
import re
import math
import nltk
import numpy as np
import jamspell
import string
import splitter
from nltk.tokenize import TreebankWordTokenizer
from nltk.stem import PorterStemmer
from nltk.stem import WordNetLemmatizer
from nltk import sent_tokenize as tokenizer
from nltk.corpus import stopwords
from wikipedia2vec import Wikipedia2Vec
import textwiser
from textwiser import TextWiser, Embedding, PoolOptions, Transformation, WordOptions
from gensim.models import Word2Vec
import gensim.downloader as api
from gensim.similarities import WmdSimilarity
import matplotlib.pyplot as plt


nltk.download('wordnet',quiet=True)
nltk.download('punkt',quiet=True)
nltk.download('stopwords',quiet=True)

# Add any utility functions here
