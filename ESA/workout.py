

# import jamspell
# import time
# from textblob import TextBlob
# print(TextBlob('aerodynam').correct())
# corrector = jamspell.TSpellCorrector()
# corrector.LoadLangModel('en.bin')
# start = time.time()
# print(corrector.FixFragment('aerodynam'))
# print(time.time() - start)

# x = 'on heat transfr in slip flow . a nuber of authors have cosidered the effect of slip on the heat transfer and skin friction in a laminar boundary layer over a flat plate . reference 1 considers this by a perturbation on the usual laminar boundarlayer analyis while some other studies.dash e.g., reference the impulsive motion of an infinite plat .'
 
# print('jamspell:Spell correction with context')
# print(corrector.FixFragment('I lke chiken & egs for breakfst!'))
# print(corrector.FixFragment('I lik chiken lgs fo lunh!'))
# print(corrector.FixFragment(x))
# print(corrector.GetCandidates(['i', 'am', 'the', 'begt', 'spell', 'cherken'], 3))
# print('textblob:')
# print(TextBlob('I lke chiken & egs for breakfst!').correct())
# print(TextBlob('I lik chiken lgs fo lunh!').correct())
# print(TextBlob(x).correct())


# import nltk
# from nltk.corpus import wordnet

# print(wordnet.synsets('associated'))


# from wikipedia2vec import Wikipedia2Vec
# import numpy as np
# wiki2vec = Wikipedia2Vec.load('enwiki_20180420_100d.pkl')

# word = 'aerodynamic'
# x = wiki2vec.get_word_vector(word)
# print(x)
# print(len(x))
# print(np.argmax(x))
# print(sum(x))
# print(wiki2vec.get_word(word))
# # print(wiki2vec.most_similar(wiki2vec.get_word(word), 5))

# word = 'heating'
# x = wiki2vec.get_word_vector(word)
# # print(x)
# print(len(x))
# print(np.argmax(x))
# print(sum(x))
# print(wiki2vec.get_word(word))
# # print(wiki2vec.most_similar(wiki2vec.get_word(word), 5))

# import string
# x = "123.hello!-k'"
# rem = str.maketrans('', '', string.digits+'\',=.!@#$()/\ ')
# rem[ord('-')] = ord(' ')
# print(rem)
# print(x.translate(rem))

# import numpy as np
# print(np.array([1,2,3]) + np.array([4,5,6]))


# import splitter

# print(splitter.split('acrothermoelasticity','en'))
# print(splitter.split('photothermoelastic'))
# print(splitter.split('nonaxisymmetric'))
# print(splitter.split('semiballistic'))
# print(splitter.split('cmsdsd'))


# import numpy as np

# # print(np.dot([1,0,0],[0,0,1])/(np.linalg.norm([1,0,0])*np.linalg.norm([0,0,1])))
# print(np.argsort(-1*np.array([1,2,3,45])))


# import numpy as np

# a = np.array([[1,2],[3,4]])
# b = [[-1,-2],[-3,-4]]
# print(a)
# print(a/np.reshape(np.linalg.norm(a,axis=1),(-1,1)))

# a = [1,2,3,4,5,6,7,8,9]
# print(a[-3:])

# q = np.reshape([1,2,3,4],(-1,1))
# w = np.array([1,2,3,4])
# print(np.dot(w,q))
# print(np.shape(w))
# print(np.linalg.norm(w[:3]))

# from nltk.corpus import stopwords

# # print(stopwords.words('english'), type(stopwords.words('english')))
# stw = {i:None for i in stopwords.words('english')}
# rem = str.maketrans('', '', '\',=.!@#$()/\ ')
# print(stw)
# print(rem)
# r = 'the the you king kong'
# print(r.translate(stw))


print(' '.join(['q','w','p']))

