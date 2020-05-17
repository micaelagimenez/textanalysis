#Importing necessary library
import nltk
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.probability import FreqDist
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk.stem import WordNetLemmatizer
from textblob import TextBlob
from nltk import ne_chunk, pos_tag
import matplotlib.pyplot as plt
import seaborn as sns

#Text to analyze
text = "Happy families are all alike; every unhappy family is unhappy in its own way. Everything was in confusion in the Oblonskys' house. The wife had discovered that the husband was carrying on an intrigue with a French girl, who had been a governess in their family, and she had announced to her husband that she could not go on living in the same house with him. This position of affairs had now lasted three days, and not only the husband and wife themselves, but all the members of their family and household, were painfully conscious of it. Every person in the house felt that there was so sense in their living together, and that the stray people brought together by chance in any inn had more in common with one another than they, the members of the family and household of the Oblonskys. The wife did not leave her own room, the husband had not been at home for three days. The children ran wild all over the house; the English governess quarreled with the housekeeper, and wrote to a friend asking her to look out for a new situation for her; the man-cook had walked off the day before just at dinner time; the kitchen-maid, and the coachman had given warning."

#Convert text to lowercase
lowertext = text.lower()
print(lowertext)

#Punctuation removal and tokenization
tokenizer = nltk.RegexpTokenizer(r"\w+")
tokenizedtxt = tokenizer.tokenize(lowertext)
print(tokenizedtxt)

#Deleting stop words
stop_words = set(stopwords.words('english'))
cleanedtext = [i for i in tokenizedtxt if not i in stop_words]
print(cleanedtext)

#Finding frequency in the tokens
FreqDist
fdist = FreqDist(cleanedtext)

#Plot of ten most frequent words
fdist.plot(10)

#Stemming
ps = PorterStemmer()
for w in cleanedtext:
    print(ps.stem(w))

#Lemmatization
lemmatizer = WordNetLemmatizer()
for word in cleanedtext:
    print(lemmatizer.lemmatize(word))

#Part of speech tagging
pos = TextBlob(lowertext)
print(pos.tags)

#Checking sentiment polarity and subjectivity
TextBlob(lowertext).sentiment

#Named entity recognition: it aims to find named entities in text and classify them into pre-defined categories 
ner = ne_chunk(pos_tag(word_tokenize(lowertext)))
print(ner)

