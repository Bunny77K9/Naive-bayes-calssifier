# load some useful Python libratries

import pandas as pd # the library for working with data tables
import re





#to download files to local machine
# from google.colab import files
# files.download("data_NBC_1.csv")

from collections import Counter # a class for counting objects (words and text labels, in our case)
# enable pandas to display large texts and look into our data
pd.options.display.max_colwidth = 300
data=pd.read_csv('C:/Users/asaelb/PycharmProjects/yandex_preparation/week1/data_NBC_1.csv')
print(data.shape) # number of rows and columns


# enable pandas to display large texts and look into our data
pd.options.display.max_colwidth = 300

#print(data.shape) # number of rows and columns
#data.head(5)
import nltk
from nltk.tokenize import  word_tokenize
#nltk.download()
from nltk.stem import PorterStemmer
porter=PorterStemmer()

# def stemSentence(sentence):
#     token_words=word_tokenize(sentence)
#     token_words
#     stem_sentence=[]
#     for word in token_words:
#         stem_sentence.append(porter.stem(word))
#         stem_sentence.append(" ")
#     return "".join(stem_sentence)
#
# import re

from string import digits
# remove_digits = str.maketrans('', '', digits)
# res = text.translate(remove_digits)
def get_words(text):
    """ This function converts the given text into an unordered and uncounted bag of words. """
    text=text.lower()
    return set(re.split('\W+', text)).difference({''})

# just an example
#get_words("this message is a long, long, long, long long long one.")

# apply this logic to texts of all messages
bags_of_words = [get_words(text) for text in data.text]
n_train = 3000
train_x, test_x, train_y, test_y = bags_of_words[:n_train], bags_of_words[n_train:], data.target[:n_train], data.target[n_train:]
# this counter will keep the number of spam and ham texts
label_counter = Counter()
# import pdb
# these counters will keep the frequency of each word in ham and spam texts
word_counters = {
    'spam': Counter(),
    'ham': Counter()}

all_words = set()

for label, words in zip(train_y, train_x):
    all_words.update(words)
    # TODO: use the `update` methods of all 3 counters, to calculate total number of
    label_counter.update([label])
    # pdb.set_trace()
    word_counters[label].update(words)
    # # TODO: use the `update` methods of all 3 counters, to calculate total number of
    # label_counter[].update(label)
    # # pdb.set_trace()
    # word_counters[label].update(label)

#assert label_counter['spam'] == 409
#assert word_counters['ham']['hello'] >= 2

def prior_probability_of_label(label):
    """ This function evaluates probability of the given label (it can be 'spam' or 'ham'), using the counters. """
    # TODO: calculate and return this probability as ratio of number of texts with this labels to number all texts
    P_label = label_counter[label] / n_train
    return P_label

p_label=prior_probability_of_label('ham')

#assert round(prior_probability_of_label('spam'), 2) == 0.14
#assert round(prior_probability_of_label('ham'), 2) == 0.86



#non zero version of w_prob

def word_probability_given_label(word, label):
    """ This function calculates probability of a word occurence in text, conditional on the label of this text. """
    # TODO: calculate and return this probability
    # as ratio of number of texts with this word and label to number of texts with this label
    p=0.1
    alpha=10**-1
    p_word_given_label=(word_counters[label][word]+alpha*p)/(label_counter[label]+p)
    return p_word_given_label

# P(text)=[text_probability_given_label(text, label) * prior_probability_of_label(label) + text_probability_given_label(text, complement label) * prior_probability_of_label(complement label))]

def text_probability_given_label(text, label):
    """ This function calculates probability of the text conditional on its label. """
    if isinstance(text, str):
        text = get_words(text)
    probability = 1.0
    # TODO: calculate the probability of text given label.
    # use a function defined above and the naive assumption of word independence
    for word in all_words:
        if word in text:
          probability=probability*word_probability_given_label(word, label)
        else:
          probability=probability*(1-word_probability_given_label(word, label))
    return probability

greeting1 = 'hello how are you'
greeting2 = 'hello teacher how are you'

#assert text_probability_given_label(greeting1, 'ham') > 0
#assert text_probability_given_label(greeting1, 'ham') < 0.0001
#assert text_probability_given_label(greeting2, 'ham') < text_probability_given_label(greeting1, 'ham')

def label_probability_given_text(text, label):
    """ This function calculates probability of the label (spam or ham) conditional on the text. """
    # TODO: calculate label probability conditional on text
    # use the Bayes rule and the functions defined above

    if label=='ham':
      other_label='spam'
    else:
      other_label='ham'

    p_text=prior_probability_of_label(label)*text_probability_given_label(text, label)+(prior_probability_of_label(other_label))*(text_probability_given_label(text, other_label))
    p_label_given_text=prior_probability_of_label(label)*text_probability_given_label(text, label)/p_text
    return p_label_given_text


text1 = 'hello how r you'
text2 = 'only today you can buy our book with 50% discount!'

#assert label_probability_given_text(text1, 'ham') + label_probability_given_text(text1, 'spam') == 1.0
#assert label_probability_given_text(text1, 'ham') > label_probability_given_text(text1, 'spam')
#assert label_probability_given_text(text1, 'ham') > label_probability_given_text(text2, 'ham')


thresholds = [0.002]
alpha=[10**-1]
P=[0.1]
for threshold in thresholds:
    test_spam_probabilities = [label_probability_given_text(text, 'spam') for text in test_x]
    test_predictions = ['spam' if spamness > threshold else 'ham' for spamness in test_spam_probabilities]

    accuracy = sum(1 if pred == fact else 0 for pred, fact in zip(test_predictions, test_y)) /len(test_y)
    print(accuracy)

assert accuracy > 0.9
