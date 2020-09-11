import nltk
from nltk.tokenize import sent_tokenize, word_tokenize
# tokenizing - split to words or sentens
# lexicon or corporas
# corporas - body of text  Journal
# lexicon - what words means

example_text = "Hello there, how are you. what are you doing? how are you? mr.smitH you are shit "

print(sent_tokenize(example_text))
print(word_tokenize(example_text))

