# - stem - core of word example riding  -> rid  (rid,riding, rider ...)

from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize

ps = PorterStemmer()
example = ["python","pythoner","pythoning","pythoned"]
new_text = "it is very important to be pythonly while you are"

word = word_tokenize(new_text)

for w in word:
    print((ps.stem(w)))
