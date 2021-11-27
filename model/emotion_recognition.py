import nltk
from nltk import PorterStemmer, LancasterStemmer, pos_tag
from nltk.corpus import wordnet as wn
from nltk.stem import WordNetLemmatizer
from collections import defaultdict

stopwords = nltk.corpus.stopwords.words("english")

# this class represents a unit for analyzing tweet (or any text in general)
class TextItem:
    # constructor
    def __init__(self, text):
        self.text = text

    def tokenize(self):
        # tokenize by space and new-line, and normalize by converting to lowercase
        tokens = [w.lower() for w in nltk.word_tokenize(self.text)]

        # remove stop words from tokens, and single-character tokens
        tokens = [w for w in tokens if w not in stopwords and len(w) > 1]

        return tokens

    def normalize(self, tokens):
        # lemmatizer based on wordnet semantic relationships
        lemmatizer = WordNetLemmatizer()

        # extract lingusitic position of each token
        tagged = pos_tag(tokens)

        #result = [lemmatizer.lemmatize(word) for word,tag in tagged if tag in ["NN", "NNS"]]
        # map position to the required input for lemmatizer, based on the first character of the tag_pos
        tag_map = defaultdict(lambda: wn.NOUN)
        tag_map['J'] = wn.ADJ
        tag_map['N'] = wn.NOUN
        tag_map['V'] = wn.VERB
        tag_map['R'] = wn.ADV

        # lemmatize tokens if are of valid pos
        result = [lemmatizer.lemmatize(token, tag_map[tag[0]]) for token, tag in tagged if tag[0] in tag_map]

        # collect semantic relationship between tokens in result
        synonyms = dict()
        for token in result:
            items = set()
            for syn in wn.synsets(token):
                for l in syn.lemmas():
                    lname = l.name()
                    if lname == token:
                        continue
                    if not lname in items:
                        items.add(lname)
            if len(items) > 0:
                synonyms[token] = items

        return result, synonyms

    def preprocessing(self):
        tokens = self.tokenize()
        self.tokens, self.synonyms = self.normalize(tokens)

        # frequency of tokens
        self.frequency = nltk.FreqDist([w.lower() for w in self.tokens])


text = """
i am feeling good today because the whether is perfect"""
t = TextItem(text)
t.preprocessing()

print("tokens = {}, synonyms = {}, frequency = {}".format(t.tokens, t.synonyms, t.frequency.items()))
