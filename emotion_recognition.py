import nltk
from nltk import PorterStemmer, LancasterStemmer, pos_tag
from nltk.corpus import wordnet as wn
from nltk.stem import WordNetLemmatizer
from collections import defaultdict

from data_loader import DataLoader

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


# text = """
# i am feeling good today because the weather is perfect"""
# t = TextItem(text)
# t.preprocessing()
#
# print("tokens = {}, synonyms = {}, frequency = {}".format(t.tokens, t.synonyms, t.frequency.items()))

class Classifier(object):

    def __init__(self):
        self.data_loader = DataLoader()

    def getLabeledDataset(self):
        dataset = self.data_loader.load2()
        size = len(dataset)
        test_set_size = int(size / 10)
        print("size of all labeled labeled_dataset = {}".format(size))
        # return test_set, train_set as train_set contains 90% of all
        return dataset[test_set_size:], dataset[:test_set_size]


class NltkClassifier(Classifier):
    def __init__(self):
        super(NltkClassifier, self).__init__()

    def preparation(self):
        plain_train_set, plain_test_set = self.getLabeledDataset()
        print("train_set = {}, test_set = {}".format(len(plain_train_set), len(plain_test_set)))

        self.train_set = []
        self.test_set = []

        for p,c in plain_train_set:
            t = TextItem(p)
            t.preprocessing()
            self.train_set.append((t,c))

        for p,c in plain_test_set:
            t = TextItem(p)
            t.preprocessing()
            self.test_set.append((t,c))

        self.classifier = nltk.NaiveBayesClassifier.train([(listToDict(t.tokens),c) for t,c in self.train_set])

        stat = {"correct": 0, "all": 0}
        for p, c in self.test_set:
            cls = self.classifier.classify(listToDict(p.tokens))
            if cls == c:
                stat["correct"] = stat["correct"] + 1
            stat["all"] = stat["all"] + 1
            #print("test = {}, c = {}".format(cls, c))

        print("stat = {}, accuracy = {}%".format(stat, 100 * stat["correct"] / stat["all"]))

def listToDict(l):
    return {l[i]: l[i] for i in range(len(l))}

nltkClassifier = NltkClassifier()
nltkClassifier.preparation()