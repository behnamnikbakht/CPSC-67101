import argparse
import pickle
import nltk
from nltk import ngrams
from nltk import PorterStemmer, LancasterStemmer, pos_tag
from nltk.corpus import wordnet as wn
from nltk.stem import WordNetLemmatizer
from collections import defaultdict

from data_loader import DataLoader

stopwords = nltk.corpus.stopwords.words("english")


class Config:
    def __init__(self, log, ngrams_factor, tokenize_not):
        self.log = log
        self.ngrams_factor = ngrams_factor
        self.tokenize_not = tokenize_not


# this class represents a unit for analyzing tweet (or any text in general)
class TextItem:
    # constructor
    def __init__(self, text):
        self.text = text

    def tokenize(self, config):
        if config.log:
            print("start tokenize {}".format(self.text))
        # tokenize by space and new-line, and normalize by converting to lowercase
        tokens = [w.lower() for w in nltk.word_tokenize(self.text)]
        if config.log:
            print("tokens after wordnet tokenize = {}".format(tokens))

        # n-grams
        tokens = [' '.join(grams) for grams in ngrams(tokens, config.ngrams_factor)]
        if config.log:
            print("after n-grams {}".format(tokens))

        # tokenize not
        if config.tokenize_not:
            temp = []
            i = 0
            while i < len(tokens) - 1:
                if tokens[i] == 'not':
                    temp.append(tokens[i] + ' ' + tokens[i+1])
                    i = i + 2
                else:
                    temp.append(tokens[i])
                    i = i + 1
            tokens = temp
            if config.log:
                print("after tokenize not {}".format(tokens))

        # remove stop words from tokens, and single-character tokens
        tokens = [w for w in tokens if w not in stopwords and len(w) > 1]
        if config.log:
            print("tokens after stop words removal = {}".format(tokens))

        return tokens

    def normalize(self, tokens):
        # lemmatizer based on wordnet semantic relationships
        lemmatizer = WordNetLemmatizer()

        # extract lingusitic position of each token
        tagged = pos_tag(tokens)

        # result = [lemmatizer.lemmatize(word) for word,tag in tagged if tag in ["NN", "NNS"]]
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

    def preprocessing(self, config):
        tokens = self.tokenize(config)
        self.tokens, self.synonyms = self.normalize(tokens)

        # frequency of tokens
        self.frequency = nltk.FreqDist([w.lower() for w in self.tokens])


class Classifier(object):

    def __init__(self):
        self.data_loader = DataLoader()

    def getLabeledDataset(self):
        dataset = self.data_loader.load1()
        size = len(dataset)
        test_set_size = int(size / 10)
        print("size of all labeled labeled_dataset = {}".format(size))
        # return test_set, train_set as train_set contains 90% of all
        return dataset[test_set_size:], dataset[:test_set_size]


class NltkClassifier(Classifier):
    def __init__(self):
        super(NltkClassifier, self).__init__()

    def preparation(self, config):
        plain_train_set, plain_test_set = self.getLabeledDataset()
        print("train_set = {}, test_set = {}".format(len(plain_train_set), len(plain_test_set)))

        self.train_set = []
        self.test_set = []

        for p, c in plain_train_set:
            t = TextItem(p)
            t.preprocessing(config)
            self.train_set.append((t, c))

        for p, c in plain_test_set:
            t = TextItem(p)
            t.preprocessing(config)
            self.test_set.append((t, c))

        self.classifier = nltk.NaiveBayesClassifier.train([(listToDict(t.tokens), c) for t, c in self.train_set])

        with open('trained_model', 'wb') as trained_model_file:
            pickle.dump(self.classifier, trained_model_file)

    def test(self):
        stat = {"correct": 0, "all": 0}
        for p, c in self.test_set:
            cls = self.classifier.classify(listToDict(p.tokens))
            if cls == c:
                stat["correct"] = stat["correct"] + 1
            stat["all"] = stat["all"] + 1
            # print("test = {}, c = {}".format(cls, c))

        print("stat = {}, accuracy = {}%".format(stat, 100 * stat["correct"] / stat["all"]))

    def predict(self, text, config):
        with open('trained_model', 'rb') as trained_model_file:
            trained_model = pickle.load(trained_model_file)
            p = TextItem(text)
            config.log = True
            p.preprocessing(config)
            return trained_model.classify(listToDict(p.tokens))


def listToDict(l):
    return {l[i]: l[i] for i in range(len(l))}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Tweet Emotion Recognition')
    parser.add_argument('--command', choices=['test', 'build', 'predict'], help='t [run test], p [predict]')
    parser.add_argument('--text', help='input text for predict')
    parser.add_argument('--ng', type=int, default=1, help='n-grams factor')
    parser.add_argument('--tokno', type=bool, default=False, help='tokenize not verbs in a single token')
    args = parser.parse_args()

    nltkClassifier = NltkClassifier()

    print("args = {}".format(args))

    config = Config(log=False, ngrams_factor=args.ng, tokenize_not=args.tokno)

    if args.command == 'build':
        nltkClassifier.preparation(config)
    elif args.command == 'test':
        nltkClassifier.preparation(config)
        nltkClassifier.test()
    elif args.command == 'predict':
        text = args.text
        result = nltkClassifier.predict(text, config)
        print("result = {}".format(result))
