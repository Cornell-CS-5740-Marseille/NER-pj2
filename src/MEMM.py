from nltk.classify import MaxentClassifier
import pickle
class MEMM():
    def __init__(self):
        self.features = []
        self.NE_type = ["PER", "LOC", "ORG", "MISC"]
        self.word_set = ["played", "on", "American", "League"]
        self.tag_set = ["VBD", "IN", "NNP", "NNP"]
        self.boi_set = ["O", "O", "B-MISC", "I-MISC"]

    def name_features(self, window_tuple):
        features = {}
        features["current_word"] = window_tuple[0]
        features["current_tag"] = window_tuple[1]
        return features
    def trainMEMM(self, dump):
        tuples = zip(self.word_set, self.tag_set, self.boi_set)
        train_set = [(self.name_features(t), t[2]) for t in tuples]
        self.maxent_classifier = MaxentClassifier.train(train_set, max_iter = 3)
        if dump:
            f = open("../models/MaxentClassifier.pickle", "wb")
            pickle.dump(self.maxent_classifier, f)
            f.close()

    def classification(self, tuple):
        testone = self.maxent_classifier.classify(self.name_features(tuple))
        print(testone)


memm_classifier = MEMM()
memm_classifier.trainMEMM(False)
memm_classifier.classification(("American", "IN"))

# import nltk
# nltk.download('names')
# from nltk.corpus import names
# import random
# def gender_features(word):
#    return {'last_letter':word[-1]}
# namesljl=[(name,'male') for name in names.words('male.txt')]
# namesljl=namesljl+[[name,'female'] for name in names.words('female.txt')]
# random.shuffle(namesljl)
# features=[(gender_features(n),g) for(n,g)in namesljl]
#
# train_set,test_set=features[5:],features[:500]
# print train_set
# classifier=nltk.MaxentClassifier.train(train_set, max_iter = 3)
# testone=classifier.classify(gender_features("Neo"))
# print testone