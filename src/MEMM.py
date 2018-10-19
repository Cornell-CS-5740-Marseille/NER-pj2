import random
import os.path
from nltk.classify import MaxentClassifier
import pickle

from src.prep import prep


class MEMM():
    def __init__(self, window_data):
        self.features = []
        self.NE_type = ["B-PER", "I-PER", "B-LOC", "I-LOC","B-ORG", "I-ORG","B-MISC", "I-MISC", "O"]
        self.word_set = ["played", "on", "American", "League"]
        self.tag_set = ["VBD", "IN", "NNP", "NNP"]
        self.boi_set = ["O", "O", "B-MISC", "I-MISC"]
        self.boi_end_list = []
        self.window_set = window_data
        self.boi_set = map(lambda x: x[1][2], window_data)
        self.max_iter = 3
        self.fname = "../models/MaxentClassifier.pickle"

    # reference from "Named entity recognition: a maximum entropy approach using global information"
    def name_features(self, window_tuple):
        features = {}
        for word_index in range(len(window_tuple)):
            word = str(window_tuple[word_index][0])
            features["Word_" + str(word_index)] = word
            features["Tag_" + str(word_index)] = window_tuple[word_index][1]
            features["Type_" + str(word_index)] = window_tuple[word_index][2]
            features["PreviousType_" + str(word_index)] = "O" if word_index == 0 else window_tuple[word_index - 1][2]
            features["InitCapPeriod_" + str(word_index)] = word[0].isupper() and word[len(word)-1] == "."
            # features["AllCapsPeriod_" + str(word_index)] = 0
            # features["ContainDigit_" + str(word_index)] = 0
            # features["TwoD_" + str(word_index)] = 0
            # features["FourD_" + str(word_index)] = 0
            # features["DigitSlash_" + str(word_index)] = 0
            # features["Dollar_" + str(word_index)] = 0
            # features["Percent_" + str(word_index)] = 0
            # features["DigitPeriod_" + str(word_index)] = 0
            # features["FirstWord_" + str(word_index)] = 0
            # features["Date_" + str(word_index)] = 0
            # features["Corporate-Suffix_" + str(word_index)] = 0
            # features["Person-Prefix_" + str(word_index)] = 0

        return features
    def trainMEMM(self, dump):
        self.end_dic = {}
        if(os.path.isfile(self.fname)):
            classifier_file = open("../models/MaxentClassifier.pickle", "rb")
            self.maxent_classifier = pickle.load(classifier_file)
            classifier_file.close()
        else:
            train_set = [(self.name_features(t), t[1][2]) for t in self.window_set]
            self.maxent_classifier = MaxentClassifier.train(train_set, max_iter = self.max_iter)

            if dump:
                f = open("../models/MaxentClassifier.pickle", "wb")
                pickle.dump(self.maxent_classifier, f)
                f.close()
        for type in self.NE_type:
            end_num = 0
            count_num = len(list(filter(lambda x: x == type, self.boi_set)))
            for end_instance in self.boi_end_list:
                if type == end_instance:
                    end_num += 1
            prob = format(end_num / (count_num + 0.0), '.5f')
            self.end_dic[type] = prob

    def classification(self, tuple):
        testone = self.maxent_classifier.classify(self.name_features(tuple))
        print(testone)

    def viterbi_search(self, word_list, tag_list):
        tuple_list = list(zip(word_list, tag_list))
        viterbi = [[0 for x in range((len(word_list) + 1))] for y in range(len(self.NE_type))]
        back_pointer = [["" for x in range((len(word_list) + 1))] for y in range(len(self.NE_type))]

        for index, type in enumerate(self.NE_type):
            probabilities = self.maxent_classifier.prob_classify(self.name_features(tuple_list[0], "start"))
            type_prob = float(probabilities.prob(type))
            viterbi[index][1] = type_prob
            # start with 0
            back_pointer[index][1] = 0

        # for word from 2 to len(word_list)
        max_score = 0
        max_previous_prob = 0
        for w_index in range(1, len(word_list)):
            # find the max viterbi
            word = tuple_list[w_index]
            for t_index, type in enumerate(self.NE_type):
                probabilities_list = [self.maxent_classifier.prob_classify(self.name_features(word, type2)) for t_index2, type2 in enumerate(self.NE_type)]
                type_prob_list = [{"key": self.NE_type[p_index], "value":float(probabilities.prob(type)) * viterbi[p_index][w_index]} for p_index, probabilities in enumerate(probabilities_list)]
                print("posterior", type, "word", word)
                random.shuffle(type_prob_list)
                type_values = list(map(lambda x: x["value"], type_prob_list))
                type_key = list(map(lambda x: x["key"], type_prob_list))
                max_score = max(type_values)
                max_index = type_values.index(max_score)
                # print type_prob_list, max_score, max_index, type_key[max_index]
                viterbi[t_index][w_index + 1] = max_score
                back_pointer[t_index][w_index + 1] = type_key[max_index]

        end_list = [float(viterbi[t_index][len(word_list)]) * float(self.end_dic[type]) for t_index, type in enumerate(self.NE_type)]
        max_end = max(end_list)
        max_previous_state = end_list.index(max_end)

        # return path
        path = [self.NE_type[max_previous_state]]
        for x in range(len(word_list) - 1):
            path.insert(0, back_pointer[max_previous_state][len(word_list) - x])
            max_previous_state = self.NE_type.index(back_pointer[max_previous_state][len(word_list) - x])

        return path



prepocessing = prep('../Project2_resources/new_train.txt')
data = prepocessing.pre_process_memm()
#
memm_classifier = MEMM(data)
memm_classifier.trainMEMM(True)

# memm_classifier.classification(("American", "IN"))

test_prepocessing = prep('../Project2_resources/validation.txt')
test_words = test_prepocessing.pre_process_hmm()
for x in range(len(test_words[0])):
    print test_words[0][x], test_words[][x]
    memm_classifier.viterbi_search(test_words[0][x], test_words[3][x])

# memm_classifier.viterbi_search(["played", "on", "American", "League"], ["VBD", "IN", "NNP", "NNP"])
