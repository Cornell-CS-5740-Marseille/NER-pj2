import random

from nltk.classify import MaxentClassifier
import pickle
class MEMM():
    def __init__(self):
        self.features = []
        self.NE_type = ["B-PER", "I-PER", "B-LOC", "I-LOC","B-ORG", "I-ORG","B-MISC", "I-MISC", "O"]
        self.word_set = ["played", "on", "American", "League"]
        self.tag_set = ["VBD", "IN", "NNP", "NNP"]
        self.boi_set = ["O", "O", "B-MISC", "I-MISC"]
        self.boi_end_list = []
        self.max_iter = 3

    # reference from http://delivery.acm.org/10.1145/1080000/1072253/p25-chieu.pdf?ip=128.84.127.125&id=1072253&acc=OPEN&key=7777116298C9657D%2EB493315FA1EC298D%2E4D4702B0C3E38B35%2E6D218144511F3437&__acm__=1539378064_64f6873daee7f2c43e003c5d77a55927
    def name_features(self, window_tuple, previous_BOI, window_feature):
        features = {}
        for word_index in range(len(window_tuple)):
            word = window_tuple[word_index][0]
            features["Word_" + str(word_index)] = window_tuple[0]
            features["Tag_" + str(word_index)] = window_tuple[0]
            features["PreviousType_" + str(word_index)] = window_tuple[0]
            features["InitCapPeriod_" + str(word_index)] = 0
            features["AllCapsPeriod_" + str(word_index)] = 0
            features["ContainDigit_" + str(word_index)] = 0
            features["TwoD_" + str(word_index)] = 0
            features["FourD_" + str(word_index)] = 0
            features["DigitSlash_" + str(word_index)] = 0
            features["Dollar_" + str(word_index)] = 0
            features["Percent_" + str(word_index)] = 0
            features["DigitPeriod_" + str(word_index)] = 0
            features["FirstWord_" + str(word_index)] = 0
            features["Date_" + str(word_index)] = 0
            features["Corporate-Suffix_" + str(word_index)] = 0
            features["Person-Prefix_" + str(word_index)] = 0

        return features
    def trainMEMM(self, dump):
        tuples = zip(self.word_set, self.tag_set, self.boi_set, ["start", "O", "O", "B-MISC"])
        train_set = [(self.name_features(t, t[3]), t[2]) for t in tuples]
        self.maxent_classifier = MaxentClassifier.train(train_set, max_iter = self.max_iter)
        self.end_dic = {}
        for type in self.NE_type:
            end_num = 0
            count_num = len(list(filter(lambda x: x == type, self.boi_set)))
            for end_instance in self.boi_end_list:
                if type == end_instance:
                    end_num += 1
            prob = format(end_num / (count_num + 0.0), '.5f')
            self.end_dic[type] = prob

        if dump:
            f = open("../models/MaxentClassifier.pickle", "wb")
            pickle.dump(self.maxent_classifier, f)
            f.close()

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

memm_classifier = MEMM()
memm_classifier.trainMEMM(False)
# memm_classifier.classification(("American", "IN"))
# print memm_classifier.viterbi_search(["played", "on", "American", "League"], ["VBD", "IN", "NNP", "NNP"])
