import random
import os.path
from nltk.classify import MaxentClassifier
import pickle
import re
from src.prep import prep


class MEMM():
    def __init__(self, window_data, trainFileName):
        self.features = []
        self.NE_type = ["B-PER", "I-PER", "B-LOC", "I-LOC","B-ORG", "I-ORG","B-MISC", "I-MISC", "O"]
        self.word_set = ["played", "on", "American", "League"]
        self.tag_set = ["VBD", "IN", "NNP", "NNP"]
        self.window_set = window_data
        self.boi_set = map(lambda x: x[1][2], window_data)
        self.boi_end_list = map(lambda y: y[1][2], filter(lambda x: x[-1][1] == "END", window_data))
        self.max_iter = 5
        self.fname = "../models/MaxentClassifier_moreFeatures_" + trainFileName + ".pickle"

    # reference from "Named entity recognition: a maximum entropy approach using global information"
    def name_features(self, window_tuple, previousBOI):
        features = {}
        word = str(window_tuple[1][0])
        features["PreviousType"] = previousBOI
        features["InitCapPeriod"] = word[0].isupper() and word[len(word) - 1] == "."
        features["OneCap"] = len(word) == 1 and bool(re.match('^[A-Z]$', word))
        features["AllCapsPeriod"] = word.isupper() and word[len(word) - 1] == "."
        features["CharSlashDigit"] = bool(re.match('^[A-Z]+[-]*[0-9]+$', word))
        features["TwoD"] = len(word) == 2 and word[0].isdigit() and word[1].isdigit()
        features["FourD"] = len(word) == 4 and word[0].isdigit() and word[1].isdigit() and word[2].isdigit() and word[3].isdigit()
        features["DigitSlash"]= bool(re.match('^[0-9]+\/[0-9]+$', word))
        features["Dollar"] = "$" in word
        features["Percent"] = "%" in word
        features["DigitPeriod"] = bool(re.match('^[0-9]+.[0-9]*$', word))

        # First Word
        features["FirstWord"] = window_tuple[0][1] == "START"

        # Date
        features["DateName"] = word in ["January", "February", "March", "April", "May", "June", "July", "August", "September", "October", "November", "December",
                                        "Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]

        # Corporate-Suffix-List
        features["Corporate-Suffix"] = 0
        for word_index in range(len(window_tuple)):
            word = str(window_tuple[word_index][0])
            features["Word_" + str(word_index)] = word
            features["Tag_" + str(word_index)] = window_tuple[word_index][1]

            features["InitCap_" + str(word_index)] = word[0].isupper()
            features["ALLCap_" + str(word_index)] = word.isupper()
            features["MixedCap_" + str(word_index)] = word.lower() != word and word.upper() != word
            features["Corporate-Suffix"] = features["Corporate-Suffix"] or (word.lower() in ["ltd.", "ltd", "associates", "inc.", "inc", "co.", "co", "corp.", "corp"] or \
                word in ["University", "Committee", "Institute", "Commission", "Plc", "Airlines"])
        return features
    def trainMEMM(self, dump):
        self.end_dic = {}
        if(os.path.isfile(self.fname)):
            classifier_file = open(self.fname, "rb")
            self.maxent_classifier = pickle.load(classifier_file)
            classifier_file.close()
        else:
            train_set = [(self.name_features(t, t[0][2]), t[1][2]) for t in self.window_set]
            self.maxent_classifier = MaxentClassifier.train(train_set, max_iter = self.max_iter)

            if dump:
                f = open(self.fname, "wb")
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

    def precision_score(self, prediction_list, actual_list, word_list):
        precision = 0
        for x in range(len(prediction_list)):
            if prediction_list[x] != actual_list[x]:
                print("wrong prediction! word:", word_list[x] , ", prediction: ", prediction_list[x], ", truth: " , actual_list[x])
            else:
                precision += 1
        precision = precision / (len(prediction_list) + 0.0)
        return precision

    def viterbi_search(self, word_list):
        viterbi = [[0 for x in range((len(word_list) + 1))] for y in range(len(self.NE_type))]
        back_pointer = [["" for x in range((len(word_list) + 1))] for y in range(len(self.NE_type))]

        feature_0 = self.name_features(word_list[0], "START")
        for index, type in enumerate(self.NE_type):
            probabilities = self.maxent_classifier.prob_classify(feature_0)
            type_prob = float(probabilities.prob(type))
            viterbi[index][1] = type_prob
            # start with 0
            back_pointer[index][1] = 0

        # for word from 2 to len(word_list)
        for w_index in range(1, len(word_list)):
            # find the max viterbi
            word_window = word_list[w_index]
            word = word_window[1][0]
            for t_index, type in enumerate(self.NE_type):
                probabilities_list = [self.maxent_classifier.prob_classify(self.name_features(word_window, type2)) for t_index2, type2 in enumerate(self.NE_type)]
                type_prob_list = [{"key": self.NE_type[p_index], "value":float(probabilities.prob(type)) * viterbi[p_index][w_index]} for p_index, probabilities in enumerate(probabilities_list)]
                # print("posterior", type, "word", word)
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



# preprocessing = prep("../Project2_resources/train.txt")
# preprocessing.pre_process_memm()