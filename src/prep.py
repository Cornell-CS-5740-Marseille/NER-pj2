import math
import json
import csv
import pandas as pd
class prep:
    def __init__(self, file1):
        self.file1 = open(file1) #'original dataset'
        self.sentence_start = '<s>'
        self.sentence_end = '</s>'

    def divide_into_validation(self, validate_percent):
        '''
        Find validation set and training set and write to file:
        input validate_percent: decide what percent of data will be used for validation
        output: new_validation_file
                new_train_file
        '''
        line_count = 0
        sentence_count = 0
        sentence = math.floor(1/validate_percent) #'use the mod to decide the sentence'

        new_train_file = open("../Project2_resources/new_train.txt", "w")
        new_validation_file = open("../Project2_resources/validation.txt", "w")
        for line in self.file1:
            if sentence_count % sentence == 0:
                new_validation_file.write(line)
            else:
                new_train_file.write(line)

            if line_count % 3 == 2: #'At the end line of a sentence token '
                sentence_count += 1
            line_count +=1

    def cross_validation(self, fold=5):
        '''
        Find validation set and training set and write to file using cross validation method
        input fold: how many folds we need, default is 5.
        output: cross_validation_'k'
                cross_train_file_'k'
        '''
        line_count = 0
        sentence_count = 0

        new_train_file = open("../Project2_resources/cross_train"+ str(k) + ".txt", "w")
        for k in range(fold):
            new_validation_file = open("../Project2_resources/cross_validation_" + str(k) + ".txt", "w")
            for line in self.file1:
                if sentence_count % sentence == k:
                    new_validation_file.write(line)
                else:
                    new_train_file.write(line)

    def convert_table_to_prob(self, table):
        '''
        Convert the times matrix into probability matrix
        input table: dictionary counting times of each index appearing
            table[word1][word2]= times word2 appearing after word1
        output table: dictionary calculating probability of each index
            probtable[word1][word2]= times word2 appearing after word1/ total appearance of word1
        '''
        for key in table:
            count = 0
            for value in table[key]:
                count += table[key][value]
            for value in table[key]:
                table[key][value] = table[key][value]*1.0/count
        return table

    def pre_process_hmm(self):
        '''
        Generate the information that HMM algorithm need
        output sentence:
        output transition_prob: probability probability dictionary using function convert_table_to_prob
        output generation_prob: probability probability dictionary using function convert_table_to_prob

        '''
        transition_table = {}
        generation_table = {}
        words = []
        tags = []
        sentence = []
        line_count = 0

        for line in self.file1: #'Using training dataset'
            if line_count % 3 == 0:  #'line of original sentence'
                words = line.split()
                sentence.append(words)   #'convert a string into a list'
            elif line_count % 3 == 2: #'line of IOB tagging'
                tags = line.split()
                prev_tag = self.sentence_start   #'initiating with <s> tag'
                for i in range(len(tags)):
                    tag = tags[i]
                    word = words[i]
                    transition_table.setdefault(prev_tag, {})
                    transition_table[prev_tag].setdefault(tag, 0)
                    transition_table[prev_tag][tag] += 1
                    prev_tag = tag    #'move forward, update the tag'
                    generation_table.setdefault(tag, {})
                    generation_table[tag][word]=generation_table[tag].setdefault(word, 0)+1
            line_count += 1

        #'Finally calculate the prob matrix using appearing times'
        transition_prob = self.convert_table_to_prob(transition_table)
        generation_prob = self.convert_table_to_prob(generation_table)
        # with open('transition_prob', 'w') as fp:
        #     json.dump(transition_prob, fp)
        # with open("generation_prob", 'w') as fp:
        #     json.dump(generation_prob, fp)
        df = pd.DataFrame(transition_prob)
        df.to_csv('transition_prob.csv')
        with open('generation_prob.csv', 'w') as csv_file:
            writer = csv.writer(csv_file)
            for key, value in generation_prob.items():
                writer.writerow([key, value])
        return [sentence, transition_prob, generation_prob]

    def isCapital(self, word):
        '''
        Return 1 if the word is capitalized.
        '''
        return 1 if len(word) > 0 and word[0].isupper() else 0

    def pre_process_memm(self):
        '''
        Generate features for MEMM algorithm
        input file1: training data file
        output:
        '''
        output = []
        line_count = 0

        for line in self.file1:
            if line_count % 3 == 0: #'original sentence lists'
                words = line.split()
            elif line_count % 3 == 1: #'POS tag lists'
                pos = line.split()
            else:
                tags = line.split() #'IOB tag lists'
                for i in range(len(tags)):
                    item = [words[i], pos[i], tags[i]]
                    #'Feature Group 1: words capitilized or not'
                    feature_prev_word_cap = self.isCapital(words[i-1]) if i > 0 else 0
                    feature_curr_word_cap = self.isCapital(words[i])
                    feature_next_word_cap = self.isCapital(words[i+1]) if i < len(tags)-1 else 0

                    features = []
                    features.append(feature_prev_word_cap)
                    features.append(feature_curr_word_cap)
                    features.append(feature_next_word_cap)
                    item.append(features) #'For each word we have several features'
                    #'item(i) = [word(i), pos(i), tags(i), prev_word(i), ...]'
                    output.append(item)

            line_count += 1
        return output


#my_prep = prep('../Project2_resources/new_train.txt')
#my_prep.pre_process_hmm()
#my_prep.pre_process_memm()
