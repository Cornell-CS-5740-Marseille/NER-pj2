import math


class prep:
    def __init__(self, file1):
        self.file1 = open(file1)
        self.sentence_start = '<s>'
        self.sentence_end = '</s>'
        self.allwords = set()

    def divide_into_validation(self, validate_percent):
        line_count = 0
        sentence_count = 0
        sentence = math.floor(1/validate_percent)

        new_train_file = open("../Project2_resources/new_train.txt", "w")
        new_validation_file = open("../Project2_resources/validation.txt", "w")
        for line in self.file1:
            if sentence_count % sentence == 0:
                new_validation_file.write(line)
            else:
                new_train_file.write(line)

            if line_count % 3 == 2:
                sentence_count += 1
            line_count +=1

    def convert_table_to_prob(self, table):
        for key in table:
            count = 0
            for value in table[key]:
                count += table[key][value]
            for value in table[key]:
                table[key][value] = table[key][value]*1.0/count
        return table

    def table_add_k_smooth(self, table, k):
        for key in table:
            for target in self.allwords:
                if target in table[key]:
                    table[key][target] += k
                else:
                    table[key][target] = k
        return table

    def pre_process_hmm_test(self):
        line_count = 0
        sentence = []
        pos = []
        number = []
        for line in self.file1:
            if line_count % 3 == 0:
                words = line.split()
                sentence.append(words)
            elif line_count % 3 == 1:
                words = line.split()
                pos.append(words)
            else:
                words = line.split()
                number.append(words)
            line_count += 1
        return [sentence, pos, number]

    def pre_process_hmm(self):
        transition_table = {}
        generation_table = {}
        words = []
        tagtag = []
        sentence = []
        line_count = 0

        for line in self.file1:
            if line_count % 3 == 0:
                words = line.split()
                sentence.append(words)
            elif line_count % 3 == 2:
                tags = line.split()
                tagtag.append(tags)
                prev_tag = self.sentence_start
                for i in range(len(tags)):
                    tag = tags[i]
                    word = words[i]
                    self.allwords.add(word)
                    if prev_tag != self.sentence_start:
                        if prev_tag in transition_table:
                            transition_table[prev_tag][tag] = transition_table[prev_tag][tag] + 1 \
                                if tag in transition_table[prev_tag] else 1
                        else:
                            transition_table[prev_tag] = {tag: 1}
                    prev_tag = tag
                    if tag in generation_table:
                        generation_table[tag][word] = generation_table[tag][word] + 1 \
                            if word in generation_table[tag] else 1
                    else:
                        generation_table[tag] = {word: 1}
            line_count += 1

        generation_table = self.table_add_k_smooth(generation_table, 0.001)
        transition_prob = self.convert_table_to_prob(transition_table)
        generation_prob = self.convert_table_to_prob(generation_table)
        return [sentence, transition_prob, generation_prob, tagtag]

    def isCapital(self, word):
        return 1 if len(word) > 0 and word[0].isupper() else 0

    def pre_process_memm(self):
        output = []
        line_count = 0

        for line in self.file1:
            if line_count % 3 == 0:
                words = line.split()
            elif line_count % 3 == 1:
                pos = line.split()
            else:
                tags = line.split()
                for i in range(len(tags)):
                    item = [words[i], pos[i], tags[i]]
                    feature_prev_word_cap = self.isCapital(words[i-1]) if i > 0 else 0
                    feature_curr_word_cap = self.isCapital(words[i])
                    feature_next_word_cap = self.isCapital(words[i+1]) if i < len(tags)-1 else 0
                    features = []
                    features.append(feature_prev_word_cap)
                    features.append(feature_curr_word_cap)
                    features.append(feature_next_word_cap)
                    item.append(features)
                    output.append(item)

            line_count += 1
        return output

    def generate_baseline(self):
        words = []
        baseline = {}
        line_count = 0

        for line in self.file1:
            if line_count % 3 == 0:
                words = line.split()
            elif line_count % 3 == 2:
                tags = line.split()
                for i in range(len(tags)):
                    tag = tags[i]
                    word = words[i]
                    if word in baseline:
                        if tag in baseline[word]:
                            baseline[word][tag] += 1
                        else:
                            baseline[word][tag] = 1
                    else:
                        baseline[word] = {tag : 1}
            line_count += 1

        for key in baseline:
            max = 0
            word = ''
            for value in baseline[key]:
                if baseline[key][value] > max:
                    max = baseline[key][value]
                    word = value
            baseline[key] = word
        return baseline

my_prep = prep('../Project2_resources/new_train.txt')
#print my_prep.pre_process_hmm()
#my_prep.pre_process_memm()
#my_prep.generate_baseline()
#print my_prep.allwords