import math


class prep:
    def __init__(self, file1):
        self.file1 = open(file1)
        self.sentence_start = '<s>'
        self.sentence_end = '</s>'

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

    def pre_process_hmm(self):
        transition_table = {}
        generation_table = {}
        words = []
        tags = []
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
                    window_words = []
                    if i == 0:
                        window_words.append(["<s>", "START", "O"])
                    else:
                        window_words.append([words[i-1], pos[i-1], tags[i-1]])
                    window_words.append([words[i], pos[i], tags[i]])
                    if i == len(tags) - 1:
                        window_words.append(["</s>", "END", "O"])
                    else:
                        window_words.append([words[i + 1], pos[i + 1], tags[i + 1]])
                    output.append(window_words)

            line_count += 1
        return output


#my_prep = prep('../Project2_resources/new_train.txt')
#my_prep.pre_process_hmm()
#my_prep.pre_process_memm()