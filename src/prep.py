import math


class prep:
    def __init__(self, file1):
        self.file1 = open(file1)
        self.sentence_start = '<s>'
        self.sentence_end = '</s>'
        self.allwords = set()
        self.alltags = ['O', 'B-PER', 'I-PER', 'B-LOC', 'I-LOC', 'B-ORG', 'I-ORG', 'B-MISC', 'I-MISC']

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
            table[key]['<unk>'] = 0
            for value in table[key]:
                if table[key][value] == 1:
                    table[key]['<unk>'] += 1
            for target in self.allwords:
                if target in table[key]:
                    table[key][target] += k
                else:
                    table[key][target] = k
        return table

    def table_add_k_smooth_table(self, table, k):
        for key in table:
            table[key]['<unk>'] = 0
            for value in table[key]:
                if table[key][value] == 1:
                    table[key]['<unk>'] += 1
            for target in self.alltags:
                if target in table[key]:
                    table[key][target] += k
                else:
                    table[key][target] = k
        return table

    def dist_table_smoothed_kneser_ney(self, table):
        reverse_dict = {}
        smoothed_count_table = {}
        all_unique_pair_counter = 0

        for key in table:
            table[key]['<unk>'] = 0
            table[key]['<a>'] = 0
            for value in table[key]:
                if table[key][value] == 1:
                    table[key]['<unk>'] += 1

        for key in table:
            for value in table[key]:
                table[key]['<a>'] += 1
                if value in reverse_dict:
                    if key in reverse_dict[value]:
                        reverse_dict[value][key] += 1
                    else:
                        reverse_dict[value][key] = 1
                    reverse_dict[value]['<a>'] += 1
                else:
                    reverse_dict[value] = {key: 1}
                    reverse_dict[value]['<a>'] = 1

        for key in table:
            all_unique_pair_counter += len(table[key]) - 1  # minus the <a> in the dict

        discount = 0.01
        for key in table:
            smoothed_count_table[key] = {}
            count_dict = table[key]
            for target in self.allwords:
                if target != '<a>':
                    count = 0
                    if target in count_dict:
                        count = count_dict[target]
                    percentage_after_discount = max(count - discount, 0) / \
                                                (count_dict['<a>'] + 0.0)
                    normalized = discount * len(count_dict) / (count_dict['<a>'] + 0.0)
                    reverse_count = 0
                    if target in reverse_dict:
                        reverse_count = len(reverse_dict[target]) - 1
                    prev = reverse_count / (all_unique_pair_counter + 0.0)
                    smoothed_prob = percentage_after_discount + normalized * prev
                    smoothed_count_table[key][target] = smoothed_prob
        table = smoothed_count_table
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
        self.allwords.add(self.sentence_start)
        self.allwords.add(self.sentence_end)

        for line in self.file1:
            if line_count % 3 == 0:
                words = line.split()
            elif line_count % 3 == 2:
                tags = line.split()
                prev_tag = self.sentence_start
                for i in range(len(tags)):
                    tag = tags[i]
                    word = words[i].lower()
                    self.allwords.add(word)
                    #if prev_tag != self.sentence_start:
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

                if len(tags) > 1 and prev_tag in transition_table:
                    transition_table[prev_tag][self.sentence_end] = transition_table[prev_tag][self.sentence_end] + 1 \
                        if self.sentence_end in transition_table[prev_tag] else 1

                tags.insert(0, self.sentence_start)
                tags.append(self.sentence_end)
                words.insert(0, self.sentence_start)
                words.append(self.sentence_end)
                tagtag.append(tags)
                sentence.append(words)

            line_count += 1

        #transition_table = self.table_add_k_smooth_table(transition_table, 0.01)
        generation_table = self.table_add_k_smooth(generation_table, 0.1)
        #print generation_table
        #generation_table = self.dist_table_smoothed_kneser_ney(generation_table)
        transition_prob = self.convert_table_to_prob(transition_table)
        generation_prob = self.convert_table_to_prob(generation_table)
        #print transition_prob
        #print generation_prob
        return [sentence, transition_prob, generation_prob, tagtag, self.allwords]

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
                    # if tags[i] != 'O':
                    #     print(words[i], pos[i], tags[i])
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

    def pre_process_memm_test(self):
        output = []
        line_count = 0

        for line in self.file1:
            if line_count % 3 == 0:
                words = line.split()
            elif line_count % 3 == 1:
                pos = line.split()
            else:
                tags = line.split()
                sentence = []
                for i in range(len(tags)):
                    window_words = []
                    if i == 0:
                        window_words.append(["<s>", "START", "START"])
                    else:
                        window_words.append([words[i-1], pos[i-1], tags[i-1]])
                    window_words.append([words[i], pos[i], tags[i]])
                    if i == len(tags) - 1:
                        window_words.append(["</s>", "END", "END"])
                    else:
                        window_words.append([words[i + 1], pos[i + 1], tags[i + 1]])
                    sentence.append(window_words)
                output.append(sentence)

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

#my_prep = prep('../Project2_resources/new_train.txt')
#my_prep.pre_process_hmm()
#my_prep.pre_process_memm()
#my_prep.generate_baseline()
#print my_prep.allwords