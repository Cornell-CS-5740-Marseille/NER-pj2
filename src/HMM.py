from src.prep import prep

class HMM:
    def __init__(self):
        self.sentence_start = '<s>'
        self.sentence_end = '</s>'



    def Viterbi(self,data):
        words = data[0]
        transition_prob = data[1]
        generation_prob = data[2]
        tags = ['B-PER', 'I-PER', 'B-LOC', 'I-LOC', 'B-ORG', 'I-ORG', 'B-MISC', 'I-MISC', 'O']
        tag_result_output = []

        for line in range(len(words)):
            score = []
            BPTR = []

            #initialization
            for i in range(len(tags)):
                s = []
                b = []
                prob_trans = transition_prob.get(self.sentence_start, 0).get(tags[i], 0)
                prob_gen = generation_prob.get(tags[i], 0).get(line[0], 0)
                s.append(prob_trans*prob_gen)
                score.append(s)
                b.append(0)
                BPTR.append(b)

            #iteration
            for t in range(1, len(line)):
                for i in range(len(tags)):
                    max = -1
                    index = -1
                    for j in range(len(tags)):
                        prob = score[j][t-1] * transition_prob.get(tags[j], 0).get(tags[i], 0)
                        if prob > max:
                            max = prob
                            index = j
                    gen_prob = generation_prob.get(tags[i], 0).get(line[t], 0)
                    score[i].append(max * gen_prob)
                    BPTR[i].append(index)

            #identify sequence
            tag_result_line = [None] * len(line)
            tag_result_line_index = [None] * len(line)
            max = -1
            index = -1
            for i in range(len(tags)):
                if score[i][len(line)-1] > max:
                    max = score[i][len(line)-1]
                    index = i
            tag_result_line_index[len(line)-1] = index
            tag_result_line[len(line)-1] = tags[index]
            for i in range(len(line)-2, 0, -1):
                tag_result_line_index[i] = BPTR[tag_result_line_index[i+1]][i+1]
                tag_result_line[i] = tags[tag_result_line_index]

            tag_result_output.append(tag_result_line)

        return tag_result_output


my_prep = prep('../Project2_resources/validation.txt')
data = my_prep.pre_process_hmm()
model = HMM()
print model.Viterbi(data)

