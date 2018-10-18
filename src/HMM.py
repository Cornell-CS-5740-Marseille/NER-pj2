from prep import prep
import sys
import math
import io

class HMM:
    def __init__(self):
        self.sentence_start = '<s>'
        self.sentence_end = '</s>'

    def Viterbi(self, words, data):
        '''
        Use Viterbi algorithm to determine tags of each word.
        input words-> words we need to tag.
        input data-> transition_prob: transition_prob matrix.
        input data-> generation_prob: generation_prob matrix.
        output tag_result: the results we get from HMM algorithm. (tagging task)
        '''

        transition_prob = data[1]
        generation_prob = data[2]
        tags = ['B-PER', 'I-PER', 'B-LOC', 'I-LOC', 'B-ORG', 'I-ORG', 'B-MISC', 'I-MISC', 'O']
        tag_result_output = []

        for line in words:
            score = [] #'record the score of tagging'
            BPTR = [] #'record the backward path'

            #initialization
            for i in range(len(tags)):
                s = []
                b = []
                prob_trans = transition_prob.get(self.sentence_start, 0) #'if not exists, then set 0'
                if prob_trans != 0:
                    prob_trans = prob_trans.get(tags[i], 0) #'if not exists, then set 0'
                prob_gen = generation_prob.get(tags[i], 0) #'if not exists, then set 0'
                #print prob_gen
                if prob_gen != 0:
                    #print line[0]
                    prob_gen = prob_gen.get(line[0], 0)  #'if not exists, then set 0'
                s.append((prob_trans) *(prob_gen))
                b.append(0)
                score.append(s)
                BPTR.append(b)
                # score=[ [s1], [s2], [s3], [s4]]
                # b=[ [0], [0], [0], [0]]

            #iteration
            for t in range(1, len(line)): # location
                for i in range(len(tags)): # tag at present tokens
                    max = -1
                    index = -1
                    for j in range(len(tags)): # tag of previous tokens
                        #prob = score[j][t-1] * transition_prob.get(tags[j], 0)
                        tran_prob = transition_prob.get(tags[j], 0)
                        if tran_prob != 0:
                            prob = tran_prob.get(tags[i], 0)
                        prob= score[i][t-1] * prob
                        if prob > max:
                            max = prob # Find the largest possible Viterbi(i,t)
                            index = j  # Record the j= argmax{Viterbi(i,t)}
                                       # =argmax{score[j][t-1]*T(j,i)*G(i,t)}=argmax{score[j][t-1]*T(j,i)}
                        gen_prob = generation_prob.get(tags[i], 0)
                        if gen_prob != 0:
                            gen_prob = gen_prob.get(line[t], 0)
                    score[i].append((max)*(gen_prob)) # Record the new score Viterbi(i,t)
                    BPTR[i].append(index) # Record the optimal path BPTR(i,t)

            #identify sequence

            tag_result_line = [None] * len(line) # Initiating tagging vector
            tag_result_line_index = [None] * len(line)
            max = -1
            index = -1
            for i in range(len(tags)): # Start from the last line
                if score[i][len(line)-1] > max:
                    max = score[i][len(line)-1]
                    index = i # Find the best tagging for the last word
            tag_result_line_index[len(line)-1] = index
            tag_result_line[len(line)-1] = tags[index]
            for i in range(len(line)-2, 0, -1): # Backward seek the optimal path
                tag_result_line_index[i] = BPTR[tag_result_line_index[i+1]][i+1]
                tag_result_line[i] = tags[tag_result_line_index[i]]

            tag_result_output.append(tag_result_line)

        return tag_result_output


my_prep = prep('../Project2_resources/new_train.txt')
data = my_prep.pre_process_hmm()
words=[]
with io.open("../Project2_resources/validation.txt", 'r', encoding='utf-8') as file1:
    i=1
    for line in file1:
        if i % 3 == 1:
            temp = line.split()
            words.append(temp)   #'convert a string into a list'
        i += 1

#print data[1]
model = HMM()
print(model.Viterbi(words,data))

# calculate the precision and recall rate
