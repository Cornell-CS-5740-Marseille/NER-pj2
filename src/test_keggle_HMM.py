from prep import prep
from HMM import HMM
import csv

my_prep = prep('../Project2_resources/train.txt')
data = my_prep.pre_process_hmm()
my_prep_test = prep('../Project2_resources/test.txt')
data_test = my_prep_test.pre_process_hmm_test()

data[0] = data_test[0]
model = HMM()
tags = model.Viterbi(data)
print data[1]
# print data_test[2]

per = ''
loc = ''
org = ''
misc = ''
prev_tag = None
prev_number = 0

for i in range(len(tags)):
    tags_line = tags[i]
    numbers_line = data_test[2][i]
    for j in range(len(tags_line)):
        tag = tags_line[j]
        number = numbers_line[j]
        if 'PER' in tag:
            if prev_tag != 'per':




with open('../output/speech_classification.csv', mode='w') as test_output:
    speech_writer = csv.writer(test_output, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
    speech_writer.writerow(['Type', 'Prediction'])
    # for i in range(len(result_arr)):
    #     speech_writer.writerow([i, result_arr[i]])
