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
# print data_test[2]

dict = {'PER': '', 'LOC': '', 'ORG': '', 'MISC': ''}
prev_tag = None
prev_number = 0
start_num = 0
print tags
for i in range(len(tags)):
    tags_line = tags[i]
    numbers_line = data_test[2][i]
    for j in range(len(tags_line)):
        tag = tags_line[j]
        number = numbers_line[j]
        #print tag, number
        if 'PER' in tag and prev_tag != 'PER':
            if prev_tag != None:
                dict[prev_tag] += start_num + '-'+ prev_number + ' '
            start_num = number
            prev_tag = 'PER'
        elif 'LOC' in tag and prev_tag != 'LOC':
            if prev_tag != None:
                dict[prev_tag] += start_num + '-' + prev_number + ' '
            start_num = number
            prev_tag = 'LOC'
        elif 'ORG' in tag and prev_tag != 'ORG':
            if prev_tag != None:
                dict[prev_tag] += start_num + '-' + prev_number + ' '
            start_num = number
            prev_tag = 'ORG'
        else:
            if prev_tag != 'MISC':
                if prev_tag != None:
                    dict[prev_tag] += start_num + '-' + prev_number + ' '
                start_num = number
                prev_tag = 'MISC'
        prev_number = number
#print dict

with open('../output/speech_classification.csv', mode='w') as test_output:
    speech_writer = csv.writer(test_output, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
    speech_writer.writerow(['Type', 'Prediction'])

    speech_writer.writerow(['PER', dict['PER']])
    speech_writer.writerow(['LOC', dict['LOC']])
    speech_writer.writerow(['ORG', dict['ORG']])
    speech_writer.writerow(['MISC', dict['MISC']])
