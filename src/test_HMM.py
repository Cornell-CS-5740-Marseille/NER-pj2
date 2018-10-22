from prep import prep
from HMM import HMM

my_prep = prep('../Project2_resources/new_train.txt')
data = my_prep.pre_process_hmm()
my_prep_test = prep('../Project2_resources/validation.txt')
data_test = my_prep_test.pre_process_hmm()

data[0] = data_test[0]
data[3] = data_test[3]
model = HMM()
tags = model.Viterbi(data)

dict = {'PER': [], 'LOC': [], 'ORG': [], 'MISC': [], 'O': []}
prev_tag = None
prev_number = 0
start_num = 0
number = 0

for i in range(len(tags)):
    tags_line = tags[i]
    for j in range(len(tags_line)):
        tag = tags_line[j]
        if 'PER' in tag and prev_tag != 'PER':
            if prev_tag != None:
                dict[prev_tag].append([start_num, prev_number])
            start_num = number
            prev_tag = 'PER'
        elif 'LOC' in tag and prev_tag != 'LOC':
            if prev_tag != None:
                dict[prev_tag].append([start_num, prev_number])
            start_num = number
            prev_tag = 'LOC'
        elif 'ORG' in tag and prev_tag != 'ORG':
            if prev_tag != None:
                dict[prev_tag].append([start_num, prev_number])
            start_num = number
            prev_tag = 'ORG'
        elif 'MISC' in tag and prev_tag != 'MISC':
            if prev_tag != None:
                dict[prev_tag].append([start_num, prev_number])
            start_num = number
            prev_tag = 'MISC'
        else:
            if tag == 'O' and prev_tag != 'O':
                if prev_tag != None:
                    dict[prev_tag].append([start_num, prev_number])
                start_num = number
                prev_tag = 'O'
        prev_number = number
        number += 1
#print dict

dict_0 = dict

dict = {'PER': [], 'LOC': [], 'ORG': [], 'MISC': [], 'O': []}
prev_tag = None
prev_number = 0
start_num = 0
number = 0
tags = data[3]

for i in range(len(tags)):
    tags_line = tags[i]
    for j in range(len(tags_line)):
        tag = tags_line[j]
        if 'PER' in tag and prev_tag != 'PER':
            if prev_tag != None:
                dict[prev_tag].append([start_num, prev_number])
            start_num = number
            prev_tag = 'PER'
        elif 'LOC' in tag and prev_tag != 'LOC':
            if prev_tag != None:
                dict[prev_tag].append([start_num, prev_number])
            start_num = number
            prev_tag = 'LOC'
        elif 'ORG' in tag and prev_tag != 'ORG':
            if prev_tag != None:
                dict[prev_tag].append([start_num, prev_number])
            start_num = number
            prev_tag = 'ORG'
        elif 'MISC' in tag and prev_tag != 'MISC':
            if prev_tag != None:
                dict[prev_tag].append([start_num, prev_number])
            start_num = number
            prev_tag = 'MISC'
        else:
            if tag == 'O' and prev_tag != 'O':
                if prev_tag != None:
                    dict[prev_tag].append([start_num, prev_number])
                start_num = number
                prev_tag = 'O'
        prev_number = number
        number += 1
#print dict
dict_1 = dict

correct = 0
predicted = 0
golden = 0
for key in dict_0:
    if key != 'O':
        arr1 = dict_0[key]
        arr2 = dict_1[key]
        i = 0
        j = 0
        while i < len(arr1) and j < len(arr2):
            if arr1[i][0] == arr2[j][0] and arr1[i][1] == arr2[j][1]:
                correct += 1
                i += 1
                j += 1
                predicted += 1
                golden += 1
            elif arr1[i][0] == arr2[j][0] and arr1[i][1] != arr2[j][1]:
                i += 1
                j += 1
                predicted += 1
                golden += 1
            elif arr1[i][0] < arr2[j][0]:
                i += 1
                predicted += 1
            elif arr1[i][0] > arr2[j][0]:
                j += 1
                golden += 1
        if i != len(arr1) - 1:
            predicted += len(arr1) - 1 - i
        if j != len(arr2) - 1:
            golden += len(arr2) - 1 - j

print correct, predicted, golden
precision = 1.0*correct/predicted
recall = 1.0*correct/golden
f1 = 2*precision*recall/(precision+recall)
print precision, recall, f1
# sum = 0
# correct = 0
# for i in range(len(data[3])):
#     for j in range(len(data[3][i])):
#         if tag[i][j] == data[3][i][j]:
#             correct = correct + 1
#         sum = sum + 1
#
# print correct
# print sum
# accuracy = float(correct) / float(sum)
# print accuracy