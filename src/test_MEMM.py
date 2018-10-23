from MEMM import MEMM
from prep import prep

prepocessing = prep('../Project2_resources/train.txt')
data = prepocessing.pre_process_memm()
#

memm_classifier = MEMM(data, "validation")
memm_classifier.trainMEMM(True)

test_prepocessing = prep('../Project2_resources/validation.txt')
test_words = test_prepocessing.pre_process_memm_test()
my_prep_test = prep('../Project2_resources/validation.txt')
data_test = my_prep_test.pre_process_hmm_test()

tags = []
actual_list = []
words = []

dict = {'PER': '', 'LOC': '', 'ORG': '', 'MISC': ''}
for x in range(len(test_words)):
    print("handling:", x)
    sentence = test_words[x]
    word_list = map(lambda x: x[1][0], sentence)
    type_list = map(lambda x: x[1][2], sentence)
    position_list = map(lambda x: x[1][2], sentence)
    # print(word_list)
    # print(type_list)
    actual_list += type_list
    words.append(word_list)

    prediction = memm_classifier.viterbi_search(sentence)
    # print(prediction)
    numbers_line = data_test[2][x]
    start_num = numbers_line[0]
    prev_number = numbers_line[0]
    prev_tag = None
    for j in range(len(prediction)):
        tag = prediction[j]
        number = numbers_line[j]
        # print tag, number
        if 'PER' in tag and prev_tag != 'PER':
            if prev_tag != None:
                dict[prev_tag] += start_num + '-' + prev_number + ' '
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
        elif 'MISC' in tag and prev_tag != 'MISC':
            if prev_tag != None:
                dict[prev_tag] += start_num + '-' + prev_number + ' '
            start_num = number
            prev_tag = 'MISC'
        else:
            if prev_tag != None and tag == 'O':
                dict[prev_tag] += start_num + '-' + prev_number + ' '
                prev_tag = None

        prev_number = number

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

print(correct, predicted, golden)
precision = 1.0*correct/predicted
recall = 1.0*correct/golden
f1 = 2*precision*recall/(precision)
print("f1 score is:", f1)
