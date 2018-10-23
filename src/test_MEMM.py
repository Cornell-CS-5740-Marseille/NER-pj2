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
words = []
def findTheResult(numberList, tagList):
    templeDict = {'PER': [], 'LOC': [], 'ORG': [], 'MISC': [], 'O': []}
    start_num = numberList[0]
    prev_number = numberList[0]
    number = 0
    prev_tag = None
    for j in range(len(tagList)):
        tag = tagList[j]
        if 'PER' in tag and prev_tag != 'PER':
            if prev_tag != None:
                templeDict[prev_tag].append([start_num, prev_number])
            start_num = number
            prev_tag = 'PER'
        elif 'LOC' in tag and prev_tag != 'LOC':
            if prev_tag != None:
                templeDict[prev_tag].append([start_num, prev_number])
            start_num = number
            prev_tag = 'LOC'
        elif 'ORG' in tag and prev_tag != 'ORG':
            if prev_tag != None:
                templeDict[prev_tag].append([start_num, prev_number])
            start_num = number
            prev_tag = 'ORG'
        elif 'MISC' in tag and prev_tag != 'MISC':
            if prev_tag != None:
                templeDict[prev_tag].append([start_num, prev_number])
            start_num = number
            prev_tag = 'MISC'
        else:
            if tag == 'O' and prev_tag != 'O':
                if prev_tag != None:
                    templeDict[prev_tag].append([start_num, prev_number])
                start_num = number
                prev_tag = 'O'
        prev_number = number
        number += 1
    return templeDict


dictResult = {'PER': {"correct": 0, "predicted": 0, "golden": 0}, 'LOC': {"correct": 0, "predicted": 0, "golden": 0}, 'ORG': {"correct": 0, "predicted": 0, "golden": 0}, 'MISC': {"correct": 0, "predicted": 0, "golden": 0}}
correct = 0
predicted = 0
golden = 0
for x in range(len(test_words)):
    print("handling:", x)
    sentence = test_words[x]
    word_list = list(map(lambda x: x[1][0], sentence))
    type_list = list(map(lambda x: x[1][2], sentence))
    # print(word_list)
    # print(type_list)

    prediction = memm_classifier.viterbi_search(sentence)
    # print(prediction)

    for x in range(len(prediction)) :
        prediction_type = prediction[x].split("-")[1] if prediction[x] != "O" else "O"
        actual_type = type_list[x].split("-")[1] if type_list[x] != "O" else "O"
        if prediction_type != 'O' or actual_type != 'O':
            if prediction[x] == type_list[x]:
                dictResult[prediction_type]["correct"] += 1
                dictResult[prediction_type]["predicted"] += 1
                dictResult[prediction_type]["golden"] += 1
            elif prediction_type == 'O':
                print(word_list[x], prediction[x], type_list[x])
                dictResult[actual_type]["golden"] += 1
            elif actual_type == 'O':
                print(word_list[x], prediction[x], type_list[x])
                dictResult[prediction_type]["predicted"] += 1
            else:
                print(word_list[x], prediction[x], type_list[x])
                dictResult[prediction_type]["predicted"] += 1
                dictResult[actual_type]["golden"] += 1

    # print(dictResult)
for key in dictResult:
    golden += dictResult[key]["golden"]
    correct += dictResult[key]["correct"]
    predicted += dictResult[key]["predicted"]

print(correct, predicted, golden)
precision = 1.0 * correct / predicted
recall = 1.0 * correct / golden
f1 = 2 * precision * recall/(precision + recall)
print("precision:", precision)
print("recall", recall)
print("f1 score is:", f1)
