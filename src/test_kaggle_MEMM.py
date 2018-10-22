import csv
from src.MEMM import MEMM
from src.prep import prep

prepocessing = prep('../Project2_resources/train.txt')
data = prepocessing.pre_process_memm()
#

memm_classifier = MEMM(data)
memm_classifier.trainMEMM(True)

test_prepocessing = prep('../Project2_resources/test.txt')
test_words = test_prepocessing.pre_process_memm_test()
my_prep_test = prep('../Project2_resources/test.txt')
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




with open('../output/speech_classification_MEMM_features.csv', mode='w') as test_output:
    speech_writer = csv.writer(test_output, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
    speech_writer.writerow(['Type', 'Prediction'])

    speech_writer.writerow(['PER', dict['PER']])
    speech_writer.writerow(['LOC', dict['LOC']])
    speech_writer.writerow(['ORG', dict['ORG']])
    speech_writer.writerow(['MISC', dict['MISC']])
