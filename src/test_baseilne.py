from prep import prep
from HMM import HMM

my_prep = prep('../Project2_resources/new_train.txt')
data = my_prep.pre_process_hmm()
my_prep_test = prep('../Project2_resources/validation.txt')
data_test = my_prep_test.pre_process_hmm()
correct = 0
sum = 0
baseline = my_prep.generate_baseline()
for i in range(len(data_test[3])):
    for j in range(len(data_test[3][i])):
        sum += 1
        tag = ''
        if data_test[0][i][j] in baseline:
            tag = baseline[data_test[0][i][j]]
        else:
            tag = 'O'
        if tag == data_test[3][i][j]:
            correct = correct + 1

print correct
print sum
accuracy = float(correct) / float(sum)
print accuracy