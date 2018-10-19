from prep import prep
from HMM import HMM

my_prep = prep('../Project2_resources/new_train.txt')
data = my_prep.pre_process_hmm()
my_prep_test = prep('../Project2_resources/validation.txt')
data_test = my_prep_test.pre_process_hmm()

data[0] = data_test[0]
data[3] = data_test[3]
model = HMM()
tag = model.Viterbi(data)
sum = 0
correct = 0
for i in range(len(data[3])):
    for j in range(len(data[3][i])):
        sum = sum + 1
        if tag[i][j] == data[3][i][j]:
            correct = correct + 1

print correct
print sum
accuracy = float(correct) / float(sum)
print accuracy