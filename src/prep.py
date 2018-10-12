import math

class prep:
    def __init__(self, file1):
        self.file1 = open(file1)

    def divide_into_validation(self, validate_percent):
        line_count = 0
        sentence_count = 0
        sentence = math.floor(1/validate_percent)

        new_train_file = open("../Project2_resources/new_train.txt", "w")
        new_validation_file = open("../Project2_resources/validation.txt", "w")
        for line in self.file1:
            if sentence_count%sentence == 0:
                new_validation_file.write(line)
            else :
                new_train_file.write(line)

            if line_count%3 == 2:
                sentence_count += 1
            line_count +=1

my_prep = prep('../Project2_resources/train.txt')
my_prep.divide_into_validation(0.2)