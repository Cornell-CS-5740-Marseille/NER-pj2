NE_type = ["PER", "LOC", "ORG", "MISC"]
feature_list = []


def load_input(input):
    word_list = []
    tags_list = []
    boi_list = []
    with open(input) as input_file:
        line_number = 0
        for line in input_file:
            line = line.rstrip()
            if "------------------------" in line:
                continue
            if line_number % 3 == 0:
                words = line.split("\t")
                # print words
            elif line_number % 3 == 1:
                tags = line.split("\t")
            else:
                bois = line.split("\t")
                word_list += words
                tags_list += tags
                boi_list += bois

                items = zip(word_list, tags_list, boi_list, previous_BOI)
                labeled_features.append(item)
                previous_BOI = boi
            line_number += 1

    return (word_list, tags_list, boi_list)


# calculate the prior (End|state) = C(state, End)/C(state)
def prior_probability(boi_list):
    for boi in BOI_list:
        for j in range(len(boi_end_list)):
            for f in boi_full_list:
                if j == 0:
                    if i == f:
                        countTag = countTag + 1
            if i == boi_end_list[j]:
                countEnd = countEnd + 1
    ProbE = format(countEnd / (countTag * 1.0), '.5f')
    dicE.update({i: {"END": ProbE}})

    countEnd = 0
    countTag = 0


print load_input("../Project2_resources/train.txt")