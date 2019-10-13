import os


def file_name_viewer(file_name):
    for name in file_name:
        print(name)

    print(len(file_name))


def read_all_file_name(filepath):
    files = []
    for i in os.listdir(filepath):
        if i.endswith(".txt"):
            files.append(i)
    return files


# Driver


training_pos_file_name = read_all_file_name("../movie-review-HW2/aclImdb/train/pos")
training_neg_file_name = read_all_file_name("../movie-review-HW2/aclImdb/train/neg")

file_name_viewer(training_neg_file_name)
