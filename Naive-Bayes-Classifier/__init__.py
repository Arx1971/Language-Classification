import os
import re
from string import punctuation


def file_name_viewer(file_name):
    for name in file_name:
        print(name)

    print(len(file_name))


def dictionary_frequency_viewer(dictionary):
    for values, key in dictionary.items():
        print(values, key)


def read_all_file_name(filepath):
    files = []
    for i in os.listdir(filepath):
        if i.endswith(".txt"):
            files.append(i)
    return files, len(files)


def set_vocabulary(review, filepath):
    vocabulary = dict()
    for review_name in review:
        with open(filepath + review_name, "r") as reviews:
            review = reviews.read()
            words = re.split(r'[\s+`=~!@#$%^&*()_+\[\]{};\--\\:"|<,./<>?^]', review)
            for word in words:
                word = word.lower()
                word = word.strip(punctuation)
                if len(word) is not 0:
                    if word in vocabulary:
                        vocabulary[word] += 1
                    else:
                        vocabulary[word] = 1
    return vocabulary


def merge_vocabulary(vocabulary_1, vocabulary_2):
    return {**vocabulary_1, **vocabulary_2}

def naive_byes_classifier_bag_of_words_model(vocabulary):

# Driver

small_training_action_corpus = read_all_file_name("../small_corpus/train/action")
small_training_comedy_corpus = read_all_file_name("../small_corpus/train/comedy")
small_test_corpus = read_all_file_name("../small_corpus/test")
small_action_corpus_vocabulary = set_vocabulary(small_training_action_corpus[0], "../small_corpus/train/action/")
small_comedy_corpus_vocabulary = set_vocabulary(small_training_comedy_corpus[0], "../small_corpus//train/comedy/")
small_corpus_vocabulary = merge_vocabulary(small_action_corpus_vocabulary, small_comedy_corpus_vocabulary)


# training_pos_file_name = read_all_file_name("../movie-review-HW2/aclImdb/train/pos")
# training_neg_file_name = read_all_file_name("../movie-review-HW2/aclImdb/train/neg")
#
# dictionary = training_vocabulary(training_neg_file_name, training_pos_file_name, "../movie-review-HW2/aclImdb/train/")
# dictionary_frequency_viewer(dictionary)
# print(len(dictionary))
