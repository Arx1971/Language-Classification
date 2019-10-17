import os
import re
from string import punctuation
from collections import Counter


#  regular expression r'[\s+`=~!@#$%^&*()_+\[\]{};\--\\:"|<,./<>?^]'

def file_name_viewer(file_name):
    for name in file_name:
        print(name)
    print(len(file_name))


def sum_of_values(dictionary):
    total = 0
    for key, values in dictionary.items():
        total += values
    return total


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
            words = review.split()
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
    x = Counter(vocabulary_1)
    y = Counter(vocabulary_2)
    x.update(y)
    return dict(x)


def naive_byes_classifier_bag_of_words_model(vocabulary, filepath, test_review, number_of_word_in_class,
                                             total_vocabulary_size):
    prob = 1.0
    for name in test_review:
        with open(filepath + name, "r") as reviews:
            review = reviews.read()
            words = review.split()
            for word in words:
                word = word.strip(punctuation)
                if word in vocabulary:
                    prob *= ((vocabulary[word] + 1) / (number_of_word_in_class + total_vocabulary_size))
                else:
                    prob *= ((1) / (number_of_word_in_class + total_vocabulary_size))
    return prob


def small_training_corpus():
    small_training_action_corpus = read_all_file_name("../small_corpus/train/action")
    small_training_comedy_corpus = read_all_file_name("../small_corpus/train/comedy")
    small_test_corpus = read_all_file_name("../small_corpus/test")
    small_action_corpus_vocabulary = set_vocabulary(small_training_action_corpus[0], "../small_corpus/train/action/")
    small_comedy_corpus_vocabulary = set_vocabulary(small_training_comedy_corpus[0], "../small_corpus//train/comedy/")
    small_corpus_vocabulary = merge_vocabulary(small_action_corpus_vocabulary, small_comedy_corpus_vocabulary)
    # print(small_corpus_vocabulary)
    # print(small_action_corpus_vocabulary)
    # print(small_comedy_corpus_vocabulary)
    total_number_of_training_files = (small_training_comedy_corpus[1] + small_training_action_corpus[1])
    total_action_training_files = small_training_action_corpus[1]
    total_comedy_training_files = small_training_comedy_corpus[1]
    action_class_prob = naive_byes_classifier_bag_of_words_model(small_action_corpus_vocabulary,
                                                                 "../small_corpus/test/",
                                                                 small_test_corpus[0],
                                                                 sum_of_values(small_action_corpus_vocabulary),
                                                                 len(small_corpus_vocabulary)) * float(
        total_action_training_files / total_number_of_training_files)

    comedy_class_prob = naive_byes_classifier_bag_of_words_model(small_comedy_corpus_vocabulary,
                                                                 "../small_corpus/test/",
                                                                 small_test_corpus[0],
                                                                 sum_of_values(small_comedy_corpus_vocabulary),
                                                                 len(small_corpus_vocabulary)) * float(
        total_comedy_training_files / total_number_of_training_files)
    print("Probabilities for Action Class: ", action_class_prob)
    print("Probabilities for Comedy Class: ", comedy_class_prob)
    if action_class_prob > comedy_class_prob:
        print("Document Belong to Action Class. ")
    else:
        print("Document Belong to Comedy Class. ")


small_training_corpus()

# training_pos_file_name = read_all_file_name("../movie-review-HW2/aclImdb/train/pos")
# training_neg_file_name = read_all_file_name("../movie-review-HW2/aclImdb/train/neg")
# dictionary = training_vocabulary(training_neg_file_name, training_pos_file_name, "../movie-review-HW2/aclImdb/train/")
# dictionary_frequency_viewer(dictionary)
# print(len(dictionary))
