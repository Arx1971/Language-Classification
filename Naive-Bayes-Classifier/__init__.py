import os
import re
from collections import Counter
from string import punctuation
import math


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
    return files


def set_vocabulary(review, filepath):
    vocabulary = dict()
    for review_name in review:
        with open(filepath + review_name, "r") as reviews:
            review = reviews.read()
            review = re.sub(r'[`=~!@#$%^&*()_+\[\]{};\\:"|<,./<>?^]', ' ', review)
            words = review.split()
            for word in words:
                word = word.lower()
                word = word.strip(punctuation)
                word = word.strip()
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
    with open(filepath + test_review, "r") as reviews:
        review = reviews.read()
        review = re.sub(r'[`=~!@#$%^&*()_+\[\]{};\\:"|<,./<>?^]', ' ', review)
        words = review.split()
        for word in words:
            word = word.lower()
            word = word.strip(punctuation)
            word = word.strip()
            if word in vocabulary:
                prob += math.log(float((vocabulary[word] + 1) / (number_of_word_in_class + total_vocabulary_size)), 2)
            else:
                prob += math.log(float((1) / (number_of_word_in_class + total_vocabulary_size)))

    return prob


def probability_method(test_files, neg_vocabulary, pos_vocabulary, filepath, training_vocabulary, total_neg_train_file,
                       total_pos_train_file):
    neg_counter_nr = 0
    pos_counter_nr = 0
    total_test_file = total_neg_train_file + total_pos_train_file
    negative_total_token = sum_of_values(neg_vocabulary)
    positive_total_token = sum_of_values(pos_vocabulary)
    for i in range(0, len(test_files)):

        neg_class_prob = naive_byes_classifier_bag_of_words_model(neg_vocabulary,
                                                                  filepath,
                                                                  test_files[i],
                                                                  negative_total_token,
                                                                  len(training_vocabulary)) + math.log(
            float(total_neg_train_file / total_test_file), 2)

        pos_class_prob = naive_byes_classifier_bag_of_words_model(pos_vocabulary,
                                                                  filepath,
                                                                  test_files[i],
                                                                  positive_total_token,
                                                                  len(training_vocabulary)) + math.log(
            float(total_pos_train_file / total_test_file), 2)

        if pos_class_prob > neg_class_prob:
            pos_counter_nr += 1
        else:
            neg_counter_nr += 1

    return neg_counter_nr, pos_counter_nr


def naive_byes_classifier():
    training_pos_file_name = read_all_file_name("../movie-review-HW2/aclImdb/train/pos")
    training_neg_file_name = read_all_file_name("../movie-review-HW2/aclImdb/train/neg")
    test_pos_file_name = read_all_file_name("../movie-review-HW2/aclImdb/test/pos")
    test_neg_file_name = read_all_file_name("../movie-review-HW2/aclImdb/test/neg")

    neg_vocabulary = set_vocabulary(training_neg_file_name, '../movie-review-HW2/aclImdb/train/neg/')
    pos_vocabulary = set_vocabulary(training_pos_file_name, '../movie-review-HW2/aclImdb/train/pos/')
    training_vocabulary = merge_vocabulary(neg_vocabulary, pos_vocabulary)
    total_neg_train_file = len(training_neg_file_name)
    total_pos_train_file = len(training_pos_file_name)

    neg_test_arr = probability_method(test_neg_file_name, neg_vocabulary, pos_vocabulary,
                                      "../movie-review-HW2/aclImdb/test/neg/", training_vocabulary,
                                      total_neg_train_file, total_pos_train_file)
    pos_test_arr = probability_method(test_pos_file_name, neg_vocabulary, pos_vocabulary,
                                      "../movie-review-HW2/aclImdb/test/pos/", training_vocabulary,
                                      total_neg_train_file, total_pos_train_file)
    pos_accuracy = pos_test_arr[1] / total_pos_train_file
    neg_accuracy = neg_test_arr[0] / total_neg_train_file
    print("Accuracy for positive class: ", pos_accuracy)
    print("Accuracy for negative class: ", neg_accuracy)
    print("Total Accuracy: ", (pos_accuracy + neg_accuracy) / 2)


def small_training_corpus():
    small_training_action_corpus = read_all_file_name("../small_corpus/train/action")
    small_training_comedy_corpus = read_all_file_name("../small_corpus/train/comedy")
    small_test_corpus = read_all_file_name("../small_corpus/test")
    small_action_corpus_vocabulary = set_vocabulary(small_training_action_corpus, "../small_corpus/train/action/")
    small_comedy_corpus_vocabulary = set_vocabulary(small_training_comedy_corpus, "../small_corpus//train/comedy/")
    small_corpus_vocabulary = merge_vocabulary(small_action_corpus_vocabulary, small_comedy_corpus_vocabulary)

    sum1 = sum_of_values(small_action_corpus_vocabulary)

    total_number_of_training_files = (len(small_training_comedy_corpus) + len(small_training_action_corpus))
    total_action_training_files = len(small_training_action_corpus)
    total_comedy_training_files = len(small_training_comedy_corpus)
    action_class_prob = naive_byes_classifier_bag_of_words_model(small_action_corpus_vocabulary,
                                                                 "../small_corpus/test/",
                                                                 small_test_corpus[0],
                                                                 sum1,
                                                                 len(small_corpus_vocabulary)) * float(
        total_action_training_files / total_number_of_training_files)

    sum2 = sum_of_values(small_comedy_corpus_vocabulary)

    comedy_class_prob = naive_byes_classifier_bag_of_words_model(small_comedy_corpus_vocabulary,
                                                                 "../small_corpus/test/",
                                                                 small_test_corpus[0],
                                                                 sum2,
                                                                 len(small_corpus_vocabulary)) * float(
        total_comedy_training_files / total_number_of_training_files)
    print("Probabilities for Action Class: ", action_class_prob)
    print("Probabilities for Comedy Class: ", comedy_class_prob)
    if action_class_prob > comedy_class_prob:
        print("Document Belong to Action Class. ")
    else:
        print("Document Belong to Comedy Class. ")


# small_training_corpus()
naive_byes_classifier()
