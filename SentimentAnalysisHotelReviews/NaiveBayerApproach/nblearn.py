#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Sep 19 13:40:26 2020

@author: huangwenhuang
"""
import sys
import re
import json
import glob
# TRAIN_DIR = sys.argv[1]
TRAIN_DIR = "../op_spam_training_data"
POSITIVE = TRAIN_DIR + "/positive*/*/*/[a-z]_*_[0-9]*.txt"
NEGATIVE = TRAIN_DIR + "/negative*/*/*/[a-z]_*_[0-9]*.txt"
TRUTHFUL = TRAIN_DIR + "/*/truthful*/*/[a-z]_*_[0-9]*.txt"
DECEPTIVE = TRAIN_DIR + "/*/deceptive*/*/[a-z]_*_[0-9]*.txt"

def get_train_files(tag):
    return glob.glob(tag)

def tokenizer(file):
    # tokenizer the review
    review = None
    SPECIAL_CHARACTER = re.compile("[.;:?,\"()\[\]\&\-\*\@]")
    with open(file, 'r') as f_open:
        review = [line.strip() for line in f_open]
        review = SPECIAL_CHARACTER.sub(" ", ' '.join(review).lower())
    return review.split()

def get_reviews(files):
    # return a list of hotel reviews
    reviews = list()
    for f in files:
        reviews.append(tokenizer(f))
    return reviews

def create_word_count_reviews(reviews):
    counter = dict()
    for review in reviews:
        for token in review:
            counter[token] = counter.get(token, 0) + 1
    return counter

def create_word_count(tag):
    data = get_train_files(tag)
    reviews = get_reviews(data)
    counter = create_word_count_reviews(reviews)
    total_num = 0
    for x in counter:
        total_num += counter[x]
    return counter, total_num

def word_filter(count):
    STOP_WORDS = {'the', 'and', 'a', 'to', 'was', 'of', 'for', 'it', 'that', 'is', 'were', 'in', 'i'}
    to_delete = []
    for key in count.keys():
        if key in STOP_WORDS:
            to_delete.append(key)
    for key in to_delete:
        count.pop(key)
    return count

def get_prior(tags):
    if len(tags) == 2:
        tag0 = len(glob.glob(tags[0])) 
        tag1 = len(glob.glob(tags[1]))
        return tag0/(tag0 + tag1), tag1/(tag0 + tag1)
    else:
        tag0 = len(glob.glob(tags[0])) 
        tag1 = len(glob.glob(tags[1]))
        tag2 = len(glob.glob(tags[2])) 
        tag3 = len(glob.glob(tags[3]))
        tag_sum = tag0 + tag1 + tag2 + tag3
        return tag0/tag_sum, tag1/tag_sum,tag2/tag_sum,tag3/tag_sum,
    
def write2file(OUTPUT = 'nbmodel.txt'):
    positive, NPos = create_word_count(POSITIVE)
    negative, NNeg = create_word_count(NEGATIVE)
    truthful, NTru = create_word_count(TRUTHFUL)
    deceptive, NDece = create_word_count(DECEPTIVE)

    positive_prior, negative_piror = get_prior([POSITIVE, NEGATIVE])
    truthful_prior, deceptive_prior = get_prior([TRUTHFUL, DECEPTIVE])
    output = open(OUTPUT, 'w')
    json.dump({ "positive_total": NPos, "positive_prior": positive_prior, "positive_word_occurrence": positive,
                "negative_total": NNeg, "negative_prior": negative_piror, "negative_word_occurrence": negative, 
                "truthful_total": NTru, "truthful_prior":  truthful_prior, "truthful_word_occurrence": truthful,  
                "deceptive_total": NDece,  "deceptive_prior": deceptive_prior,"deceptive_word_occurrence": deceptive
                }, output, indent=1)
    output.close()

def main():
    write2file()
    
if __name__ == "__main__":
    main()