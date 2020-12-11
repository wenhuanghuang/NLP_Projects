#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Oct  2 21:01:49 2020

@author: huangwenhuang
"""
import sys
import re
import json
import glob
import random
import copy

# TRAIN_DIR = sys.argv[1]
TRAIN_DIR = "./op_spam_training_data"

POSITIVE = TRAIN_DIR + "/positive*/*/*/[a-z]_*_[0-9]*.txt"
NEGATIVE = TRAIN_DIR + "/negative*/*/*/[a-z]_*_[0-9]*.txt"
TRUTHFUL = TRAIN_DIR + "/*/truthful*/*/[a-z]_*_[0-9]*.txt"
DECEPTIVE = TRAIN_DIR + "/*/deceptive*/*/[a-z]_*_[0-9]*.txt"

def get_train_files(tag):
    # return training data links
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

def create_word_count(reviews):
    counter = dict()
    for review in reviews:
        for token in review:
            counter[token] = counter.get(token, 0) + 1
    return counter

def create_word_features(tags, D):
    files = []
    for tag in tags:
        files += get_train_files(tag)
    reviews = get_reviews(files)
    counter = create_word_count(reviews)
    total_num = 0
    for x in counter:
        total_num += counter.get(x)
        counter[x] = 0
    return counter.keys(), counter, files

def word_filter(count):
    STOP_WORDS = {'the', 'and', 'a', 'to', 'was', 'of', 'for', 'it', 'that', 'is', 'were', 'in', 'i'}
    to_delete = []
    for x in count.keys():
        if (x in STOP_WORDS):
            to_delete.append(x)
    for y in to_delete:
        count.pop(y)
    return count

def feature_vector(file, weights, posneg=True):
    fv = dict()
    token_review = tokenizer(file)
    for token in token_review:
        if token in weights:
            fv[token] = fv.get(token, 0) + 1
    label = 1
    if posneg and file.find("negative") != -1:
        label = -1
    if posneg == False and file.find("deceptive") != -1:
        label = -1
    
    return fv, label

def calculate_activation(fv, w, b):
    a = 0
    for feature in fv:
        a += fv[feature] * w[feature] + b
    return a

def update_weight(w,x,y):
    for feature in x:
        if feature in w:
            w[feature] = w[feature] + x[feature] * y
    return w

def training(D, MaxIter, tags, posneg):
    features, weights, data_files = create_word_features(tags,D)
    b = 0
    cached_weights = copy.deepcopy(weights)
    cached_b = 0
    c = 1
    for Iter in range(MaxIter):
        random.shuffle(data_files)
        for td in data_files:
            x, y = feature_vector(td, weights,posneg)
            a = calculate_activation(x, weights, b)
            if y * a <= 0:
                weights = update_weight(weights, x, y)
                b = b + y
                cached_weights = update_weight(cached_weights, x, c*y)
                cached_b = cached_b + c*y
            c+=1
    # print(weights,b)
    for w in cached_weights:
        cached_weights[w] = weights[w] - cached_weights[w] / c
    cached_b = b - cached_b / c
    return weights, b, cached_weights, cached_b
            
def write2file(OUTPUT1="vanillamodel.txt", OUTPUT2="averagedmodel.txt"):
    posneg_w, posneg_b, posneg_cw, posneg_cb = training(1000, 100, [POSITIVE, NEGATIVE], True)
    trudec_w, trudec_b, trudec_cw, trudec_cb = training(1000, 100, [TRUTHFUL, DECEPTIVE], False)
    output1 = open(OUTPUT1, 'w')
    json.dump({ "pos_neg_w": posneg_w, "pos_neg_b": posneg_b,
                "tru_dec_w": trudec_w, "tru_dec_b": trudec_b 
                }, output1, indent=1)
    output1.close()
    
    output2 = open(OUTPUT2, 'w')
    json.dump({ "pos_neg_w": posneg_cw, "pos_neg_b": posneg_cb,
                "tru_dec_w": trudec_cw, "tru_dec_b": trudec_cb 
                }, output2, indent=1)
    output2.close()
    
def main():
    write2file()
    print()
        
if __name__ == "__main__":
    main()

