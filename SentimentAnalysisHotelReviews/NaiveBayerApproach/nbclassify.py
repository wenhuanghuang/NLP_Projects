#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Sep 24 16:05:11 2020

@author: huangwenhuang
"""

import sys
import re
import glob
import json
import math
from nblearn import tokenizer, get_reviews
# TEST_DIR = sys.argv[1]
TEST_DIR = "../op_spam_test_data"
POSITIVE = ["positive_total", "positive_prior", "positive_word_occurrence"]
NEGATIVE = ["negative_total", "negative_prior", "negative_word_occurrence"]
TRUTHFUL = ["truthful_total", "truthful_prior", "truthful_word_occurrence"]
DECEPTIVE = ["deceptive_total", "deceptive_prior","deceptive_word_occurrence"]
                      
def get_test_data(root_dir):
    folder = root_dir + "/*/*/*/*.txt"
    return glob.glob(folder)

def get_word_count(tag, MODEL_FILE = 'nbmodel.txt'):
    with open(MODEL_FILE) as f:
        para = json.load(f)
        total = para[tag[0]]
        counter = para[tag[2]]
    return total, counter

def get_prior(tag, MODEL_FILE = 'nbmodel.txt'):
    with open(MODEL_FILE) as f:
        para = json.load(f)
        prior = para[tag[1]]
    return prior

def add_one(tags):
    # add one smoothing
    Ntag0, tag0count = get_word_count(tags[0])
    Ntag1, tag1count = get_word_count(tags[1])
    tag0missing = []
    tag1missing = []
    for i in tag1count:
        if i not in tag0count:
            tag0missing.append(i)
    for j in tag0count:
        if j not in tag1count:
            tag1missing.append(j)
            
    b_tag0 = len(tag0count) + len(tag0missing)
    b_tag1 = len(tag1count) + len(tag1missing)
    
    Ntag0 += b_tag0
    Ntag1 += b_tag1

    for key in tag0count:
        tag0count[key] = tag0count.get(key) + 1
    for key in tag1count:
        tag1count[key] = tag1count.get(key) + 1

    for missing0 in tag0missing:
        tag0count[missing0] = 1
    for missing1 in tag1missing:
        tag1count[missing1] = 1

    tag0count = conditional_probability(tag0count, Ntag0)
    tag1count = conditional_probability(tag1count, Ntag1)

    return tag0count, Ntag0, tag1count, Ntag1    

def conditional_probability(word_count, total_number):
    # print(total_number)
    for key in word_count:
        word_count[key] = math.log(word_count[key] / total_number)
    return word_count

def classify_pos_neg(reviews):
    
    tag0count, Ntag0, tag1count, Ntag1 = add_one([POSITIVE, NEGATIVE]) 
    results = []
    for review in reviews: 
        score_tag0 = math.log(get_prior(POSITIVE))
        score_tag1 = math.log(get_prior(NEGATIVE))
       
        for token in review:
 
            if token in tag0count:
                score_tag0 += tag0count[token]
            if token in tag1count:
                score_tag1 += tag1count[token]

        if score_tag0 > score_tag1:
            results.append("positive")
        else:
            results.append("negative")
    return results

def classify_tru_dec(reviews):

    tag0count, Ntag0, tag1count, Ntag1 = add_one([TRUTHFUL, DECEPTIVE]) 
    results = []
    for review in reviews: 
        score_tag0 = math.log(get_prior(POSITIVE))
        score_tag1 = math.log(get_prior(NEGATIVE))
        for token in review:
            if token in tag0count:
                score_tag0 += tag0count[token]
            if token in tag1count:
                score_tag1 += tag1count[token]
        if score_tag0 > score_tag1:
            results.append("truthful")
        else:
            results.append("deceptive")
    return results 

def write2file(OUTPUT = 'nboutput.txt'):
    files = get_test_data(TEST_DIR)
    reviews = get_reviews(files)
    pos_neg_results = classify_pos_neg(reviews)
    tru_dec_results = classify_tru_dec(reviews)
    file = open(OUTPUT, "w")
    for index in range(len(tru_dec_results)):
        file.write(str(tru_dec_results[index]) + " " + str(pos_neg_results[index] + " " + files[index]) + "\n")
    file.close()
def main():
    write2file()
    
if __name__ == "__main__":
    main()

