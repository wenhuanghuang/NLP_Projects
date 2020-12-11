#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Oct  3 11:09:42 2020

@author: huangwenhuang
"""
import sys
import re
import glob
import json
from perceplearn import tokenizer, calculate_activation
# MODEL_FILE = sys.argv[1]
MODEL_FILE = 'averagedmodel.txt'
POSNEG = ["pos_neg_w", "pos_neg_b"]
TRUDEC = ["tru_dec_w", "tru_dec_b"]
# TEST_DIR = sys.argv[2]
TEST_DIR = "./op_spam_test_data"

OUTPUT = "percepoutput.txt"

def get_test_data(root_dir):
    folder = root_dir + "/*/*/*/*.txt"
    return glob.glob(folder)

def get_weight_bias(tag, model_file):
    with open(model_file) as f:
        para = json.load(f)
        w = para[tag[0]]
        b = para[tag[1]]
    return w, b

def feature_vector(file, weights):
    fv = dict()
    token_review = tokenizer(file)
    for token in token_review:
        if token in weights:
            fv[token] = fv.get(token, 0) + 1
    return fv

def classify(files):
    posneg_w, posneg_b = get_weight_bias(POSNEG, MODEL_FILE)
    trudec_w, trudec_b = get_weight_bias(TRUDEC, MODEL_FILE)
    results_posneg = []
    results_trudec = []
    for file in files:
        label1 = "positive"
        label2 = "truthful"
        posneg_fv = feature_vector(file, posneg_w)
        # print(posneg_fv)
        a_posneg = calculate_activation(posneg_fv, posneg_w, posneg_b)
        if (a_posneg < 0): label1 = "negative"
        trudec_fv = feature_vector(file, trudec_w)
        a_trudec = calculate_activation(trudec_fv,trudec_w, trudec_b)
        if (a_trudec < 0): label2 = "deceptive"
        results_posneg.append(label1)
        results_trudec.append(label2)
    return results_posneg, results_trudec

def write2file():
    files = get_test_data(TEST_DIR)
    pos_neg_results, tru_dec_results = classify(files)
    file = open(OUTPUT, "w")
    for index in range(len(tru_dec_results)):
        file.write(str(tru_dec_results[index]) + " " + str(pos_neg_results[index] + " " + files[index]) + "\n")
    file.close()
    
def main():
    write2file()
if __name__ == "__main__":
    main()
