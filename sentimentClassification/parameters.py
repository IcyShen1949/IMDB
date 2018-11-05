# -*- coding: utf-8 -*-
# @Time    : 18-1-2 下午12:56
# @Author  : Icy Shen
# @Email   : SAH1949@126.com
import os
data_path = "./data/"
source_data_path = os.path.join(data_path, "labeledTrainData.tsv")
source_Undata_path = os.path.join(data_path, "unlabeledTrainData.tsv")
source_test_path = os.path.join(data_path, "testData.tsv")
sample_file = os.path.join(data_path, 'sampleSubmission.csv')
target_pos_data = os.path.join(data_path, "./subdata/full-train-pos.txt")
target_neg_data = os.path.join(data_path, "./subdata/full-train-neg.txt")
target_un_data = os.path.join(data_path, "./subdata/train-unsup.txt")
target_test_pos_data = os.path.join(data_path, "./subdata/test-pos.txt")
target_test_neg_data = os.path.join(data_path, "./subdata/test-neg.txt")
valid_pos = os.path.join(data_path, "./subdata/valid-pos.txt")
valid_neg = os.path.join(data_path, "./subdata/valid-neg.txt")
small_train_pos = os.path.join(data_path, "./subdata/small-train-pos.txt")
small_train_neg = os.path.join(data_path, "./subdata/small-train-neg.txt")
small_train_size = 10000
liblinear = "./liblinear-1.96"
lr = 0.5
n_epochs=15
alpha = 1e-6
ngram = "123"
modeltype = ["gru", 'lstm']
out_small_nbsvm = "./scores/NBSVM-VALID"
out_full_nbsvm = "./scores/NBSVM-TEST"
out_small_gru = './scores/PASSAGE_GR-VALID'
out_full_gru = './scores/PASSAGE_GR-TEST'
out_small_lstm = './scores/PASSAGE_LSTM-VALID'
out_full_lstm = './scores/PASSAGE_LSTM-TEST'
out_small_LR = './scores/LR-VALID'
out_full_LR= './scores/LR-TEST'
all_classifier = ["NBSVM", "PARAGRAPH", "PASSAGE_GR", "LR", 'PASSAGE_LSTM']
score_path = "./scores/"