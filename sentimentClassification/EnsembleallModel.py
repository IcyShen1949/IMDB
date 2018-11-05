# -*- coding: utf-8 -*-
# @Time    : 17-12-29 下午4:51
# @Author  : Icy Shen
# @Email   : SAH1949@126.com

import pandas as pd
import os
import re
from bs4 import BeautifulSoup
from collections import Counter
import numpy as np
from passage.preprocessing import Tokenizer
from passage.layers import Embedding, GatedRecurrent, LstmRecurrent, Dense
from passage.updates import Adadelta
from passage.models import RNN
from tqdm import tqdm
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
from sklearn.feature_extraction.text import TfidfVectorizer
import parameters as prm


def normalize_data(source_path, target_path, Dtype ):
    source_data = pd.read_csv(source_path, escapechar="\\", delimiter='\t')
    for index, row in source_data.iterrows():
        text = BeautifulSoup(row['review'], 'html.parser').get_text()
        temp_review = re.sub("[^a-zA-Z]", " ", text )
        words = temp_review.lower().split()
        review = ' '.join(words)
        if (Dtype == "pos" and row['sentiment'] == 1) or (Dtype == "neg" and row['sentiment'] == 0) or (Dtype == -1):
                with open(target_path, "a") as f:
                    f.write(review)
                    f.write('\n')


def split_train_valid(datafile, size, tar_train, tar_valid):
    with open(datafile, 'r') as f_sour:
        text = f_sour.readlines()
    train = text[:size]
    with open(tar_train, 'w') as f_train:
        for sen in train:
            f_train.write(sen)

    valid = text[size:]
    with open(tar_valid, 'w') as f_valid:
        for sen in valid:
            f_valid.write(sen)


def pre_data():

    if not os.path.exists(prm.target_pos_data):
        normalize_data(prm.source_data_path, prm.target_pos_data, Dtype = "pos")
    if not os.path.exists(prm.target_neg_data):
        normalize_data(prm.source_data_path, prm.target_neg_data, Dtype="neg")
    if not os.path.exists(prm.target_un_data):
        normalize_data(prm.source_Undata_path, prm.target_un_data, -1 )
    if not os.path.exists(prm.target_test_pos_data):
        normalize_data(prm.source_test_path, prm.target_test_pos_data, -1 )
    if not os.path.exists(prm.target_test_neg_data):
        os.system("touch " + prm.target_test_neg_data)
    if (not os.path.exists(prm.small_train_pos)) or (not os.path.exists(prm.valid_pos)):
        split_train_valid(prm.target_pos_data, prm.small_train_size, prm.small_train_pos, prm.valid_pos)
    if (not os.path.exists(prm.small_train_neg)) or (not os.path.exists(prm.valid_neg)):
        split_train_valid(prm.target_neg_data, prm.small_train_size, prm.small_train_neg, prm.valid_neg)


def tokenize(sen, grams):

    words = sen.split()
    tokens = []
    for gram in grams:
        for i in range(len(words) - gram + 1):
            tokens += ["-*-".join(words[i:(i+gram)])]
    return tokens


def build_dict(file, gram):

    dic = Counter()
    text = open(file).readlines()
    for sentence in tqdm(text):
        dic.update(tokenize(sentence, gram))
    return dic


def compute_ratio(pc, nc, alpha):
    all_tokens = list(set(list(pc.keys()) + list(nc.keys())))
    dic = dict((t, i ) for i, t in enumerate(all_tokens) )
    dic_len = len(dic)
    p, n = np.ones(dic_len) * alpha, np.ones(dic_len) * alpha
    for t in tqdm(all_tokens):
        p[dic[t]] += pc[t]
        n[dic[t]] += nc[t]
    p /= abs(p).sum()
    n /= abs(n).sum()
    log_pn = np.log(p / n)
    return dic, log_pn


def process_nbsvmData(p, n, d, r, tar_, ngram):
    output = []
    for beg_line, f in zip(["1", "0"], [p, n]):
        text = open(f).xreadlines()
        for l in tqdm(text):
            tokens = tokenize(l, ngram)
            indexes = []
            for t in tokens:
                try:
                    indexes += [d[t]]
                except KeyError:
                    pass
            indexes = list(set(indexes))
            indexes.sort()
            line = [beg_line]
            for i in indexes:
                line += ["%i:%f" % (i + 1, r[i])]
            output += [" ".join(line)]
    output = "\n".join(output)
    with open(tar_, "w") as f:
        f.writelines(output)


def nbsvm(ptrain, ntrain, ptest, ntest, out, liblinear, ngram):

    ngram = [int(i) for i in ngram]
    print ("counting....")
    pos_counts = build_dict(ptrain, ngram)
    neg_counts = build_dict(ntrain, ngram)

    print ("compute ratio....")

    dic, r_pn = compute_ratio(pos_counts, neg_counts, prm.alpha)

    print("process data...")
    process_nbsvmData(ptrain, ntrain, dic, r_pn, 'train-nbsvm.txt', ngram)
    process_nbsvmData(ptest, ntest, dic, r_pn, 'test-nbsvm.txt', ngram)

    print("train & predict")
    trainsvm = os.path.join(liblinear, "train")
    predictsvm = os.path.join(liblinear, "predict")

    os.system("chmod +x " + trainsvm )
    os.system("chmod +x " + predictsvm )
    os.system(trainsvm + " -s 0 train-nbsvm.txt model.logreg")
    os.system(predictsvm + " -b 1 test-nbsvm.txt model.logreg " + out)
    os.system("rm model.logreg train-nbsvm.txt test-nbsvm.txt")


def read_file(filename, label):
    with open(filename, 'r') as f:
        text = f.readlines()
        X = [sen.lower() for sen in text]
        Y = [label] * len(X)
    return X,Y


def splitXy(pos_file, neg_file):
    pos_X, pos_y = read_file(pos_file, 1)
    neg_X, neg_y = read_file(neg_file, 0)
    X = pos_X + neg_X
    y = pos_y + neg_y
    return X, y


def LR(pos_train, neg_train, pos_test, neg_test, out):
    train_X, train_Y = splitXy(pos_train, neg_train)
    test_X, test_Y = splitXy(pos_test, neg_test)

    tfidf_impLR = TfidfVectorizer(min_df=3, max_features=None, strip_accents='unicode', analyzer='word',
                                  token_pattern=r'\w{1,}',
                                  ngram_range=(1, 2), use_idf=1, smooth_idf=1, sublinear_tf=1, stop_words='english')
    X = train_X + test_X
    lentrain = len(train_X)
    tfidf_impLR.fit(X)
    data_pro_impLR = tfidf_impLR.transform(X)
    train_x_impLR = data_pro_impLR[:lentrain]
    test_x_impLR = data_pro_impLR[lentrain:]

    model_impLR = LogisticRegression(penalty='l2', dual=True, tol=0.0001,
                               C=1, fit_intercept=True, intercept_scaling=1.0,
                               class_weight=None, random_state=None)
    print("20 Fold CV Score: ", np.mean(cross_val_score(model_impLR, train_x_impLR, train_Y,
                                                        cv=20, scoring='roc_auc')))
    model_impLR.fit(train_x_impLR, train_Y)
    print("Predicting ...")
    result = model_impLR.predict_proba(test_x_impLR)[:, 1]
    predY = [0 if y < 0.5 else 1 for y in result]
    with open(out, "w") as f:
        for label, pos_prob, neg_prob in zip(predY, result, 1 - result):
            f.write("%d %f %f\n" % (label, pos_prob, neg_prob))
    return 0


def rnn(pos_train, neg_train, pos_test, neg_test, out, Mtype):

    print("Loading data ...")
    train_X, train_Y = splitXy(pos_train, neg_train)
    test_X, test_Y = splitXy(pos_test, neg_test)

    tokenizer = Tokenizer(min_df = 10, max_features = 100000)
    train_X_vec = tokenizer.fit_transform(train_X)
    test_X_vec = tokenizer.transform(test_X)

    print("build the %s model ..." % Mtype)
    if Mtype == "gru":
        layers = [
            Embedding(size=256, n_features=tokenizer.n_features),
            GatedRecurrent(size=512, activation='tanh', gate_activation='steeper_sigmoid',
                           init='orthogonal', seq_output=False, p_drop=0.75),
            Dense(size=1, activation='sigmoid', init='orthogonal')
        ]
    else:
        layers = [
            Embedding(size=256, n_features=tokenizer.n_features),
            LstmRecurrent(size=512, activation='tanh', gate_activation='steeper_sigmoid',
                          init='orthogonal', seq_output=False, p_drop=0.75),
            Dense(size=1, activation='sigmoid', init='orthogonal')
        ]

    model = RNN(layers=layers, cost='bce', updater=Adadelta(prm.lr))
    model.fit(train_X_vec , train_Y, n_epochs = prm.n_epochs)

    print("Predicting ...")
    test_pred = model.predict(test_X_vec).flatten()

    predY = [0 if y < 0.5 else 1 for y in test_pred ]
    with open(out, "w") as f:
        for label, pos_prob, neg_prob in zip(predY, test_pred, 1 - test_pred):
            f.write("%d %f %f\n" % (label, pos_prob, neg_prob))


def load_clssifier_result(Dtype, classifiers, path = prm.score_path):
    assert Dtype in ['TEST', "VALID"]
    prob = []
    for c in classifiers:
        data = np.loadtxt(os.path.join(path, "-".join([c, Dtype])))
        prob += [data[:, 1]]
    x = np.vstack(prob).T
    y = np.vstack([np.ones((x.shape[0] // 2, 1)), np.zeros((x.shape[0] // 2, 1))])
    return x, y


def load_individual_result(Dtype, classifiers, path = prm.score_path):
    sample_submission_df = pd.read_csv(prm.sample_file)
    ids = sample_submission_df['id'].values
    for c in classifiers:
        data = np.loadtxt(os.path.join(path, "-".join([c, Dtype])))
        result_df = pd.DataFrame(np.asarray([ids, data[:, 1]]).T)
        result_df.to_csv(prm.data_path + "-".join([c, Dtype]) + '.csv', index=False, header=['id', 'sentiment'])


def accuracy(classifier_id, data):
    x, y = data
    output = [classifier_id[i] * x[:, i] for i in range(len(classifier_id))]
    pred = np.vstack(output).sum(0)
    acc = ((pred > 0.5) == y.T).mean()
    return acc


def ensemble(data, classifier):
    output = []
    # x, y = data
    for i, c in enumerate(classifier):
        k = np.zeros(len(classifier))
        k[i] = 1
        acc = accuracy(k, data)
        output += [acc]
    k_value = np.array(output)
    k_value /= k_value.sum()
    best = accuracy(k_value, data)
    return k_value, best


def pred_prob(key, data):
    x, y = data
    output = [key[i] * x[:, i] for i in range(len(key))]
    return np.vstack(output).sum(0)


def build_ensemble():
    all_classifier = prm.all_classifier
    valid, test = load_clssifier_result("VALID", all_classifier), load_clssifier_result("TEST", all_classifier)
    k, _ = ensemble(valid, all_classifier)
    result = pred_prob(k, test)
    pd.DataFrame(result.T).to_csv("./data/out_test_ensemble_value.csv", index = False, header = ['pos'])


def generate_submission():
    assembled_file = os.path.join(prm.data_path, 'out_test_ensemble_value.csv')
    submission_file = os.path.join(prm.data_path, 'out_test_ensemble.csv')
    sample_submission_df = pd.read_csv(prm.sample_file)
    assembled_df = pd.read_csv(assembled_file)
    ids = sample_submission_df['id'].values
    probs = assembled_df['pos'].values
    result_df = pd.DataFrame(np.asarray([ids, probs]).T)
    result_df.to_csv(submission_file, index=False, header=['id', 'sentiment'])


def build_nbsvm():
    nbsvm(prm.small_train_pos, prm.small_train_neg, prm.valid_pos,
          prm.valid_neg, prm.out_small_nbsvm, prm.liblinear, prm.ngram)
    os.system("tail -n 5000 ./scores/NBSVM-VALID > tmp; mv tmp ./scores/NBSVM-VALID")
    nbsvm(prm.target_pos_data, prm.target_neg_data, prm.target_test_pos_data,
          prm.target_test_neg_data, prm.out_full_nbsvm, prm.liblinear, prm.ngram)
    os.system("tail -n 25000 ./scores/NBSVM-TEST > tmp; mv tmp ./scores/NBSVM-TEST")


def build_paragraph():
    os.system("chmod +x iclr15/scripts/*.sh")
    os.system("./iclr15/scripts/paragraph.sh")


def build_passage_rnn():
    rnn(prm.small_train_pos, prm.small_train_neg, prm.valid_pos, prm.valid_neg,
        prm.out_small_gru, prm.modeltype[0])
    rnn(prm.target_pos_data, prm.target_neg_data, prm.target_test_pos_data, prm.target_test_neg_data,
        prm.out_full_gru, prm.modeltype[0])

    rnn(prm.small_train_pos, prm.small_train_neg, prm.valid_pos, prm.valid_neg,
        prm.out_small_lstm, prm.modeltype[1])
    rnn(prm.target_pos_data, prm.target_neg_data, prm.target_test_pos_data, prm.target_test_neg_data,
        prm.out_full_lstm, prm.modeltype[1])


def build_LR():
    LR(prm.small_train_pos, prm.small_train_neg, prm.valid_pos,
       prm.valid_neg, prm.out_small_LR)
    LR(prm.target_pos_data, prm.target_neg_data, prm.target_test_pos_data,
       prm.target_test_neg_data, prm.out_full_LR)


if __name__ == "__main__":
    pre_data()
    build_nbsvm()
    build_paragraph()
    build_passage_rnn()
    build_LR()
    build_ensemble()
    load_individual_result("TEST", prm.all_classifier, path = prm.score_path )
    generate_submission()
    print('end')