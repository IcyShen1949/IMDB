# -*- coding: utf-8 -*-
# @Time    : 17-12-28 下午3:02
# @Author  : Icy Shen
# @Email   : SAH1949@126.com

# https://www.kaggle.com/c/word2vec-nlp-tutorial/data
from sklearn.linear_model import LogisticRegression
import pandas as pd
import numpy as np
from bs4 import BeautifulSoup
import re
from tqdm import tqdm
from sklearn.feature_extraction.text import TfidfVectorizer
import os
from nltk.corpus import stopwords
from sklearn.model_selection import cross_val_score, GridSearchCV
import xgboost as xgb
import operator
import matplotlib.pyplot as plt
from wordcloud import WordCloud,STOPWORDS,ImageColorGenerator
data_path = './data/'
stopwords_set = set(stopwords.words('english'))
def filter_text(review, remove_stopwords):
    text = BeautifulSoup(review, 'html.parser').get_text()
    text = re.sub('n\'t', ' not', text )
    text_good = re.sub("[^a-zA-Z]", " ", text)
    words = text_good.lower().split()
    if remove_stopwords:
        words = [w for w in words if not w in stopwords_set]
    return words

def pro_data():
    train_ini = pd.read_csv(data_path + "labeledTrainData.tsv", header = 0, delimiter = '\t')
    test_ini = pd.read_csv(data_path + "testData.tsv", header = 0, delimiter = '\t')
    train_words = []
    for i in tqdm(range(len(train_ini['review']))):
        train_words.append(' '.join(filter_text(train_ini['review'][i], False)))
    train_pro = pd.DataFrame(data = train_ini['id'], columns = ['id'])
    train_pro['sentiment'] = train_ini['sentiment']
    train_pro['review'] = train_words
    train_pro.to_csv(data_path + "train_pro.csv", index = False)

    test_words = []
    for i in tqdm(range(len(test_ini['review']))):
        test_words.append(' '.join(filter_text(test_ini['review'][i], False)))

    test_pro = pd.DataFrame(data = test_ini['id'], columns = ['id'])
    test_pro['review'] = test_words
    test_pro.to_csv(data_path + 'test_pro.csv', index = False)


def NaiveBayes(train_x_nb, test_x_nb, label_nb, test_nb):
    from sklearn.naive_bayes import  MultinomialNB
    model_NB = MultinomialNB()
    model_NB.fit(train_x_nb, label_nb )
    print ("model naive bayes 20 Fold CV Score:", np.mean(cross_val_score(model_NB, train_x_nb, label_nb,
                                                      cv = 20, scoring = "roc_auc")))
    test_pred_NB = model_NB.predict(test_x_nb)
    out_test_NB = pd.DataFrame(test_pred_NB, columns = ['sentiment'])
    out_test_NB['id'] = test_nb['id']
    out_test_NB = out_test_NB[['id', 'sentiment']]
    out_test_NB.to_csv(data_path + "out_test_NB .csv", index=False)
    print('model naive bayes finished')

def LogisticReg(train_x_LR, test_x_LR, label_LR, test_LR):
    print ('build logistic regression model')
    LR =  LogisticRegression()
    model_LR_improve = GridSearchCV(LR, {'C':[10, 30]}, scoring='roc_auc', cv=20, n_jobs = -1)
    model_LR_improve.fit(train_x_LR, label_LR)
    print("pred test")
    test_pred_LR = model_LR_improve.predict(test_x_LR)
    out_test_LR = pd.DataFrame(data = test_pred_LR, columns = ['sentiment'])
    out_test_LR['id'] = test_LR['id']
    out_test_LR = out_test_LR[['id', 'sentiment']]
    out_test_LR.to_csv(data_path + "out_test_LR.csv", index = False)
    print("model logistic regression finished!")

def plot_importance(Importance, d):
    Importance = sorted(Importance.items(), key=operator.itemgetter(1))
    df = pd.DataFrame(Importance, columns=['feature', 'fscore'])
    df['fscore'] = df['fscore'] / df['fscore'].sum()
    name = [d[int(x[0].strip('f'))] for x in Importance]
    df['feature'] = name
    df.to_csv("importance.csv", index = False)
    plt.figure()
    N = 50
    X = df.tail(N)['feature'].values.tolist()
    Y = df.tail(N)['fscore'].values.tolist()
    plt.bar(X, Y, 0.8, color="green")
    plt.xticks(df.tail(N)['feature'].values.tolist(), rotation = 90, fontsize=5)
    plt.savefig('./figure/importance.png', dpi=144)
# data = data.sort_values(by = ['fscore'])
    text = df.feature.values
    wc = WordCloud( background_color = 'white',
                    max_words = len(text),
                    stopwords = STOPWORDS,
                    max_font_size = 50,
                    random_state = 30,            
                    )

    Text = " ".join(text)
    wc.generate(Text)
    plt.imshow(wc)
    plt.axis('off')
    plt.savefig("./figure/wordCloud.png", dpi=144)

def XGB(train_xgb, test_xgb):
    print("tfidf of xgb")
    tfidf_impLR = TfidfVectorizer(min_df=3, max_features=None, strip_accents='unicode', analyzer='word',
                                  token_pattern=r'\w{1,}',
                                  ngram_range=(1, 2), use_idf=1, smooth_idf=1, sublinear_tf=1, stop_words='english')
    data_all_impLR = train_xgb['review'].values.tolist() + test_xgb['review'].values.tolist()
    lentrain = len(train_xgb)
    tfidf_impLR.fit(data_all_impLR)
    tfidf_impLR.get_feature_names()
    data_pro_impLR = tfidf_impLR.transform(data_all_impLR)
    train_x_xgb= data_pro_impLR[:lentrain]
    test_x_xgb= data_pro_impLR[lentrain:]
    label_xgb= train_xgb['sentiment']

    dic = tfidf_impLR.get_feature_names()
    small_train_xgb_size = 20000
    small_train_X = train_x_xgb[:small_train_xgb_size ]
    small_train_y = label_xgb[:small_train_xgb_size ]
    valid_x = train_x_xgb[small_train_xgb_size:]
    valid_y = label_xgb[small_train_xgb_size:]

    xgb_val = xgb.DMatrix(valid_x, label=valid_y)
    xgb_train = xgb.DMatrix(small_train_X, label=small_train_y)

    xgb_params ={'booster':'gbtree', 'objective': 'binary:logistic', 'gamma':0.1, 'max_depth':30,
                 'lambda':2, 'subsample':0.7, 'colsample_bytree':0.7, 'min_child_weight':5, 'silent':0 ,
                 'eta': 0.007, 'seed':1000, 'nthread':32, 'eval_metric': 'auc'}
    num_rounds = 20000
    plst = list(xgb_params.items())
    watchlist = [(xgb_train, 'train'), (xgb_val, 'val')]
    model_xgb = xgb.train(plst, xgb_train, num_rounds, watchlist, early_stopping_rounds=100)
    print "best best_ntree_limit", model_xgb.best_ntree_limit

    dtest = xgb.DMatrix(test_x_xgb)
    importance = model_xgb.get_fscore()

    result = model_xgb.predict(dtest, ntree_limit=model_xgb.best_ntree_limit)
    out_test_xgb = pd.DataFrame(data={"id": test["id"], "sentiment": result})
    out_test_xgb.to_csv(os.path.join(data_path, 'out_test_xgb.csv'), index=False, quoting=3)
    plot_importance(importance, dic)
    print("model xgb finished!")


def improve_LogisticRegression(train_impLR, test_impLR):# score: 0.95351
    print("tfidf of improved logistic regression")
    tfidf_impLR = TfidfVectorizer(min_df=3, max_features=None, strip_accents='unicode', analyzer='word', token_pattern=r'\w{1,}',
                            ngram_range=(1, 2), use_idf=1, smooth_idf=1, sublinear_tf=1, stop_words='english')
    data_all_impLR = train_impLR['review'].values.tolist() + test_impLR['review'].values.tolist()
    lentrain = len(train_impLR)
    tfidf_impLR.fit(data_all_impLR)
    tfidf_impLR.get_feature_names()
    data_pro_impLR = tfidf_impLR.transform(data_all_impLR)
    train_x_impLR = data_pro_impLR[:lentrain]
    # dic = tfidf_impLR.get_feature_names()
    test_x_impLR = data_pro_impLR[lentrain:]
    label_impLR = train_impLR['sentiment']

    print('build improved logistic regression model')
    model_impLR = LogisticRegression(penalty='l2', dual=True, tol=0.0001,
                               C=1, fit_intercept=True, intercept_scaling=1.0,
                               class_weight=None, random_state=None)
    print("20 Fold CV Score: ", np.mean(cross_val_score(model_impLR, train_x_impLR , label_impLR,
                                                        cv=20, scoring='roc_auc')))
    model_impLR.fit(train_x_impLR, label_impLR)

    result = model_impLR.predict_proba(test_x_impLR)[:, 1]
    out_test_impLR = pd.DataFrame(data={"id": test["id"], "sentiment": result})
    out_test_impLR.to_csv(os.path.join(data_path, 'out_test_impLR.csv'), index=False, quoting=3)
    print("model improved logistic Regression finished!")

if __name__  == "__main__":
    if (not os.path.exists(data_path + 'train_pro.csv') ) and (not os.path.exists(data_path + 'test_pro.csv')):
        pro_data()
    print('read data...')
    train = pd.read_csv(data_path + "train_pro.csv")
    test = pd.read_csv(data_path + "test_pro.csv")

    # print("tfidf")
    # tfidf = TfidfVectorizer(min_df = 2, strip_accents='unicode', analyzer='word', token_pattern=r'\w{1,}',
    #               ngram_range=(2, 3), use_idf=1, smooth_idf=1, sublinear_tf=1, stop_words='english')
    # data_all = train['review'].tolist() + test['review'].tolist()
    # tfidf.fit(data_all)
    # data_pro = tfidf.transform(data_all)
    # train_x = data_pro[:len(train)]
    # test_x = data_pro[len(train):]
    # label = train['sentiment']

    # NaiveBayes(train_x, test_x, label, test)
    # LogisticReg(train_x, test_x, label, test)
    # improve_LogisticRegression(train, test)
    XGB(train, test)