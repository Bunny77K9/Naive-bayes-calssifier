''' naive bayes version :9 ; date: 21/01 13:07'''

from sklearn.feature_extraction.text import CountVectorizer
import sklearn
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.datasets import fetch_20newsgroups
from sklearn.pipeline import make_pipeline
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.model_selection import train_test_split, GridSearchCV
import numpy as np
import re
from sklearn.naive_bayes import MultinomialNB


class NaiveBayes(BaseEstimator, ClassifierMixin):

    def __init__(self, alpha=1, step_size=1):
        self.alpha = alpha
        self.step_size = step_size

    def fit(self, x, y):
        self.classes = list(set(y))
        x = x.toarray()
        priors = np.zeros(len(self.classes))
        N = x.shape[0]
        cond_prob=np.zeros((len(self.classes),x.shape[1]))
        for class_k in self.classes:
            N_c = x[y == class_k].shape[0]
            priors[class_k] = N_c / N
            counter_in_class = x[y == class_k]
            N_w=x.sum(axis=0)+x.shape[1]
            object_counter_in_class = counter_in_class.sum(axis=0)
            cond_prob[class_k, :] = (object_counter_in_class + self.alpha) / (N_w + self.step_size * self.alpha)
        self.priors=priors
        self.log_priors=np.log(priors)
        self.cond_prob=cond_prob
        return

    def predict(self,x):
        self.probs=np.zeros((x.shape[0],len(self.classes)))
        x=x.toarray()
        prediction=np.zeros(x.shape[0])
        for i in range(x.shape[0]):
            self.probs[i]=self.prdict_class(x[i])
        return np.argmax(self.probs,axis=1)

    def prdict_class(self,sample):
        score = np.zeros(len(self.classes))
        for class_k in self.classes:
            score[class_k] = np.log(self.priors[class_k])
            probabilites=self.cond_prob[class_k, np.where(sample != 0)]
            score[class_k] += sum(np.log(probabilites[0]))
        return score

    def predict_log_proba(self, x):
        _=self.predict(x)
        return self.probs


if __name__ == "__main__":

    dataset = fetch_20newsgroups(shuffle=True, random_state=1,
                                 remove=('headers', 'footers', 'quotes'))
    X = dataset.data#[:1000]
    y = dataset.target#[:1000]



    def preprocess_text(text):
        text = re.sub('((www\.[^\s]+)|(https?://[^\s]+))', 'URL', text)
        text = re.sub('@[^\s]+', 'USER', text)
        text = text.lower().replace("ё", "е")
        text = re.sub('[^a-zA-Zа-яА-Я1-9]+', ' ', text)
        text = re.sub(' +', ' ', text)
        text = re.sub('!', ' ', text)
        text = re.sub('$', ' ', text)
        return text.strip()


    #data = [preprocess_text(t) for t in X]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3,
                                                        random_state=10)

    # model_count = make_pipeline(CountVectorizer(stop_words='english'), NaiveBayes())
    # model_tdif_vect = make_pipeline(TfidfVectorizer(stop_words='english'),NaiveBayes())
    #
    # model_multinomial=make_pipeline(TfidfVectorizer(max_df=0.7, max_features=1000, norm='l2', preprocessor=None, smooth_idf=False,stop_words='english', use_idf=True),MultinomialNB())
    # models=[model_count,model_tdif_vect,model_multinomial]
    # models_name = [  '_count','_tdif', 'multinomial']
    # for model,name in zip(models,models_name):
    #     model.fit(X_train,y_train)
    #     y_test_pred = model.predict(X_test)
    #     accuracy=sklearn.metrics.accuracy_score(y_test, y_test_pred)
    #     print('accuracy for model {} is {}:'.format(name,accuracy))
    #
    # print('break point')
    ## compare to TfidfVectorizer preprocessing
    #''' we get better accuracy with tfidf instead of count, and in general sklearn's model is better-but not that much'''


    # #plot the learning curve:
    # from sklearn.model_selection import learning_curve
    # import matplotlib.pyplot as plt
    # from sklearn.model_selection import ShuffleSplit
    # #plt=plt.plot()
    # train_sizes=[0.1,0.2,0.5,0.7,0.8]#np.linspace(0.1, 1.0, 5)
    # #cv = ShuffleSplit(n_splits=100, test_size=0.2, random_state=0)
    #
    # Tfid=TfidfVectorizer(stop_words='english')
    # data_mat=Tfid.fit_transform(X)
    #
    # train_sizes, train_scores, test_scores= learning_curve(MultinomialNB(),data_mat, y,cv=len(train_sizes),n_jobs=1,train_sizes=train_sizes,shuffle=True)
    # train_scores_mean = np.mean(train_scores, axis=1)
    # train_scores_std = np.std(train_scores, axis=1)
    # test_scores_mean = np.mean(test_scores, axis=1)
    # test_scores_std = np.std(test_scores, axis=1)
    #
    # # Plot learning curve
    # plt.grid()
    # plt.fill_between(train_sizes, train_scores_mean - train_scores_std,
    #                      train_scores_mean + train_scores_std, alpha=0.1,
    #                      color="r")
    # plt.fill_between(train_sizes, test_scores_mean - test_scores_std,
    #                      test_scores_mean + test_scores_std, alpha=0.1,
    #                      color="g")
    # plt.plot(train_sizes, train_scores_mean, 'o-', color="r",
    #              label="Training score")
    # plt.plot(train_sizes, test_scores_mean, 'o-', color="g",label="Cross-validation(test) score")
    # plt.legend(loc="best")
    #
    # plt.show()
    # plt.xlabel('samples')
    # plt.ylabel('accuracy score')
    # print('break point')
    #answer: the model is in the variance regime -it is starting to get closer to bias regime only at last point


    #Grid search hyper parameter optimization:
    from sklearn.metrics import classification_report

    model_multinomial = make_pipeline(TfidfVectorizer(stop_words='english'), MultinomialNB())
    tuned_parameters={'tfidfvectorizer__max_df': [0.7, 0.75, 0.8, 1.0],'tfidfvectorizer__max_features':[None,500,2500,5000,7500,10000], 'tfidfvectorizer__norm':['l1', 'l2'],'tfidfvectorizer__use_idf':[True,False], 'tfidfvectorizer__smooth_idf':[True,False]}

    grid_search = GridSearchCV(model_multinomial, tuned_parameters)
    grid_search.fit(X_train, y_train)
    best_parameters = grid_search.best_estimator_.get_params()
    print(classification_report(y_test, grid_search.predict(X_test), digits=4))
    print(best_parameters)
    #so we got

    print('break point')


    #test with optimization

    model_tdif_opt_vect = make_pipeline(TfidfVectorizer(max_df=0.7, max_features=1000, norm='l2', preprocessor=None, smooth_idf=False,stop_words='english', use_idf=True),NaiveBayes())
    model_tdif_vect = make_pipeline(TfidfVectorizer(stop_words='english'),NaiveBayes())

    model_tdif_opt_vect.fit(X_train, y_train)
    y_test_pred = model_tdif_opt_vect.predict(X_test)
    opt_accuracy = sklearn.metrics.accuracy_score(y_test, y_test_pred)
    print('accuracy for opt model is {}:'.format( opt_accuracy))
    model_tdif_vect.fit(X_train, y_train)
    y_test_pred_ = model_tdif_vect.predict(X_test)
    accuracy = sklearn.metrics.accuracy_score(y_test, y_test_pred_)
    print('accuracy for model is {}:'.format(accuracy))
    print('break point')

    #Optional: Model interpretability

    def calc_p_c_given_xi(model):
        priors=model.class_log_prior_
        probs=model.feature_log_prob_
        ppc=np.exp(priors).reshape(-1,1)*np.exp(probs)
        prob_c_given_xi=ppc/np.sum(ppc,axis=0)
        return prob_c_given_xi


    # model = make_pipeline(TfidfVectorizer(max_df=0.7, max_features=1000, norm='l2', preprocessor=None, smooth_idf=False,stop_words='english', use_idf=True),MultinomialNB())
    # model.fit(X_train, y_train)
    # p_c_w = calc_p_c_given_xi(model)

    '''
    Optional: Model
    interpretability
    '''
    from sklearn.naive_bayes import MultinomialNB
    random_state = 42
    shuffle = True
    remove = []
    import pandas as pd

    data_train = fetch_20newsgroups(subset='train',
                                shuffle=shuffle,
                                random_state=random_state,
                                remove=remove)
    data_test = fetch_20newsgroups(subset='test',
                                shuffle=shuffle,
                                random_state=random_state,
                                remove=remove)
    target_names = data_train.target_names

    vectorizer=TfidfVectorizer(stop_words='english')
    vectorizer.fit(data_train.data)

    x_trs=vectorizer.fit_transform(data_train.data)
    x = vectorizer.transform(data_test.data)

    model = MultinomialNB()
    model.fit(x_trs,data_train.target)
    pred=model.predict(x)
    p_c_given_model_per_word=calc_p_c_given_xi(model)


    x_c_not_good=x.toarray()[np.where(pred!=data_test.target)][0]
    pred_for_c   = pred[np.where(pred!=data_test.target)][0]
    actual_class = data_test.target[np.where(pred != data_test.target)][0]