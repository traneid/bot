import joblib
import pandas as pd
from sklearn.model_selection import train_test_split
import string
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import SnowballStemmer
import nltk
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import precision_score, recall_score, precision_recall_curve
from sklearn.model_selection import GridSearchCV
import threading
import time
import pickle

import pprint

nltk.download('punkt')
nltk.download('stopwords')


class Checker:

    @staticmethod
    def is_int_positive(value: str) -> bool:
        return value.isnumeric()

    @staticmethod
    def is_int_negative(value: str) -> bool:
        return value.startswith('-') and value[1:].isnumeric()

    @staticmethod
    def is_float_positive(value: str) -> bool:
        return value.replace('.', '', 1).isnumeric()

    @staticmethod
    def is_float_negative(value: str) -> bool:
        return value.startswith('-') and value[1:].replace('.', '', 1).isnumeric()

    @classmethod
    def is_int(cls, value: str) -> bool:
        return cls.is_int_positive(value) or cls.is_int_negative(value)

    @classmethod
    def is_float(cls, value: str) -> bool:
        return cls.is_float_positive(value) or cls.is_float_negative(value)

    @staticmethod
    def is_number_positive(value: str) -> bool:
        return value.replace('.', '', 1).isnumeric()

    @staticmethod
    def is_number_negative(value: str) -> bool:
        return value.startswith('-') and value[1:].replace('.', '', 1).isnumeric()

    @classmethod
    def is_number(cls, value: str):
        return cls.is_int(value) or cls.is_float(value)

    @classmethod
    def is_zero(cls, value: str):
        return cls.is_number(value) and not value.replace('.', '', 1).replace('0', '').replace('-', '', 1)


class IiPy:
    def __init__(self):
        self.learning_model = {}

        self.normalization(self.learning_model)
        self.finished_model = self.model_training(self.learning_model)

    def start(self, handling):
        print(handling)
        thr_list = []
        # start = time.time()
        temp = []

        tt = []
        start = time.time()
        for i in self.finished_model:
            thr = threading.Thread(target=self.response_prediction,
                                   args=(self.finished_model, self.learning_model, handling, i, tt), )
            thr_list.append(thr)
            thr.start()

        for i in thr_list:
            i.join()

        # if len(tt) > 1:
        #     for i in tt:
        #         if i != 'Корректировка данных':
        #             # temp.append(f'Категория: {i}')
        #             return f'Категория: {i}' , time.time() - start
        #             # print("Категория:", i)
        # else:
        if len(tt) != 0:
            return f'Категория: {tt[-1]}', time.time() - start
        else:
            return f'Категория: не найдена ', time.time() - start
            # temp.append(f'Категория: {tt[-1]}')
            # print("Категория:", tt[-1], index)
        # end = time.time() - start

        # print(end)
        # print(temp)

    def response_prediction(self, finished_model: dict, learning_model: dict, handling, i, temp) -> None:
        # 'Корректировка данных': 708,
        f_0_92 = {
            'АВТОГРАФ': 692,
            'Заказ на ТС': 560,
            'Штрафы': 707,

            'Служебная записка': 650,
            'График работ': 669,
            'Лицензии не запускается': 694,
            'АРМ Кассира планшет': 680,
            'Сбис': 686,
            'Топливо': 677

        }

        answer = finished_model[i].predict([handling])
        probas_pred = finished_model[i].predict_proba(learning_model[i][-1]["Описание"])[:, 1]
        prec_c_10, rec_c_10, thresholds_c_10 = precision_recall_curve(y_true=learning_model[i][-1]["index"],
                                                                      probas_pred=probas_pred)

        threshold_index = f_0_92[i]

        prec_10 = precision_score(y_true=learning_model[i][-1]["index"],
                                  y_pred=probas_pred > thresholds_c_10[threshold_index])

        rec_10 = recall_score(y_true=learning_model[i][-1]["index"],
                              y_pred=probas_pred > thresholds_c_10[threshold_index])
        print(answer)
        if answer[-1] == 1:
            temp.append(i)

    @staticmethod
    def tokenize_sentence(sentence: str, remove_stop_words: bool = True)->list:
        snowball = SnowballStemmer(language="russian")
        russian_stop_words = set(stopwords.words("russian"))
        bead_symbols = {'``', "''", '+', '-', 'pecom.ru', '«', '»', 'доб.', '“', '”', 'тел.', ',', 'автопэк', '..',
                        'www.avtopek.ru', '.'}

        tokens = word_tokenize(sentence, language="russian")
        tokens = [token.lower() for token in tokens if token not in string.punctuation]

        temp = []
        for token in tokens:
            for rep_symbol in bead_symbols:
                token = token.replace(rep_symbol, '')

            if token == 'уважением':
                break
            elif token:
                temp.append(token)

        if remove_stop_words:
            temp = [token for token in temp if token not in russian_stop_words]

        stemmed_tokens = [snowball.stem(token) for token in temp]

        return stemmed_tokens

    def normalization(self, learning_model: dict) -> None:
        df_keys = pd.read_excel('answer_techer_v3.xlsx', sheet_name=None).keys()
        for key in df_keys:
            df = pd.read_excel('answer_techer_v3.xlsx', sheet_name=key)
            df["index"] = df["index"].astype(int)
            df["Описание"] = df["Описание"].astype(str)
            train_df, test_df = train_test_split(df, test_size=1000, random_state=0)
            learning_model[key] = [train_df, test_df]

    def model_training(self, learning_model: dict) -> dict:
        finished_model = {}
        for i in learning_model:
            vectorizer = TfidfVectorizer(tokenizer=lambda x: IiPy.tokenize_sentence(x, remove_stop_words=True))

            features = vectorizer.fit_transform(learning_model[i][0]["Описание"])

            model = LogisticRegression(random_state=0)
            model.fit(features, learning_model[i][0]["index"])

            model_pipeline_c_10 = Pipeline([
                ("vectorizer", TfidfVectorizer(tokenizer=lambda x: IiPy.tokenize_sentence(x, remove_stop_words=True))),
                ("model", LogisticRegression(random_state=0, C=10.))
            ]
            )
            finished_model[i] = model_pipeline_c_10.fit(learning_model[i][0]["Описание"],
                                                        learning_model[i][0]["index"])

        return finished_model


