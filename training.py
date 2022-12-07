from collections import Counter
from re import A
import re
from nltk.stem.snowball import SnowballStemmer
stemmer = SnowballStemmer(language='french')
#python -m spacy download fr_core_news_sm
from spacy.lang.fr.examples import sentences

import nltk
import pandas as pd
import numpy as np
# Appel des bibliothèques
from numpy import dot
from numpy.linalg import norm
from nltk import word_tokenize
from nltk.corpus import stopwords
from gensim.models import Word2Vec
import multiprocessing
from multiprocessing import Pool
import os
import spacy
import sent2vec
from sent2vec.vectorizer import Vectorizer
#vectorizer = Vectorizer()
from pathlib import Path
import time
import xlrd

class Preprocessing:

    def __init__(self):
        self.final_french_stopwords_list = stopwords.words('french')
        self.final_english_stopwords_list = stopwords.words('english')
        self.path_in = Path("Resultats/in/")
        self.path_out = Path("Resultats/out/")



    """
    Partie Importation dataset
    """
    # TODO à modifier
    def read_data(self):
        # Import file7
        dataset = pd.read_csv(self.path_in / "demande_conso_avec_labellisation.csv", sep ="|", engine="python", encoding="utf-8")
        correspondance_id_cat_id = pd.read_csv(self.path_in / "correspondances_id_categ_id_worktype.csv", sep=",")
        equivalence_forum = pd.read_excel(self.path_in / "EquivalencesForumConstruire - v29.xlsx",
                usecols= ["affiliate_cat_id","maison_cat_id","maison_wkt_id", "affiliate_cat_price", "affiliate_cat_lib",
                "maison_cat_lib", "maison_wkt_lib",	"liste_mots_positifs", "liste_mots_negatifs", "stopwords",
                "post_action", "returned_categ", "returned_wkt", "callcenter"], dtype={"maison_cat_id": float,
            "maison_wkt_id": float, "affiliate_cat_price": float, "affiliate_cat_lib":str,
            "maison_cat_lib": str, "maison_wkt_lib": str, "liste_mots_positifs": str, "returned_wkt": float, "callcenter":str,
            "liste_mot_negatifs": str, "stop_words": str, "post_action": str, "returned_categ": float})
        equivalence_devisplus = pd.read_excel(self.path_in / "EquivalencesDevisplus_v1.xlsx",
                                              usecols=["affiliate_id", "affiliate_cat_id", 'maison_cat_id',
                                                       'maison_wkt_id', 'affiliate_cat_price',
                                                       'affiliate_cat_lib', 'maison_cat_lib', 'maison_wkt_lib',
                                                       'created_at', 'liste_mots_positifs',
                                                       'liste_mots_negatifs', 'stopwords', 'post_action',
                                                       'returned_categ', 'returned_wkt', 'callcenter'])
        equivalence_forum.dropna(subset=["maison_cat_id"], inplace=True)
        supplement_word = pd.read_excel(self.path_in / "table_matching_categs.xlsx")
        for row in supplement_word.index:
                try:
                    if supplement_word.loc[row, "lib_cwt"] == supplement_word.loc[row, "lib_cwt"] and "->" in supplement_word.loc[row, "lib_cwt"]:
                        supplement_word.loc[row, "lib_cwt"] = supplement_word.loc[row, "lib_cwt"].split("->")[0]
                    elif supplement_word.loc[row, "lib_cwt"] != supplement_word.loc[row, "lib_cwt"]:
                        supplement_word.drop(index =row,inplace=True)
                    else : continue
                except TypeError:
                    print('oups')
        importance = dataset.groupby("Origin")["caseId"].count()
        categorie = dataset.groupby("Categorie")["caseId"].count()
        categorie_main = categorie.sort_values(ascending=False)[:10]
        categorie_main.to_excel(self.path_out / "main_categories.xlsx")
        dataset_test = dataset.loc[dataset["Categorie"].isin(categorie_main.index.tolist())]
        dataset_test["sentence"] = ""
        for row in dataset_test.index:
            if  dataset_test.loc[row, "message_rdv"] == dataset_test.loc[row, "message_rdv"]:
                dataset_test.loc[row, "sentence"] = dataset_test.loc[row, "message_rdv"]
            else: dataset_test.loc[row, "sentence"] = dataset.loc[row, "message_from_ping"]

        dataset_test.dropna(subset="sentence", inplace=True)
        dataset_test["sentence"] = dataset_test["sentence"].apply(lambda x: x.replace("\n", ""))
        dataset_test["sentence"]= dataset_test["sentence"].apply(lambda x: x.lower())
        dataset_test["sentence"] = dataset_test["sentence"].apply(lambda x: x.replace("bonjour", ""))

        equivalence_forum["maison_cat_id"] = equivalence_forum["maison_cat_id"].astype(int)
        equivalence_devisplus.dropna(subset=["maison_cat_id"], inplace=True)
        equivalence_forum["affiliate_cat_id"] = equivalence_forum["affiliate_cat_id"] .apply(lambda x: int(x) if x == x else print(x))
        equivalence_devisplus["maison_cat_id"] = equivalence_devisplus["maison_cat_id"].apply(lambda x: int(x) if x == x else print(x))
        fob = pd.read_excel(self.path_in / "Fob_Francisco.xlsx", usecols=["Desc Avant", "Desc Après"])

        self.dict_transition_forum = dict(zip(equivalence_forum["affiliate_cat_id"],
                                            equivalence_forum["maison_cat_id"]))
        self.dict_transition_devisplus = dict(zip(equivalence_devisplus["affiliate_cat_id"],
                                                  equivalence_devisplus["maison_cat_id"]))
        return equivalence_forum, equivalence_devisplus, fob, dataset_test, categorie_main

    # TODO à modifier
    def import_consolided_data(self):
        consolided_data = pd.read_csv(self.path_out/"consolided_data.csv", sep="!", engine="python")
        return consolided_data



    @staticmethod
    def arrange_word(dataset):
        dataset = dataset.apply(lambda x: x.replace("\n", ""))
        return dataset

    """
    Partie Tokénization
    """
    # TODO modifier nom colonne
    def tokenize_sentence(self, scrap, lemmatize= True):
        # Define regex
        regex = re.compile('[^A-zÀ-ú]+')
        # Replace unwanted symbols
        scrap["web"] = scrap["web"].apply(lambda x: regex.sub(' ', x))
        #
        scrap["web"] = scrap["web"].apply(lambda x: x.replace("  ", " "))
        if lemmatize == True:
            #scrap = self.lemmatize_sentence(scrap)
             scrap = self.tache_parallele(scrap, self.lemmatize_sentence)
        # Get tokenize sentences
        tokenized_sentences = self.get_token(scrap)
        return tokenized_sentences

    # TODO à modifier
    @staticmethod
    def get_token(x):
        # Tokenize all sentences
        x["token"] = x["web"].apply(lambda x: word_tokenize(x, language="french"))
        return x

    """
    Partie Pré-processing Modulable
    """
    def take_all_preprocess(self, data):
        data = self.stop_word_removal(data)
        data = self.lemmatize_sentence(data)
        data = self.return_stem(data)
        data.to_csv(self.path_out / 'final_data.csv', sep="¬")
        return data



    def stop_word_removal(self, raw_sentence):
        # Stop Words removal in french
        # TODO arrange stop_word_removal
        self.final_french_stopwords_list.extend(self.final_english_stopwords_list)
        self.final_french_stopwords_list = np.unique(self.final_french_stopwords_list)
        for row in raw_sentence.index:
                start_time = time.time()
                raw_sentence.at[row, "sentence_stop_word"] = " ".join([x for x in raw_sentence.at[row, "sentence"].split(" ")
                                                if len(x) > 2 and x not in self.final_french_stopwords_list and x==x])
                end_time = time.time()
                print('delta = {}'.format(end_time - start_time))
        print("fin de la suppression des stop_word français")
        return raw_sentence        # TODO arrange stop_word_removal

    @staticmethod
    def lemmatize_sentence(raw_sentence):
        # Load lemmatize vocab for french language
        nlp = spacy.load('fr_core_news_sm')
        # Get all lemmatized format
        raw_sentence["lemmatize_without_stp_wrd"] = " "
        nlp.max_length = 1030000000
        i = 0
        for doc in raw_sentence["sentence"]:
                print(i)
                i += 1
                lemmatize = []
                try:
                        lemma = nlp(doc)
                        lemmatize.extend([token.lemma_ for token in lemma if token.lemma_ not in lemmatize])
                        raw_sentence.loc[raw_sentence["sentence"] == doc, "lemmatize_without_stp_wrd"] = " ".join([token for token in lemmatize])
                except ValueError:
                    print("oups, erreur de valeur")
                except TypeError:
                    print("oups, erreur de type")
        raw_sentence["lemmatize_with_stp_wrd"] = " "
        for doc in raw_sentence["sentence_stop_word"]:
                print(i)
                i += 1
                lemmatize = []
                try:
                        lemma = nlp(doc)
                        lemmatize.extend([token.lemma_ for token in lemma if token.lemma_ not in lemmatize])
                        raw_sentence.loc[raw_sentence["sentence_stop_word"] == doc, "lemmatize_with_stp_wrd"] = " ".join([token for token in lemmatize])
                except ValueError:
                    print("oups, erreur de valeur")
                except TypeError:
                    print("oups, erreur de type")
        return raw_sentence

    def return_stem(self, sentence):
        nlp = spacy.load('fr_core_news_sm')
        sentence["stemmed_sentence"] = ""
        sentence["stemmed_with_stp_word"] = ""
        for row in sentence.index:
            doc = nlp(sentence.loc[row, "sentence"])
            sentence.loc[row, 'stemmed_sentence'] = " ".join([stemmer.stem(X.text) for X in doc])
            doc = nlp(sentence.loc[row, "sentence_stop_word"])
            sentence.loc[row, "stemmed_with_stp_word"] = " ".join([stemmer.stem(X.text) for X in doc])
        return sentence

    def export_preprocessing(self, final_sentences):
        final_sentences.to_csv(self.path_out/"final_sentences_after_preprocessing.csv", sep="!")

    """
    Partie Autre
    """
    def tache_parallele(self, df, fonction):
        # create a pool for multiprocess;;ing
        # split your dataframe to execute on these pools
        pool = Pool(multiprocessing.cpu_count() - 1)
        splitted_df = np.array_split(df, len(pool._pool))
        # execute in parallel:
        split_df_results = pool.map(fonction, splitted_df)
        # combine your results
        df = pd.concat(split_df_results)
        pool.close()
        pool.join()
        return df

