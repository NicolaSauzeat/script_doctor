import multiprocessing
from multiprocessing import Pool
import pickle
from numpy import linalg
import math
import gensim
from gensim.models import Word2Vec
import torch
import seaborn
import pandas as pd
import sklearn
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
from transformers import CamembertForSequenceClassification, CamembertTokenizer,CamembertModel, CamembertConfig, AdamW
from sentence_transformers import models, losses
from sentence_transformers import SentencesDataset, LoggingHandler, SentenceTransformer, util
from sentence_transformers.evaluation import LabelAccuracyEvaluator
from transformers import AutoTokenizer, AutoModelForTokenClassification
from sentence_transformers.readers import *
from transformers import pipeline
from sklearn import metrics
from pathlib import Path
import multiprocessing
from multiprocessing import Pool
from training import Preprocessing
preprocessing = Preprocessing()
from nltk import word_tokenize
from sklearn.feature_extraction.text import CountVectorizer
import nltk
import numpy as np
nltk.download('punkt')
import itertools
from scipy import spatial
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from sklearn import metrics
from joblib import dump, load
import json

class Compute:

    def __init__(self):
        self.pool = Pool(multiprocessing.cpu_count() - 1)
        self.path_in = Path("Resultats/in/")
        self.path_out = Path("Resultats/out/")
        self.batch = 32


    """
    Partie Importation
    """
    def load_final_data(self):
        clear_sentences =pd.read_csv(self.path_out/"final_sentences_after_preprocessing.csv",sep="!")
        clear_sentences["stop_word_removal"] = clear_sentences["stop_word_removal"].apply(
        lambda x: x.replace(" ","").replace("'", "").replace("[", "").replace("]", "").split(","))
        hs_code = pd.read_excel(self.path_in/"NC2017.xls")
        hs_code_final = hs_code.loc[(hs_code["CODE NC\n2017"].str.len() >=4) & (hs_code["CODE NC\n2017"].str.len() <6)]
        hs_code_final.rename(columns={"CODE NC\n2017":"code_nc_2017", "LIBELLE AUTOSUFFISANT NC 2017":"libelle_nc_2017"}, inplace=True)
        return clear_sentences, hs_code_final

    @staticmethod
    def load_bert_model():
        # On importe la version pre-entrainée de camemBERT 'base'
        # TODO change value of num_labels with len of hs_categories
        model = CamembertForSequenceClassification.from_pretrained(
            'camembert-base',
            num_labels=2)
        return model


    """
    Partie Préprocessing + Initialisation
    """
    def preprocess_bert_model(self, final_sentences):
        validation_set = pd.read_csv(self.path_in /"jeu_annote_final.csv", sep="!", usecols=[2,3,4,5,6,7,8,9,10,11,12])
        validation_set_final = final_sentences.merge(validation_set, left_on="siren", right_on="siren")
        validation_set_final.drop(columns=validation_set_final.columns[0:2], inplace=True)
        return validation_set_final


    # TODO relancer avec les phrases préprocessées
    def initialize_bert_model(self, validation_set):
        tokenizer, encoded_batch, config, model, optimizer, epochs, device= self.initialize_model(validation_set)
        sentiments = self.encode_validation_set(validation_set)
#        all_sentences = validation_set['encoded_sentence'].tolist()
#        validation_set.apply(lambda x: self.best_match(x["encoded_sentence"],all_sentences, x.index), axis=1)
#        util.cos_sim(validation_set["encoded_sentence"], list_worktype_categ_embedding)
        train_dataset, validation_set = self.split_dataset(encoded_batch, sentiments, split_border=int(len(sentiments) * 0.9))
        train_dataloader, validation_dataloader = self.initialize_dataloader(train_dataset, validation_set)
        model = self.train_bert_model(optimizer, epochs, device, encoded_batch, tokenizer, model, train_dataloader)
        #self.evaluate(validation_dataloader, sentiments, model,split_border=int(len(sentiments) * 0.9))
        return validation_dataloader, train_dataloader


    def make_dict(self, x):
        if x== "Travaux de maçonnerie":
            return 0
        elif x == 'Peinture intérieure':
            return 1
        elif x == "Pompe à chaleur":
            return 2
        elif x == "Climatisation réversible":
            return 3
        elif x == "Fenêtre PVC":
            return 4
        elif x == "Salle de bains":
            return 5
        elif x == "Multiservice / Petits travaux":
            return 6
        elif x == "Entreprise générale de rénovation":
            return 7
        elif x == "Carrelage/Carreleur":
            return 8
        elif x == 'Cuisiniste':
            return 9

    def best_match(self, embedding, list_worktype_categ, list_worktype_categ_embedding):
        list_scores = util.cos_sim(embedding, list_worktype_categ[0])
        idx_best_score = np.argmax(list_scores)
        return list_worktype_categ[idx_best_score], float(list_scores[idx_best_score])


    """
    @staticmethod
    def initialize_training_dataset(sentiments):
        # On transforme la liste des sentiments en tenseur
        # On calcule l'indice qui va delimiter nos datasets d'entrainement et de validation
        # On utilise 80% du jeu de donnée pour l'entrainement et les 20% restant pour la validation
        split_border = int(len(sentiments) * 0.8)
        train_dataset = TensorDataset(
            encoded_batch['input_ids'][:split_border],
            encoded_batch['attention_mask'][:split_border],
            sentiments[:split_border])

        validation_dataset = TensorDataset(
            encoded_batch['input_ids'][split_border:],
            encoded_batch['attention_mask'][split_border:],
            sentiments[split_border:])
        return train_dataset, validation_dataset
        """
    """
    def initialize_dataloader(self, validation_dataset):
        # On cree les DataLoaders d'entrainement et de validation
        # Le dataloader est juste un objet iterable
        # On le configure pour iterer le jeu d'entrainement de façon aleatoire et creer les batchs.
        train_dataloader = DataLoader(
            train_dataset, sampler=RandomSampler(train_dataset), batch_size=self.batch)
        validation_dataloader = DataLoader(
            validation_dataset,
            sampler=SequentialSampler(validation_dataset), batch_size=self.batch)
        return train_dataloader, validation_dataloader
    """



    """
    Partie Entraînement de modèle
    """
    # TODO interessant
    """
    def train_bert_model(self, validation, categories):
        tokenizer, encoded_batch, sentiments = self.initialize_bert_model(validation_set, hs_code)
        train_dataset, validation_dataset = self.initialize_training_dataset(encoded_batch, sentiments)
        train_dataloader, validation_dataloader = self.initialize_dataloader(train_dataset, validation_dataset)
        model = self.load_bert_model()
        return validation_dataset, train_dataset
    """
    def train_w2v_model(self, *cols, **kwargs):
        # Word2Vec model training
        # Save model
        self["Desc Avant"] = self["Desc Avant"].apply(lambda x: word_tokenize(x, language="french"))
        #sentences =[ [*set(sentences)] for sentences in self["Desc Avant"]]
        #hs_code["token"] = hs_code["libelle_nc_2017"].apply(lambda x: word_tokenize(x, language="french"))
        # TODO del tok in hs_code["token"] if len(tok) <2
        #test = [tok for tok in hs_code["token"] if len(tok)>2]
        #sentences.extend(hs_code["libelle_nc_2017"].tolist())

        model_cbow = Word2Vec(sentences=self["Desc Avant"],sg=1, min_count=1)
        model_cbow.build_vocab(self["Desc Avant"])
        model_cbow.train(self["Desc Avant"], total_examples=model_cbow.corpus_count, epochs=200)
        word_vectors = model_cbow.wv
        results= self.similarity()
        #model_cbow.wv.cosine_similarities(self["Desc Avant"].iloc[0], self["Desc Avant"])
        df = pd.DataFrame(data=self["Desc Avant"], index=[row for row in range(0,len(self["Desc Avant"]))])
        df.at['score'] = 0
        model_cbow.save(self.path_out/"cbow_model_one.model")

    def tagged_document(self,list_of_list_of_words):
        for i, list_of_words in enumerate(list_of_list_of_words):
            yield gensim.models.doc2vec.TaggedDocument(list_of_words, [i])

    def readWeights(self, frequency_file):
        data = open(frequency_file,"r").read()
        data = data.split('\n')
        del data[-1]
        self.word_weight = {}
        for d in data:
            wordpair = d.split(" ")
            wordpair[0] = wordpair[0][:-1]
            self.word_weight[wordpair[0]] = int(wordpair[1])
    """
    Partie Vectorisation 
    """


    def get_vector(self, sentence):
        # sentence = self.clear_sentence(sentence)
        vectors = [self.model_cbow.wv[w] for w in word_tokenize(sentence, language='french')
                   if w in self.model.wv]

        weights = [1 / (math.log(self.word_weight.get(w, 10) + 0.00001)) for w in
                   word_tokenize(sentence, language='french')
                   if w in self.model.wv]
        v = np.zeros(self.model.vector_size)
        if (len(vectors) > 0):
            v = np.average(vectors, axis=0, weights=weights)  # Commande pour moyenne pondérée
            # v = (np.array([sum(x) for x in zip(*vectors)])) / v.size # Commande pour moyenne normale
        return v




    """
    Partie Calcul
    
    """
    def avg_feature_vector(sentence, model, num_features, index2word_set):
        words = sentence.split()
        feature_vec = np.zeros((num_features,), dtype='float32')
        n_words = 0
        for word in words:
            if word in index2word_set:
                n_words += 1
                feature_vec = np.add(feature_vec, model[word])
        if (n_words > 0):
            feature_vec = np.divide(feature_vec, n_words)
        return feature_vec


    def similarity(self, first_sentence, second_sentence):
        first_vector = self.get_vector(first_sentence)
        second_vector = self.get_vector(second_sentence)
        score = 0
        if first_vector.size > 0 and second_vector.size > 0:
            if (linalg.norm(first_vector) * linalg.norm(second_vector)) > 0:
                score = np.dot(first_vector, second_vector) / (linalg.norm(first_vector) * linalg.norm(second_vector))  # Similarité cosinus
        return score


    def get_score(df, model_cbow):
        model_cbow.readWeights("statistiques.txt")
        for i, row in df.iterrows():
            df.at[i, 'score'] = model_cbow.similarity(row["description"], row["WEB"])
            # scores.append(score)
            # cedcodes[idx] = sentence
            print(i)
            # df.at[i, 'score'] = score
        # df.to_csv('scores_juste2.csv', sep="!")#D
        return df

    """
    Partie CLustering
    """
    def get_classification(self, x_train, y_train):
        classifier = sklearn.ensemble.RandomForestClassifier(n_estimators=1000, random_state=0)
        classifier.fit(x_train, y_train)
        return classifier


    def get_results(self, X_test,classifier ):
        y_pred = classifier.predict(X_test)

    """
    Autres
    """
    def tache_parallele(self, df, fonction):
        # create a pool for multiprocessing
        # split your dataframe to execute on these pools
        splitted_df = np.array_split(df, len(self.pool._pool))
        # execute in parallel:
        split_df_results = self.pool.map(fonction, splitted_df)
        # combine your results
        df = pd.concat(split_df_results)
        self.pool.close()
        self.pool.join()
        return df


    # TODO try to use categories embedded as center of cluster
    def tf_idf(self, *sentence):
        vectorizer = TfidfVectorizer()
        test = [x for x in sentence[0]["sentence"] if len(x) <10]
        X = vectorizer.fit_transform(sentence[0]['sentence'].tolist())

        true_k = 10
        model = KMeans(n_clusters=true_k, init='k-means++', max_iter=100, n_init=1)
        model.fit(X)
        order_centroids = model.cluster_centers_.argsort()[:, ::-1]
        terms = vectorizer.get_feature_names()
        for i in range(10):
            print("Cluster %d:" % i),
            for ind in order_centroids[i, :10]:
                print(' %s' % terms[ind]),
            print
        dump(model, self.path_out/'tf_idf_model_naive.joblib')
        vectorizer = TfidfVectorizer(ngram_range=(2,4))

        X = vectorizer.fit_transform(sentence[0]['sentence'].tolist())

        model = KMeans(n_clusters=20, init='k-means++', max_iter=100, n_init=1)
        model.fit(X)
        order_centroids = model.cluster_centers_.argsort()[:, ::-1]
        terms = vectorizer.get_feature_names()
        for i in range(20):
            print("Cluster %d:" % i),
            for ind in order_centroids[i, :10]:
                print(' %s' % terms[ind]),
            print
        dump(model, self.path_out/'tf_idf_model_naive_ngramm.joblib')

        vectorizer = TfidfVectorizer(ngram_range=(2, 4))

        X = vectorizer.fit_transform(sentence[0]['lemmatize_with_stp_wrd'].tolist())

        model = KMeans(n_clusters=20, init='k-means++', max_iter=100, n_init=1)
        model.fit(X)
        dump(model, self.path_out / 'tf_idf_model_ngramm_non_naive.joblib')

        print("Top terms per cluster:")
        order_centroids = model.cluster_centers_.argsort()[:, ::-1]
        terms = vectorizer.get_feature_names()
        for i in range(20):
            print("Cluster %d:" % i),
            for ind in order_centroids[i, :10]:
                print(' %s' % terms[ind]),
            print



        vectorizer = TfidfVectorizer(ngram_range=(1, 3))

        X = vectorizer.fit_transform(sentence[0]['lemmatize_with_stp_wrd'].tolist())

        model = KMeans(n_clusters=20, init='k-means++', max_iter=100, n_init=1)
        model.fit(X)
        dump(model, self.path_out / 'tf_idf_model_less_ngramm_non_naive.joblib')

        print("Top terms per cluster:")
        order_centroids = model.cluster_centers_.argsort()[:, ::-1]
        terms = vectorizer.get_feature_names()
        for i in range(20):
            print("Cluster %d:" % i),
            for ind in order_centroids[i, :10]:
                print(' %s' % terms[ind]),
            print

        print("\n")
        print("Prediction")
        Y = vectorizer.transform(["Monsieur bougna souhaite rénover le totalité de un appartement de un superficie 35m2 ; cuisine salle de bain couloir plafond. le consommateur n'être disponible que à partir de 11h00 .n'hésiter pas à l'appeler pour convenir avec ce dernier de un heure de passage précise ."])
        prediction = model.predict(Y)
        print(prediction)
        Y = vectorizer.transform(["My cat is hungry."])
        prediction = model.predict(Y)
        print(prediction)


    def initialize_model(self, validation_set):
        tokenizer = CamembertTokenizer.from_pretrained('camembert-base',
                                                       do_lower_case=True, max_length=len(max(validation_set["lemmatize_with_stp_wrd"])),
                                                       sep_token=".", padding=True)
        encoded_batch = tokenizer.batch_encode_plus(validation_set["lemmatize_with_stp_wrd"].values.tolist(),
                                                    dd_special_tokens=True,
                                                    max_length=len(max(validation_set["lemmatize_with_stp_wrd"])),
                                                    padding=True,
                                                    truncation=True,
                                                    return_attention_mask=True,
                                                    return_tensors='pt')
        config = CamembertConfig.from_pretrained("camembert-base", output_hidden_states=True)
        model_similarity = CamembertModel.from_pretrained("camembert-base", config=config)
        model_classification = CamembertForSequenceClassification.from_pretrained('camembert-base', num_labels = 10)
        optimizer = AdamW(model_classification.parameters(), lr = 2e-5, eps = 1e-8)
        epochs=3
        device = torch.device("cpu")
        return tokenizer, encoded_batch, config, model_classification, optimizer, epochs, device

    def split_dataset(self, encoded_batch, sentiments, split_border):
        train_dataset = TensorDataset(
        encoded_batch['input_ids'][:split_border],
        encoded_batch['attention_mask'][:split_border],
        sentiments[:split_border])

        validation_dataset = TensorDataset(
        encoded_batch['input_ids'][split_border:],
        encoded_batch['attention_mask'][split_border:],
        sentiments[split_border:])
        return train_dataset, validation_dataset

    def initialize_dataloader(self, train_dataset, validation_dataset):
        train_dataloader = DataLoader(
        train_dataset,
        sampler = RandomSampler(train_dataset),
        batch_size = 2)

        validation_dataloader = DataLoader(
        validation_dataset,
        sampler = SequentialSampler(validation_dataset),
        batch_size = 2)

        return train_dataloader, validation_dataloader

    def encode_validation_set(self, validation_set):
        #validation_set.drop_duplicates(subset=["caseId", "sentence"], inplace=True)
        validation_set.reset_index(inplace=True, drop=True)
        index = pd.Index([i for i in range(0, len(validation_set))])
        validation_set.set_index(index)
        # TODO a retravailler
        validation_set["encoded_categories"] = validation_set["Categorie"].apply(lambda x: self.make_dict(x))
        sentiments = torch.tensor(validation_set["encoded_categories"].values.tolist())
        return sentiments

    def preprocess(raw_reviews, sentiments=None):
        encoded_batch = TOKENIZER.batch_encode_plus(raw_reviews, truncation = True, pad_to_max_length = True,
        return_attention_mask = True, return_tensors = 'pt')
        if sentiments:
            sentiments = torch.tensor(sentiments)
            return encoded_batch['input_ids'], encoded_batch['attention_mask'], sentiments
        return encoded_batch['input_ids'], encoded_batch['attention_mask']

    def predict(self, validation_set,  model, split_border):
        model.load_state_dict(torch.load(self.path_out/"sentiments.pt"))
        device = torch.device("cpu")
        with torch.no_grad():
                model.eval()
        #input_ids, attention_mask = preprocess(reviews)
        #inputs_id = list(zip(*test_dataloader))[0][0]
        #attention_mask = list(zip(*test_dataloader))[0][1]
        #inputs_id = []
        #attention_mask = []
        sentiment = []
        #inputs_id = test_dataloader.dataset.tensors[0]
        #attention_mask = test_dataloader.dataset.tensors[1]
        #sentiment = test_dataloader.dataset.tensors[2]
        #inputs_id = test_dataloader.dataset.tensors[0].to(device)
        #attention_mask = test_dataloader.dataset.tensors[1].to(device)
        #inputs_ids = encoded_batch['input_ids'][split_border:]
        #attention_mask = encoded_batch['attention_mask'][split_border:]
        #tokenizer = CamembertTokenizer.from_pretrained('camembert-base',
        #                                               do_lower_case=True, max_length=len(max(validation_set["sentence"])),
        #                                               sep_token=".", padding=True)

        #encoded_batch = tokenizer.batch_encode_plus(validation_set["encoded_sentence"][split_border:],
        #add_special_tokens = True,
        #max_length = len(max(validation_set["sentence"])),
        #padding = True,
        #truncation = True,
        #return_attention_mask = True,
        #return_tensors = 'pt')


        retour = model(validation_set.dataset.tensors[:][0].to(device),attention_mask=validation_set.dataset.tensors[:][1].to(device))
        results=[]
        for batch in validation_set:
            retour= model(batch[0].to(device), attention_mask= batch[1].to(device))
            results.extend(torch.argmax(retour[0], dim=1))
        return torch.argmax(retour[0], dim=1)

    def evaluate(self, encoded_batch, sentiments, model,split_border):
        predictions = self.predict(encoded_batch, model,split_border)
        print(metrics.f1_score(sentiments[split_border:], predictions, average='weighted', zero_division=0))
        seaborn.heatmap(metrics.confusion_matrix(sentiments, predictions))

    def train_bert_model(self, optimizer, epochs, device, encoded_batch, tokenizer, model, train_dataloader):
        for epoch in range(0, epochs):
            print("")
            print(f'########## Epoch {epoch + 1} / {epochs} ##########')
            print('Training...')
            # On initialise la loss pour cette epoque
            total_train_loss = 0
            # On met le modele en mode 'training'
            # Dans ce mode certaines couches du modele agissent differement
            model.train()
            # Pour chaque batch
            training_stats = []
            for step, batch in enumerate(train_dataloader):
                # On fait un print chaque 40 batchs
                if step % 40 == 0 and not step == 0:
                    print(f'  Batch {step}  of{len(train_dataloader)}.')
                # On recupere les donnees du batch
                input_id = batch[0].to(device)
                attention_mask = batch[1].to(device)
                sentiment = batch[2].to(device)
                # On met le gradient a 0
                model.zero_grad()
                # On passe la donnee au model et on recupere la loss et le logits (sortie avant fonction d'activation)
                loss, logits = model(input_id,
                token_type_ids = None,
                attention_mask = attention_mask,
                labels = sentiment).loss,\
                model(input_id,
                token_type_ids = None,
                attention_mask = attention_mask,
                labels = sentiment).logits
                # On incremente la loss totale
                 # .item() donne la valeur numerique de la loss
                total_train_loss += loss.item()
                 # Backpropagtion
                loss.backward()

                  # On actualise les parametrer grace a l'optimizer
                optimizer.step()
                  # On calcule la  loss moyenne sur toute l'epoque
                avg_train_loss = total_train_loss / len(train_dataloader)
                print("")
                print("  Average training loss: {0:.2f}".format(avg_train_loss))
                  # Enregistrement des stats de l'epoque
                training_stats.append(
                {
                'epoch': epoch + 1,
                'Training Loss': avg_train_loss,
                }
                )
                print("Model saved!")

        with open(self.path_out / "stat_entrainement.json", "x") as outfile:
            json.dump(training_stats, outfile)
        torch.save(model.state_dict(), self.path_out /"sentiments_with_preprocess.pt")
        return model
    def vectorize_with_sklearn(self, documents):
        vectorizer = CountVectorizer(max_features=1500, min_df=5, max_df=0.7 )#stop_words=stopwords.words('english'))
        X = vectorizer.fit_transform(documents).toarray()
        return X


    def initialize_ner(self, dataset_test):
        tokenizer = CamembertTokenizer.from_pretrained("Jean-Baptiste/camembert-ner")
        model = AutoModelForTokenClassification.from_pretrained("Jean-Baptiste/camembert-ner")
        nlp = pipeline('ner', model=model, tokenizer=tokenizer, aggregation_strategy="simple")
        dataset_test.dropna(subset=["message_rdv"], inplace=True)
        dataset_test.dropna(subset=["nom_rdv"], inplace=True)
        test = []
        for doc, index in zip(dataset_test["message_rdv"], dataset_test["caseId"]):
            test.append((nlp(doc), index))

        personne = []
        for row in test:
            try:
                if len(row[0]) > 1:
                    for item in  row[0]:
                        if item["entity_group"] == "PER":
                            personne.append((item["entity_group"], item["word"], item["score"],row[1]))
                elif len(row[0]) == 1:
                        for item in row[0]:
                            if item["entity_group"] == "PER":
                                personne.append((item["entity_group"], item["word"], item["score"],row[1]))
                            else: continue
                else: continue
            except KeyError:
                print("oups")
            except TypeError:
                print("oups")
        dataframe_personne = pd.DataFrame(personne, columns=['entite', 'personne', 'score', "caseId"])
        dataframe_personne.to_csv(self.path_out/"ner_test.csv",sep="!")

        print("hi")