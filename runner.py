# Import required libraries
import torch
from transformers import BertTokenizer, BertModel

from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer
from nltk.stem import WordNetLemmatizer
import nltk
import re
import pandas_profiling as pp
import pandas as pd
import os

# Load Dataset
DATASET = "phishing_site_urls.csv"
DATA = []
assert os.path.exists(DATASET)
url_data = pd.read_csv(DATASET)
# Dropping duplicate entries
print("Entries before dropping duplicates: ", len(url_data))
url_data = url_data.drop_duplicates()
print("Entries after dropping duplicates: ", len(url_data))
url_data.head()


from nltk.corpus import stopwords
def removeStopWords(df, targetCol, resultCol):
    """
    Works InPlace
    """
    nltk.download('stopwords')
    print(stopwords.words('english'))
    stopWords=list(set(stopwords.words("english")))
    target = getattr(df, targetCol)
    url_data[resultCol] = target.astype(str)
    url_data[resultCol]=url_data[resultCol].apply(lambda x: "#".join(re.split(r"\W+", x)))
    df[resultCol]=df[resultCol].apply(lambda x:[word for word in x.split("#") if word not in stopWords])
    print(df.head())
    return df

def lemmatize(df, targetCol, resultCol):
    """
    Works InPlace
    """
    lemObj = WordNetLemmatizer()
    target = getattr(df, targetCol)
    df[resultCol] = target.astype(str)
    df[resultCol] = df[targetCol].map(lambda x: [lemObj.lemmatize(word) for word in x])
    print(df.head())
    return df

def balanceLabels(df):
    bad_balance = df[df["Label"] == "bad"]
    good_balance = df[df["Label"] == "good"].sample(n=len(bad_balance))
    balanced = good_balance.append(bad_balance)
    balanced = balanced.sample(frac=1)
    print(balanced.head())
    return balanced

def featureGeneration(df, targetCol, maxFeatures=500):
    vecObj = TfidfVectorizer(ngram_range=(1,1), max_features=maxFeatures)

    unigramdataGet= vecObj.fit_transform(df[targetCol].astype('str'))
    unigramdataGet = unigramdataGet.toarray()
    vocab = vecObj.get_feature_names_out()
    features=pd.DataFrame(np.round(unigramdataGet, 1), columns=vocab)
    features[features>0] = 1
    return features

def dataSplitter(df, features, randomState=21, testSize=0.2, shuffle=True):
    y = df["Label"]
    x_train,x_test,y_train,y_test =  train_test_split(features,y,random_state=randomState,
                                                      test_size=testSize, shuffle=shuffle)
    return x_train,x_test,y_train,y_test

url_data = removeStopWords(url_data, "URL", "cleanURL")
url_data = lemmatize(url_data, "cleanURL", "lemURL")
url_data_bal = balanceLabels(url_data)
# features_bal = featureGeneration(url_data_bal, "lemURL", maxFeatures=512)

# Check if CUDA is available and set device accordingly
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Load pre-trained BERT tokenizer and model onto device
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertModel.from_pretrained('bert-base-uncased').to(device)

# Define a function to create feature vectors for sentences
def get_bert_features(sentence):
    # Tokenize sentence and truncate to maximum sequence length
    tokenized = tokenizer(sentence, max_length=512, truncation=True, padding='max_length', return_tensors='pt').to(device)

    # Pass input IDs through BERT model to get embeddings
    with torch.no_grad():
        outputs = model(**tokenized)
        embeddings = outputs[0]

    # Average embeddings to create feature vector
    features = torch.mean(embeddings, dim=1).squeeze()

    return features.cpu().numpy()


import os
import csv
allLemData = url_data["lemURL"].tolist()
allLabelData = url_data["Label"]
nlpFeatures = []
import time
print("Starting feature extraction")
starttime = time.time()
import pdb
pdb.set_trace()
with open("processedData.csv", "w") as f:
    fields = ["staticFeatures", "nlpFeatures", "Label"]
    writer = csv.writer(f, delimiter=',')
    writer.writerow(fields)
    for idx in range(len(allLemData)):
        if os.path.exists("hook.txt"):
            import pdb
            pdb.set_trace()
        try:
            label = 0 if allLabelData[idx] == "good" else 1
            line = [allLemData[idx], get_bert_features(" ".join(allLemData[idx])), label]
            writer.writerow(line)
        except Exception as e:
            print("Skipped entry: ", idx, "\nException: ", e)
        if idx % 1000 == 0:
            dumpArr = np.array(nlpFeatures)
            np.save("nlpFeatures.npy", dumpArr)
            print("Time: {}; IDX: {}".format(time.time() - starttime, idx))
# Example usage
# sentence = "The quick brown fox jumps over the lazy dog."
# features = get_bert_features(sentence)
# print(features)

