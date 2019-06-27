#!/usr/bin/env python
# coding: utf-8

# # Import Libraries

# In[2]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import nltk


# # Read Preprocessed File

# In[3]:


preprocess = pd.read_csv('intermediate_dataset/preprocessed2.csv', encoding='latin1')


# In[4]:


preprocessed_df = preprocess.copy(deep=True)
preprocessed_df['Tag'] = preprocessed_df['Tag'].apply(lambda x: str(x).split(','))


# In[5]:


all_tags = [item for sublist in preprocessed_df['Tag'].values for item in sublist]


# In[6]:


tag_set = set(all_tags)
unique_tags = list(tag_set)


# # PLOT WORDCLOUD

# In[7]:


from wordcloud import WordCloud

def word_cloud(tags):
    u = ' '.join(tags)
    # print(u)
    wc=WordCloud(background_color='black',max_font_size=60).generate(u)
    plt.figure(figsize=(16,12))
    plt.imshow(wc, interpolation='bilinear')


# In[8]:


word_cloud(unique_tags)
len(unique_tags)


# # FIND THE MOST FREQUENT TAGS

# In[9]:


freq = nltk.FreqDist(all_tags)

frequent_tags = freq.most_common(100)

freq_tag = []
for tag in frequent_tags:
    freq_tag.append(tag[0])


# In[10]:


figure, ax = plt.subplots(figsize=(15, 10))
freq.plot(100, cumulative=False)


# # Separate Dataset According to Frequent Tags

# In[11]:


def most_frequent(t, frequent):
    
    tags = []
    for i in range(len(t)):
        if t[i] in frequent:
            tags.append(t[i])
    return tags


# In[12]:


preprocessed_df['Tag'] = preprocessed_df['Tag'].apply(lambda x: most_frequent(x, freq_tag))
preprocessed_df['Tag'] = preprocessed_df['Tag'].apply(lambda x: x if len(x)>0 else None)


# In[13]:


# Drop nan values from the dataset
new_preprocess = preprocessed_df.dropna(subset=['Tag', 'Title'], inplace=False)
dataframe = new_preprocess.dropna(subset=['Title'], inplace=False)


# In[ ]:





# In[18]:


dataframe


# # Classification Process

# In[24]:


# Importing sklearn required libraries
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.feature_extraction.text import TfidfVectorizer
from scipy.sparse import coo_matrix, hstack
from sklearn.svm import LinearSVC


# In[20]:


X1 = dataframe['Body']
X2 = dataframe['Title']
y = dataframe['Tag']


# In[21]:


y


# In[25]:


multilabel_binarizer = MultiLabelBinarizer()
y_bin = multilabel_binarizer.fit_transform(y)


# In[27]:





# In[31]:


def calculate_tfidf_vector(data):
    
    vectorizer_X = TfidfVectorizer(analyzer = 'word', token_pattern=r"(?u)\S\S+", max_features=10000)
    X_tfidf = vectorizer_X.fit_transform(data)
    
    return X_tfidf


# In[32]:


X1_tfidf = calculate_tfidf_vector(X1)
X2_tfidf = calculate_tfidf_vector(X2)
X_tfidf = hstack([X1_tfidf, X2_tfidf])


# # Creating Validation Set

# In[33]:


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X_tfidf, y_bin, test_size = 0.2, random_state = 0)


# In[34]:


X_train.shape


# In[35]:


from sklearn.multiclass import OneVsRestClassifier
from sklearn.metrics import classification_report


# In[36]:


def apply_model(classifier, X_Train, y_Train, X_Test, y_Test):
    
    clf = OneVsRestClassifier(classifier)
    clf.fit(X_Train, y_Train)
    y_Pred = clf.predict(X_Test)
    
    accuracy = clf.score(X_Test, y_Test)
    
    return accuracy, y_Pred


# # Multinomial Naive Bayes

# In[37]:


from sklearn.naive_bayes import MultinomialNB
mn = MultinomialNB()
mn_accuracy, mn_y_pred = apply_model(mn, X_train, y_train, X_test, y_test)


# # Support Vector Machine

# In[38]:


svc = LinearSVC()
svc_accuracy, svc_y_pred = apply_model(svc, X_train, y_train, X_test, y_test)


# # Bernoulli Naive Bayes

# In[39]:


from sklearn.naive_bayes import BernoulliNB
bnb = BernoulliNB()
bnb_accuracy, bnb_y_pred = apply_model(bnb, X_train, y_train, X_test, y_test)


# In[40]:


print(mn_accuracy, svc_accuracy, bnb_accuracy)


# In[41]:


print(classification_report(y_test, mn_y_pred))
print(classification_report(y_test, svc_y_pred))


# In[38]:


from sklearn.metrics import precision_recall_curve
from sklearn.metrics import average_precision_score

clf = OneVsRestClassifier(svc)
clf.fit(X_train, y_train)
y_score = clf.decision_function(X_test)
# For each class
precision = dict()
recall = dict()
average_precision = dict()
n_classes = y_bin.shape[1]
for i in range(n_classes):
    precision[i], recall[i], _ = precision_recall_curve(y_test[:, i],
                                                        y_score[:, i])
    average_precision[i] = average_precision_score(y_test[:, i], y_score[:, i])
    
precision["micro"], recall["micro"], _ = precision_recall_curve(y_test.ravel(), y_score.ravel())
average_precision["micro"] = average_precision_score(y_test, y_score, average="micro")
print('Average precision score, micro-averaged over all classes: {0:0.2f}'.format(average_precision["micro"]))


# In[ ]:


plt.figure()
from sklearn.utils.fixes import signature
step_kwargs = ({'step': 'post'} if 'step' in signature(plt.fill_between).parameters else {})
plt.step(recall['micro'], precision['micro'], color='r', alpha=0.2, where='post')
plt.fill_between(recall["micro"], precision["micro"], alpha=0.2, color='r', **step_kwargs)

plt.xlabel('Recall')
plt.ylabel('Precision')
plt.ylim([0.0, 1.05])
plt.xlim([0.0, 1.0])
plt.title('Precision-Recall Curve(SVM)')


# In[42]:


# from yellowbrick.classifier import PrecisionRecallCurve
# viz = PrecisionRecallCurve(MultinomialNB())
# viz.fit(X_train, y_train)
# viz.score(X_test, y_test)
# viz.poof()


# In[ ]:




