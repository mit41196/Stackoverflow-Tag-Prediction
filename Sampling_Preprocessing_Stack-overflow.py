#!/usr/bin/env python
# coding: utf-8

# # Import Libraries

# In[1]:


import pandas as pd
import matplotlib.pyplot as plt
from bs4 import BeautifulSoup
from sklearn.model_selection import train_test_split
import re
import string
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk.stem import WordNetLemmatizer
lemmatizer = WordNetLemmatizer()
stemmer = PorterStemmer()
import sys
sys.setrecursionlimit(3000)

from sklearn.metrics import recall_score
from sklearn.metrics import precision_score
from sklearn.metrics import f1_score
from sklearn.metrics import hamming_loss
from sklearn.metrics import jaccard_similarity_score


# In[2]:


languages = ['vb.net', 'c#', 'c++', '.net']


# # Load Datasets

# In[3]:


questions = pd.read_csv('stacksample/Questions.csv', encoding = 'latin1')
# answers = pd.read_csv('stacksample/Answers.csv', encoding = 'latin1')
tags = pd.read_csv('stacksample/Tags.csv', encoding = 'latin1')


# In[ ]:





# # Sampling Functions

# In[4]:


def sort_by_date(df):
    
    dataframe = df.copy(deep=True)
    sep = 'T'
    print(dataframe)
    for i in range(dataframe.shape[0]):
        head, sep, tail = dataframe.iloc[i]['CreationDate'].partition(sep)
        dataframe.at[i, 'CreationDate'] = head
    
    sorted_dates = dataframe.sort_values(by='CreationDate')
    
    return sorted_dates


# In[5]:


def sampling(df):
    
    dataframe = df.copy(deep=True)
    sample = dataframe[(dataframe['CreationDate'] >= '2014-01-01')].reset_index()
    sample = sample.iloc[:,1:]
    
    return sample


# # MERGING TWO DATAFRAME

# In[6]:


def generate_dataset(df, tag):
    
    data = df.copy(deep=True)
    ta = tag.copy(deep=True)

    ta = ta.astype(str)
    ta = ta.groupby('Id')['Tag'].apply(', '.join).reset_index()
    
    data.Id = data.Id.astype(str)
    ta.Id = ta.Id.astype(str)
    data = pd.merge(data, ta, on='Id')
#     data.to_csv('intermediate_dataset/all_data_questions_with_tags.csv', index=False)

    dataframe = data[['Id', 'Title', 'Body', 'Tag']]
#     dataframe.to_csv('intermediate_dataset/all_questions_only_with_tags.csv', index=False)
    
    return data, dataframe


# # TRAIN-TEST SPLIT

# In[7]:


def training_testing_split(df):
    
    dataframe = df.copy(deep=True)
    print(dataframe)
    X_train, X_test, y_train, y_test = train_test_split(dataframe[['Id', 'OwnerUserId', 'CreationDate', 'ClosedDate', 'Score', 'Title', 'Body']], dataframe[['Tag']], test_size=0.33, shuffle=False, random_state=0)
    return X_train, X_test, y_train, y_test


# # Preprocessing Functions

# In[8]:


# Reference: https://stackoverflow.com/questions/43018030/replace-apostrophe-short-words-in-python
def expansion_apostrophes(sentence):

    sentence = re.sub(r"won\'t", "will not", sentence)
    sentence = re.sub(r"can\'t", "can not", sentence)
    sentence = re.sub(r"let\'s", "let us", sentence)
    sentence = re.sub(r"shan\'t", "shall not", sentence)

    sentence = re.sub(r"n\'t", " not", sentence)
    sentence = re.sub(r"\'re", " are", sentence)
    sentence = re.sub(r"\'s", " is", sentence)
    sentence = re.sub(r"\'d", " would", sentence)
    sentence = re.sub(r"\'ll", " will", sentence)
    sentence = re.sub(r"\'t", " not", sentence)
    sentence = re.sub(r"\'ve", " have", sentence)
    sentence = re.sub(r"\'m", " am", sentence)
    
    return sentence


# In[9]:


def remove_html_tags(question_body):

    question = BeautifulSoup(question_body, 'html.parser')
    question = question.get_text(separator=" ")
    
    return question


# In[10]:


def checknumber(x):
    try:
        float(x)
        return True
    except ValueError:
        return False


# In[11]:


def remove_single_letter(str):
    single_list=['c','r'] 
    return(' '.join( [p for p in str.split() if (len(p)>1 or (p in single_list)) ] ))
    


# In[12]:


def lowercase(str):
    return(str.lower())


# In[13]:


def remove_numbers(str):
    return(' '.join([p for p in str.split() if (not(checknumber(p)))]))


# In[14]:


def remove_punctuation(str):
    x=""
    punc=set(string.punctuation)
  
    for word in str.split():
        if word in languages:
            x=x+" "+word
        else:
            x=x+" "+''.join(c for c in word if c not in punc)
    return x


# In[15]:


def tokenization(str):
    return(str.split())


# In[16]:


def stopwords_removal(str):
    stopWords = set(stopwords.words('english')) 
    stop=[]
    for i in range(len(str)):
        #print(str[i])
        if(str[i] not in stopWords):
            stop.append(str[i])
    return stop


# In[17]:


def stemming(str):
    stemmed=[]
   # print(str)
    for i in range(len(str)):
        stemmed.append(lemmatizer.lemmatize(str[i]))
    return stemmed
                    


# In[18]:


def remove_numbers_from_list(str):
    notnum=[]
    for i in range(len(str)):
        if not(checknumber(str[i])):
            notnum.append(str[i])
    return notnum


# In[19]:


def preprocessing(df):
    
    dataframe = df.copy(deep=True)
    
    for i in range(dataframe.shape[0]):
        
        question = dataframe.iloc[i]['Body']
        title=dataframe.iloc[i]['Title']
        
        # Removing HTML Tags from the question body!
        question = remove_html_tags(question)
        
         # Expansion of apostrophes like won't = will not etc.
        expanded = expansion_apostrophes(question)
        expandedT = expansion_apostrophes(title)
        
        #Case folding
        title=lowercase(expandedT)
        question=lowercase(expanded)
        
       
        #Remove Punctuation
        titlePN=remove_punctuation(title)
        bodyPN=remove_punctuation(question)
       
         
        #Removing single letters
        titleRSL=remove_single_letter(titlePN)
        bodyRSL=remove_single_letter(bodyPN)
        #print(titleRSL)
        
        #Remove numbers
        titleRN=remove_numbers(titleRSL)
        bodyRN=remove_numbers(bodyRSL)
       
        
        #Tokenization
        titletoken=tokenization(titleRN)
        bodytoken=tokenization(bodyRN)
       
        #stopwords removal
        titleSR=stopwords_removal(titletoken)
        bodySR=stopwords_removal(bodytoken)
        #print(titleSR)
        
        #Remove numbers from list after tokenization
        titleRNo=remove_numbers_from_list(titleSR)
        bodyRNo=remove_numbers_from_list(bodySR)
        
        #stemming
        titleSM=stemming(titleRNo)
        bodySM=stemming(bodyRNo)
        #print(titleSM)
        
        titleSM = ' '.join(titleSM)
        bodySM = ' '.join(bodySM)
        
        dataframe.at[i, 'Body'] = bodySM
        dataframe.at[i,'Title'] = titleSM
    return dataframe


# In[ ]:





# In[20]:


def extract_languages(l1, l2, l3, df):
    
    ls = []
    for i in range(df.shape[0]):
        tags = df.iloc[i]['Tag']
        tags = tags.split(",")
        
        if l1 in tags or l2 in tags or l3 in tags:
            ls.append(df.iloc[i])
    return ls
            


# In[21]:


def extract_code(code):
    
    codes = []
    soup = BeautifulSoup(code,'html.parser')
    for i in soup.find_all('code'):
        codes.append(str(i))
        i.decompose()
    cd = (''.join(codes))
    return str(soup).encode("utf-8"), cd


# In[22]:


def calculate_tfidf_vector(data):
    
    vectorizer_X = TfidfVectorizer(analyzer = 'word', token_pattern=r"(?u)\S\S+", max_features=10000)
    tfidf = vectorizer_X.fit(data)
    X_tfidf = tfidf.transform(data)
    return X_tfidf, tfidf


# # Merging the dataframe

# In[23]:


dmerged,dfmerged=generate_dataset(questions,tags)


# # Sampling Related Function Calls

# In[24]:


sorted_dates = sort_by_date(dmerged)


# In[25]:


sampled_data = sampling(sorted_dates)
print(sampled_data)


# In[ ]:


# sampled_data.to_csv("sampled_data.csv", index=False)


# # Creating Training Set and Testing Set

# In[27]:


X_train, X_test, y_train, y_test = training_testing_split(sampled_data)


# In[28]:


training_set = pd.concat([X_train, y_train], axis=1)
testing_set = pd.concat([X_test, y_test], axis=1)
print(training_set.shape)
print(testing_set.shape)
training_set.to_csv('intermediate_datasets/training_set.csv', index=False)
testing_set.to_csv('intermediate_datasets/testing_set.csv', index=False)


# In[ ]:


training_tag = training_set[['Id', 'Title', 'Body', 'Tag']]
testing_tag = testing_set[['Id', 'Title', 'Body', 'Tag']]
training_set.to_csv('intermediate_datasets/tags_training.csv', index=False)
testing_set.to_csv('intermediate_datasets/tags_testing.csv', index=False)


# # Preprocessing Related Function Calls

# In[29]:


dff=preprocessing(training_tag)


# In[ ]:


dff.to_csv('intermediate_datasets/preprocessed.csv', index=False)


# In[30]:


language_data = extract_languages("java", "javascript", "c#", sampled_data)


# In[31]:


codes_data = pd.DataFrame(language_data)


# In[ ]:


# for i in range(codes_data.shape[0]):
#     extract_code(codes_data.iloc[i][6])


# In[ ]:


codes_data.to_csv('intermediate_datasets/java_js_c#_wo_preprocessing_all.csv', index=False)


# In[ ]:


codes_data


# In[32]:


code_only = codes_data.copy()
tb_only = codes_data.copy()


# In[33]:


code_d = codes_data.values


# In[34]:


code_body = []
normal_body = []
for i in range(code_d.shape[0]):
    if (i%500)==0:
        print(i)
    codes = extract_code(code_d[i][6])
    code_part = remove_html_tags(codes[1])
    title_body_part = remove_html_tags(codes[0])

    code_body.append(code_part)
    normal_body.append(title_body_part)


# In[35]:


title_body_only = pd.DataFrame(normal_body)
title_body_only.shape


# In[36]:


final_title_body = pd.DataFrame()
final_title_body = pd.concat([final_title_body, pd.DataFrame(codes_data['Title']).reset_index(drop=True), title_body_only], axis=1, ignore_index=True).reset_index(drop=True)


# In[37]:


code_only_df = pd.DataFrame(code_body)
print(code_only_df.shape)
labels = code_d[:, 7]
final_code = pd.DataFrame()
final_code = pd.concat([code_only_df, final_title_body, pd.DataFrame(labels)], axis=1, ignore_index=True).reset_index(drop=True)


# In[38]:


final_code_nonnull = final_code[final_code[0]!=''].reset_index(drop=True)


# In[ ]:


final_code_nonnull


# In[39]:


label_y = []
lang = ["java", "javascript", "c#"]
for i in range(final_code_nonnull.shape[0]):
    
    ta = final_code_nonnull[3][i].split(",")
    l = [x for x in ta if x in lang]
    label_y.append(l[0])
#     final_code_nonnull.at[i, 3] = l[0]


# In[40]:


final_df = pd.DataFrame()
final_df = pd.concat([final_df, final_code_nonnull, pd.DataFrame(label_y)], axis=1, ignore_index=True).reset_index(drop=True)


# In[41]:


final_df.shape


# In[ ]:


final_code_nonnull.to_csv("Final_Code_Body_All.csv", index=False)


# In[42]:


final_df_list = final_df.copy()

for i in range(final_df_list.shape[0]):
    
    if i%500==0:
        print(i)
    q1 = final_df_list[3][i]
    q2 = final_df_list[4][i]
    q1 = q1.split(",")
    q2 = q2.split(",")
    final_df_list.at[i,3] = q1
    final_df_list.at[i,4] = q2


# In[322]:


final_df_list


# In[323]:


final_df_list.to_csv("Final_wo_preprocessing_JAVA_C#_JAVASCRIPT.csv", index=False)


# In[388]:


final_df_list


# In[ ]:





# In[43]:


X_code = pd.DataFrame(final_df[0])
y_code = pd.DataFrame(final_df_list[4])


# In[44]:


for i in range(X_code.shape[0]):
    numbers = remove_numbers(X_code.iloc[i][0])
    lowers = lowercase(numbers)
    tokenized = remove_punctuation(lowers)
    tokenized = tokenization(tokenized)
    tokenized = remove_numbers_from_list(tokenized)
    tokenized = ' '.join(tokenized)
    X_code.at[i,0] = tokenized


# In[45]:


U_code = pd.DataFrame()
U_code = pd.concat([U_code,X_code,y_code], axis=1, ignore_index=True).reset_index(drop=True)


# In[46]:


# tempX = pd.DataFrame(U[0])
# tempY = pd.DataFrame(U[1])


# In[47]:


# Importing sklearn required libraries
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.feature_extraction.text import TfidfVectorizer
from scipy.sparse import coo_matrix, hstack
from sklearn.svm import LinearSVC


# In[48]:


multilabel_binarizer = MultiLabelBinarizer()
y_bin = multilabel_binarizer.fit_transform(U_code[1].values)


# In[49]:


def evaluation_metrics(y_actual, y_predicted):
    
    accuracy = accuracy_score(y_actual, y_predicted)
    print("Accuracy:", accuracy)
    recall = recall_score(y_actual, y_predicted, average='micro')
    print("Recall:", recall)
    precision = precision_score(y_actual, y_predicted, average='micro')
    print("Precision:", precision)
    f1 = f1_score(y_actual, y_predicted, average='micro')
    print("F1-Score:", f1)
    hamming = hamming_loss(y_actual, y_predicted)
    print("Hamming Loss:", hamming)
    jaccard = jaccard_similarity_score(y_actual, y_predicted)
    print("Jaccard Similarity Score:", jaccard)
    return accuracy, recall, precision, f1, hamming, jaccard


# # ONLY CODE PART

# In[50]:


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(U_code[0].values, y_bin, test_size = 0.3, random_state = 0)


# In[51]:


X_train_tfidf, tfidf = calculate_tfidf_vector(X_train.ravel())
X_test_tfidf = tfidf.transform(X_test.ravel())


# In[52]:


def apply_model(classifier, X_Train, y_Train, X_Test, y_Test):
    
    clf = OneVsRestClassifier(classifier)
    clf.fit(X_Train, y_Train)
    y_Pred = clf.predict(X_Test)
    
    accuracy = clf.score(X_Test, y_Test)
    
    return accuracy, y_Pred, clf


# In[53]:


from sklearn.multiclass import OneVsRestClassifier
from sklearn.metrics import classification_report


# In[54]:


from sklearn.naive_bayes import MultinomialNB
mn = MultinomialNB()
mn_accuracy, mn_y_pred, mn_code = apply_model(mn, X_train_tfidf, y_train, X_test_tfidf, y_test)


# In[55]:


print(mn_accuracy)
print(classification_report(y_test, mn_y_pred))


# In[57]:


from sklearn.metrics import accuracy_score
mn_code_accuracy, mn_code_recall, mn_code_precision, mn_code_f1, mn_code_hamming, mn_code_jaccard = evaluation_metrics(y_test, mn_y_pred)


# In[58]:


svc = LinearSVC()
svc_accuracy, svc_y_pred_code, clf_svc_code = apply_model(svc, X_train_tfidf, y_train, X_test_tfidf, y_test)


# In[59]:


print(svc_accuracy)
print(classification_report(y_test, svc_y_pred_code))


# In[60]:


svc_code_accuracy, svc_code_recall, svc_code_precision, svc_code_f1, svc_code_hamming, svc_code_jaccard = evaluation_metrics(y_test, svc_y_pred_code)


# In[61]:


final_df_list


# In[62]:


final_df_title_body = final_df_list[[1,2,3]]
final_df_title_body = final_df_title_body.rename(columns={1: 'Title', 2: 'Body', 3:'Tag'})


# In[63]:


full = preprocessing(final_df_title_body)


# In[89]:


full


# In[90]:


X_title = pd.DataFrame(full['Title'])
X_body = pd.DataFrame(full['Body'])
y_body = pd.DataFrame(full['Tag'])


# In[91]:


final_df_title_body


# In[92]:


y_body = pd.DataFrame(final_df_title_body['Tag'])


# In[93]:


U_title = pd.DataFrame()
U_title = pd.concat([U_title,X_title,y_body], axis=1, ignore_index=True).reset_index(drop=True)

U_body = pd.DataFrame()
U_body = pd.concat([U_body, X_body, y_body], axis=1, ignore_index=True).reset_index(drop=True)


# # TITLE PART ONLY

# In[149]:


multilabel_binarizer_title = MultiLabelBinarizer()
y_bin_title = multilabel_binarizer_title.fit_transform(U_title[1].values)


# In[153]:


y_bin_title.shape


# In[150]:


X_train_title, X_test_title, y_train_title, y_test_title = train_test_split(U_title[0].values, y_bin_title, test_size = 0.3, random_state = 0)


# In[151]:


X_train_title_tfidf, tfidf_title = calculate_tfidf_vector(X_train_title.ravel())
X_test_title_tfidf = tfidf_title.transform(X_test_title.ravel())


# In[152]:


mn_accuracy, mn_y_pred = apply_model(mn, X_train_title_tfidf, y_train_title, X_test_title_tfidf, y_test_title)
print(mn_accuracy)
# print(classification_report(y_test, mn_y_pred))


# In[88]:


svc = LinearSVC()
svc_accuracy, svc_y_pred = apply_model(svc, X_train_title_tfidf, y_train_title, X_test_title_tfidf, y_test_title)
print(svc_accuracy)
# print(classification_report(y_test, svc_y_pred))


# # BODY PART ONLY

# In[63]:


multilabel_binarizer_body = MultiLabelBinarizer()
y_bin_body = multilabel_binarizer_body.fit_transform(U_body[1].values)


# In[107]:





# In[64]:


X_train_body, X_test_body, y_train_body, y_test_body = train_test_split(U_body[0].values, y_bin_body, test_size = 0.3, random_state = 0)


# In[65]:


X_train_body_tfidf, tfidf_body = calculate_tfidf_vector(X_train_body.ravel())
X_test_body_tfidf = tfidf_body.transform(X_test_body.ravel())


# In[66]:


mn_accuracy, mn_y_pred = apply_model(mn, X_train_body_tfidf, y_train_body, X_test_body_tfidf, y_test_body)
print(mn_accuracy)
# print(classification_report(y_test, mn_y_pred))


# In[67]:


svc = LinearSVC()
svc_accuracy, svc_y_pred = apply_model(svc, X_train_body_tfidf, y_train_body, X_test_body_tfidf, y_test_body)
print(svc_accuracy)
# print(classification_report(y_test, svc_y_pred))


# # BODY & TITLE PART 

# In[68]:


U_title_body = pd.DataFrame()
U_title_body = pd.concat([U_title_body,X_title,X_body,y_body], axis=1, ignore_index=True).reset_index(drop=True)


# In[70]:


X_train_title_body, X_test_title_body, y_train_title_body, y_test_title_body = train_test_split(U_title_body[0].values, y_bin_body, test_size = 0.3, random_state = 0)


# In[75]:


X_train_title_body_tfidf, tfidf_title_body = calculate_tfidf_vector(X_train_title_body.ravel())
X_test_title_body_tfidf = tfidf_title_body.transform(X_test_title_body.ravel())


# In[76]:


mn_accuracy, mn_y_pred = apply_model(mn, X_train_title_body_tfidf, y_train_title_body, X_test_title_body_tfidf, y_test_title_body)
print(mn_accuracy)
# print(classification_report(y_test, mn_y_pred))


# In[77]:


svc = LinearSVC()
svc_accuracy, svc_y_pred = apply_model(svc, X_train_title_body_tfidf, y_train_title_body, X_test_title_body_tfidf, y_test_title_body)
print(svc_accuracy)
# print(classification_report(y_test, svc_y_pred))


# In[93]:


print(svc_y_pred)
print(y_test_title_body)


# In[80]:


from sklearn.metrics import jaccard_similarity_score
jaccard_similarity_score(y_test_title_body, svc_y_pred)


# In[104]:


c=0
for j in range (len(y_test_title_body)):
    count=0
    for k in range (len(y_test_title_body[j])):
        if(svc_y_pred[j][k]==y_test_title_body[j][k]==1):
            count=count+1
    if(count>=11):
        c=c+1
print(c)
print(c/len(svc_y_pred))


# In[105]:


for i in range(y_test_title_body.shape[0]):
    print(y_test_title_body[i])


# # NeW

# In[95]:


all_tags = [item for sublist in final_df_list[3].values for item in sublist]
freq = nltk.FreqDist(all_tags)

frequent_tags = freq.most_common(100)

freq_tag = []
for tag in frequent_tags:
    freq_tag.append(tag[0])


# In[96]:


freq_tag


# In[97]:


def most_frequent(t, frequent):
    
    tags = []
    for i in range(len(t)):
        if t[i] in frequent:
            tags.append(t[i])
    return tags


# In[86]:


frequent_df = full.copy()
print(frequent_df)
frequent_df['Tag'] = frequent_df['Tag'].apply(lambda x: str(x).split(','))
# frequent_df


# In[99]:


frequent_df = full.copy()


# In[100]:


frequent_df['Tag'] = frequent_df['Tag'].apply(lambda x: most_frequent(x, freq_tag))
frequent_df['Tag'] = frequent_df['Tag'].apply(lambda x: x if len(x)>0 else None)


# In[103]:


# Drop nan values from the dataset
new_preprocess = frequent_df.dropna(subset=['Tag', 'Title'], inplace=False)
dataframe = new_preprocess.dropna(subset=['Title'], inplace=False)


# In[105]:


multilabel_binarizer_title = MultiLabelBinarizer()
y_bin_title = multilabel_binarizer_title.fit_transform(dataframe['Tag'].values)
y_bin_title.shape


# In[106]:


X_train_title, X_test_title, y_train_title, y_test_title = train_test_split(dataframe['Title'].values, y_bin_title, test_size = 0.3, random_state = 0)


# In[107]:


X_train_title_tfidf, tfidf_title = calculate_tfidf_vector(X_train_title.ravel())
X_test_title_tfidf = tfidf_title.transform(X_test_title.ravel())


# In[108]:


mn_accuracy_title, mn_y_pred_title, mn_clf_title = apply_model(mn, X_train_title_tfidf, y_train_title, X_test_title_tfidf, y_test_title)
print(mn_accuracy_title)


# In[109]:


mn_title_accuracy, mn_title_recall, mn_title_precision, mn_title_f1, mn_title_hamming, mn_title_jaccard = evaluation_metrics(y_test_title, mn_y_pred_title)


# In[110]:


svc = LinearSVC()
svc_accuracy_title, svc_y_pred_title, svc_clf_title = apply_model(svc, X_train_title_tfidf, y_train_title, X_test_title_tfidf, y_test_title)
print(svc_accuracy_title)


# In[111]:


svc_title_accuracy, svc_title_recall, svc_title_precision, svc_title_f1, svc_title_hamming, svc_title_jaccard = evaluation_metrics(y_test_title, svc_y_pred_title)


# In[ ]:


c=0
for j in range (len(y_test_title)):
    count=0
    for k in range (len(y_test_title[j])):
        if(mn_y_pred[j][k]==y_test_title[j][k]==1):
            count=count+1
    if(count>1):
        c=c+1
print(c)
print(c/len(svc_y_pred))


# In[152]:


import gc
gc.collect()


# # NEW BODY PART

# In[112]:


multilabel_binarizer_body = MultiLabelBinarizer()
y_bin_body = multilabel_binarizer_body.fit_transform(dataframe['Tag'].values)


# In[113]:


X_train_body, X_test_body, y_train_body, y_test_body = train_test_split(dataframe['Body'].values, y_bin_body, test_size = 0.3, random_state = 0)


# In[114]:


X_train_body_tfidf, tfidf_body = calculate_tfidf_vector(X_train_body.ravel())
X_test_body_tfidf = tfidf_body.transform(X_test_body.ravel())


# In[115]:


mn_accuracy_body, mn_y_pred_body, mn_clf_body = apply_model(mn, X_train_body_tfidf, y_train_body, X_test_body_tfidf, y_test_body)
print(mn_accuracy_body)


# In[116]:


mn_body_accuracy, mn_body_recall, mn_body_precision, mn_body_f1, mn_body_hamming, mn_body_jaccard = evaluation_metrics(y_test_body, mn_y_pred_body)


# In[117]:


svc = LinearSVC()
svc_accuracy_body, svc_y_pred_body, svc_clf_body = apply_model(svc, X_train_body_tfidf, y_train_body, X_test_body_tfidf, y_test_body)
print(svc_accuracy_body)


# In[118]:


svc_body_accuracy, svc_body_recall, svc_body_precision, svc_body_f1, svc_body_hamming, svc_body_jaccard = evaluation_metrics(y_test_body, svc_y_pred_body)


# # NEW TITLE & BODY COMBINED

# In[119]:


X_train_title_body, X_test_title_body, y_train_title_body, y_test_title_body = train_test_split(dataframe[['Title', 'Body']], y_bin_body, test_size = 0.3, random_state = 0)


# In[120]:


y_train_title_body.shape


# In[121]:


import numpy as np
from scipy.sparse import coo_matrix, hstack
X_train_title_body_tfidf1, tfidf_title_body1 = calculate_tfidf_vector(X_train_title_body['Title'].ravel())
X_test_title_body_tfidf1 = tfidf_title_body1.transform(X_test_title_body['Title'].ravel())


X_train_title_body_tfidf2, tfidf_title_body2 = calculate_tfidf_vector(X_train_title_body['Body'].ravel())
X_test_title_body_tfidf2 = tfidf_title_body2.transform(X_test_title_body['Body'].ravel())

X_train_tfidf_combined = hstack([X_train_title_body_tfidf1, X_train_title_body_tfidf2])
X_test_tfidf_combined = hstack([X_test_title_body_tfidf1, X_test_title_body_tfidf2])


# In[122]:


y_train_title_body.shape


# In[123]:


mn_accuracy_tb, mn_y_pred_tb, clf_mn_tb = apply_model(mn, X_train_tfidf_combined, y_train_title_body, X_test_tfidf_combined, y_test_title_body)
print(mn_accuracy_tb)


# In[124]:


mn_tb_accuracy, mn_tb_recall, mn_tb_precision, mn_tb_f1, mn_tb_hamming, mn_tb_jaccard = evaluation_metrics(y_test_title_body, mn_y_pred_tb)


# In[125]:


svc = LinearSVC()
svc_accuracy_tb, svc_y_pred_tb, clf_svc_tb = apply_model(svc, X_train_tfidf_combined, y_train_title_body, X_test_tfidf_combined, y_test_title_body)
print(svc_accuracy_tb)


# In[127]:


accuracy_score(y_test_title_body, svc_y_pred_tb)


# In[128]:


svc_tb_accuracy, svc_tb_recall, svc_tb_precision, svc_tb_f1, svc_tb_hamming, svc_tb_jaccard = evaluation_metrics(y_test_title_body, svc_y_pred_tb)


# # Merged Voting

# In[129]:


svc_y_pred_code


# In[132]:


code_classes = multilabel_binarizer.classes_
tb_classes = multilabel_binarizer_body.classes_


# In[ ]:





# In[135]:


new_predictions =  np.copy(svc_y_pred_tb)
tb = tb_classes.tolist()
for i in range(code_classes.shape[0]):
    ls = []
    lan = code_classes[i]
    ind = tb.index(lan)
    tb1 = svc_y_pred_tb[:, ind]
    code1 = svc_y_pred_code[:, i]
    for j in range(tb1.shape[0]):
        if code1[j] == 1 and tb1[j] == 0:
            new_predictions[j][i] = 1


# In[136]:


from sklearn.metrics import accuracy_score
accuracy_score(y_test_title_body, new_predictions)


# In[137]:


combined_accuracy, combined_recall, combined_precision, combined_f1, combined_hamming, combined_jaccard = evaluation_metrics(y_test_title_body, new_predictions)


# # NEW TITLE AND CODE PART

# In[139]:


da = dataframe[['Title']]
final_df_list = final_df_list[[0]]
dfs = pd.DataFrame()
dfs = pd.concat([dfs, da, final_df_list], axis=1, ignore_index=True).reset_index(drop=True)


# In[141]:


X_train_title_code, X_test_title_code, y_train_title_code, y_test_title_code = train_test_split(dfs[[0, 1]], y_bin_body, test_size = 0.3, random_state = 0)


# In[146]:


X_train_title_code


# In[147]:


X_train_title_code_tfidf1, tfidf_title_code1 = calculate_tfidf_vector(X_train_title_code[0].ravel())
X_test_title_code_tfidf1 = tfidf_title_code1.transform(X_test_title_body['Title'].ravel())

X_train_title_code_tfidf2, tfidf_title_code2 = calculate_tfidf_vector(X_train_title_code[1].ravel())
X_test_title_code_tfidf2 = tfidf_title_code2.transform(X_test_title_code[1].ravel())

X_train_tfidf_combined_tc = hstack([X_train_title_code_tfidf1, X_train_title_code_tfidf2])
X_test_tfidf_combined_tc = hstack([X_test_title_code_tfidf1, X_test_title_code_tfidf2])


# In[148]:


mn_accuracy_tc, mn_y_pred_tc, clf_mn_tc = apply_model(mn, X_train_tfidf_combined_tc, y_train_title_code, X_test_tfidf_combined_tc, y_test_title_code)
print(mn_accuracy_tb)


# In[149]:


mn_tc_accuracy, mn_tc_recall, mn_tc_precision, mn_tc_f1, mn_tc_hamming, mn_tc_jaccard = evaluation_metrics(y_test_title_code, mn_y_pred_tc)


# In[150]:


svc = LinearSVC()
svc_accuracy_tc, svc_y_pred_tc, clf_svc_tc = apply_model(svc, X_train_tfidf_combined_tc, y_train_title_code, X_test_tfidf_combined_tc, y_test_title_code)
print(svc_accuracy_tc)


# In[151]:


svc_tc_accuracy, svc_tc_recall, svc_tc_precision, svc_tc_f1, svc_tc_hamming, svc_tc_jaccard = evaluation_metrics(y_test_title_code, svc_y_pred_tc)


# In[ ]:


# mnb_only_title_precision  = []
# svc_only_title_precision = []
# mnb_only_body_precision = []
# svc_only_body_precision = []
# mnb_title_body_precision = []


# In[ ]:


# X = np.arange(5)
# plt.bar(X + 0.00, data[0], color = 'b', width = 0.25)
# plt.bar(X + 0.25, data[1], color = 'g', width = 0.25)
# plt.bar(X + 0.50, data[2], color = 'r', width = 0.25)


# In[ ]:




