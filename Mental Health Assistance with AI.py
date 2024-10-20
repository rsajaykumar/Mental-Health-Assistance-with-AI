#!/usr/bin/env python
# coding: utf-8

# In[16]:


import pandas as pd
import numpy as np


# In[90]:


import os
for dirname, _, filenames in os.walk('\kaggle.input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))


# In[157]:


import json
with open ('H:\Datathon\intents2.json', 'r') as f:
    data = json.load(f)
df = pd.DataFrame(data['intents'])
df


# In[158]:


dic = {"tag":[], "patterns":[], "responses":[]}
df = pd.DataFrame.from_dict(dic)
for i in range (len(df)):
    ptrns = df[df.index == i]['patterns'].values[0]
    rspns = df[df.index == i]['responses'].values[0]
    tag = df[df.index == i]['tag'].values[0]
    for j in range(len(ptrns)):
        dic['tag'].append(tag)
        dic['patterns'].append(ptrns[j])
        dic['responses'].append(rspns)
        
df= pd.DataFrame.from_dict(dic)
df


# In[119]:


df['tag'].unique()


# In[161]:


import pandas as pd
import plotly.graph_objects as go
intent_counts = df['tag'].value_counts()
fig = go.Figure(data=[go.Bar(x=intent_counts.index, y=intent_counts.values)])
fig.update_layout(title='Distribution of Intents', xaxis_title='Intents', yaxis_title='Count')
fig.show()


# In[148]:


df['pattern_count'] = df['patterns'].apply(lambda x: len(x))
df['response_count'] = df['responses'].apply(lambda x: len(x))
avg_pattern_count = df.groupby('tag')['pattern_count'].mean()
avg_response_count = df.groupby('tag')['response_count'].mean()

fig = go.Figure()
fig.add_trace(go.Bar(x=avg_pattern_count.index, y=avg_pattern_count.values, name='Average Pattern Count'))
fig.add_trace(go.Bar(x=avg_response_count.index, y=avg_response_count.values, name='Average Response Count'))
fig.update_layout(title='Pattern and Response Analysis', xaxis_title='Intents', yaxis_title='Average Count')
fig.show()


# In[149]:


from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC
from sklearn.metrics import classification_report
import plotly.graph_objects as go


# In[150]:


df['pattern_count'] = df['patterns'].apply(lambda x: len(x))
df['response_count'] = df['responses'].apply(lambda x: len(x))
avg_pattern_count = df.groupby('tag')['pattern_count'].mean()
avg_response_count = df.groupby('tag')['response_count'].mean()

fig = go.Figure()
fig.add_trace(go.Bar(x=avg_pattern_count.index, y=avg_pattern_count.values, name='Average Pattern Count'))
fig.add_trace(go.Bar(x=avg_response_count.index, y=avg_response_count.values, name='Average Response Count'))
fig.update_layout(title='Pattern and Response Analysis', xaxis_title='Intents', yaxis_title='Average Count')
fig.show()


# In[156]:


import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC
from sklearn.metrics import classification_report
import plotly.graph_objects as go

X = df['patterns'].astype(str)
y = df['tag']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

vectorizer = TfidfVectorizer()
X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

model = SVC()
model.fit(X_train_vec, y_train)

y_pred = model.predict(X_test_vec)

report = classification_report(y_test, y_pred, output_dict=True, zero_division=1)

# Filter out the 'accuracy' and 'macro avg' keys
report = {k: v for k, v in report.items() if k not in ['accuracy', 'macro avg', 'weighted avg']}

labels = list(report.keys())
evaluation_metrics = ['precision', 'recall', 'f1-score']
metric_scores = {metric: [report[label][metric] for label in labels] for metric in evaluation_metrics}

fig = go.Figure()

for metric in evaluation_metrics:
    if metric in metric_scores and len(metric_scores[metric]) == len(labels):
        fig.add_trace(go.Bar(name=metric, x=labels, y=metric_scores[metric]))
    else:
        print(f"Warning: Metric '{metric}' has mismatched data")

fig.update_layout(title='Intent Prediction Model',
                  xaxis_title='Intent',
                  yaxis_title='Score',
                  barmode='group')

fig.show()


# In[155]:


from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB 
import numpy as np
def predict_intent(user_input):
    user_input_vec = vectorizer.transform([user_input])
    intent = model.predict(user_input_vec)[0]
    return intent
def generate_response(intent):
    if intent == 'greeting':
        response = "Hello! How can I assist you today?"
    elif intent == 'farewell':
        response = "Goodbye! Take care."
    elif intent == 'question':
        response = "I'm sorry, I don't have the information you're looking for."
    else:
        response = "I'm here to help. Please let me know how I can assist you."
    return response
while True:
    user_input = input("User: ")
    intent = predict_intent(user_input)
    response = generate_response(intent)
    print("Chatbot:", response)


# In[ ]:




