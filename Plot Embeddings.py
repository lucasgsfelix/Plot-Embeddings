#!/usr/bin/env python
# coding: utf-8

# In[1]:


import gensim
import numpy as np
import matplotlib.pyplot as plt
from gensim.models import Word2Vec


# In[2]:


import pandas as pd
import texthero as hero
from sklearn.manifold import TSNE
import tqdm


# In[19]:


df = pd.read_table("trip_advisor_dataset.csv", sep=';')


# In[4]:


df_yelp = pd.read_table("manual_reviews.csv", sep=';')


# In[ ]:


df['dataset'] = 'TripAdvisor'

df_yelp['dataset'] = 'Yelp'

df = pd.concat([df_yelp, df])


# In[ ]:


# pré processamentos

df['text'] = hero.clean(df['text'])

df = df[['text', 'trip type']]

df = df.dropna(subset=['text'])

df['text'] = df['text'].str.lower()

sentencas = df['text'].str.split(' ').values.tolist()

# Treinamento do modelo Word2Vec
trip_advisor_model = Word2Vec(sentencas, window=5, min_count=1, workers=10)


# In[11]:


model = trip_advisor_model

legend = {0: 'Leisure', 1: 'Work'}

# Plotando os embeddings
plt.figure(figsize=(20, 20))

sne = TSNE(n_components=2, random_state=42, perplexity=30, distance_metrics='cosine')

for dataset in tqdm.tqdm(df['dataset'].unique())

    for trip_type in df['trip type'].unique():


        category_df = df[(df['trip type'] == trip_type) & (df['dataset'] == dataset)].sample(100)

        categoy_words = ' '.join(category_df['text'].tolist()).split(' ')

        # Recupere os embeddings para as palavras escolhidas
        embeddings = np.array([model[word] for word in categoy_words if word in model.wv.vocab])

        embeddings_2d = tsne.fit_transform(embeddings)

        for i, palavra in enumerate(categoy_words):


            plt.scatter(embeddings_2d[i, 0], embeddings_2d[i, 1], marker='o',
                        label=dataset + ' ' + legend[trip_type])
            #plt.text(embeddings[i, 0] + 0.01, embeddings[i, 1] + 0.01, palavra, fontsize=9)

plt.xlabel('Dimensão 1')
plt.ylabel('Dimensão 2')
plt.title('Representação de Word Embeddings')
plt.legend()
plt.savefig("word_embeddings.png", format='png')


# In[13]:




