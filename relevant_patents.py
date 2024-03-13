#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import pandas as pd
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Patent veritabanı dosyasını oku, sadece "Title" ve "Claims" sütunlarını kullan
df = pd.read_excel(r"C:\Users\Administrator\Downloads\patentdatabase.XLSX", usecols=["Title", "Claims"])

# Tokenizasyon
df["Text"] = df["Title"] + " " + df["Claims"]
df["Tokenized_Text"] = df["Text"].apply(word_tokenize)

queries = ['unmanned aerial vehicle', 'unmanned aircraft', 'intelligent aerial vehicle', 'drone', 'remotely piloted vehicle', 'unmanned aircraft system']

# Sorguları tokenleştir
tokenized_queries = [word_tokenize(query) for query in queries]

# Tokenleştirilmiş metinleri string'e dönüştürme
df["Processed_Text"] = df["Tokenized_Text"].apply(lambda x: ' '.join(x))

# İlgili patentlerin başlıklarını ve benzerlik skorlarını saklayacak bir liste oluştur
relevant_patents_data = []

# Her sorgu için benzerlik skorlarını hesapla ve ilgili patentlerin başlıklarını ve skorlarını sakla
for query, tokenized_query in zip(queries, tokenized_queries):
    print(f"Query: {query}")
    
    # Sorguları ve patentleri vektöre dönüştür
    vectorizer = TfidfVectorizer()
    query_vector = vectorizer.fit_transform([query])
    patent_vectors = vectorizer.transform(df["Processed_Text"])
    
    # Kosinüs benzerliğini hesapla
    similarities = cosine_similarity(query_vector, patent_vectors).flatten()
    
    # Benzerlik skoru 0.5'ten büyük olan patentleri filtrele
    relevant_patents = df[similarities > 0.5]
    
    # İlgili patentleri sakla
    for _, row in relevant_patents.iterrows():
        relevant_patents_data.append({"Patent Title": row['Title'], "Similarity Score": similarities[_]})

# Veri çerçevesini oluştur
relevant_patents_df = pd.DataFrame(relevant_patents_data)

# Veri çerçevesini Excel dosyasına kaydet
relevant_patents_df.to_excel("relevant_patents.xlsx", index=False)

