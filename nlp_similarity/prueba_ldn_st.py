import requests
from bs4 import BeautifulSoup
import re
from nltk import sent_tokenize
import pandas as pd
import streamlit as st
from keybert import KeyBERT
from keyphrase_vectorizers import KeyphraseCountVectorizer
import pickle

# Streamlit title and instructions
st.title("Keyword Extraction from Web Articles")
st.write("Enter a URL to extract keywords from the article.")

# User input for URL
url = st.text_input("Paste the URL:")

if url:
    # Web scraping and text extraction
    get_url = requests.get(url)
    get_text = get_url.text
    soup = BeautifulSoup(get_text, "html.parser")
    all_content = soup.find("section", class_="article__body col-md-8")
    text_elements = all_content.find_all(['p'])
    
    docs = []
    emoji_pattern = re.compile(r'[\U0001F600-\U0001F64F\U0001F300-\U0001F5FF\U0001F680-\U0001F6FF\U0001F700-\U0001F77F\U0001F1E0-\U0001F1FF\U00002702-\U000027B0\U000024C2-\U0001F251]+')

    for p in text_elements:
        if p.get_text(strip=True) == '':
            continue
        if not p.find('img') and not emoji_pattern.search(p.get_text()):
            if not p.find_parent('blockquote') and not p.find_parent('iframe', class_='instagram-media instagram-media-rendered'):
                docs.append(p.get_text())

    # Keyword extraction
    kw_model = KeyBERT()
    vectorizer = KeyphraseCountVectorizer(spacy_pipeline='en_core_web_trf')
    kw_to_check = [kw_model.extract_keywords(docs=docs, vectorizer=vectorizer)]
    
    selected_keywords = []
    
    for sublist in kw_to_check[0]:
        for keyword, score in sublist:
            if score >= 0.6:
                selected_keywords.append((keyword, score))
    
    unique_keywords = {}
    for keyword, score in selected_keywords:
        if keyword not in unique_keywords or score > unique_keywords[keyword]:
            unique_keywords[keyword] = score
    
    sorted_keywords = sorted(unique_keywords.items(), key=lambda x: x[1], reverse=True)
    keyphrase = [(keyword, score) for keyword, score in sorted_keywords]
    keyphrase

    # # Display extracted keywords
    # st.write("Extracted Keywords:")
    # for keyword, score in keyphrase:
    #     st.write(f"- {keyword} (Score: {score:.2f})")

    # # Load processed data and URLs
    # with open('ldn_gastro.pkl', 'rb') as f:
    #     processed_contents_pp = pickle.load(f)

    # with open('urls_gastro.pkl', 'rb') as f:
    #     urls = pickle.load(f)

    # # Keyword matching with stored data
    # results = []
    
    # for url2, sublist in zip(urls, processed_contents_pp):
    #     for kw, score in keyphrase:
    #         keyword_found = False
    #         anchor_phrase = None
    #         for sentence in sublist:
    #             if kw.lower() in sentence.lower():
    #                 keyword_found = True
    #                 anchor_phrase = sentence
    #                 break
    #         results.append((url, kw, score, url2, anchor_phrase))

    # filtered_results = [result for result in results if result[4] is not None]
    # df = pd.DataFrame(filtered_results, columns=['kw_url', 'keywords', 'scores', 'is_in_url', 'anchor_phrase'])

    # # Display results in a table
    # st.write("Keyword Matching Results:")
    # st.dataframe(df)
