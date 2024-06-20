import pandas as pd
import numpy as np
import spacy
from collections import Counter
from unidecode import unidecode
from bertopic import BERTopic
from umap import UMAP
from sklearn.feature_extraction.text import TfidfVectorizer
import gdown
import streamlit as st

# Configurações Globais
URL_TO_DATA = 'https://drive.google.com/uc?id=1JS-Wqs4DchAskjO1Rm8qCAOuZIsHazcx'
VOWELS = "aeiouy"
OUTPUT_FILE = 'ted_talks_en.csv'

# Baixar o arquivo CSV do Google Drive
gdown.download(URL_TO_DATA, OUTPUT_FILE, quiet=False)

# Lista personalizada de stop words adicionais
CUSTOM_STOP_WORDS = set([
    'like', 'go', 'laughter', 'people', 'think', 'know', 'say', 'want', 'thing', 'time', 'look',
    'come', 'right', 'year', 'work', 'feel', 'life', 'need', 'start', 'good', 'tell', 'actually',
    'talk', 'world', 'applause', 'little', 'find', 'number', 'kind', 'happen', 'story', 
    'minute', 'idea', 'change', 'thank', 'hand', 'room', 'love', 'take', 'long', 
    'help', 'mean', 'question', 'maybe', 'call', 'turn', 'mind', 'write', 'well', 'get', 'person', 
    'learn', 'Laughter', 'word', 'body', 'power', 'great', 'live', 'study', 'fact', 
    'yeah', 'hard', 'try', 'ask', 'different', 'see', 'speak', 'make', 'book', 'family', 
    'place', 'hold', 'hear', 'friend', 'able', 'today', 'leave', 'system', 'moment', 'percent', 
    'second', 'school', 'important', 'stop', 'somebody', 'remember', 
    'give', 'woman', 'course', 'realize', 'night', 'head', 'point', 'high', 'line', 'picture', 
    'listen', 'watch', 'happy', 'pretty', 'bring', 'open', 'heart', 'sort', 'okay', 
    'answer', 'read', 'stand', 'example', 'week', 'have', 'away', 'kid', 'spend', 
    'amazing', 'inside', 'create', 'close', 'begin', 'walk', 'sense', 
    'eye', 'reason', 'probably', 'student', 'email', 'face', 'child', 'stay', 
    'draw', 'month', 'couple', 'single', 'group', 'wrong', 'lose', 'real', 'left', 'step', 'affair', 
    'wait', 'drive', 'hour', 'day','allow', 'conversation', 'sure', 'instead',
    'level', 'early', 'play', 'matter', 'true', 'sound', 'social', 'experiment', 'later', 
    'easy', 'grow', 'paper', 'interview', 'everybody', 'interesting', 'half', 
    'case', 'original', 'move', 'follow', 'show', 'stuff', 'fast', 'send', 'water', 
    'process', 'self', 'order', 'breath', 'money', 'public', 'mother', 'house', 'dream', 'control', 
    'light', 'build', 'tunnel', 'creative', 'imagine', 'million', 'basically', 
    'attention', 'online', 'physical', 'simple', 'response',
    'past', 'home', 'doctor', 'powerful', 'exactly', 'care', 'share', 'black', 'pull', 'push', 
    'lie', 'camera', 'cool', 'possible', 'fake', 'fail', 'normal', 'fine', 
    'reward', 'North', 'end', 'phone', 'test', 'guy', 'plan', 'piece', 'young', 
    'shame', 'term', 'blood', 'lead', 'century', 'break', 'People', 'small', 'space', 'notice', 
    'record', 'decision', 'guess', 'release', 'model', 'choose',
    'sit', 'outside', 'focus', 'foot', 'deal', 'likely', 'decide', 'situation', 'train', 
    'suppose', 'entire', 'completely', 'check', 'partner', 'poem', 'deep', 'girl', 
    'seven', 'parent', 'condition', 'large', 'white', 'truth', 'lot', 'glass', 
    'morning', 'bad', 'way', 'lot', 'job', 'form', 'inbox', 'non', 'box'
])

# Funções Auxiliares
def count_syllables(word):
    count = 0
    if word[0] in VOWELS:
        count += 1
    for index in range(1, len(word)):
        if word[index] in VOWELS and word[index - 1] not in VOWELS:
            count += 1
    if word.endswith("e"):
        count -= 1
    if count == 0:
        count += 1
    return count

def preprocess_text(text, nlp):
    doc = nlp(text)
    tokens = [
        token.lemma_ for token in doc if token.pos_ == 'NOUN' and not token.is_stop and not token.is_punct and 
        len(token) > 2 and token.lemma_ not in CUSTOM_STOP_WORDS and count_syllables(token.lemma_) >= 3
    ]
    return " ".join(tokens)

def get_top_n_words(corpus, n=10):
    vectorizer = TfidfVectorizer(stop_words='english', max_features=n)
    X = vectorizer.fit_transform(corpus)
    feature_array = np.array(vectorizer.get_feature_names_out())
    tfidf_sorting = np.argsort(X.toarray()).flatten()[::-1]
    top_n = feature_array[tfidf_sorting][:n]
    return top_n

# Código Principal
def main():
    st.title("Análise de TED Talks")
    nlp = spacy.load('en_core_web_sm')
    ted_data = pd.read_csv(OUTPUT_FILE)
    ted_data_sorted = ted_data.sort_values(by='views', ascending=False)
    top_50_talks = ted_data_sorted.head(50)
    top_50_talks['processed_transcript'] = top_50_talks['transcript'].apply(lambda x: preprocess_text(x, nlp))

    umap_model = UMAP(n_neighbors=5, n_components=10, metric='cosine', random_state=42)
    topic_model = BERTopic(umap_model=umap_model, min_topic_size=2, nr_topics='auto')
    topics, probs = topic_model.fit_transform(top_50_talks['processed_transcript'])

    top_50_talks['top_words'] = top_50_talks['processed_transcript'].apply(lambda x: ", ".join(get_top_n_words([x])))

    # Exibindo os dados no Streamlit
    st.dataframe(top_50_talks[['title', 'speaker_1', 'views', 'topics', 'top_words']])

if __name__ == "__main__":
    main()
