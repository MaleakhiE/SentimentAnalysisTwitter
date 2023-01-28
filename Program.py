import numpy as np
import nltk
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
import pandas as pd
from sklearn.svm import SVC
import re
from nltk.stem import WordNetLemmatizer
import streamlit as st
import matplotlib.pyplot as plt
from wordcloud import WordCloud
import tweepy

st.set_page_config(
    page_title="Information Retrieval",
    layout="wide",
    initial_sidebar_state="expanded"
)
st.markdown(""" <style>
#MainMenu {visibility: hidden;}
footer {visibility: hidden;}
</style> """, unsafe_allow_html=True)
st.set_option('deprecation.showPyplotGlobalUse', False)

writepreprocessing = []
BOW2 = []
bearer_token = 'AAAAAAAAAAAAAAAAAAAAAMwIiAEAAAAAKx8hO494M5rvmNpQ8YdEVv' \
                 'JTu1U%3DM2MU4UfQHtxGfirs29iRqgcn5k25lZ7uu4eVPXQFp91dHwniW2'
lemma = WordNetLemmatizer()
swords = stopwords.words("english")

def crawling(bearer_token):
    client = tweepy.Client(bearer_token=bearer_token)
    input = st.text_input('Input Query',placeholder='#hastagcrawling, dan lang:(en,ind) (Contoh : #covid lang:en)')
    query = input
    result = st.number_input('Jumlah Input Result Crawling (Minimal 10)',min_value=10)
    try :
        tweets = client.search_recent_tweets(query=query, max_results=int(result))

        data = []
        for n in tweets.data:
            symbols = "!\"#$%&*+-/:;<=>?@[\]^_`{|}~[],\n"
            for i in symbols:
                n = np.char.replace(str(n), i, '')
            data.append(n)

        hasil_crawling = {'OriginalTweet': data}
        write = pd.DataFrame(hasil_crawling)
        write.to_csv('hasilcrawling.csv',index=False)
    except :
        st.text('')


def preprocessing():
    upload_file = st.file_uploader("Pilih File Hasil Crawling")
    try :
        data = pd.read_pickle("Data Train.pkl")
        tes = pd.read_csv(upload_file)
        x_test = tes["OriginalTweet"]
        cleanedData = []
        x_train = data["OriginalTweet"]
        y_train = data["Sentiment"]

        lemma = WordNetLemmatizer()
        swords = stopwords.words("english")
        for text in x_train:
            # Cleaning links
            text = re.sub(r'http\S+', '', text)
            # Cleaning everything except alphabetical and numerical characters
            text = re.sub("[^a-zA-Z0-9]", " ", text)
            # Tokenizing and lemmatizing
            text = nltk.word_tokenize(text.lower())
            text = [lemma.lemmatize(word) for word in text]
            # Removing stopwords
            text = [word for word in text if word not in swords]
            # Joining
            text = " ".join(text)
            cleanedData.append(text)

        data_test = []
        prediksi = []
        cleanedDataa = []
        global BOW2
        for n in x_test:
            uji = [n]
            data_test.append(n)
            for text in uji:
                # Cleaning links
                text = re.sub(r'http\S+', '', str(text))
                # Cleaning everything except alphabetical and numerical characters
                text = re.sub("[^a-zA-Z0-9]", " ", text)
                # Tokenizing and lemmatizing
                text = nltk.word_tokenize(text.lower())
                text = [lemma.lemmatize(word) for word in text]
                # Removing stopwords
                text = [word for word in text if word not in swords]
                # Joining
                text = " ".join(text)
                cleanedDataa.append(text)
            vectorizer = TfidfVectorizer()
            BOW = vectorizer.fit_transform(cleanedData)
            BOW2 = vectorizer.transform(uji)
            model = SVC().fit(BOW, y_train)
            predictions = model.predict(BOW2)

            if predictions[0] == 0:
                predic = "Negative"
            elif predictions[0] == 1:
                predic = "Neutral"
            elif predictions[0] == 2:
                predic = "Positive"

            prediksi.append(predic)

        hasil_data = {
            'OriginalTweet': data_test,
            'Sentiment': prediksi
        }

        global writepreprocessing
        writepreprocessing = pd.DataFrame(hasil_data)
        writepreprocessing.to_csv('test3.csv', index=False)
        st.header('Hasil Sentiment Analysis')
        st.write(writepreprocessing)
        print(writepreprocessing)
    except :
        st.info('Masukan Dataset')

def viewcloud():
    datanegatif = []
    datapositif = []
    datanetral = []
    katanegatif = []
    katapositif = []
    katanetral = []
    try :
        tes = pd.read_csv("Hasil Sentiment Analysis.csv")
        negatif = tes.loc[tes["Sentiment"]== 'Negative']
        positif = tes.loc[tes["Sentiment"] == 'Positive']
        netral = tes.loc[tes["Sentiment"] == 'Neutral']
        for a in range(len(negatif)):
            datanegatif.append(negatif.iloc[a]["OriginalTweet"])
        for a in range(len(positif)):
            datapositif.append(positif.iloc[a]["OriginalTweet"])
        for a in range(len(netral)):
            datanetral.append(netral.iloc[a]["OriginalTweet"])

        st.header('Word Cloud')
        st.markdown("""---""")
        colu1, colu2, colu3 = st.columns(3)
        with colu1 :
            st.text('Kalimat - Kalimat Negatif')
            st.write(datanegatif)
        with colu2:
            st.text('Kalimat - Kalimat Netral')
            st.write(datanetral)
        with colu3 :
            st.text('Kalimat - Kalimat Positif')
            st.write(datapositif)

        for i in datanegatif :
            text = re.sub(r'http\S+', '', str(i))
            # Cleaning everything except alphabetical and numerical characters
            text = re.sub("[^a-zA-Z0-9]", " ", text)
            # Tokenizing and lemmatizing
            text = nltk.word_tokenize(text.lower())
            text = [lemma.lemmatize(word) for word in text]
            # Removing stopwords
            text = [word for word in text if word not in swords]
            katanegatif = ','.join(map(str, text))

        for i in datapositif :
            text = re.sub(r'http\S+', '', str(i))
            # Cleaning everything except alphabetical and numerical characters
            text = re.sub("[^a-zA-Z0-9]", " ", text)
            # Tokenizing and lemmatizing
            text = nltk.word_tokenize(text.lower())
            text = [lemma.lemmatize(word) for word in text]
            # Removing stopwords
            text = [word for word in text if word not in swords]
            katapositif = ','.join(map(str, text))

        for i in datanetral :
            text = re.sub(r'http\S+', '', str(i))
            # Cleaning everything except alphabetical and numerical characters
            text = re.sub("[^a-zA-Z0-9]", " ", text)
            # Tokenizing and lemmatizing
            text = nltk.word_tokenize(text.lower())
            text = [lemma.lemmatize(word) for word in text]
            # Removing stopwords
            text = [word for word in text if word not in swords]
            katanetral = ','.join(map(str, text))

        col1, col2, col3 = st.columns(3)
        with col1 :
            try :
                st.text('Word Cloud Negatif')
                wordcloud = WordCloud().generate(katanegatif)

                # Display the generated image:
                plt.imshow(wordcloud, interpolation='bilinear')
                plt.axis("off")
                plt.show()
                st.pyplot()
            except :
                st.info ('Tidak ada data')
        with col2:
            try :
                st.text('Word Cloud Netral')
                wordcloud = WordCloud().generate(katanetral)

                # Display the generated image:
                plt.imshow(wordcloud, interpolation='bilinear')
                plt.axis("off")
                plt.show()
                st.pyplot()
            except :
                st.info ('Tidak ada data')
        with col3:
            try :
                st.text('Word Cloud Positif')
                wordcloud = WordCloud().generate(katapositif)

                # Display the generated image:
                plt.imshow(wordcloud, interpolation='bilinear')
                plt.axis("off")
                plt.show()
                st.pyplot()
            except :
                st.info ('Tidak ada data')
    except :
        st.text('')

crawling(bearer_token)
preprocessing()
viewcloud()