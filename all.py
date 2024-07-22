import re
import string
import emoji
import pickle
import os
import streamlit as st
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.corpus import stopwords
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.embeddings import OllamaEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_community.chat_models import ChatOllama
from langchain_core.runnables import RunnablePassthrough
from langchain.retrievers.multi_query import MultiQueryRetriever
from langchain.prompts import ChatPromptTemplate



# Define the functions as you already have them
def load_pdf():
    loader = PyPDFLoader('cyberbullying007.pdf')
    return loader.load()

def preprocess_text(text):
    text = text.lower()
    text = re.sub(r"(?:\@|https?\://)\S+", "", text)
    text = emoji.replace_emoji(text, replace='')
    text = decontract(text)
    text = text.translate(str.maketrans('', '', string.punctuation))
    english_stopwords = set(stopwords.words('english'))
    text = " ".join(word for word in text.split() if word not in english_stopwords)
    text = re.sub(r"\s\s+", " ", text)
    return text

def decontract(text):
    text = re.sub(r"won't", "will not", text)
    text = re.sub(r"can't", "cannot", text)
    text = re.sub(r"n't", " not", text)
    text = re.sub(r"'re", " are", text)
    text = re.sub(r"'s", " is", text)
    text = re.sub(r"'d", " would", text)
    text = re.sub(r"'ll", " will", text)
    text = re.sub(r"'t", " not", text)
    text = re.sub(r"'ve", " have", text)
    text = re.sub(r"'m", " am", text)
    return text

def predict_cyberbullying(text):
    cleaned_text = preprocess_text(text)
    tfidf_text = loaded_tfidf_vectorizer.transform([cleaned_text])
    label_index = loaded_rf_model.predict(tfidf_text)[0]
    label_categories = {
        4: 'Religion',
        0: 'Age',
        2: 'Gender',
        1: 'Ethnicity',
        3: 'Not Cyberbullying'
    }
    cyberbullying_label = label_categories[label_index]
    return cyberbullying_label

def init_vectorstore():
    try:
        persist_directory = "./chroma_db_cyber"
        create_directory_if_not_exists(persist_directory)
        
        vectorstore = Chroma(
            persist_directory=persist_directory,       
            embedding_function=ollama_embed  
        )
        return vectorstore
    except Exception as e:
        print(f"Error initializing vectorstore: {e}")
        return None

def create_directory_if_not_exists(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)

# Initialize components
data = load_pdf()
text_splitter = RecursiveCharacterTextSplitter(chunk_size=7500, chunk_overlap=100)
chunks = text_splitter.split_documents(data)
ollama_embed = OllamaEmbeddings(model="nomic-embed-text")
vectorstore_disk = init_vectorstore()

if vectorstore_disk:
    local_model = "mistral:7b"
    llm = ChatOllama(model=local_model)
    QUERY_PROMPT = PromptTemplate(
        input_variables=["question"],
        template="""You are an assistant for identifying types of cyberbullying in user tweets and providing prevention strategies based on these identifications. Your task is to detect the type of cyberbullying in the input tweet, categorizing it based on religion, gender, age, ethnicity, or other forms of cyberbullying. If the input tweet is not cyberbullying, specify that it is not cyberbullying. Additionally, provide prevention strategies and advice on how to deal with the identified type of cyberbullying according to the data provided.

        Given a user tweet, generate five different versions of the tweet to retrieve relevant documents from a vector database. These alternative versions should help overcome some limitations of distance-based similarity search. Provide these alternative versions separated by newlines.

        Original question: {question}"""
    )
    retriever = MultiQueryRetriever.from_llm(
        vectorstore_disk.as_retriever(),
        llm,
        prompt=QUERY_PROMPT
    )

    template = """Answer the question based ONLY on the following context:
    {context}
    Question: {question}
    """

    prompt = ChatPromptTemplate.from_template(template)

    chain = (
        {"context": retriever, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )
else:
    print("Vectorstore initialization failed. Please check the logs for errors.")

# Load models
with open('rf_model-1-test.pkl', 'rb') as model_file:
    loaded_rf_model = pickle.load(model_file)

with open('tfidf_vectorizer-1-test.pkl', 'rb') as vectorizer_file:
    loaded_tfidf_vectorizer = pickle.load(vectorizer_file)

# Streamlit code
st.title("Cyberbullying Prediction and Guidance System")

st.markdown(
    """
    <style>
        /* Add your custom CSS styles here */
        .text-area-container textarea {
            background-color: #f0f0f0 !important;
            padding: 10px;
            border-radius: 10px;
            box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
            margin-bottom: 20px;
            border: none;
            resize: none;
        }
        .button-container {
            display: flex;
            justify-content: center;
            margin-top: 20px;
        }
        .predict-button {
            background-color: #228B22;
            color: white;
            padding: 10px 20px;
            border: none;
            border-radius: 5px;
            cursor: pointer;
            transition: background-color 0.3s;
            text-align: center;
            width: 100%;
            box-sizing: border-box;
        }
        .predict-button:hover {
            background-color: #196619;
        }
        .clear-button {
            background-color: #0000FF;
            color: white;
            padding: 10px 20px;
            border: none;
            border-radius: 5px;
            cursor: pointer;
            transition: background-color 0.3s;
            text-align: center;
            width: 100%;
            box-sizing: border-box;
        }
        .clear-button:hover {
            background-color: #0000CC;
        }
    </style>
    """,
    unsafe_allow_html=True
)

if 'text_input' not in st.session_state:
    st.session_state.text_input = ''

st.markdown('<div class="text-area-container">', unsafe_allow_html=True)
text_input = st.text_area("Enter Text:", value=st.session_state.text_input, height=200)
st.markdown('</div>', unsafe_allow_html=True)

col1, col2 = st.columns([1, 1])

with col1:
    if st.button("Predict"):
        if text_input:
            st.session_state.text_input = text_input

            st.write(f"**Input Text:** {st.session_state.text_input}")
            
            cyberbullying_label = predict_cyberbullying(text_input)
            st.write(f"**Cyberbullying Label:** {cyberbullying_label}")

            if vectorstore_disk:
                response = chain.invoke({"question": text_input})
                detailed_analysis = response if response else "No additional analysis available."
                st.write(f"**Detailed Analysis:** {detailed_analysis}")
            else:
                st.write("Error: Vectorstore is not initialized. Please check the logs for errors.")
        else:
            st.write("Please enter some text to analyze.")

with col2:
    if st.button("Clear"):
        st.session_state.text_input = ''
        st.experimental_rerun()

#streamlit run all.py