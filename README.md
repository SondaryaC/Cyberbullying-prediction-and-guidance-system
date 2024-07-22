# 🚀 Cyberbullying Prediction and Guidance System

Welcome to the **Cyberbullying Prediction and Guidance System** - a cutting-edge application designed to identify and provide strategies against various forms of cyberbullying. This project leverages advanced technologies and machine learning models to create a safer online environment.

## 🌟 Features
- **🔍 Detection of Cyberbullying Types**: Accurately identifies cyberbullying based on religion, gender, age, ethnicity, and more.
- **🧹 Text Preprocessing**: Robust text cleaning including decontracting phrases, removing stopwords, and handling emojis.
- **🤖 Machine Learning Models**: Utilizes a Random Forest Classifier with TfidfVectorizer for precise classification.
- **📚 Document Embeddings & Vector Store**: Integrates OllamaEmbeddings and Chroma vector store for efficient document retrieval.
- **🎨 Interactive User Interface**: Built with Streamlit for an intuitive and user-friendly experience.
- **📈 Retriever-Augmented Generation (RAG)**: Enhances document retrieval and context-aware responses.
- **💡 Large Language Models (LLM)**: Utilizes state-of-the-art LLMs for generating effective prevention strategies.

## 💻 Technologies Used
- **Programming Languages**: Python
- **Frameworks**: Streamlit, LangChain
- **Libraries**: Scikit-learn, NLTK, PyPDF2, Chroma, OllamaEmbeddings
- **Machine Learning**: Random Forest Classifier, TfidfVectorizer

## 📥 Installation

1. **Clone the repository**:
    ```sh
    git clone https://github.com/yourusername/cyberbullying-prediction-system.git
    cd cyberbullying-prediction-system
    ```

2. **Create and activate a virtual environment**:
    ```sh
    python3 -m venv venv
    source venv/bin/activate  # For Windows use `venv\Scripts\activate`
    ```

3. **Install the required dependencies**:
    ```sh
    pip install -r requirements.txt
    ```

4. **Download necessary models and embeddings**:
    ```sh
    pip install langchain-community
    pip install pypdf chromadb
    ```

## 🚀 Usage

1. **Ensure all necessary models and data files are in place**, including the trained Random Forest model (`rf_model-1-test.pkl`) and TfidfVectorizer (`tfidf_vectorizer-1-test.pkl`).

2. **Place the `cyberbullying007.pdf` file** in the root directory for PDF processing.

3. **Run the Streamlit application**:
    ```sh
    streamlit run all.py
    ```

4. **Open your web browser** and navigate to `http://localhost:8501` to interact with the application.

## 📁 Project Structure
cyberbullying-prediction-system/
│

├── all.py # Main application script

├── model-test.ipynb # Notebook for model training and testing

├── ollama_chat.ipynb # Notebook for integrating Ollama and vector store

├── requirements.txt # Python dependencies

├── rf_model-1-test.pkl # Trained Random Forest model

├── tfidf_vectorizer-1-test.pkl # Trained TfidfVectorizer

├── cyberbullying007.pdf # PDF file for document processing

├── cyberbullying_tweets.csv # Csv dataset for  cyberbullyingprediction

└── README.md # Project README file


## 🙌 Contributing
We welcome contributions! Please follow these steps:

1. **Fork the repository**.
2. **Create a new branch**.
3. **Make your changes**.
4. **Submit a Pull Request**.

---

Feel free to reach out if you have any questions or suggestions. Together, let's make the internet a safer place for everyone! 🌐

---

**Developed with ❤️ by Sondarya Chauhan**

[![LinkedIn](https://www.linkedin.com/in/yourprofile/)

