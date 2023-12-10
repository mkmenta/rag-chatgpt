# Retrieval Augmented Generation Lab with ChatGPT
This is a simple lab I have implemented to test Knowledge Augmented or Retrieval Augmented Generation (RAG) with Large Language Models. In particular, I am using LangChain, Streamlit, and OpenAI ChatGPT API. This project has been an excuse to try RAG and LangChain.

Run it with:
```
streamlit run chat.py
```

Disclaimer: This is a personal project without any guarantee and I am not planning to maintain it.

## Requirements
I have fixed the requirements with the versions I have used during the development. But the code will probably work with previous and newer versions. 

Install the requirements with
```
python3 -m pip install -r requirements.txt
```

You will also need to create a `.env` file with the content:
```
OPENAI_API_KEY="YOUR OPENAI API KEY"
```

## Example
After running the streamlit app with
```
streamlit run chat.py
```
access to the website. 

The repository has a sample of documents in the `data/test_workers` folder. You can ask ChatGPT about them:
![Example of the web interface and ChatGPT using the content of the documents to answer](/misc/example.jpg)
