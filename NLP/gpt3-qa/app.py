import streamlit as st
from streamlit_chat import message
import pandas as pd
from transformers import AutoTokenizer, AutoModel, BertForMaskedLM, BertTokenizer
from sklearn.cluster import MiniBatchKMeans
import tokenizers
from typing import *
import torch
import time
import numpy as np
import plotly.express as px
from sklearn.manifold import TSNE
from open_ai_api import GPT3API


def read_q_a_dataset(file_name : str) -> List[Tuple[str, str]]:
    q_a_pairs = list()
    with open(file_name, "r") as f:
        for x in f.readlines():
            row = x.strip().split("\t")
            for i in range(1, len(row)):
                q_a_pairs.append([row[0], row[i]])
    return q_a_pairs


def vectorize_and_save(
    dataset: List[Tuple[str, str]],
    model: AutoModel, 
    tokenizer: AutoTokenizer, 
    path_and_filename: str,
    from_index: int = 0, 
) -> None:
    BATCH_SIZE = 100
    for i in range(from_index, len(dataset), BATCH_SIZE):
        st.text(f'Batch {i} -> {i + BATCH_SIZE - 1}')
        encodings = predict(
            model=model,
            tokenizer=tokenizer,
            sentence=["[CLS] " + q + " [SEP] " + a + " [SEP]" for q, a in dataset[i:i + BATCH_SIZE]]
        )
        torch.save(encodings, f'{path_and_filename}_{i}.pt')
        del encodings

@st.cache
def join_encoding(
    to_index: int,
    filename: str,
) -> torch.Tensor:
    BATCH_SIZE = 100
    encoded_data = torch.load(f'data_encoded/{filename}_0.pt')
    for i in range(100, to_index, BATCH_SIZE):
        encoded = torch.load(f'data_encoded/{filename}_{i}.pt')
        encoded_data = torch.cat([encoded_data, encoded])
    return encoded_data


@st.cache
def generate_kmeans(num_of_clusters: int, encoded_qa_data):
    kmeans = MiniBatchKMeans(n_clusters=num_of_clusters).fit(encoded_qa_data.detach().numpy().astype('double'))
    # get_clusters(num_of_clusters, encoded_qa_data.detach().numpy())
    return kmeans


@st.cache
def calculate_tsne(data):
    return TSNE(
        n_components=2,
        learning_rate='auto',
        init = 'random',
        perplexity = 3
    ).fit_transform(data)


def get_gpt_answer(prompt: str, gpt3_api: GPT3API) -> str:
    return gpt3_api.send_prompt(prompt=prompt)


def get_gpt_with_clustering_answer(prompt: str) -> str:
    return "Yo"


def main() -> None:
    # st.set_page_config(layout="wide")
    st.title('Clustering-based QA')
    
    max_length_tokens: int = st.slider("Maximum number of tokens in answer: ", 1, 4000, 256)
    
    gpt3_api = GPT3API(max_length_tokens=max_length_tokens)
    
    if 'q_asked' not in st.session_state:
        st.session_state['q_asked'] = []

    if 'q_answered_gpt' not in st.session_state:
        st.session_state['q_answered_gpt'] = []
        
    if 'q_answered_gpt_with_clustering' not in st.session_state:
        st.session_state['q_answered_gpt_with_clustering'] = []


    prompt = st.text_area("Question: ")
    
    if st.button("Ask question"):
        st.session_state.q_asked.append(prompt)
        st.session_state.q_answered_gpt.append(
            get_gpt_answer(prompt=prompt, gpt3_api=gpt3_api)
        )
        st.session_state.q_answered_gpt_with_clustering.append(
            get_gpt_with_clustering_answer(prompt=prompt)
        )
    
 
    idx = 0
    for q, a_gpt, a_gpt_with_clustering in zip(
        reversed(st.session_state.q_asked), 
        reversed(st.session_state.q_answered_gpt),
        reversed(st.session_state.q_answered_gpt_with_clustering)
    ):
        message(q, is_user=True, key=f"left_q_{idx}")
        message(f"GPT-3:\n{a_gpt}", is_user=False, key=f"left_a_gpt_{idx}")
        message(f"GPT-3 with clusters:\n{a_gpt_with_clustering}", is_user=False, key=f"left_a_gpt_with_clustering_{idx}")
        idx += 1
            

if __name__ == '__main__':
    main()