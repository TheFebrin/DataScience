import streamlit as st
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


@st.cache(hash_funcs={tokenizers.Tokenizer: lambda x: x.__hash__})
def load_herbert() -> Tuple[AutoTokenizer, AutoModel]:
    tokenizer = AutoTokenizer.from_pretrained("allegro/herbert-large-cased")
    model = AutoModel.from_pretrained("allegro/herbert-large-cased")
    return tokenizer, model

def read_q_a_dataset(file_name : str) -> List[Tuple[str, str]]:
    q_a_pairs = list()
    with open(file_name, "r") as f:
        for x in f.readlines():
            row = x.strip().split("\t")
            for i in range(1, len(row)):
                q_a_pairs.append([row[0], row[i]])
    return q_a_pairs

def predict(
    model: AutoModel, 
    tokenizer: AutoTokenizer,
    sentence: List[str],
) -> torch.tensor:
    return model(
        **tokenizer.batch_encode_plus(sentence,
        padding='longest',
        add_special_tokens=True,
        return_tensors='pt'
        )
    )['pooler_output']
    #.values().mapping['pooler_output']

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


#st.cache(suppress_st_warning=True)
#def get_clusters(n_clusters : int, dataset : np.array) -> KMeans:
#    return KMeans(n_clusters=n_clusters).fit(dataset.astype('double'))

def main() -> None:
    st.header('Clustering-based question answering agent for Polish')

    st.text('Loading models ...')
    time_start = time.time()
    herbert_tokenizer, herbert_model = load_herbert()
    st.text(f'Model loaded in {time.time() - time_start:.2f}s')

    q_a_dataset = read_q_a_dataset("./data/task2_questions_with_answers.tsv")

    #if st.checkbox('Vectorize and save datset'):
    #    vectorize_and_save(
    #        dataset=q_a_dataset,
    #        model=herbert_model,
    #        tokenizer=herbert_tokenizer,
    #        path_and_filename='data_encoded/q_a/encoded'
    #    )
    #    st.text('DONE!!!')
    #    st.stop()
    q_a_dataset = pd.DataFrame(q_a_dataset)

    with st.expander("Show questions"):
        st.dataframe(q_a_dataset)

    encoded_qa_data = join_encoding(
            to_index=4000,
            filename="q_a/encoded" 
        )

    st.text(f'Chosen dataset size: {len(encoded_qa_data)}')
        
    num_of_clusters: int = st.slider('select number of clusters', 30, 800, 400)
    with st.spinner("Clustering..."):
        kmeans = generate_kmeans(num_of_clusters, encoded_qa_data)


    X_embedded = calculate_tsne(data=encoded_qa_data.detach().numpy().astype('double'))
    fig = px.scatter(
        pd.DataFrame({"x": X_embedded[:, 0], "y": X_embedded[:, 1], "class": kmeans.labels_, "question": q_a_dataset[0]}),
        x="x", y="y", color="class",
        hover_data=["question"]
    )
    st.plotly_chart(fig)

    question: str = st.text_area('question:')

    if question:
        encoded_q = predict(
            model=herbert_model,
            tokenizer=herbert_tokenizer,
            sentence = ["CLS " + question + " SEP "]
        )
        encoded_q = encoded_q[0].detach().numpy()
        predicted_cluster = kmeans.predict(X = [encoded_q])

        cluster_map = pd.DataFrame()
        cluster_map['data_index'] = q_a_dataset.index.values
        cluster_map['cluster'] = kmeans.labels_
        group_elements_idx = cluster_map[cluster_map.cluster == predicted_cluster[0]]['data_index']
        group_elements = q_a_dataset.iloc[group_elements_idx]

        st.write(f"cluster ID: {predicted_cluster[0]} |  | elements in this group: {group_elements.shape[0]} | randomly selected 5 pairs:")
        
        # randomly select n elements from group
        to_print_out = group_elements.sample(n=5)
        for index, row in to_print_out.iterrows():
            with st.expander(f"{row[1]}"):
                st.write(f"{row[0]}")

        #st.write(group_elements)

if __name__ == '__main__':
    main()