import streamlit as st
from enum import Enum, auto
from transformers import AutoTokenizer, AutoModel, BertForMaskedLM, BertTokenizer, pipeline
import tokenizers
from typing import *
from tqdm import tqdm
import torch
import time
import json
import os
import numpy as np
from datasets import load_dataset
from sklearn.neighbors import KNeighborsClassifier


@st.cache(hash_funcs={tokenizers.Tokenizer: lambda x: x.__hash__})
def load_herbert() -> Tuple[AutoTokenizer, AutoModel]:
    tokenizer = AutoTokenizer.from_pretrained("allegro/herbert-large-cased")
    model = AutoModel.from_pretrained("allegro/herbert-large-cased")
    return tokenizer, model


@st.cache
def load_polbert() -> Tuple[BertTokenizer, BertForMaskedLM]:
    model = BertForMaskedLM.from_pretrained("dkleczek/bert-base-polish-cased-v1")
    tokenizer = BertTokenizer.from_pretrained("dkleczek/bert-base-polish-cased-v1")
    return tokenizer, model


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


def process_line(line) -> Tuple[str, str]:
    data = json.loads(line)
    return data['queries']['pl'], data['answers']['pl'][0]['text']


@st.cache
def read_mkqa_dataset(filename: str):
    data = []
    with open(f'data/{filename}', 'r', encoding='utf8') as f:
        for l in f.readlines():
            data.append(process_line(l))
    return data


@st.cache
def read_dyk_dataset(name: str) -> List[Tuple[str, str]]:
    dataset = load_dataset(name)
    test = [
        (dataset['test']['question'][i].strip('\"'), dataset['test']['answer'][i].strip('\"'))
        for i in range(len(dataset['test']))
    ]
    train = [
        (dataset['train']['question'][i].strip('\"'), dataset['train']['answer'][i].strip('\"'))
        for i in range(len(dataset['train']))
    ]
    return train + test


@st.cache
def read_poleval_dataset() -> List[Tuple[str, str]]:
    dataset = []
    for prefix in ('a', 'b', 'c'):
        questions = []
        with open(f'data/poleval/{prefix}_questions.txt', 'r', encoding='utf8') as f:
            for l in f.readlines():
                questions.append(l.replace('\n', '').strip())
        answers = []
        with open(f'data/poleval/{prefix}_answers.txt', 'r', encoding='utf8') as f:
            for l in f.readlines():
                answers.append(l.replace('\n', '').strip())
        
        dataset += list(zip(questions, answers))
    return dataset


@st.cache
def read_sitcoms_dataset() -> List[Tuple[str, str]]:
    dataset = []
    for filename in list(filter(lambda x: x.find('.txt') != -1, os.listdir('data/sitcoms'))):
        with open(f'data/sitcoms/{filename}', 'r', encoding='utf8') as f:
            text = f.readlines()
            text = list(map(lambda x: x.replace('-', '').strip(), text))
            text = list(filter(
                lambda x: len(x) > 1 and x.find('-->') == -1 and not x.isdigit() and x.find(' > ') == -1,
                text
            ))
            text = list(map(lambda x: x.replace('<i>', '').replace('</i>', '').strip(), text))
            questions = text.copy()
            questions.insert(0, 'start')
            answers = text.copy()
        dataset += list(zip(questions, answers))
    return dataset


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
            sentence=[q for q, a in dataset[i:i + BATCH_SIZE]]
        )
        torch.save(encodings, f'{path_and_filename}_{i}.pt')
        del encodings


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

def find_answer_idx(neigh: KNeighborsClassifier, encoded_q : np.ndarray, n_neighbors: int) -> np.ndarray:
    
    dist, idx = neigh.kneighbors([encoded_q], n_neighbors=n_neighbors, return_distance=True)
    
    sorted_idx = np.argsort(dist).flatten()
    idx = idx.flatten()

    return idx[sorted_idx].flatten()

def main() -> None:
    st.header('Retrieval based conversational agent for Polish')
    with st.expander('Used datasets:'):
        st.markdown(
            """
            * Czy wiesz?: https://huggingface.co/datasets/dyk?fbclid=IwAR1dcUr1A3BLLrqWAKiaFySFLPOzyslcCC1H1ucdWVEJmScS2QT3RkKi-Z0
            * mkqa: https://huggingface.co/datasets/mkqa?fbclid=IwAR3qcPXYIK6zPAnRGL2Ka_lAW_hOhBJASparZ572mPbAQHZpQVpKctiLobg
            * poleval contest: https://github.com/poleval/2021-question-answering, https://github.com/poleval/2021-question-answering/tree/secret/test-B, http://2021.poleval.pl/tasks/task4
            """
        )

    time_start = time.time()
    st.text('Loading models ...')
    herbert_tokenizer, herbert_model = load_herbert()
    polbert_tokenizer, polbert_model = load_polbert()
    polbert_pipeline = pipeline('fill-mask', model=polbert_model, tokenizer=polbert_tokenizer)
    st.text(f'Model loaded in {time.time() - time_start:.2f}s')

    time_start = time.time()
    mkqa_dataset = read_mkqa_dataset(filename='mkqa.jsonl')
    dyk_dataset = read_dyk_dataset(name="allegro/klej-dyk")
    poleval_dataset = read_poleval_dataset()
    sitcoms_dataset = read_sitcoms_dataset()

    # if st.checkbox('Vectorize and save datset'):
    #     vectorize_and_save(
    #         dataset=sitcoms_dataset,
    #         model=herbert_model,
    #         tokenizer=herbert_tokenizer,
    #         path_and_filename='data_encoded/sitcoms/encoded'
    #     )
    # st.text('DONE!!!')
    # st.stop()
    
    dataset: List[Tuple[str, str]] = mkqa_dataset + dyk_dataset + poleval_dataset + sitcoms_dataset
    st.text(f'Dataset size: {len(dataset)}')
    
    encoded_mkqa_data = join_encoding(
        to_index=10000,
        filename="mkqa/encoded" 
    )
    
    encoded_dyk_data = join_encoding(
        to_index=5200,
        filename="dyk/encoded"
    )
    
    encoded_poleval_data = join_encoding(
        to_index=6000,
        filename="poleval/encoded"
    )
    
    sitcoms_data = join_encoding(
        to_index=12600,
        filename="sitcoms/encoded"
    )
    
    dataset_dict = {
        "mkqa" : encoded_mkqa_data.detach().numpy(),
         "dyk" : encoded_dyk_data.detach().numpy(),
         "poleval" : encoded_poleval_data.detach().numpy(),
         "sitcoms" : sitcoms_data.detach().numpy()
    }

    # make container with 2 columns in width ratio 4/3
    col1, col2 = st.columns([4,3]) 

    with col1:
        N_NEIGHBORS: int = st.slider('select k neighbours', 1, 10, 5)
    with col2:
        selected_datasets = st.multiselect(
            "select one or more dataset",
            ["mkqa", "dyk", "poleval", "sitcoms"],
            )
        
    # default selected option (if none selected)
    if len(selected_datasets) == 0:
        selected_datasets = ["mkqa"]
        
    X = np.vstack([dataset_dict[name] for name in selected_datasets])

    y = [
        i for i in range(
            sum([dataset_dict[name].shape[0] for name in selected_datasets])
        )
    ]

    neigh = KNeighborsClassifier(n_neighbors=N_NEIGHBORS, metric='cosine')
    neigh.fit(X, y)
    
    st.text(f'Dataset loaded and KNN fit in {time.time() - time_start:.2f}s')

    st.subheader('Ask HuBert:')
    question: str = st.text_area('question:')

    if question:
        encoded_q = predict(
            model=herbert_model,
            tokenizer=herbert_tokenizer,
            sentence = [question]
        )
        encoded_q = encoded_q[0].detach().numpy()

        indexes = find_answer_idx(
            neigh=neigh, 
            encoded_q=encoded_q,
            n_neighbors=N_NEIGHBORS
        )

        for idx in indexes:
            if dataset[idx][1]:
                with st.expander(f"{dataset[idx][1]}"):
                    st.write(f"{dataset[idx][0]}")
    
    st.subheader('PolBert can fill the gap in the sentence for you:')
    st.code(
        """
        Example usage:
        prompt: Adam Mickiewicz wielkim polskim <FILL> był.

        > Adam Mickiewicz wielkim polskim pisarzem był.
        > Adam Mickiewicz wielkim polskim człowiekiem był.
        > Adam Mickiewicz wielkim polskim bohaterem był.
        > Adam Mickiewicz wielkim polskim mistrzem był.
        > Adam Mickiewicz wielkim polskim artystą był.
        """
    )
    prompt: str = st.text_area('prompt:', key='prompt')
    if len(prompt) > 0:
        for preds in polbert_pipeline(prompt.replace("<FILL>", polbert_pipeline.tokenizer.mask_token)):
            if isinstance(preds, list):
                for pred in preds:
                    st.text(f"{pred['sequence']} (score: {pred['score']:.3f})")
                st.text('===')
            else:
                st.text(f"{preds['sequence']} (score: {preds['score']:.3f})")


if __name__ == '__main__':
    main()