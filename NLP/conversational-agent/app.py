import streamlit as st
from enum import Enum, auto
from transformers import AutoTokenizer, AutoModel
from typing import *
from tqdm import tqdm
import torch
import time
import json


class ModelName(Enum):
    POLBERT = 'PolBert'
    HERBERT = 'HerBert'


def load_herbert() -> Tuple[AutoTokenizer, AutoModel]:
    tokenizer = AutoTokenizer.from_pretrained("allegro/herbert-large-cased")
    model = AutoModel.from_pretrained("allegro/herbert-large-cased")
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
    ).values().mapping['pooler_output']


def process_line(line) -> Tuple[str, str]:
    data = json.loads(line)
    return data['queries']['pl'], data['answers']['pl'][0]['text']


@st.cache
def read_dataset():
    data = []
    with open('data/mkqa.jsonl', 'r', encoding='utf8') as f:
        for l in f.readlines():
            data.append(process_line(l))
    return data


def vectorize_and_save(
    dataset: List[Tuple[str, str]],
    model: AutoModel, 
    tokenizer: AutoTokenizer, 
) -> None:
    BATCH_SIZE = 100
    for i in range(400, len(dataset), BATCH_SIZE):
        st.text(f'Batch {i} -> {i + BATCH_SIZE - 1}')
        encodings = predict(
            model=model,
            tokenizer=tokenizer,
            sentence=[q for q, a in dataset[i:i + BATCH_SIZE]]
        )
        torch.save(encodings, f'data_encoded/encoded_{i}.pt')
        del encodings


def read_vectorized_data(filename: str):
    t = torch.load(filename)
    st.text(t)
    st.text(t.shape)


def main() -> None:
    st.header('Retrieval based conversational agent for Polish')

    time_start = time.time()
    st.text('Loading model')
    tokenizer, model = load_herbert()
    dataset = read_dataset()
    st.text(f'Model loaded in {time.time() - time_start}s')

    # read_vectorized_data('data_encoded/encoded.pt')
    # st.stop()

    if st.checkbox('Vectorize and save datset'):
        vectorize_and_save(
            dataset=dataset,
            model=model,
            tokenizer=tokenizer
        )

    model_name: str = st.radio(
        "Choose model:", 
        [ModelName.POLBERT.value, ModelName.HERBERT.value]
    )

    if model_name == ModelName.POLBERT.value:
        pass
    elif model_name == ModelName.HERBERT.value:
        pass
    else:
        raise RuntimeError('Wrong model.')

    st.text(f'Chosen model: {model_name}')

    st.subheader('Input your prompt:')
    prompt: str = st.text_area('prompt:')

    # prediction = predict(
    #     model=model,
    #     tokenizer=tokenizer,
    #     sentence=[prompt]
    # )
    # st.text(prediction)





if __name__ == '__main__':
    main()