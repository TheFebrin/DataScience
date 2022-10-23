import pandas as pd
from tqdm import tqdm 
from transformers import *

def process_line(line):
    processed_line: List[str] = line.replace('\n', '').strip().split('  ===>  ')
    assert len(processed_line) == 2
    return processed_line


def main():
    data1 = []
    with open('data/pseudo_dialogs_ver1.txt', 'r', encoding='utf8') as f:
        for x in f.readlines():
            data1.append(process_line(x))

    
def test_polbert():
    """
    https://huggingface.co/dkleczek/bert-base-polish-uncased-v1?text=dupa
    https://github.com/kldarek/polbert/blob/master/LM_testing.ipynb
    """
    model = BertForMaskedLM.from_pretrained("dkleczek/bert-base-polish-cased-v1")
    tokenizer = BertTokenizer.from_pretrained("dkleczek/bert-base-polish-cased-v1")
    nlp = pipeline('fill-mask', model=model, tokenizer=tokenizer)

    print(f"Adam Mickiewicz wielkim polskim {nlp.tokenizer.mask_token} był.")
    for pred in nlp(f"polacy nie gęsi {nlp.tokenizer.mask_token} mają"):
        print(pred)


def test_herbert():
    """
    https://huggingface.co/allegro/herbert-large-cased
    """
    tokenizer = AutoTokenizer.from_pretrained("allegro/herbert-large-cased")
    model = AutoModel.from_pretrained("allegro/herbert-large-cased")

    output = model(
        **tokenizer.batch_encode_plus(
            [
                "A potem szedł środkiem drogi w kurzawie, bo zamiatał nogami, ślepy dziad prowadzony przez tłustego kundla na sznurku.",
                "A potem leciał od lasu chłopak z butelką, ale ten ujrzawszy księdza przy drodze okrążył go z dala i biegł na przełaj pól do karczmy.",
                "a",
                "b",
                "c",
            ],
        padding='longest',
        add_special_tokens=True,
        return_tensors='pt'
        )
    )
    print(dir(output))
    print(output.keys())
    print('\n\n===========\n\n')
    print(output.values().mapping['last_hidden_state'].shape)
    print('\n\n===========\n\n')
    print(output.values().mapping['pooler_output'].shape)
    print('\n\n===========\n\n')
    print(dir(output.values().mapping))



if __name__ == "__main__":
    # main()
    test_herbert()

   