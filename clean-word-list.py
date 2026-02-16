import csv
import os
import subprocess
from string import punctuation
import pandas as pd
import spacy
import spacy_transformers
from wordfreq import zipf_frequency

def count_csv_elements_in_file(filepath):

    total_elements = 0

    with open(filepath, "r", encoding="utf-8") as file:
        csv_reader = csv.reader(file)
        for row in csv_reader:
            total_elements += len(row)

    return total_elements

language = []
total_words = []

for path, subdirs, files in os.walk("raw-word-lists"):
    for name in files:
        filename = (os.path.join(path, name))
        language += [name.split(".")[0]]
        total_words += [count_csv_elements_in_file(filename)]

pd.DataFrame({
    "language": language,
    "total_words": total_words
})

spacy_models = {
    "Catalan": "ca_core_news_trf",
    "Croatian": "hr_core_news_lg",
    "Danish": "da_core_news_trf",
    "Dutch": "nl_core_news_lg",
    "English": "en_core_web_trf",
    "Finnish": "fi_core_news_lg",
    "French": "fr_dep_news_trf",
    "German": "de_dep_news_trf",
    "Greek": "el_core_news_lg",
    "Italian": "it_core_news_lg",
    "Lithuanian": "lt_core_news_lg",
    "Macedonian": "mk_core_news_lg",
    "Norwegian": "nb_core_news_lg",
    "Polish": "pl_core_news_lg",
    "Portuguese": "pt_core_news_lg",
    "Romanian": "ro_core_news_lg",
    "Russian": "ru_core_news_lg",
    "Slovenian": "sl_core_news_trf",
    "Spanish": "es_dep_news_trf",
    "Swedish": "sv_core_news_lg",
    "Ukrainian": "uk_core_news_trf"
}

for model in spacy_models.values():
    result = subprocess.run(['python', '-m', 'spacy', 'download', f'{model}'],
                       capture_output=True, text=True)
    print(result.stdout)
    print(result.stderr)

os.mkdir("data")

for language in spacy_models.keys():
    try:
        os.mkdir(f"data/{language}")
        print(f"Directory {language} created")
    except FileExistsError:
        print(f"Directory {language} already exists")

def load_and_clean_word_list(language: str) -> pd.DataFrame:
    with open(f"raw-word-lists/{language}/{language}.txt", "r", encoding="utf-8") as f:
        word_list = f.read().split(",")

    word_df = pd.DataFrame({
        "word": word_list
    })

    word_df["word"] = word_df["word"].str.strip(punctuation)

    return word_df

load_and_clean_word_list("English")

nlp = spacy.load(spacy_models["English"], disable=["parser", "ner", "textcat"])

def add_lemma(df: pd.DataFrame,
              nlp,
              batch_size: int = 1000) -> pd.DataFrame:
    docs = nlp.pipe(df["word"].tolist(), batch_size=batch_size)
    lemmas = [doc[0].lemma_ for doc in docs]
    df["lemma"] = pd.DataFrame(lemmas, index=df.index)
    return df

add_lemma(
    load_and_clean_word_list("English")[:100], nlp
)

def add_word_frequencies(df: pd.DataFrame,
                         language: str) -> pd.DataFrame:
    language_group = spacy_models[language].split("_")[0]
    df["zipf_freq_lemma"] = [zipf_frequency(w, language_group) for w in df["lemma"]]
    return df

add_word_frequencies(
    add_lemma(
        load_and_clean_word_list("English")[:100], nlp
), "English")

def clean_up_and_export(df: pd.DataFrame, language: str) -> None:
    df = (
        df.loc[df.groupby("lemma", sort=False)["zipf_freq_lemma"].idxmax()]
        .reset_index(drop=True)
    )

    df = df[(df["zipf_freq_lemma"] > 0)]

    df.loc[:, "word_difficulty"] = pd.cut(
        df["zipf_freq_lemma"],
        bins = [-float("inf"), 2.0, 4.0, float("inf")],
        labels = ["advanced", "intermediate", "beginner"],
        include_lowest = True,
        right = True
    )

    df = df.drop(columns=["word", "zipf_freq_lemma"])
    df = df.rename(columns = {
        "lemma": "word"
    })

    df.to_json(f"data/{language}/word-list-cleaned.json", orient="index")

clean_up_and_export(
    add_word_frequencies(
        add_lemma(
            load_and_clean_word_list("English")[:100], nlp),
        "English"),
    "English")

def create_clean_word_list(language: str) -> None:
    nlp = spacy.load(spacy_models[language], disable=["parser", "ner", "textcat"])

    print("Load in dataset")
    lang_df = load_and_clean_word_list(language)

    print("Lemmatise words")
    lang_df = add_lemma(lang_df, nlp)

    print("Add the word frequencies")
    lang_df = add_word_frequencies(lang_df, language)

    print("Do the final clean ups and export to file")
    clean_up_and_export(lang_df, language)

    return None

create_clean_word_list("Spanish")

import json

language_raw = []
total_words_raw = []

for path, subdirs, files in os.walk("raw-word-lists"):
    for name in files:
        filename = (os.path.join(path, name))
        language_raw += [name.split(".")[0]]
        total_words_raw += [count_csv_elements_in_file(filename)]

raw_data = pd.DataFrame({
    "language": language_raw,
    "type": ["Raw"] * 21,
    "total_words_raw": total_words_raw,
})

language_clean = []
total_words_clean = []

for path, subdirs, files in os.walk("data"):
    for name in files:
        filename = (os.path.join(path, name))
        language_clean += [path.split("/")[1]]
        with open(filename, "r", encoding="utf-8") as f:
            data = json.load(f)
            total_words_clean += [len(data.keys())]

clean_data = pd.DataFrame({
    "language": language_clean,
    "type": ["Clean"] * 21,
    "total_words_raw": total_words_clean,
})

pd.concat([raw_data, clean_data])