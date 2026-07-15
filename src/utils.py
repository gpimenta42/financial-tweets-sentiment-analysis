
from matplotlib import pyplot as plt
import pandas as pd
import string
import re
import numpy as np
from collections import Counter
import os
import nltk
import json
import torch
from tqdm import tqdm

from nltk.corpus import stopwords
from nltk.stem import SnowballStemmer
from nltk.stem.wordnet import WordNetLemmatizer

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_selection import SelectKBest, chi2


from sklearn.linear_model import LogisticRegression

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    precision_recall_fscore_support,
)
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.model_selection import StratifiedKFold

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

import re


import torch
import tensorflow as tf
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

from transformers import AutoTokenizer, AutoModel


pattern_meaning_dict = {
    "$words mentions": r"\$[A-Za-z]{1,5}(?:pr[A-Z])?",
    "$numbers mentions": r"\$\d+(?:\.\d+)?",
    "Negative number mentions": r"(?<![a-zA-Z0-9])-\d+(?:\.\d+)?\b(?!\d)",
    "Positive percentage mentions": r"(?<!-)\b\d+(?:\.\d+)?%",
    "Dates mentions": r"\b\d{1,2}[-/\.]\d{1,2}[-/\.]\d{2,4}\b|"  # 12/05/2024 or 12.05.2024
    r"\b(?:\d{1,2}(?:st|nd|rd|th)?\s)?(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec|January|February|March|April|May|June|July|August|September|October|November|December)[a-z]*\.?\s\d{2,4}\b|"  # 5th Jan 2023 or 5th January 2023
    r"\b(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec|January|February|March|April|May|June|July|August|September|October|November|December)\s\d{1,2}(?:st|nd|rd|th)?\b|"  # Jan 5th or January 5th
    r"\b\d{1,2}[-](?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*[-]\d{2,4}\b|"  # 12-Jan-2024
    r"\b(?:Mon|Tues|Tue|Wed|Thu|Thurs|Fri|Sat|Sun|Monday|Tuesday|Wednesday|Thursday|Friday|Saturday|Sunday)\b|"  # weekdays
    r"\b(?:today|tomorrow|yesterday|tonight|tonite|weekend|next\s(?:week|month|year|Monday|Tuesday|Wednesday|Thursday|Friday|Saturday|Sunday))\b|"  # relative words
    r"\b(?:in\s\d+\s(?:days?|weeks?|months?|years?))\b|"  # e.g. in 2 days
    r"\bon\s(?:the\s)?\d{1,2}(?:st|nd|rd|th)?\b|"  # on the 5th
    r"\b(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec|January|February|March|April|May|June|July|August|September|October|November|December)\b",  # month names,
    "Positive number mentions": r"(?<!-)\b\d{1,3}(?:,\d{3})*(?:\.\d+)?\b",
    "Links mentions": r'https://[^\s"]+',
    "Unicode mentions": r"[\U00010000-\U0010FFFF]",
    "User mentions (@user)": r"@[\w_]+",
    "Hashtag mentions (#hashtag)": r"#\w+",
    "Multi-exclamation mentions": r"(\!)\1+",
    "Exclamation mentions": r"(\!)+",
    "Multi-stop mentions": r"(\.)\1+",
    "Multi-question mentions": r"\?{2,}",
    "Question mentions": r"\?+",
    "Marketscreener mentions": r"\bmarketscreener\b|\bmarket screener\b",
}

pattern_substitution_dict = {
    "Positive percentage mentions": "percent",
    "Negative number mentions": "drop",
    "Dates mentions": "date",
    "$words mentions": "stock",
    "$numbers mentions": "price",
    "Positive number mentions": "x",
    "Multi-exclamation mentions": "wow",
    "Links mentions": "",
    "Unicode mentions": "",
    "User mentions (@user)": "",
    "Hashtag mentions (#hashtag)": "",
    "Multi-stop mentions": "",
    "Marketscreener mentions": "",
}


def analyze_regex_mentions(
    df, text_col="text", target_col="label", pattern=None, pattern_name="Pattern"
):
    colors = ["#9FE2BF", "#FFBF00"]

    # Boolean mask for rows where pattern is found
    mask = df[text_col].str.contains(pattern, regex=True)
    matched_df = df[mask]

    match_df_len = len(matched_df)

    print(
        f"Tweets with {pattern_name.upper()}\nTotal {match_df_len} ({round(match_df_len / len(df) * 100, 2)}%)\nExamples:\n"
    )
    for index, row in matched_df.sample(min(match_df_len, 5)).iterrows():
        print(f"Text: {row[text_col]}\nLabel: {row[target_col]}")
        print("-" * 60)

    matched_dist = matched_df[target_col].value_counts(normalize=True)
    overall_dist = df[target_col].value_counts(normalize=True)

    # Sort labels by overall count
    sorted_labels = overall_dist.sort_values(ascending=False).index

    # Reindex both distributions to ensure same label order
    matched_dist = matched_dist.reindex(sorted_labels, fill_value=0)
    overall_dist = overall_dist.reindex(sorted_labels, fill_value=0)

    # Combine into DataFrame
    dist_df = pd.DataFrame({"All": overall_dist, "Matched": matched_dist})

    dist_df.plot(
        kind="bar",
        figsize=(8, 5),
        title=f"Target Distribution: All vs Matched Pattern ({pattern_name}), n = {match_df_len}",
        color=colors,
    )
    plt.xlabel("Label")
    plt.ylabel("Proportion")
    plt.xticks(rotation=0)
    plt.legend(title="Dataset")
    plt.tight_layout()
    plt.show()

    return None


def get_top_tweet_bigrams(corpus, n=None):
    vec = CountVectorizer(ngram_range=(2, 2)).fit(corpus)
    bag_of_words = vec.transform(corpus)
    sum_words = bag_of_words.sum(axis=0)
    words_freq = [(word, sum_words[0, idx]) for word, idx in vec.vocabulary_.items()]
    words_freq = sorted(words_freq, key=lambda x: x[1], reverse=True)
    return words_freq[:n]


def count_non_printable(text):
    return sum(1 for char in text if not char.isprintable())


# ---------------------------------------------------------------------------------
# BAR PLOTS
# ---------------------------------------------------------------------------------
def visualize_frequency(df, col):
    plt.figure(figsize=(12, 5))

    # KDE by label
    plt.subplot(1, 2, 1)
    sns.kdeplot(data=df, x=col, hue="label", palette=sns.color_palette("Set2", 3))
    plt.xlabel(col)
    plt.ylabel("Density")
    plt.title(f"KDE plot of {col} by label")

    # Boxplot for all
    plt.subplot(1, 2, 2)
    sns.boxplot(x=df[col], color="#87bde1")
    plt.title(f"Boxplot of {col}")

    plt.tight_layout()
    plt.show()


# ---------------------------------------------------------------------------------
# WORD, CHARACTER COUNT
# ---------------------------------------------------------------------------------
def avg_word(sentence):
    words = re.findall(r"\b\w+\b", sentence)
    return sum(len(word) for word in words) / len(words) if words else 0


def avg_word_length(df):
    df["avg_word"] = df["text"].apply(avg_word)
    return df[["text", "avg_word"]].head()


# ---------------------------------------------------------------------------------
# SPECIAL CHARACTERS: HASHTAG, DOLLAR TICKER, EMOJIS, MENTIONS, QUOTATIONS
# ---------------------------------------------------------------------------------
def extract_hashtags(df):
    df["hashtags"] = df["text"].apply(
        lambda x: [word for word in x.split() if word.startswith("#")]
    )
    return df[["text", "hashtags"]].head()


def extract_dollar(text):
    # Find $ tickers
    return re.findall(r"\$\w+", text)


def is_valid_ticker(word):
    # Keep only the ones that are not numbers
    return re.match(r"^\$[A-Za-z]{2,10}$", word) is not None


def extract_mentions(text):
    # Find @s
    return re.findall(r"\@\w+", text)


def extract_emojis(text):
    # Get list with emojis
    return [char for char in text if emoji.is_emoji(char)]


def extract_quoted_content(text):
    # This makes sure that it won't take 'It' as a quotation
    double_quotes = re.findall(r'"(.*?)"', text)
    single_quotes = re.findall(r"(?<!\w)'(.*?)'(?!\w)", text)
    return double_quotes + single_quotes


# ------------------------------------------------------------------------------------------------------------
# TOKENIZE FOR GLOVE EMBEDDING
# ------------------------------------------------------------------------------------------------------------


def average_embedding(text, model, dim):
    words = text.split()
    vectors = []
    for word in words:
        if word in model:
            vectors.append(model[word])
    if vectors:
        # is a vector
        return np.mean(vectors, axis=0)
    else:
        return np.zeros(dim)


# def average_embedding_transform(X): # So we can incorporate it, it's a wrapper for the average embedding
#     return np.array([average_embedding(text, glove_embedder, dim) for text in X])


def make_embedding_transformer(glove_embeddings, dim):
    def transformer(X):
        return np.array([average_embedding(text, glove_embeddings, dim) for text in X])

    return FunctionTransformer(transformer, validate=False)


# ------------------------------------------------------------------------------------------------------------
# CREATE REGEX FLAGS
# ------------------------------------------------------------------------------------------------------------
def create_regex_flag(
    word_mentions=True,
    number_mentions=True,
    negative_number_mentions=True,
    positive_percentage_mentions=True,
    dates_mentions=True,
    positive_number_mentions=True,
    links_mentions=True,
    unicode_mentions=True,
    user_mentions=True,
    hashtag_mentions=True,
    multi_exclamation_mentions=True,
    multi_stop_mentions=True,
    marketscreener_mentions=True,
):
    return {
        "$words mentions": word_mentions,
        "$numbers mentions": number_mentions,
        "Negative number mentions": negative_number_mentions,
        "Positive percentage mentions": positive_percentage_mentions,
        "Dates mentions": dates_mentions,
        "Positive number mentions": positive_number_mentions,
        "Links mentions": links_mentions,
        "Unicode mentions": unicode_mentions,
        "User mentions (@user)": user_mentions,
        "Hashtag mentions (#hashtag)": hashtag_mentions,
        "Multi-exclamation mentions": multi_exclamation_mentions,
        "Multi-stop mentions": multi_stop_mentions,
        "Marketscreener mentions": marketscreener_mentions,
    }


# ------------------------------------------------------------------------------------------------------------
# PREPROCESSING WITH REGEX FLAGS
# ------------------------------------------------------------------------------------------------------------


def processing_text(
    text,
    regex_flags,
    re_expressions=None,
    substitution=None,
    remove_stopwords=True,
    remove_punct=True,
    stem=False,
    lemmatize=True,
    regex=True,
    lowercase=True,
):


    text = "".join([ch for ch in text if ch in string.printable])

    if regex and re_expressions:
        for name, pattern in re_expressions.items():
            if regex_flags is not None and not regex_flags.get(name, False):
                continue  # skip

            substitute = substitution.get(name, "")
            text = re.sub(pattern, substitute, text)

    if remove_stopwords:
        stop_words = set(stopwords.words("english"))
        """Some stopwords are relevant: down; up"""
        remove_words = [
            "down",
            "up",
            "above",
            "below",
            "not",
            "no",
            "but",
            "don't",
            "doesn't",
            "doesn'",
        ]
        for word in remove_words:
            if word in stop_words:
                stop_words.remove(word)
        text = " ".join(
            [word for word in text.split() if word.lower() not in stop_words]
        )

    if remove_punct:
        exclude = set(string.punctuation)
        text = "".join(ch for ch in text if ch not in exclude)

    if lowercase:
        text = text.lower()

    if lemmatize:
        lemmatizer = WordNetLemmatizer()
        text = " ".join([lemmatizer.lemmatize(word) for word in text.split()])

    if stem:
        stemmer = SnowballStemmer("english")
        text = " ".join([stemmer.stem(word) for word in text.split()])

    return text


# ------------------------------------------------------------------------------------------------------------
# TO GET THE RESULTS
# ------------------------------------------------------------------------------------------------------------


def get_results(y_true, y_pred):
    """
    It takes:
    - y_true: true labels
    - y_pred: predicted labels

    Returns a dictionary with:
        Accuracy, Macro-averaged Precision/recall/F1, and per-class Precision/Recall/F1 scores.


    """
    results = {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision_macro": precision_score(
            y_true, y_pred, average="macro", zero_division=0
        ),
        "recall_macro": recall_score(y_true, y_pred, average="macro", zero_division=0),
        "f1_macro": f1_score(y_true, y_pred, average="macro", zero_division=0),
        "precision_per_class": precision_score(
            y_true, y_pred, average=None, zero_division=0
        ),
        "recall_per_class": recall_score(y_true, y_pred, average=None, zero_division=0),
        "f1_per_class": f1_score(y_true, y_pred, average=None, zero_division=0),
    }
    return results


# ------------------------------------------------------------------------------------------------------------
# FUNCTION TO SAVE TO CSV ORGANISED
# ------------------------------------------------------------------------------------------------------------

preprocess_configurations = [
    {
        "stem": False,
        "lemmatize": False,
        "remove_stopwords": False,
        "regex": False,
        "lowercase": False,
        "remove_punct": False,
    },
    # simply preprocess with/without regex
    {
        "stem": False,
        "lemmatize": False,
        "remove_stopwords": True,
        "regex": False,
        "lowercase": True,
        "remove_punct": True,
    },
    {
        "stem": False,
        "lemmatize": False,
        "remove_stopwords": True,
        "regex": True,
        "lowercase": True,
        "remove_punct": True,
    },
    # Aplying only one
    {
        "stem": False,
        "lemmatize": False,
        "remove_stopwords": False,
        "regex": False,
        "lowercase": False,
        "remove_punct": True,
    },
    {
        "stem": False,
        "lemmatize": False,
        "remove_stopwords": False,
        "regex": False,
        "lowercase": True,
        "remove_punct": False,
    },
    {
        "stem": False,
        "lemmatize": False,
        "remove_stopwords": False,
        "regex": True,
        "lowercase": False,
        "remove_punct": False,
    },
    {
        "stem": False,
        "lemmatize": False,
        "remove_stopwords": True,
        "regex": False,
        "lowercase": False,
        "remove_punct": False,
    },
    # 2 things
    {
        "stem": False,
        "lemmatize": False,
        "remove_stopwords": False,
        "regex": False,
        "lowercase": True,
        "remove_punct": True,
    },
    {
        "stem": False,
        "lemmatize": False,
        "remove_stopwords": False,
        "regex": True,
        "lowercase": True,
        "remove_punct": True,
    },
    # stemming a lemmatization
    {
        "stem": True,
        "lemmatize": False,
        "remove_stopwords": True,
        "regex": False,
        "lowercase": True,
        "remove_punct": True,
    },  # stem
    {
        "stem": False,
        "lemmatize": True,
        "remove_stopwords": True,
        "regex": False,
        "lowercase": True,
        "remove_punct": True,
    },  # lemma
    {
        "stem": True,
        "lemmatize": False,
        "remove_stopwords": True,
        "regex": True,
        "lowercase": True,
        "remove_punct": True,
    },
    {
        "stem": False,
        "lemmatize": True,
        "remove_stopwords": True,
        "regex": True,
        "lowercase": True,
        "remove_punct": True,
    },
]

def save_results_to_csv(
    embedder,
    config_number,
    config_id,
    model_name,
    results,
    results_file,
    tested_configs,
):

    df_new = pd.DataFrame(
        [
            {
                "config_number": embedder + "_" + config_number,  # ADDED THIS
                "config": config_id,
                "model": model_name,
                "accuracy": results["accuracy"],
                "recall_macro": results["recall_macro"],
                "precision_macro": results["precision_macro"],
                "f1_score_macro": results["f1_macro"],
                "precision_per_class": results["precision_per_class"].tolist(),
                "recall_per_class": results["recall_per_class"].tolist(),
                "f1_per_class": results["f1_per_class"].tolist(),
            }
        ]
    )

    if os.path.exists(results_file):
        df_new.to_csv(results_file, mode="a", header=False, index=False)
    else:
        df_new.to_csv(results_file, index=False)

    tested_configs.add(config_id)
    # print(f"[SAVED] Results for config: {config_id}")


# ------------------------------------------------------------------------------------------------------------
# FUNCTION TO TEST DIFFERENT PREPROCESSING + FEATURE EXTRACTION
# ------------------------------------------------------------------------------------------------------------


def test_feature_engineering(
    df_train,
    preprocessing_configs,
    pattern_meaning_dict,
    pattern_substitution_dict,
    feature_extractor,
    feature_extractor_name,
    params,
    results_file,
    tested_configs,
    model_name,
    embedder_name,
    regex_flags,
    model,
):

    stratified_kf = StratifiedKFold(
        n_splits=5, shuffle=True, random_state=42
    )  
    
    for config_idx, config in enumerate(preprocessing_configs):
        stem = config["stem"]
        lemmatize = config["lemmatize"]
        remove_stopwords = config["remove_stopwords"]
        regex = config["regex"]
        lower = config["lowercase"]
        punct = config["remove_punct"]
        
        
        
                    

        for fold_idx, (train_index, val_index) in enumerate(
            stratified_kf.split(df_train["text"], df_train["label"])
        ):
            config_id = (
                f"cv={fold_idx+1}"
                f"|remove_stopwords={remove_stopwords}"
                f"|stem={stem}"
                f"|lemmatize={lemmatize}"
                f"|regex={regex}"
                f"|lowercase={lower}"
                f"|remove_punct={punct}"
                f"|feature_extractor={feature_extractor_name}"
                f"- {params}",
                f"|{model_name}"
                )
            
            if str(config_id) in tested_configs:
                print(f"Skipping config already tested - configuration {config_id}")
                continue
            
            if fold_idx == 0:
                print(f"\nRunning Config {config_id}")
            
            
            X_train, X_val = (
                df_train["text"].iloc[train_index],
                df_train["text"].iloc[val_index],
            )
            y_train, y_val = (
                df_train["label"].iloc[train_index],
                df_train["label"].iloc[val_index],
            )

            X_train_processed = X_train.apply(
                lambda x: processing_text(
                    x,
                    remove_stopwords=remove_stopwords,
                    stem=stem,
                    lemmatize=lemmatize,
                    regex=regex,
                    re_expressions=pattern_meaning_dict,
                    substitution=pattern_substitution_dict,
                    lowercase=lower,
                    remove_punct=punct,
                    regex_flags=regex_flags,
                )
            )
            X_val_processed = X_val.apply(
                lambda x: processing_text(
                    x,
                    remove_stopwords=remove_stopwords,
                    stem=stem,
                    lemmatize=lemmatize,
                    regex=regex,
                    re_expressions=pattern_meaning_dict,
                    substitution=pattern_substitution_dict,
                    lowercase=lower,
                    remove_punct=punct,
                    regex_flags=regex_flags,
                )
            )

            if regex:
                train_mask = X_train_processed.str.strip().astype(bool)
                X_train_processed = X_train_processed[train_mask]
                y_train_filtered = y_train[train_mask]

                val_mask = X_val_processed.str.strip().astype(bool)
                X_val_processed = X_val_processed[val_mask]
                y_val_filtered = y_val[val_mask]
            else:
                y_train_filtered = y_train
                y_val_filtered = y_val

            # Model
            pipeline = Pipeline(
                [("embedding", feature_extractor), 
                 ("classifier", model)]
            )

            pipeline.fit(X_train_processed, y_train_filtered)
            y_pred = pipeline.predict(X_val_processed)

            results = get_results(y_val_filtered, y_pred)

            print(
                f"CV{fold_idx+1} - F1 Macro: {results['f1_macro']:.4f}"
            )

            save_results_to_csv(
                embedder_name,
                config_number=str(config_idx + 1),
                config_id=config_id,
                model_name="LogisticRegression",
                results=results,
                results_file=results_file,
                tested_configs=tested_configs,
            )



# ------------------------------------------------------------------------------------------------------------
# FUNCTION FOR HUFFING FACE TRANSFORMER BASED EMBEDDINGS
# ------------------------------------------------------------------------------------------------------------

class HF_Transformer_Embedding(BaseEstimator, TransformerMixin):
    def __init__(self, model_name, batch_size=16, max_length=128):
        self.model_name = model_name
        self.batch_size = batch_size
        self.max_length = max_length

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu") # It will use GPU if available, else CPU

        #Uses the tokenizer for this model
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)

        # Use automodel from hugging face
        self.model = AutoModel.from_pretrained(model_name)
        self.model.to(self.device)
        self.model.eval()

    def transform(self, X, y=None): # Given X=list of text
        all_embeddings = [] # Empty list to store the number representations for the text
        
        tokenizer = AutoTokenizer.from_pretrained(self.model_name)
    
    
        train_len = X.map(lambda x: len(tokenizer(x)["input_ids"]))
        max_len = train_len.max()

        # Loop over input text
        for i in range(0, len(X), self.batch_size):

            batch_texts = X[i:i+self.batch_size] # We will process in batches

            inputs = self.tokenizer(batch_texts.tolist(), # Conver the batch of sentences to token IDs
                                    padding=True, # same length
                                    truncation=True, #remove if sentence too long
                                    max_length=max_len, 
                                    return_tensors="pt")

            # Move to processor
            inputs = {k: v.to(self.device) for k, v in inputs.items()}


            with torch.no_grad(): # no training so no need to calculate gradients
                outputs = self.model(**inputs) # pass tokens through model to get output vetores

                # From the last hidden state, extract the cls token embedding
                cls_embeddings = outputs.last_hidden_state[:, 0, :].cpu().numpy()

                all_embeddings.append(cls_embeddings) # add to the list

        return np.vstack(all_embeddings)

    def fit(self, X, y=None):
        return self


# Helper function to format parameters of grid search for config
def format_params_for_config(params):
    return "|".join([f"{k}={v}" for k, v in params.items()])

def extract_features_chi2(X_train, y_train, top_features=100):
    vectorizer = TfidfVectorizer(min_df=4)
    vectorizer.fit(X_train)
    vec_X_train = vectorizer.transform(X_train)
    feature_names = vectorizer.get_feature_names_out()

    ch2 = SelectKBest(chi2, k=top_features)
    ch2.fit(vec_X_train, y_train)
    X_train_ch2 = ch2.transform(vec_X_train)

    feat_index = ch2.get_support(indices=True)
    feat_score = ch2.scores_[feat_index]
    dict_feats = dict(zip(feat_index, feat_score))
    dict_names = [(feature_names[idx], value) for idx, value in dict_feats.items()]
    ngrams_sorted = sorted(dict_names, key=lambda x: -x[1])
    return [w for w, i in ngrams_sorted]
