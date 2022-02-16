#!/usr/bin/env python
from typing import Callable, Generator, Iterable

import pandas as pd
from sklearn.feature_extraction import DictVectorizer
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import BernoulliNB
import argparse

""" Binary classification for a Sarcastic or Not Sarcastic News Article"""


class NewsHeadlineInstance:
    """ Return instance of News Headline data containing label and headline"""

    def __init__(self, label: int, headline: str, link: str) -> None:
        self.label: int = label
        self.headline: str = headline
        self.link: str = link

    def __repr__(self) -> str:
        return f"label: {self.label}; headline: {self.headline}"

    def __str__(self) -> str:
        return self.__repr__()

    @classmethod
    def from_series(cls, series_inst: pd.Series) -> "NewsHeadlineInstance":
        return NewsHeadlineInstance(series_inst['is_sarcastic'], series_inst['headline'], series_inst['article_link'])


def load_news_article_instances(df: pd.DataFrame) -> Generator[NewsHeadlineInstance, None, None]:
    """ Return an instance of the dataframe using a generator"""
    for _, row in df.iterrows():
        yield NewsHeadlineInstance.from_series(row)


def unigram_features(instance: NewsHeadlineInstance) -> dict[str, float]:
    """ Return a feature count dictionary for unigram features """
    return {token: 1.0 for token in instance.headline}


def bigram_features(instance: NewsHeadlineInstance) -> dict[str, float]:
    """ Return a feature count dictionary for bigram features"""
    features = {}
    for i in range(len(instance.headline) - 1):
        bigram = (instance.headline[i], instance.headline[i + 1])
        features[str(bigram)] = 1.0

    return features


def get_features_and_labels(instances: Iterable[NewsHeadlineInstance],
                            feature_generator: Callable[[NewsHeadlineInstance],
                                                        dict[str]]) -> tuple[list[dict[str]], list[int]]:
    """ Return a tuple of the features and labels for each instance within the dataset. """
    features = []
    labels = []
    for instance in instances:
        features.append(feature_generator(instance))
        labels.append(instance.label)
    return features, labels


def train_and_eval(train: pd.DataFrame, dev: pd.DataFrame,
                   feature_generator: Callable[[NewsHeadlineInstance], dict[str]]) -> tuple[float, str]:
    """ Train and Evaluate Development Set """
    vectorizer = DictVectorizer()

    # Train
    train_instances = load_news_article_instances(train)
    train_features, train_labels = get_features_and_labels(train_instances, feature_generator)
    train_feature_vectors = vectorizer.fit_transform(train_features, train_labels)
    model = BernoulliNB()
    model.fit(train_feature_vectors, train_labels)

    # Evaluate
    dev_instances = load_news_article_instances(dev)
    dev_features, dev_labels = get_features_and_labels(dev_instances, feature_generator)
    dev_feature_vectors = vectorizer.transform(dev_features)
    dev_predicted_labels = model.predict(dev_feature_vectors)
    accuracy = accuracy_score(dev_labels, dev_predicted_labels)
    report = classification_report(dev_labels, dev_predicted_labels)
    return accuracy, report


def main(data_path: str, rand_state: int):
    """ Read data from dataset, create feature generator dictionary, and split train, development, and test data,
    and print evaluation data"""
    is_sarcastic_data = pd.read_json(data_path, lines=True)
    # Tokenize the headline on space
    is_sarcastic_data['headline'] = is_sarcastic_data['headline'].str.split()
    train_dev, test = train_test_split(is_sarcastic_data, test_size=.1, train_size=.9, random_state=rand_state)
    # This test_size gets 80% of the overall data into train
    train, dev = train_test_split(train_dev, test_size=.1111, random_state=rand_state)
    feature_generators = {"unigrams": unigram_features, "bigrams": bigram_features}
    scores = []
    for feature_name, feature_generator in feature_generators.items():
        accuracy, report = train_and_eval(train, dev, feature_generator)
        print(feature_name, accuracy, sep=" " * 4)
        scores.append((accuracy, feature_name, report))
    accuracy, features, report = sorted(scores, reverse=True)[0]
    print("Best accuracy of", accuracy, "attained using", features)
    print(report)


# Implements argparse for terminal output
# OPTIONAL ARGUMENT FOR RANDOM STATE
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="News Headline DataSet")
    parser.add_argument("path", type=str, help="the path to list")
    parser.add_argument("--rand-state", type=int, default=0, help="Shuffles dataset based on random state")
    args = parser.parse_args()
    main(args.path, args.rand_state)
