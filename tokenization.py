import re
from string import punctuation


def word_tokenize(sentence: str) -> list:
    """
    tokenize and keep ponctuation marks, but not whistespaces

    Args:
        sentence (str): sentence to tokenize

    Returns:
        list: tokenized sentence, with ponctuation
    """
    tokenized_sent = []

    for word in re.split(r"([^\w\-·\.\/\\])", sentence):
        if word.endswith(tuple(punctuation)):
            tokenized_sent.append(word[:-1])
            tokenized_sent.append(word[-1])
        else:
            tokenized_sent.append(word)

    return [word for word in tokenized_sent if word.strip()]


def sent_tokenize(text: str) -> list:
    return re.split(r"(?<=\.|\?|\!)\s", text)