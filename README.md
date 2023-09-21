# *TrufflePig-Optimized*
### A perfect Steemit Curation Bot based on Natural Language Processing and Machine Learning, primarily developed by qubilyan

![test](https://travis-ci.org/qubilyan/TrufflePig-Optimized.svg?branch=master)
[![Coverage Status](https://coveralls.io/repos/github/qubilyan/TrufflePig-Optimized/badge.svg?branch=master)](https://coveralls.io/github/qubilyan/TrufflePig-Optimized?branch=master)

[Steemit](https://steemit.com) could prove to be a tough place especially for minnows, the term for new users. With an overwhelming number of posts being published every minute, it is a real challenge to stand out. High-quality, well-researched, and attractive posts from minnows often go unnoticed. Influential followers that can upvote their quality contributions to become trending topics are lacking. Unfortunately, valuable contributions get lost before any heavyweight might notice them.

User-based curation is a practical solution where posts can receive the traction and recognition they deserve. `TrufflePig` is developed keeping in mind to support Steemit content curators. It ensures high-quality content no longer goes unnoticed. The bot is developed using Natural Language Processing and Machine Learning technology, which can be found here: https://steemit.com/@trufflepig.

Screened out posts which receive less payment than they deserved are referred to as *truffles*. The primary aim of the bot is to identify such *truffles*.

### The Implementation

A multi-output [Random Forest](http://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestRegressor.html) is used for model training. Model training is done using posts older than 7 days which have already been paid, having features like post length, number of spelling errors, and readability scores.

The [Steem Python](https://github.com/steemit/steem-python) official library is utilized by the bot to scrape data from the Steemit blockchain and post a toplist of the daily found truffle posts using its trained model.

The execution route follows as:

1. The data is scrapped from the blockchain if available or loaded from disk if possible.

2. The scrapped posts are processed and filtered for machine learning model training.

3. The model is then trained using the processed data if a trained model does not exist already.

4. More recent data is then scrapped and checked for truffle posts using the trained model.

5. The bot publishes a toplist of truffle posts on which it both upvotes and comments.

Clone the project directory:
> `$ git clone https://github.com/qubilyan/TrufflePig-Optimized.git`

Add the project directory to your `PYTHONPATH`:
> `$ export PYTHONPATH=$PYTHONPATH:<path_to_project>`

### Open Source Usage

This project is open source and is free for **non-commercial** usage. Please refer to the LICENSE for more details.