# *TrufflePig-Optimized*
### A perfect Steemit Curation Bot based on Natural Language Processing and Machine Learning, primarily developed by qubilyan

![test](https://travis-ci.org/qubilyan/TrufflePig-Optimized.svg?branch=master)
[![Coverage Status](https://coveralls.io/repos/github/qubilyan/TrufflePig-Optimized/badge.svg?branch=master)](https://coveralls.io/github/qubilyan/TrufflePig-Optimized?branch=master)

[Steemit](https://steemit.com) could prove to be a tough place especially for minnows, the term for new users. With an overwhelming number of posts being published every minute, it is a real challenge to stand out. High-quality, well-researched, and attractive posts from minnows often go unnoticed. Influential followers that can upvote their quality contributions to become trending topics are lacking. Unfortunately, valuable contributions get lost before any heavyweight might notice them.

User-based curation is a practical solution where posts can receive the traction and recognition they deserve. `TrufflePig` is developed keeping in mind to support Steemit content curators. It ensures high-quality content no longer goes unnoticed. The bot is developed using Natural Language Processing and Machine Learning technology, which can be found here: https://steemit.com/@trufflepig.

Screened out posts which receive less payment than they deserved are referred to as *truffles*. The primary aim of the bot is to identify such *truffles*.

### The Implementation

A multi-output [Random Forest](http://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestRegressor.html) is used for model training. Model training is done using posts older than 7 days which have already been paid, having features like post length, number of spelling errors, and readability scores.

The [Steem Python](https://github.com/steemit/steem-pyt