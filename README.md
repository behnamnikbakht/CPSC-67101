# CPSC-67101

---
This is the project repository for the course Design and Implementation of Database Systems (CPSC-671) at university of Calgary

---
Author: Behnam Nikbakhtbideh

Date: November 2021

---

## Introduction
Emotion Recognition is a kind of classification problem to detect what emotion the human actor has in media or text.
According to [1], these emotions are limited to six basic categories: <b>happiness, surprise, sadness, anger, fear, and disgust</b>.
In fact, other complex emotional states can be inferred via these universal categories with respect to situations like culture and sex.

Emotion Recognition in Tweets is a kind of emotion recognition that focuses on tweets as input data.
Tweets have some characteristics that make them distinctive from other resources.
One feature is the multimedia nature of tweets that might contain image, audio and video.
But the majority of works in this domain concentrate on text.
The other feature related to social media is that when it comes to analyse data, some special inputs like emoji should be considered in NLP.
Also gaining benefit from the nature of social networks is considered in some of the research.

## Related Work
In terms of input type, techniques in this area could be categorized in three main sections:
### Multi-modal
Most of the work in emotion recognition focus on text because of the fact that tweets are mostly in text.
But in some papers like [4], it tries to fuse both visual and textual models to get a more comprehensive result.
### Based on social networking characteristics
Some papers try to utilize information that can be gained by social relationships and behaviour.
For example in [5], the focus is on social media features like user’s opinions, amount of user activities (number of tweets for example), and user’s behaviour (for example, periods of being active or inactive). 
### Text-based
Works in this domain concentrate on NLP models to analyze tweets as textual data.
In [2] it uses a non-supervised learning model that requires no manual annotation.
This technique starts with a couple of hashtags like #angry & #happy, or emoticons and assign a label to each tweet, then extends its training set automatically.
The limitation of this technique is that such initial data is not present in all kinds of tweets and it will cause a bias in the training model.

Also in [3], it uses another kind of non-supervised learning model that is based on linguistic and acoustic features of language to automatically assign labels to each tweet.
It seems that this technique doesn’t remove the need for learning, and only transfers it to another level which might decrease accuracy.

On the other hand, supervised models try to construct a model based on training data.
Multiple techniques like SVM, decision-trees and Bayesian models, and also deep-learning models for text classification are provided in this section.

Supervised solutions could be devided in two general parts: <b>statistical models</b> and <b>NLP models</b>.

Statistical models concentrate on entropy features of the language and mainly use solutions that are based on the N-Gram models.

On the other hand, in NLP solutions, the very basic infrastructure of work is individual words.
These techniques are using analysis based on tf-idf and text similarity (like Levenshtein distance).
Also an important part of these kind of solutions is pre-processing in terms in Stemming or Lemmatization or so.

## Model
To build a model, I first selected an open-source library [6] and analyzed its accuracy against a dataset [7] with 16000 classified text entries.
Due to some differences between these two models, 14696 entries are analyzed and among them, only 5425 items are classified correctly which makes about 37%, and the overall time was about 16min.
[The source code is available here](comparison/cmp_text2emotion.py)

Because of the fact that tweets are different from ordinary text, I run [this](comparison/cmp2_text2emotion.py) comparison against another dataset [8] specific to tweets with 40,000 entries.
Also despite the previous examination, I ignored direct translation of classes.
For example, one category like “worry” in the dataset might be related to multiple categories like “fear, sadness, anger” in the model.
By the way, here is a part of the result of this comparison:
```json
{
    "Sad": {
      "empty": 168,
      "sadness": 1505,
      "worry": 1998,
      "hate": 356,
      "neutral": 1267,
      "love": 317,
      "surprise": 294,
      "fun": 207,
      "relief": 229,
      "happiness": 518,
      "enthusiasm": 120,
      "boredom": 45,
      "anger": 18
  }
}
```
Meaning that if we consider {sadness, worry, hate} as one group, it will result in about 55% of accuracy.
For happiness, if we consider {enthusiasm, love, happiness, fun, relief} in one class, it will result in 45% of accuracy.

To know what should be done, first we should find how this library (and most of the similar libraries) work.
The general flow is as follows:
### Preparation
Removing stop words and redundant characters except emoji characters.
#### Stemming
This is to reduce words to their structural roots.
For example, both two words {book, books} will turn into “book”.
Most of the well-known facilities in this part are based on statistical analysis without considering the meaning of words.
For example, considering the PorterStemmer in nltk library, following transforms are applied:
```
Homes → home
Winning → win
Alone → alon
```
So the result might be semantically problematic.
#### Lemmatization
On the other hand, lemmatization considers the meaning of the word, especially when it uses wordnet network.
#### Tokenization
It is to split text in a set of words.
The simplest way is to split by space characters.
But it could get more strength by wordnets as well.
For example, “european union” might be considered as two or one tokens.
#### Additional processing
like considering “not” as the opposite semantic value.
### Model (train or predict/evaluate)
Multiple solutions using TF-IDF, Bayesian, and SVM could be used to make a relationship between text, words (tokens), and classes.
In the following, we construct a model that uses Bayesian model for train/test.


## References
[1] Du, S., Tao, Y., &#38; Martinez, A. M. (2014). Compound facial expressions of emotion. Proceedings of the National Academy of Sciences, 111(15), E1454–E1462. https://doi.org/10.1073/PNAS.1322355111
[2] SintsovaValentina, &#38; PuPearl. (2016). Dystemo. ACM Transactions on Intelligent Systems and Technology (TIST), 8(1). https://doi.org/10.1145/2912147
[3] Hines, C., Sethu, V., &#38; Epps, J. (2015). Twitter: A new online source of automatically tagged data for conversational speech emotion recognition. ASM 2015 - Proceedings of the 1st International Workshop on Affect and Sentiment in Multimedia, Co-Located with ACM MM 2015, 9–14. https://doi.org/10.1145/2813524.2813529
[4] Lin, C., Hu, P., Su, H., Li, S., Mei, J., Zhou, J., &#38; Leung, H. (2020). SenseMood: Depression detection on social media. ICMR 2020 - Proceedings of the 2020 International Conference on Multimedia Retrieval, 407–411. https://doi.org/10.1145/3372278.3391932
[5] Shen, G., Jia, J., Nie, L., Feng, F., Zhang, C., Hu, T., Chua, T. S., &#38; Zhu, W. (2017). Depression detection via harvesting social media: A multimodal dictionary learning solution. IJCAI International Joint Conference on Artificial Intelligence, 0, 3838–3844. https://doi.org/10.24963/IJCAI.2017/536
[6] aman2656/text2emotion-library. (n.d.). Retrieved November 26, 2021, from https://github.com/aman2656/text2emotion-library
[7] Emotions dataset for NLP | Kaggle. (n.d.). Retrieved November 26, 2021, from https://www.kaggle.com/praveengovi/emotions-dataset-for-nlp?select=train.txt
[8] text_emotion | Kaggle. (n.d.). Retrieved November 26, 2021, from https://www.kaggle.com/maysaasalama/text-emotion/version/1
