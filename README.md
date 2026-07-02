# SMS Spam Classifier

For this task I built a simple text classifier that predicts whether an SMS message is spam or not. Here's a rundown of what I did and why.

## Dataset

I used the SMS Spam Collection Dataset from Kaggle. It has around 5,500 real SMS messages, each labeled as either spam or ham (a normal message). I picked this one mainly because it's real text rather than something generated, it's a good size to work with without taking forever to train, and it's a pretty standard dataset for this kind of task, so I could sanity-check my results against how others typically approach it.

## Cleaning the data

The CSV came with a few extra empty columns I didn't need, so I dropped those and kept just the message and its label, renaming them to something clearer. After that I checked for missing values and duplicate rows and removed them. I also lowercased everything and stripped out numbers and punctuation, since I wanted the model to treat something like "Free" and "free!!!" as the same word rather than two different ones.

## Splitting the data

I split the dataset into 80% for training and 20% for testing. One thing worth mentioning: the dataset is pretty imbalanced, about 87% of the messages are ham and only 13% are spam. So I used a stratified split to make sure that ratio stayed consistent in both the training and test sets, rather than risking one set ending up with barely any spam examples.

## Turning text into numbers

Models can't work with raw text, so I used TF-IDF to convert the messages into numeric features. This basically scores words based on how important they are to a specific message relative to the rest of the dataset, so common words end up mattering less automatically. I made sure to fit the TF-IDF vectorizer only on the training data and then apply it separately to the test data, so the model wasn't getting any indirect exposure to the test set before evaluation.

## Model

I went with Multinomial Naive Bayes. It's a simple, fast algorithm that tends to work well as a baseline for text classification, so it felt like a reasonable choice here rather than jumping straight to something more complex.

## Results

| Metric      | Ham   | Spam  |
|-------------|-------|-------|
| Precision   | 0.94  | 1.00  |
| Recall      | 1.00  | 0.59  |
| F1-score    | 0.97  | 0.74  |

Overall accuracy was 94.8%.

## What I noticed in the results

The model was pretty cautious. It never once mislabeled a normal message as spam, its precision on spam was a perfect 1.00. But its recall on spam was only 0.59, so it missed 54 out of 131 actual spam messages in the test set, letting them through as ham instead.

This is honestly a good example of why accuracy alone doesn't tell the full story. Because the dataset leans heavily toward ham, a model could still post a high accuracy score just by favoring "ham" as a guess more often. Looking at precision and recall separately for each class gave a much better sense of what the model was actually getting right and wrong.

## A couple of notes

The CSV needed to be loaded with latin-1 encoding, it threw a UnicodeDecodeError with the default UTF-8, most likely because of some special characters in a few of the messages. I also kept preprocessing fairly minimal, just lowercasing and stripping punctuation and numbers, since the dataset was already reasonably clean and TF-IDF handles down-weighting common, low-value words on its own.
