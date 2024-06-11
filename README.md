# LLM_StockPredictor
## Overview

## Notebooks
### 1. Univariate Model
Filename: `univariate.ipynb`
Description: This notebook focuses on using a Long Short-Term Memory (LSTM) neural network to predict stock prices based on a univariate time series.

### 2. FinGPT Sentiment Retrieval
Filename: `retrieve_sentiment.ipynb`
Description: This notebook demonstrates the use of a Large Language Model (LLM), specifically FinGPT, to retrieve and analyze sentiments of financial news articles using the LangChain framework. We asked FinGPT to generate a sentiment score for each news headline in the dataset. The sentiment score ranges from -10 to 10, where -1 indicates a negative sentiment, 0 indicates a neutral sentiment, and 1 indicates a positive sentiment.

### 3. Predicting Stock Prices with News Headlines without Looking Back at the Data
Filename: `news_rnn.ipynb`
Description: This notebook predicts stock prices using an LSTM neural network and financial news headlines which uses BERT embeddings. This uses the current news headlines and does not account for looking back at the previous stock prices and news headlines.

### 4. Predicting Stock Prices with News Headlines with Look Back
Filename: `lookback_news_rnn.ipynb`
Description: This notebook predicts stock prices using an LSTM neural network and financial news headlines which uses BERT embeddings. This uses a look back window of 50 days of news headlines and stock prices to predict the next day's stock price in the training data and a 30 day window in the testing data. The reason for a longer window is they provide more data points to learn from, allowing the model to capture long-term trends and dependencies.

Stock market dynamics can change over time, so a shorter test window ensures the model is evaluated on its ability to adapt to the most recent market conditions.

### 5. Predicting Stock Prices with News Headlines with Look Back and Sentiment Analysis
Filename: `sentiment_rnn.ipynb`
Description: Here, aside from using the look back window of 50 days of news headlines embeddings and stock prices to predict the next day's stock price, we also include sentiment analysis of the news headlines using FinGPT. The sentiment analysis gives a score from the range -10 to 10, where -10 is the most negative sentiment and 10 is the most positive sentiment. This score is then used as an additional feature in the model. 

### 6. Predicting Stock Prices with News Headlines with Look Back and Moving Average
Filename: `bert_MA.ipynb`
Description: In this notebook, we predict stock prices using an LSTM neural network and financial news headlines which uses BERT embeddings. We use a look back window of 50 days of news headlines and stock prices to predict the next day's stock price in the training data and a 30 day window in the testing data. We also include a moving average of the stock prices of 10 days as an additional feature in the model.

## Results

| Notebooks          	| Train RMSE         	| Test RMSE          	|
|--------------------	|--------------------	|--------------------	|
| bert_MA            	| 10.83293233874024  	| 816.5899384610334  	|
| lookback_news_rnn: 	| 15.457514085649745 	| 309.5794180019174  	|
| univariate         	| 31.743497936455558 	| 164.36006533558697 	|