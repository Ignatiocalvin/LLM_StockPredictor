# LLM_StockPredictor
## Overview
The impact of Natural Language Processing (NLP) algorithms in predicting
stock market prices, especially price shocks.

Commodity price shocks are times when the prices for commodities have
drastically increased or decreased over a short time. Typically, the stock market
and economic performance are aligned. Thus, when the stock market is performing well, it is usually a function of a growing economy.

Stock market declines have the potential to diminish wealth across personal
and retirement investment portfolios. Consequently, individuals witnessing a depreciation in their portfolio value are inclined to curtail their expenditures. 

With this project, we aim to develop a model that analyzes news headlines
and predicts stock market crashes based on text data. Having such a model may
help us anticipate the stock market movement to better manage our wealth and
prepare for adverse economic events. We chose to work with NLP algorithms and
fine-tune existing pre-trained LLMs, as classical autoregressive models show
poor predictive capacity during price shocks. Using textual data as input such
as news headlines may help the model adjust its predictions to keep up with
drastically changing trends. Aside from that, we used also analyzed the effects of sentiment analysis and moving averages as features used for prediction. The models are evaluated based on the root mean squared error (RMSE) of the predicted stock prices. 

## Data

The data is obtained from the Kaggle dataset [Daily News for Stock Market Prediction](https://www.kaggle.com/aaron7sun/stocknews). The dataset contains historical news headlines from Reddit WorldNews Channel and Dow Jones Industrial Average (DJIA) stock prices. The news headlines are from 2008 to 2016, and the stock prices are from 2008 to 2016. Combined_News_DJIA.csv contains the top 25 news headlines and the corresponding stock prices for each day. upload_DJIA_table.csv contains the stock prices for each day. 

The project is divided into the following sections:

## Notebooks

### 1. FinGPT Sentiment Retrieval
Filename: `retrieve_sentiment.ipynb`
Description: This notebook demonstrates the use of a LLM, specifically FinGPT, to retrieve and analyze sentiments of financial news articles using the LangChain framework. We asked FinGPT to generate a sentiment score for each news headline in the dataset. The sentiment score ranges from -10 to 10, where -1 indicates a negative sentiment, 0 indicates a neutral sentiment, and 1 indicates a positive sentiment.

### 2. Univariate Model
Filename: `univariate.ipynb`
Description: This notebook focuses on using a LSTM neural network to predict stock prices based on a univariate time series.

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

### 7. Predicting Stock Prices with FastText Embeddings
Filename: `lookback_fasttext.ipynb`
Description: This notebook predicts stock prices using an LSTM neural network and financial news headlines which uses FastText embeddings. We use a look back window of 50 days of news headlines and stock prices to predict the next day's stock price in the training data and a 30 day window in the testing data. The FastText embeddings are used as an alternative to BERT embeddings.

### 8. Predicting Stock Prices with XGBoost
Filename: `xgboost.ipynb`
Description: This notebook predicts stock prices using an XGBoost model. The model uses the stock prices and news headlines as features to predict the stock prices. The model is trained on the training data and evaluated on the testing data.





## Results
For our models, we used the RMSE as the evaluation metric. The RMSE is a measure of the differences between the predicted values and the actual values. It gives us an idea of how well the model is performing in terms of predicting the stock prices.

The table below shows the RMSE values for the training and testing data for each of the models.

<center>

| Notebooks 	        | Train RMSE 	        | Test RMSE 	        |
|:---:	                |:---:	                |:---:	                |
| univariate 	        | 31.743497936455558 	| 164.36006533558697 	|
| xgboost 	            | 0.9986960112179987 	| 749.0485733114635 	|
| lookback_xgboost 	    | 3.5651031645488147 	| 200.72583615414294 	|
| bert_MA 	            | 10.83293233874024 	| 816.5899384610334 	|
| news_rnn            	| 622.6200037110209  	| 741.4623210266183 	|
| lookback_news_rnn 	| 15.457514085649745 	| 309.5794180019174 	|
| sentiment_rnn 	    | 27.86876269801595 	| 1143.883111213618 	|
| lookback_fasttext 	| 13.962031174946512 	| 169.93569939132175 	|

</center>   

From the table, we can see that the `lookback_xgboost` model has the lowest RMSE value for the testing data, indicating that it is the best performing model among the four. The `bert_MA` model has the lowest RMSE value for the training data, but it has the highest RMSE value for the testing data, indicating that it may be overfitting the training data.

Having news ruin the data dependency, ultimately they act as noise for the data. The `univariate` model, which only uses the stock prices, performs the best among the models. This suggests that the stock prices themselves contain enough information to predict future stock prices, and the addition of news headlines does not significantly improve the model's performance. 


| Notebooks 	        | Loss Graph 	                        | Test Predictions	                |
|:---:	                |:---:	                                |:---:	                            |
| univariate 	        | ![alt text](images/image-2.png) 	    | ![alt text](images/image-6.png) 	|
| xgboost 	            | 0.9986960112179987 	                | ![alt text](images/image-11.png)  |
| lookback_xgboost 	    | 3.5651031645488147 	                | ![alt text](images/image-10.png)  |
| bert_MA 	            | ![alt text](images/image.png) 	    | ![alt text](images/image-5.png) 	|
| news_rnn            	| ![alt text](images/image-3.png)       | ![alt text](images/image-8.png) 	|
| lookback_news_rnn 	| ![alt text](images/image-1.png) 	    | ![alt text](images/image-7.png) 	|
| sentiment_rnn 	    | ![alt text](images/image-4.png)       | ![alt text](images/image-9.png) 	|
| lookback_fasttext 	| ![alt text](images/image-12.png) 	    | ![alt text](images/image-13.png) 	|

