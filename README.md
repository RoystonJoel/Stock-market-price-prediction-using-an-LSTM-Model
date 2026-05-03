
---

## 📂 File Summaries

### 1. `1_sentiment_Analysis.ipynb`
#### Financial News Sentiment Analysis

This notebook performs sentiment analysis on financial news articles related to a specific company within a given date range. It leverages the Hugging Face `datasets` library for data loading and processing, and the `ProsusAI/finbert` model for sentiment prediction.

##### Workflow:

1.  **Configuration**: Define the `start_date_str`, `end_date_str`, and `company` for filtering news articles.
2.  **Library Imports**: Import necessary libraries such as `os`, `matplotlib.pyplot`, `pandas`, `datetime`, and `datasets`.
3.  **Data Loading**: Load the `sabareesh88/FNSPID_nasdaq` dataset in streaming mode.
4.  **Data Preprocessing**: Drop irrelevant columns like `Luhn_summary`, `Textrank_summary`, `Lexrank_summary`, and `Article`.
5.  **Sentiment Model Loading**: Load the `ProsusAI/finbert` pre-trained model and its tokenizer for sentiment analysis.
6.  **News Article Filtering**: Filter the dataset to include only articles within the specified date range and related to the chosen company.
7.  **Sentiment Scoring**: Apply the FinBERT model to calculate positive, negative, and neutral sentiment scores for the LSA summaries of the filtered articles.
8.  **Data Integration**: Add the calculated sentiment scores back to the filtered dataset.
9.  **Daily Aggregation**: Convert the filtered dataset to a Pandas DataFrame, parse dates, and aggregate sentiment scores by day to get `avg_positive_score`, `avg_negative_score`, and `avg_neutral_score`.
10. **Validation and Saving**: Perform data validation to ensure all dates are accounted for in the aggregation. If validation passes, save the daily aggregated sentiment scores to a CSV file (`daily_sentiment.csv`).

##### Resources

*   **Dataset**: [FNSPID_nasdaq](https://huggingface.co/datasets/sabareesh88/FNSPID_nasdaq) on Hugging Face
*   **Sentiment Model**: [ProsusAI/finbert](https://huggingface.co/ProsusAI/finbert) on Hugging Face

### 2. `2_time_series_analysis.ipynb`
#### Time Series Analysis of Sentiment and Stock Price Data

This notebook performs a time series analysis on a dataset containing sentiment scores and stock price information. The goal is to explore trends, correlations, and time series components to understand the relationship between sentiment and stock price movements.

##### Dataset

The analysis uses the `sentiment_and_price_data.csv` file, which includes:
-   `day`: Date of the data point.
-   `avg_positive_score`, `avg_negative_score`, `avg_neutral_score`: Average sentiment scores for the day.
-   `open`, `high`, `low`, `close`, `adjclose`: Stock price metrics.
-   `volume`: Trading volume.

##### Tasks Performed

1.  **Data Loading and Preprocessing**: Loads the data and converts the 'day' column to datetime objects, setting it as the DataFrame index.
2.  **Visualization of Trends**: Plots sentiment scores and the 'close' price over time to observe their individual trends and how they move in relation to each other.
3.  **Correlation Analysis**: Calculates and visualizes the correlation matrix between sentiment scores and the 'close' price using a heatmap to identify relationships.
4.  **Time Series Decomposition**: Decomposes the 'close' price into trend, seasonal, and residual components to understand underlying patterns and seasonality.


### 3. `3_AAPL_Price_data.ipynb`
#### Notebook: Apple Stock Price Analysis with Sentiment Data

This notebook performs an analysis of Apple (AAPL) stock historical prices by integrating it with daily sentiment scores. The goal is to prepare a consolidated dataset that can be used for further analysis, such as predicting stock movements based on sentiment.

##### Table of Contents

1.  **Data Loading**: Loading Apple stock price data from KaggleHub and daily sentiment data from a CSV file.
2.  **Data Filtering**: Filtering the stock price data to a specific date range.
3.  **Data Merging**: Combining the filtered stock price data with the daily sentiment data.
4.  **Data Cleaning**: Handling missing values, specifically forward-filling price data for non-trading days (weekends).
5.  **Data Validation**: Performing checks to ensure data integrity after merging and cleaning.
6.  **Output**: Saving the cleaned and merged dataset to a CSV file.

##### Data Sources

*   **Apple Stock Historical Price**: Sourced from KaggleHub via `caesarmario/apple-stock-historical-price`.
*   **Daily Sentiment**: Loaded from a local `daily_sentiment.csv` file.

##### Key Steps

*   **`kagglehub`**: Used to access and load the Apple stock price dataset.
*   **`pandas`**: Extensively used for data manipulation, cleaning, and merging.
*   **Date Filtering**: Stock data is filtered to the period `2022-03-20` to `2023-03-19`.
*   **`ffill()`**: Forward-fill method applied to stock price columns to impute missing values for non-trading days.
*   **Validation Checks**: Ensures no business days are missing and that all dates in the merged DataFrame are consistent with the sentiment data.

This notebook provides a foundational dataset for time-series analysis and machine learning tasks involving stock prices and sentiment.


### 4. `4_LTSM.ipynb`
#### LSTM Stock Price Prediction with Sentiment Analysis

This notebook demonstrates how to build and evaluate Long Short-Term Memory (LSTM) neural networks for stock price prediction, incorporating sentiment analysis as an additional predictive feature. The project progresses through several stages, from data loading and preprocessing to model training, evaluation, and feature importance analysis.

##### Notebook Overview:

##### 1. Data Loading and Initial Preprocessing
- Loads `sentiment_and_price_data.csv` into a pandas DataFrame.
- Drops irrelevant columns (`ingested_at_utc`).
- Converts the `day` column to datetime objects and sets it as the DataFrame index.

##### 2. Feature Engineering and Preprocessing (7-day Sentiment)
- Calculates 7-day rolling averages for `avg_positive_score`, `avg_negative_score`, and `avg_neutral_score`.
- Creates a target variable (`target_close`) by shifting the `close` price by -1 (for next-day prediction).
- Handles missing values introduced by rolling windows and shifting.
- Scales all features and the target variable using `MinMaxScaler`.
- Reshapes the data into a 3D format (samples, time steps, features) suitable for LSTM.
- Splits the data into training and testing sets while preserving temporal order.

##### 3. LSTM Model Training (7-day Sentiment)
- Builds and compiles an LSTM model using Keras with a `Dense` output layer.
- Trains the model on the prepared training data.

##### 4. Evaluation and Visualization (7-day Sentiment)
- Generates predictions on the test set.
- Inverse transforms predictions and actual values back to their original scale.
- Calculates the Root Mean Squared Error (RMSE) to quantify model accuracy.
- Visualizes actual vs. predicted stock prices using a line plot.

##### 5. Feature Importance Analysis (7-day Sentiment)
- Implements a permutation importance method to assess the impact of each feature (including 7-day rolling sentiment) on the model's performance.
- Visualizes feature importance using a horizontal bar chart.

##### 6. Multi-Window Sentiment Feature Engineering
- Calculates 3-day and 14-day rolling averages for sentiment scores to explore different sentiment horizons.
- Cleans missing values after creating new rolling features.

##### 7. Model Retraining and Comparison (3-day and 14-day Sentiment)
- Trains two new LSTM models, one for 3-day sentiment features and another for 14-day sentiment features.
- Ensures consistent data preparation (scaling, reshaping, splitting) for both models.

##### 8. Performance Evaluation (All Rolling Sentiment Models)
- Calculates RMSE for the 3-day, 7-day, and 14-day sentiment window models.
- Generates a comparative plot showing actual vs. predicted prices for all three models to identify the most effective sentiment window.

##### 9. No-Window Model Training (Raw Sentiment)
- Prepares a feature set using raw daily sentiment scores (`avg_positive_score`, `avg_negative_score`, `avg_neutral_score`) alongside base price features.
- Trains a baseline LSTM model with these raw sentiment features.

##### 10. Performance Comparison (All Models)
- Calculates RMSE for the No-Window model.
- Generates a comprehensive visualization comparing actual prices against predictions from the No-Window, 3-Day, 7-Day, and 14-Day sentiment models.

##### 11. Feature Importance Analysis (No-Window Model)
- Conducts permutation importance for the no-window model to understand the impact of raw daily sentiment signals.
- Visualizes the feature importance for the no-window model.

##### Conclusion:
This notebook provides a detailed comparative study of how different approaches to incorporating sentiment data (raw daily vs. various rolling averages) can influence the predictive accuracy of LSTM models for stock price forecasting. The analysis helps in understanding the optimal sentiment window size and the relative importance of sentiment features compared to traditional price indicators.

---

*This workflow is part of research conducted at Auckland University of Technology (AUT) on integrating sentiment analysis with deep learning for financial forecasting.*

---