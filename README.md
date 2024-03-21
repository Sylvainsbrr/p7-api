Sentiment Analysis for Social Media - Air Paradis Project
Overview
This project was initiated in response to Air Paradis' request for a prototype AI product capable of predicting the sentiment associated with a tweet. Recognizing the challenge Air Paradis faces with maintaining a positive public perception on social media, we aimed to develop an advanced, custom AI model that can help anticipate potential PR crises by analyzing social media sentiments.

Objectives
Develop a prototype AI model that predicts sentiment from tweets.
Implement a methodology presentation for a non-technical audience, highlighting the custom advanced model approach.
Explore multiple modeling approaches, including simple custom models, advanced custom models, and a BERT-based model.
Demonstrate the application of MLOps principles throughout the model development process.
Project Structure
src/: Contains all the source code for the project, including data preprocessing, model training scripts, and utility functions.
models/: Stores the trained model files for easy access and versioning.
notebooks/: Jupyter notebooks for exploratory data analysis, model development, and results visualization.
docs/: Documentation and presentation materials explaining the project methodology, results, and MLOps approach.
tests/: Automated tests ensuring model reliability and API stability.
requirements.txt: Lists all the dependencies required to run the project.
Approach
Model Development
Simple Custom Model: A logistic regression model to serve as a baseline for sentiment prediction.
Advanced Custom Model: A deep neural network model incorporating different word embeddings (Word2Vec, GloVe) to predict tweet sentiment.
BERT Model: Utilizing a pre-trained BERT model to assess its performance and suitability for sentiment analysis in our context.
MLOps Implementation
Utilization of MLFlow for experiment tracking, model versioning, and deployment.
Continuous deployment pipeline setup for model serving via an API, integrating automated unit tests and cloud deployment (considering free-tier limits).
