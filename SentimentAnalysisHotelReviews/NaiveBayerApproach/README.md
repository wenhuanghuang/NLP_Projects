## Sentiment Classification of Hotel Reviews 

### Corpus:
* 400 truthful positive reviews from TripAdvisor 
* 400 deceptive positive reviews from Mechanical Turk 
* 400 truthful negative reviews from Expedia, Hotels.com, Orbitz, Priceline,
  TripAdvisor and Yelp 
* 400 deceptive negative reviews from Mechanical Turk 
More info in README file under data folders. 

### Task:
The task is to implement naive bayes with python to classify hotel reviews as truthful or deceptive, positive or negative. 

### File Structure:
* nblearn.py -- learn features from data
* nbmodel.txt -- output of nblearn.py, weights for each word 
* nbclassify.py -- classifier
* nboutput.txt -- output of nbclassify.py, prediction for test data 

### Results: 

<p>
  <img src="https://github.com/wenhuanghuang/NLP_Projects/blob/main/SentimentAnalysisHotelReviews/NaiveBayerApproach/Results.png" width="300">
</p>

This is an assignment from USC CSCI544 Natural Language Processing 
