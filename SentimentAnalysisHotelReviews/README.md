## Sentiment Classification of Hotel Reviews 

### Corpus:
* 400 truthful positive reviews from TripAdvisor 
* 400 deceptive positive reviews from Mechanical Turk 
* 400 truthful negative reviews from Expedia, Hotels.com, Orbitz, Priceline,
  TripAdvisor and Yelp 
* 400 deceptive negative reviews from Mechanical Turk 

More info in README file under data folders. 

### Task: 
The task is to implement perceptron algorithmn with python to classify hotel reviews as truthful or deceptive, positive or negative. 
Another approach is Naive Bayes, see more details in `NaiveBayerApproach` folder

### File structure: 
* perceplearn.py -- learn features from the data
* averagemodel.txt / vanillamodel.txt -- output of perceplearn.py, weights of each word
* percepclassify.py -- classifier for final predictions 
* percepoutput.txt -- output of percepclassify.py, prediction for test data
* op_spam_test_data -- test data 
* op_spam_training_data -- training data 

### Results: 

<p>
  <img src="https://github.com/wenhuanghuang/NLP_Projects/blob/main/SentimentAnalysisHotelReviews/Results.png" width="300">
</p>

This is an assignment from USC CSCI544 Natural Language Processing 
