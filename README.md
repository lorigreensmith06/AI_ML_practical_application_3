# Practical Application 3 
### Lori Smth
#### Link to file : https://github.com/lorigreensmith06/AI_ML_practical_application_3/blob/main/prompt_III.ipynb

![image info](images/bank_prediction_results.png)


In this practical application 4 models were compared: 
* K Nearest Neighbors
* Linear Regression
* SVC
* DecisionTrees

## The Business Problem


The data was collected from Portugese marketing campaigns that recorded information about customers and whether they agreed to invest in a a long-term loan that was being sold to them.
The data in this dataset was very imbalanced which was challenging.  The data shows that 88% of the customers did not agree to put their money in a long-term loan.  That can be expected in sales data and is not unusual.

## Evaluating the data

**Baseline accuracy on the test set is 88%.**

The data is highly imbalanced, with 88% of customers not agreeing to the loan and only 12% agreeing. This imbalance posed challenges for some models, particularly in detecting the minority class.

JamesSteinEncoder was used to encode the categorical features. While the dataset did not contain any missing values, some fields included "unknown" as a category. I experimented with replacing these "unknown" values using the mode of each feature, but this approach often led to lower accuracy. As a result, I chose to allow JamesSteinEncoder to handle the some "unknown" values directly, which resulted in better performance.


## The Base Models

The base models had about 90% accuracy on the test data and much lower recall.  

![image info](images/model_comparison_results.png)

Also looking at the coefficients of logistic regression, we can find some useful information. Based on the results, March appears to be the best time to run a marketing campaign, while May, November, and June may be the least effective months for outreach.




![image info](images/coefficients_results.png)

## Improving the Models

The first thing I did was to go back to the data and to try to improve the model using the data. There was no missing data, but there were a lot of unknown fields.  Because I was using the JamesSteinEncoder, the efforts to clean the data didn't seem to make much difference because I suspect that the JamesSteinEncoder was creating a separate category for the "unknown" data.  I did replace the data in columns, "job", "marital", and "education", but left the "default" data to be handled by the encoder.  I did not see an improvement in the models with those changes.

The effort that had the most effect was tuning the hyper-parameters in grid search.  
I added quite a few hyper-parameters to improve performance.  
The decision tree had an improvement with some of the changes. 
They didn't have a huge effect, but I was able to improve the model sometimes by a percentage point so it did make a difference.



### K-Nearest Neighbors (KNN)
How it works: Looks at the "k" closest data points and predicts the majority vote.

**Best parameters:** 
**n_neighbors = 7**
This means the model checks the 7 closest customers to decide if a new one is likely to say "yes."

**Accuracy: ~0.905**
It performs well but is sensitive to the number of neighbors and can struggle with large datasets.

#### Logistic Regression
How it works: Estimates the probability of a customer saying "yes" or "no" based on a linear combination of input features.

**Best parameters:**

**C = 0.1:** Smaller values mean more regularization, helping avoid overfitting.

**penalty = 'l2':** This regularization discourages large coefficients, which stabilizes the model.

**Accuracy: ~0.910**
A solid, interpretable model that provides clear coefficients to understand feature impact (e.g., March and prior campaign success).

#### Decision Tree


**Best parameters:**

**criterion = 'entropy':** Splits the data based on information gain (how much uncertainty is reduced).

**max_depth = 5**: Limits the tree to 5 layers to prevent overfitting.


**Accuracy: ~0.914**
The best performing model on accuracy. It's also intuitive to understand when visualized, but may generalize poorly if too deep.


* from imblearn.over_sampling import SMOTE
* from imblearn.pipeline import Pipeline as ImbPipeline
* *from imblearn.over_sampling import RandomOverSampler
* *from imblearn.under_sampling import RandomUnderSampler

The final thing that I tried was using some sampling tools to resample the data from the imblearn libraries. This was useful in improving the recall.
That recall value is very important in this instance because the problem has more to lose if the bank loses a sale than spending some resources to have a sales person call the customer.
If the goal was to find the highest recall this was actually very successful.  I was able to find a model with 95% recall.

The accuracy took a hit and went down the 83%, but the RandomUnderSampler produced excellent results for recall.  This means that there are a lot of false positives, but in this case it would just mean that a sales person would get more rejections but would also get more sales. 

![image info](images/sampling_results.png)

## Conclusions
In all cases, using sampling methods reduced the overall accuracy, but significantly increased the recall score.

While accuracy decreased, that isn't necessarily a concern in this business case. A false positive means reaching out to someone who may not convert — which is acceptable if it means we catch more of the people who would convert. What’s more important is minimizing false negatives, since missing potential customers who are likely to say "yes" is a lost opportunity.

So even though the sampling models performed below the baseline in terms of accuracy, they provided valuable insights that can help guide the sales team more effectively.

SMOTE achieved a 90% recall

RandomOverSampler reached 92% recall

RandomUnderSampler had the highest recall at 93%

Because we have a large majority class, Random Under Sampling is feasible — we can afford to drop some of the majority class without losing critical information. In this case, a simpler, underfit model can actually work well by improving our ability to detect potential converters.




## Recommendations

My recommendation is to use the model that resulted in the highest recall. I would also recommend running the campaign in March, which showed a strong correlation with positive responses. Additionally, I suggest focusing on customers who responded positively to previous campaigns, as the coefficients indicated a high correlation with success for those individuals.