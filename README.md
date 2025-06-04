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
The data was collected from Portugese marketing campaigns that recorded information about customers and wheter they agreed to invest in a a long-term loan that was being sold to them.
The data in this dataset was very imbalanced which was challenging.  The data shows that 88% of the customers did not agree to put their money in a long-term loan.  That can be expected in sales data and is not unusual.

## Evaluating the data

##  The Dummy Data

## The Base Models

![image info](images/model_comparison_results.png)

## Improving the Models

The first thing I did was to go back to the data and to try to improve the model using the data.  I did not make a lot of improvement using the methods that I tried. There was no missing data, but there were a lot of unkown fields.  Because I was using the JamesSteinEncoder, the efforts to clean the data didn't seem to make much difference because I suspect that the JamesSteinEncoder was creating a separate category for the "unknown" data.

The effort that had the most affect was tuning the hyper-parameters in grid search.  
I added quite a few features.  
The decision tree had an improvement with some of the changes. 
They didn't have a huge effect, but I was able to improve the model sometimes by a percentage point so it did make a difference.


from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline
from imblearn.over_sampling import RandomOverSampler
from imblearn.under_sampling import RandomUnderSampler

The final thing that I tried was using some sampling tools to resample the data.  from the imblearn libraries This was useful in improving the recall.
That value is very important in this instance because the problem has more to lose if the bank loses a sale than spending some resources to have a sales person call the customer.
If the goal was to find the highest recall this was actually very successful.  I was able to find a model with 95% recall.

The accuracy took a hit and went down the 83%, but the RandomUnderSampler produced good results for recall.  This means that there are a lot of false positives, but in this case it would just mean that a sales person would get more rejections but would also get more sales. 
![image info](images/sampling_results.png)

## Conclusions
In all cases the accuracy went down and the precision went down, while the recall accuracy went up.

Even thought the accuracy went down for this business case let's look at what this actually means. 
False positive means that you may have to sell to someone who doesn't convert. 
That's ok because sales people are used to trying to convert to people who say no.  
What we want is fewer false negatives because we don't want to miss selling to people who might convert. 
So even though the accuracy went down below the baseline, SMOTE gives us some useful information that can be used by the sales team. 
It’s okay to contact more people if it means catching the ones who would actually say yes.

Use SMOTE produced a 90% recall.
The RandomUnderSampler actually had the highest recall at 95%.  
The RandomOverSampler had a 94% accuracy.

Because we have a huge majority class we can afford to get rid of some of the majority class and so underfitting works in this case.

## Recommendations

My recommendation is to use the model that resulted in the highest recall which would be the data found using the 