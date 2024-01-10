# ML Workflow
- part

## How a machine learns

## Lab
- 3 phases of ML workflows
    - data preparation, model development, and model serving.

- model evaluation - (help you to interpret the training results)


![Alt text](image-1.png)

#### confusion matrix
- The true positives are 100%
    - This represents the percentage of people that the model predicts would repay their loan and actually paid it back.
    - In other words the model is 100% accurate in predicting the number of people that would pay back their loan.

- The true negatives are 87%
    - This represents the percentage of people that the model predicts would not repay their loan and indeed did not pay it back.
    - In other words the model is 87% accurate in predicting the number of people that would not pay back their loan.

- The false negatives are 0%.
    - This represents the percentage of people that the model predicts would not repay their loan but actually paid it back.

- the false positives are 13%
    - This representing the percentage of people that the model predicts would repay their loan but actually did not pay it back.


- As a general principle, it’s good to have high true positives and true negatives, and low false positives and false negatives.


- different ways to improve the performance of a model : 
    - using a more accurate data source
    - using a larger dataset
    - choosing a different type of ML model
    - tuning the hyperparameters.


#### precision-recall curve 

- **confidence threshold** determines how a machine learning model counts the positive cases
    - A higher threshold increases the precision but decreases recall.
    - A lower threshold decreases the precision, but increases recall.

- Scnario: Moving the confidence threshold to zero produces the highest recall of 100% and the lowest precision of 50%.
    - That means the model predicts that 100% of loan applicants will be able to repay a loan they take out.
        However, actually only 50% of people were able to repay the loan.
    - Using this threshold to identify the default cases in this example can be risky, 
        because it means that you can only get half of the loan investment back.

- scnario:  other extreme by moving the threshold to 1. This will produce the highest precision of 100% with the lowest recall of 1%.
    - It means that of all the people who were predicted to repay the loan, 100% of them actually did.
        However, you rejected 99% of loan applicants by only offering loans to 1% of them.
    - That’s a pretty big business loss for your company.


- **These are both extreme examples, but it’s important that you always try to set an appropriate threshold for your model.**




### Questions
- A hospital uses the machine learning technology of Google to help pre-diagnose cancer by feeding historical patient medical data to the model. 
    The goal is to identify as many potential cases as possible. Which metric should the model focus on?
    - Feature importance
    - Confusion matrix
    - Precision
    - Recall **correct**

- A farm uses the machine learning technology of Google to detect defective apples in their crop, like those with irregular sizes or scratches. The goal is to identify only the apples that are actually bad so that no good apples are wasted. Which metric should the model focus on?

    - Feature importance
    - Confusion matrix
    - Precision **correct**
    - Recall