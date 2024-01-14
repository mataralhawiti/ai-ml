# Defining ML Modele

- optimization always requires a standard by which to say we're improving, we'll discuss loss functions

- ML models are mathematical functions with :
    - **parameters** and **hyperparameters**

    - *parameter* : is a real valued variable that changes during model training
    - *hyperparameter*: is a setting that we set before training and which doesn't change afterwards


- **linear models** were one of the first sorts of ML models 
    - In a linear model, small changes in the *independent variables*, or ***features*** as we refer to them in *machine learning*, yield the same amount of change in the *dependent* variable or label regardless of where that change takes place in the input space.

    ![liner](image-5.png)

    - ***Explain** :
        - formula used to model the relationship is simply : *image*
            - **M** captures the amount of *change* we observe in our **label** in response to a small change in our **feature**.

    - This same concept of a relationship defined by a fixed ratio change between labels and features can be extended to arbitrarily hide dimensionality, 
    
        both with respect to the inputs and the outputs, meaning ***we can build models that accept many more features as input, model multiple labels simultaneously or both***.
        - When we increase the *dimensionality* of the input, our slope term M must become *n-dimensional*. We call this new term the **weight**.
        - Visually this process yields the n-dimensional *generalization of a line*, which is called a **hyperplane**
        ![hyperplane](image-6.png)