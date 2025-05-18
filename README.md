Credit Risk Analysis: Logistic Regression Model

Overview
This project uses a logistic regression model to analyze credit risk based on historical loan data. The dataset includes borrower information and loan outcomes. The primary goal is to determine how effectively logistic regression can classify loans as either healthy or high-risk.

The analysis was conducted using Python and key machine learning libraries. The original dataset (lending_data.csv) was split into training and testing sets to train and evaluate the model.

Technologies Used
• Python 3.9+
• Pandas
• NumPy
• Scikit-learn (sklearn)
• Jupyter Notebook
• GitHub Copilot
• ChatGPT

Steps Completed

Split the Data into Training and Testing Sets

Loaded lending_data.csv into a Pandas DataFrame

Created feature matrix X by dropping the loan_status column

Created label vector y using the loan_status column

Split the data into training and testing sets using train_test_split() with a random_state of 1

Create and Train Logistic Regression Model

Imported and instantiated a LogisticRegression model with a random_state of 1 and increased max_iter to 200 to prevent convergence warnings

Trained the model using the training data (X_train, y_train)

Model Evaluation

Made predictions using X_test

Evaluated model performance with:
• Accuracy score
• Confusion matrix
• Classification report (precision, recall, F1-score)

Model Performance Summary
Accuracy: ~0.99
The model shows strong performance in predicting healthy loans (label 0).
There is also good predictive ability for high-risk loans (label 1), though results may vary depending on class balance in the dataset.
