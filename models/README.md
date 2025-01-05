# Model Testing

## General Insights
1. Overall Accuracy:
    - The Stacking Classifier performs the best with an accuracy of 95.69%, followed closely by Extra Trees (95.42%) and SVM (95.14%). These models seem to be highly reliable for this task.
    - Logistic Regression (92.78%) and Naive Bayes (91.25%) are still good performers but slightly less accurate compared to the top performers.
    - The Decision Tree and Random Forest models perform similarly with accuracy around 89-94%, which is still solid but not as high as the ensemble models like Stacking and Extra Trees.

2. Classification Report (Precision, Recall, F1-Score):
    - Precision, recall, and F1-scores across all models indicate strong harmful class detection with precision and recall generally above 90% for harmful messages. This shows that these models can effectively identify harmful content.
    - The harmful class tends to have better recall (more true positives), which is ideal for applications like detecting harmful behavior. Safe class recall is slightly lower in many cases, indicating that some safe messages might be misclassified as harmful.

3. ROC AUC Scores:
    - Stacking Classifier has the highest ROC AUC score (0.988), indicating that it has the best ability to distinguish between harmful and safe messages across different thresholds.
    - SVM and Extra Trees also have high ROC AUC scores (0.987 and 0.984), indicating strong discriminatory power between classes.
    - Naive Bayes and Logistic Regression also show very good ROC AUC values (0.977-0.985), which suggests they perform reasonably well for imbalanced class prediction.

## Model Specific Insights
- Stacking Classifier: This model stands out for its high accuracy and ROC AUC, showing that combining multiple base models (Random Forest, SVC) gives it superior generalization power.
- SVM: With slightly lower accuracy than Stacking but still very good, SVM shows that it's quite capable in terms of precision and recall for both classes.
- Extra Trees: Similar to Random Forest, but typically a bit more robust in terms of accuracy and recall for harmful messages.
- Logistic Regression: A reliable model with a slightly lower accuracy, but it's still highly effective in predicting harmful messages. It might be a good option when you need simpler, more interpretable models.
- Decision Tree: While it has lower accuracy compared to ensemble methods, it's still decent for basic use cases. However, its performance is more sensitive to overfitting, which might explain why it's less accurate than the others.
- Naive Bayes: Slightly less precise than others but still effective, especially in terms of handling the harmful messages. It might be useful for simpler, fast models.
- Perceptron-Based Models (e.g., MLPClassifier): The MLPClassifier shows good performance in accuracy and ROC AUC, indicating that neural network-based models can also be quite effective.


## Random Forest Classifier
```
Accuracy: 0.9388888888888889

Classification Report:
               precision    recall  f1-score   support

     harmful       0.91      0.98      0.94       385
        safe       0.97      0.90      0.93       335

    accuracy                           0.94       720
   macro avg       0.94      0.94      0.94       720
weighted avg       0.94      0.94      0.94       720


ROC AUC: 0.9790889707307617

Model Testing: 
prediction     expected      
-----------------------
   harmful      harmful
      safe         safe
      safe         safe
   harmful      harmful
      safe      harmful
   harmful      harmful
   harmful      harmful
      safe         safe
   harmful         safe
```

## Logistic Regression
```
Accuracy: 0.9277777777777778

Classification Report:
               precision    recall  f1-score   support

     harmful       0.95      0.91      0.93       385
        safe       0.90      0.95      0.92       335

    accuracy                           0.93       720
   macro avg       0.93      0.93      0.93       720
weighted avg       0.93      0.93      0.93       720


ROC AUC: 0.9847683659623958

Model Testing: 
prediction     expected      
-----------------------
   harmful      harmful
   harmful         safe
      safe         safe
   harmful      harmful
      safe      harmful
   harmful      harmful
   harmful      harmful
   harmful         safe
   harmful         safe
```
## Decision Tree
```
Accuracy: 0.8902777777777777

Classification Report:
               precision    recall  f1-score   support

     harmful       0.88      0.92      0.90       385
        safe       0.90      0.86      0.88       335

    accuracy                           0.89       720
   macro avg       0.89      0.89      0.89       720
weighted avg       0.89      0.89      0.89       720


ROC AUC: 0.9100368288427989

Model Testing: 
prediction     expected      
-----------------------
      safe      harmful
      safe         safe
      safe         safe
   harmful      harmful
      safe      harmful
   harmful      harmful
   harmful      harmful
   harmful         safe
   harmful         safe
```

## Extra Trees
```
Accuracy: 0.9541666666666667

Classification Report:
               precision    recall  f1-score   support

     harmful       0.94      0.98      0.96       385
        safe       0.97      0.93      0.95       335

    accuracy                           0.95       720
   macro avg       0.96      0.95      0.95       720
weighted avg       0.95      0.95      0.95       720


ROC AUC: 0.9841868579182013

Model Testing: 
prediction     expected      
-----------------------
   harmful      harmful
      safe         safe
      safe         safe
      safe      harmful
      safe      harmful
   harmful      harmful
      safe      harmful
      safe         safe
   harmful         safe
```

## Stacked Classifiers (StackingClassifier)
```
Accuracy: 0.9569444444444445

Classification Report:
               precision    recall  f1-score   support

     harmful       0.96      0.96      0.96       385
        safe       0.95      0.96      0.95       335

    accuracy                           0.96       720
   macro avg       0.96      0.96      0.96       720
weighted avg       0.96      0.96      0.96       720


ROC AUC: 0.9880209342895909
Model Testing: 
prediction     expected      
-----------------------
   harmful      harmful
      safe         safe
      safe         safe
   harmful      harmful
      safe      harmful
      safe      harmful
   harmful      harmful
      safe         safe
   harmful         safe
```

## Support Vector Machine (SVM)
```
Accuracy: 0.9513888888888888

Classification Report:
               precision    recall  f1-score   support

     harmful       0.96      0.95      0.95       385
        safe       0.94      0.96      0.95       335

    accuracy                           0.95       720
   macro avg       0.95      0.95      0.95       720
weighted avg       0.95      0.95      0.95       720


ROC AUC: 0.9874394262453963

Model Testing: 
prediction     expected      
-----------------------
   harmful      harmful
      safe         safe
      safe         safe
   harmful      harmful
      safe      harmful
      safe      harmful
   harmful      harmful
      safe         safe
   harmful         safe
```

## Naive Bayes (MultinomialNB)
```
Accuracy: 0.9125

Classification Report:
               precision    recall  f1-score   support

     harmful       0.90      0.94      0.92       385
        safe       0.93      0.88      0.90       335

    accuracy                           0.91       720
   macro avg       0.91      0.91      0.91       720
weighted avg       0.91      0.91      0.91       720


ROC AUC: 0.9779414615235511

Model Testing: 
prediction     expected
-----------------------
   harmful      harmful
   harmful         safe
   harmful         safe
   harmful      harmful
   harmful      harmful
   harmful      harmful
   harmful      harmful
   harmful         safe
   harmful         safe
```

## Perceptron-Based Models (e.g., Neural Networks with MLPClassifier)
```
Accuracy: 0.9430555555555555

Classification Report:
               precision    recall  f1-score   support

     harmful       0.93      0.97      0.95       385
        safe       0.96      0.92      0.94       335

    accuracy                           0.94       720
   macro avg       0.94      0.94      0.94       720
weighted avg       0.94      0.94      0.94       720


ROC AUC: 0.9883969761581702

Model Testing: 
prediction     expected      
-----------------------
   harmful      harmful
   harmful         safe
      safe         safe
   harmful      harmful
   harmful      harmful
   harmful      harmful
   harmful      harmful
   harmful         safe
   harmful         safe
```