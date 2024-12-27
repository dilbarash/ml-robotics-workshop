### Summary of Findings:

1. Model Performance:
   - The classification model was evaluated based on three key metrics:
     - Accuracy: This metric represents the overall proportion of correct predictions made by the model. A higher accuracy value indicates that the model is correctly classifying sensor readings as normal, warning, or critical.
     - Precision: This measures how many of the predicted "positive" classifications (e.g., warning or critical) were actually correct. A high precision indicates that the model minimizes false positives.
     - Recall: This measures how many of the actual "positive" instances (e.g., warning or critical) the model correctly identified. A high recall indicates that the model minimizes false negatives.

2. Confusion Matrix:
   - The **confusion matrix** visualizes the number of correct and incorrect predictions for each class (normal, warning, critical). It shows where the model is performing well and where it may be misclassifying certain readings.
   - If the model performs well, the diagonal values (representing correct classifications) should be high, while off-diagonal values (representing misclassifications) should be low.

3. Visualizations:
   - The confusion matrix was visualized using a heatmap, making it easier to interpret the classification performance across different classes.

   ![download (3)](https://github.com/user-attachments/assets/e19ecb01-5fb0-4815-8094-b02797c532c3)
