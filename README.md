# Acasă Real Estate Price Predictor

This project is a real estate price prediction tool named **Acasă Real Estate Price Predictor**. It uses various machine learning models to predict the price of real estate units based on specific features like transaction date, house age, distance to the nearest railway station, and the number of convenience stores nearby.

## Features

- **Data Exploration and Visualization**: 
  - Load and explore the dataset.
  - Visualize data distributions and relationships using histograms and scatter plots.

- **Data Preprocessing**: 
  - Handle missing values using different strategies.
  - Normalize and standardize features.
  - Split the dataset into training and testing sets, with stratified sampling based on certain features.

- **Model Training**:
  - Multiple models are implemented, including:
    - Linear Regression
    - Decision Tree Regressor
    - Random Forest Regressor
    - Gradient Boosting Regressor (final model)
  - Pipeline creation for streamlined preprocessing and model fitting.

- **Model Evaluation**:
  - Evaluate models using RMSE (Root Mean Square Error).
  - Implement cross-validation for more robust performance evaluation.

- **Model Saving and Loading**:
  - Save the trained model using `joblib`.
  - Load the model for future predictions.

## Requirements

- Python 3.6+
- Required Python packages:
  - `pandas`
  - `numpy`
  - `matplotlib`
  - `scikit-learn`
  - `joblib`

You can install the required packages using:

```bash
pip install pandas numpy matplotlib scikit-learn joblib
```

## Usage

1. **Data Loading**: The dataset should be in CSV format and named `data.csv`. Place it in the same directory as the script.
   
2. **Training the Model**: Run the script to preprocess the data, train the model, and evaluate its performance.

3. **Making Predictions**: After training, the model can predict house prices based on new data. You can use the saved model to make predictions on new data by providing appropriate features.

4. **Testing**: The model is tested against a test set, and the final RMSE is computed.

## Example

To predict a house price using the trained model:

```python
from joblib import load
import numpy as np

model = load('Siri.joblib') 
features = np.array([[-1.69732575,  0.18277164,  0.88477281, -0.37510886, -0.48993566, -1.38876478]])
prediction = model.predict(features)
print(prediction)
```

## Project Structure

- `data.csv`: The dataset file.
- `Acasă Real Estate Price Predictor.ipynb`: The main Jupyter Notebook with code and explanations.
- `Siri.joblib`: The saved model file after training.

## Conclusion

This project demonstrates the process of building a real estate price prediction model using Python and scikit-learn. By following the steps outlined, you can understand data preprocessing, model training, and evaluation, and how to make predictions with a trained model.

## License

This project is licensed under the MIT License.
