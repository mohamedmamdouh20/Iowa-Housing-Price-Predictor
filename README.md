# Iowa Housing Price Predictor

This project utilizes various regression techniques to predict house prices based on the Iowa Housing dataset. It includes a Streamlit app for interactive predictions and model evaluation.

## Techniques Used

- **Multiple Linear Regression**
- **Lasso Regression**
- **Ridge Regression**
- **Simple Linear Regression**

## Evaluation Metrics

- **R-squared Score**: 0.9
- **Mean Squared Error (MSE)**
- **Mean Absolute Error (MAE)**

## Files

- `model.py`: Code for training and evaluating the regression models.
- `app.py`: Streamlit application for interactive predictions.
- `data/`: (Optional) Folder containing the Iowa Housing dataset.
- `requirements.txt`: Dependencies for running the code.

## Getting Started

1. **Clone the repository**:
   ```bash
   git clone https://github.com/yourusername/IowaHousingPricePredictor.git
   ```

2. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Run the Streamlit app**:
   ```bash
   streamlit run app.py
   ```

4. **Open the app**: Navigate to the address provided by Streamlit in your browser to start using the app.

## Usage

Input house features into the Streamlit app to receive price predictions from various regression models and view evaluation metrics.
