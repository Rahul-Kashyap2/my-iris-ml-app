# My Iris ML Project

This repository demonstrates a multi-page Flask app using the classic Iris dataset. 
We showcase:

1. **Two Binary Classifications**:
   - Setosa vs. Not Setosa
   - Versicolor vs. Virginica

2. **Dynamic K-Means Clustering**:
   - User can input the number of clusters (k) and see the cluster plot (2D scatter) on the fly.

## Folder Structure

- `train_models.py`: Script that trains and saves two Logistic Regression models (for binary classification).
- `app.py`: Flask web app with 3 sub-pages:
  1. Setosa vs. Not Setosa
  2. Versicolor vs. Virginica
  3. K-Means clustering
- `requirements.txt`: Python dependencies.
- `.env`: Contains the **ngrok auth token** (ignored by git).
- `.gitignore`: Ensures `.env` and other unwanted files are not committed.

## How To Run (Locally or Colab)
1. Clone this repo:  
   `git clone https://github.com/YOUR_USERNAME/YOUR_REPO_NAME.git`
2. Install dependencies:  
   `pip install -r requirements.txt`
3. Create a `.env` file with your `NGROK_AUTH_TOKEN`.
4. Train models:  
   `python train_models.py`  
   This saves the model files locally.
5. Launch the web app:  
   `python app.py`
6. You’ll see instructions in the console for accessing your app via **ngrok**.

## Notes
- Make sure you do **not** commit your `.env` file! 
- The `.env` file is in `.gitignore`, so your token remains safe.

