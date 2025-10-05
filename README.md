# iOracle

**Predicting Apple Stock Prices with Ensemble Machine Learning**

A time-series forecasting project that combines LSTM neural networks and Random Forest models to predict Apple (AAPL) stock prices 5 days into the future based on 30 days of historical data and technical indicators.

Developed as part of Le Wagon's Data Science Bootcamp (24 weeks)

## Project Overview

iOracle demonstrates an end-to-end machine learning workflow from data collection to deployment. The project showcases ensemble learning methods, feature engineering with technical indicators, custom time-series cross-validation, and production deployment on Google Cloud Platform with an interactive Streamlit interface.

### Results
- Prediction horizon: 5 days ahead
- Input window: 30 days of historical data
- Evaluation metric: Mean Absolute Error (MAE) in dollars
- Deployment: Real-time predictions via FastAPI and Streamlit

## Architecture

### Machine Learning Pipeline

The pipeline processes raw stock data through feature engineering, feeds it into two separate models (LSTM and Random Forest), and combines their predictions using a linear regression meta-learner.

### Technical Stack

**Machine Learning**
- TensorFlow/Keras for LSTM implementation
- scikit-learn for Random Forest, preprocessing, and evaluation
- ta (Technical Analysis Library) for technical indicators

**Data Processing**
- yfinance for real-time stock data
- pandas and numpy for data manipulation
- joblib for model serialization

**Deployment**
- FastAPI for REST API backend
- Streamlit for interactive frontend
- Google Cloud Storage for model persistence
- Google Cloud Run for serverless deployment
- Plotly for interactive visualizations

## Feature Engineering

The model uses 12 engineered features calculated over multiple time windows:

**Moving Average (MA)**: 14, 50, 200 days - Trend identification
**RSI (Relative Strength Index)**: 14, 50, 200 days - Momentum indicator
**Bollinger Band Width**: 14, 50, 200 days - Volatility measure
**VIX**: Market volatility index
**Volume**: Trading volume
**Adj Close**: Adjusted closing price

All features are scaled using StandardScaler to ensure consistent model performance.

## Model Details

### LSTM Model
- Architecture: 2 LSTM layers (20 units each) with 2 Dense layers
- Input shape: (30 timesteps, 12 features)
- Regularization: L1 regularization (0.01)
- Optimizer: RMSprop
- Loss: Mean Squared Error (MSE)
- Early stopping with patience of 5 epochs

### Random Forest Regressor
- Hyperparameter tuning via RandomizedSearchCV (100 iterations)
- Custom time-series cross-validation
- Search space includes tree count (200-2000), max depth (10-110), min samples split, feature selection methods, and bootstrap settings

### Ensemble Meta-Model
- Linear Regression combining LSTM and Random Forest predictions
- Outputs final 5-day ahead price prediction

## Project Structure

```
iOracle/
├── iOracle/                  # Core ML package
│   ├── data.py              # Yahoo Finance data fetching
│   ├── preproc.py           # Feature engineering and preprocessing
│   ├── lstm.py              # LSTM model class
│   ├── random_forest.py     # Random Forest trainer
│   ├── TimedSplit.py        # Custom time-series CV
│   └── GBQ.py               # Google BigQuery utilities
│
├── api/
│   └── fast.py              # FastAPI endpoint
│
├── notebooks/               # Jupyter notebooks for experimentation
│   ├── iOracle Deep Learning Ivan.ipynb
│   ├── iOracle Random Forest Regressor - Brendan.ipynb
│   ├── Combined Linear Model.ipynb
│   └── ...
│
├── app.py                   # Streamlit web interface
├── predict.py              # Main prediction orchestration
├── requirements.txt        # Python dependencies
├── Dockerfile             # Container configuration
└── Makefile              # Build and test automation
```

## Getting Started

### Prerequisites
- Python 3.7+
- Google Cloud credentials (for model loading)

### Installation

1. Clone the repository
```bash
git clone https://github.com/{your-username}/iOracle.git
cd iOracle
```

2. Create virtual environment
```bash
python -m virtualenv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies
```bash
pip install -r requirements.txt
```

4. Set up Google Cloud credentials (if using stored models)
```bash
# Place your service-account-file.json in the project root
export GOOGLE_APPLICATION_CREDENTIALS="service-account-file.json"
```

### Running the Application

**Streamlit Web App**
```bash
streamlit run app.py
```
Visit http://localhost:8501 to view the interactive dashboard.

**FastAPI Backend**
```bash
make run_api
# or
uvicorn api.fast:app --reload
```
API available at http://localhost:8000/predict/?ticker_name=aapl

**Command Line Prediction**
```bash
python predict.py
```

## Model Training

The training process is documented in Jupyter notebooks:

1. Data Collection: `notebooks/iOracle Data.ipynb`
2. Random Forest Development: `notebooks/iOracle Random Forest Regressor - Brendan.ipynb`
3. LSTM Development: `notebooks/iOracle Deep Learning Ivan.ipynb`
4. Ensemble Model: `notebooks/Combined Linear Model.ipynb`

To retrain models:
```bash
python iOracle/random_forest.py
python iOracle/lstm.py
```

## Example Output

The Streamlit app displays:
- Historical actual vs predicted prices (30-day comparison)
- 5-day future predictions with Bollinger Bands
- Mean Absolute Error (MAE) for model performance
- Interactive Plotly charts with zoom and hover functionality

## Development

### Running Tests
```bash
make clean install test
```

### Code Quality
```bash
make black        # Format code
make check_code   # Linting with flake8
```

### Docker Deployment
```bash
docker build -t ioracle .
docker run -p 8000:8000 ioracle
```

## Key Technical Implementations

**Time-series data leakage prevention**: Custom TimedSplit class ensures no future data leaks into training sets

**Sequential data preparation**: Custom preprocessing pipeline creates 30-day rolling windows for LSTM input

**Feature alignment**: Synchronized feature engineering between Random Forest and LSTM models for consistent ensemble predictions

**Model versioning**: Google Cloud Storage integration for model persistence and versioning

**Real-time pipeline**: Live data fetching and preprocessing for production predictions

## Limitations and Future Work

Current implementation focuses on Apple stock only. Future enhancements could include:
- Multi-stock support
- Sentiment analysis from financial news
- Attention mechanisms in LSTM architecture
- Confidence intervals for predictions
- Comprehensive backtesting framework
- Automated model retraining pipeline

## Acknowledgments

Developed as part of Le Wagon Data Science Bootcamp (24 weeks). Team members: Ivan (LSTM development), Derrick (AutoARIMA experiments), Brendan (Random Forest implementation).

Technical resources: Technical Analysis Library (ta) for financial indicators, Yahoo Finance (yfinance) for stock data API.
