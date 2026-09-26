# IPL Match Winner Prediction System

## Description
This project is an IPL Match Winner Prediction System that uses a Machine Learning model to forecast the winner between two teams based on historical data. It features a Flask web interface for ease of use, connecting a robust data pipeline, a machine learning component, and a frontend presentation.

## Features
- **Predict Match Winners:** Get win probabilities for matches using a Random Forest Classifier.
- **Statistics Dashboard:** View detailed team statistics, top batsmen, and top bowlers based on historical IPL data.
- **Head-to-Head Comparisons:** View how two teams have performed against each other historically.
- **Prediction History:** See a historical log of all predictions made through the interface.
- **API Endpoints:** Get raw data access for teams and venues.

## Tech Stack
- **Python**: Core backend language.
- **Flask**: Web framework for building the application.
- **SQLite**: Lightweight database for storing matches, deliveries, and predictions.
- **scikit-learn**: Machine learning library for Random Forest model.
- **Bootstrap 5**: CSS framework for styling templates.

## Project Structure
```
IPL Project/
├── app.py
├── requirements.txt
├── README.md
├── database/
│   ├── __init__.py
│   └── db_setup.py
├── ml/
│   ├── __init__.py
│   ├── train_model.py
│   └── predict.py
├── static/
└── templates/
    ├── index.html
    ├── result.html
    ├── stats.html
    └── history.html
```

## Installation

Follow these steps to set up the project locally:

1. **Clone the repository**
2. **Install the required packages:**
   ```bash
   pip install -r requirements.txt
   ```
3. **Set up the database:**
   ```bash
   python database/db_setup.py
   ```
4. **Train the ML model:**
   ```bash
   python ml/train_model.py
   ```
5. **Run the application:**
   ```bash
   python app.py
   ```
6. **Access the application:** Open `http://127.0.0.1:5000` in your web browser.

## Screenshots
*(Add screenshots of your application here)*

## How it works
The ML model predicts match outcomes based on multiple features. It uses a `RandomForestClassifier` trained on historical ball-by-ball and match summary data. When a user queries a prediction via the frontend, the server queries the pre-trained model and returns win probabilities for both teams alongside a predicted winner.

### Features Used in the Model

| Feature | Description |
|---------|-------------|
| `team1` | The first team playing in the match |
| `team2` | The second team playing in the match |
| `venue` | The stadium where the match takes place |
| `toss_winner` | The team that won the toss |
| `toss_decision` | The decision made by the toss-winning team (bat/field) |

## Future Improvements
- Real-time match data scraping and active integration.
- Expanded player-level metrics in predicting outcomes.
- Advanced hyperparameter tuning and model ensemble to increase accuracy.

