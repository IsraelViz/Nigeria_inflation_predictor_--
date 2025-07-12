from flask import Flask, render_template, request
import numpy as np
import pandas as pd
import joblib
import plotly.graph_objects as go
import plotly.io as pio

app = Flask(__name__)

# Load model and dataset
model = joblib.load("inflation_model.pkl")
df = pd.read_csv("NigeriaInflationRates.csv")

# Process date and historical stats
df['Date'] = pd.to_datetime(df['Year'].astype(str) + '-' + df['Month'].astype(str))
df = df.sort_values(by='Date')
latest_known = df.iloc[-1]['Inflation_Rate']
mean_inflation = df['Inflation_Rate'].mean()

@app.route('/')
def home():
    return render_template("index.html")

@app.route('/predict', methods=['POST'])
def predict():
    try:
        input_features = [
            float(request.form['Crude_Oil_Price']),
            float(request.form['Production']),
            float(request.form['Crude_Oil_Export']),
            float(request.form['CPI_Food']),
            float(request.form['CPI_Energy']),
            float(request.form['CPI_Health']),
            float(request.form['CPI_Transport']),
            float(request.form['CPI_Communication']),
            float(request.form['CPI_Education']),
        ]

        prediction = model.predict([input_features])[0]

        # Create comparison bar chart
        bar_fig = go.Figure(data=[
            go.Bar(name='Inflation Rate', x=['Predicted', 'Last Recorded', 'Average'], y=[prediction, latest_known, mean_inflation],
                   marker_color=['blue', 'orange', 'green'])
        ])
        bar_fig.update_layout(
            title="Inflation Comparison",
            yaxis_title="Inflation Rate (%)",
            width=700, height=400
        )
        chart_html = pio.to_html(bar_fig, full_html=False)

        comparison_text = f"""
        <b>Predicted Inflation Rate:</b> {prediction:.2f}%<br>
        <b>Last Recorded Inflation Rate:</b> {latest_known:.2f}%<br>
        <b>Historical Average Inflation Rate:</b> {mean_inflation:.2f}%
        """

        return render_template("index.html", prediction_text=comparison_text, chart=chart_html)

    except Exception as e:
        return render_template("index.html", prediction_text=f"Error: {str(e)}", chart=None)

if __name__ == '__main__':
    app.run(debug=True)
