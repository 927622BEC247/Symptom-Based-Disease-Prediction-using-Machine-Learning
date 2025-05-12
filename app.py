import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from flask import Flask, render_template, request

app = Flask(__name__)

# Load the dataset
df = pd.read_csv("data/your_data.csv")

# Strip any extra spaces from column names (to avoid issues with column names containing spaces)
df.columns = df.columns.str.strip()

# Check if the 'Disease' column exists, and handle the drop accordingly
if 'Disease' in df.columns:
    # Features (X) and target (y)
    X = df.drop(columns='Disease')
    y = df['Disease']
else:
    print("'Disease' column not found in the DataFrame.")
    # Handle the error gracefully (you could set a default behavior here)
    exit()  # Terminate the script if 'Disease' column is missing

# Proceed with model training or prediction
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Initialize and train the KNN classifier
model = KNeighborsClassifier(n_neighbors=3)
model.fit(X_train, y_train)

# Prediction logic for later use
@app.route('/predict', methods=['POST'])
def predict():
    # Assuming you get the symptoms from the form
    symptoms = request.form.getlist('symptoms')

    # Preprocessing input and predicting disease
    # You will need to adjust this part depending on how your symptoms are used in the model

    prediction = model.predict([symptoms])  # This part depends on your input format

    # Assuming you have a way to get prediction accuracy (for example purposes)
    accuracy = model.score(X_test, y_test) * 100  # Model accuracy on test data

    return render_template('result.html', prediction=prediction[0], accuracy=accuracy)

# Flask app run (ensure your Flask setup is correct)
if __name__ == '__main__':
    app.run(debug=True)
