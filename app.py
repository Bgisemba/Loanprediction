import joblib
from flask import Flask, render_template, request

# Load the pre-trained model
model = joblib.load('model.pkl')

# Initialize Flask app
app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict/', methods=['POST'])
def predict():
    if request.method == 'POST':
        # Collect form inputs
        Gender = request.form.get('Gender')
        Married = request.form.get('Married')
        Dependents = request.form.get('Dependents')
        Education = request.form.get('Education')
        Self_Employed = request.form.get('Self_Employed')
        ApplicantIncome = request.form.get('ApplicantIncome')
        CoapplicantIncome = request.form.get('CoapplicantIncome')
        LoanAmount = request.form.get('LoanAmount')
        Loan_Amount_Term = request.form.get('Loan_Amount_Term')
        Credit_History = request.form.get('Credit_History')
        Property_Area = request.form.get('Property_Area')

        # Map categorical variables to numerical values
        CATEGORY_MAPPINGS = {
            "Gender": {"Male": 1, "Female": 0},
            "Married": {"Yes": 1, "No": 0},
            "Education": {"Graduate": 1, "Not Graduate": 0},
            "Self_Employed": {"Yes": 1, "No": 0},
            "Property_Area": {"Urban": 2, "Semiurban": 1, "Rural": 0}
        }

        try:
            # Convert inputs to numerical format
            Gender = CATEGORY_MAPPINGS["Gender"][Gender]
            Married = CATEGORY_MAPPINGS["Married"][Married]
            Education = CATEGORY_MAPPINGS["Education"][Education]
            Self_Employed = CATEGORY_MAPPINGS["Self_Employed"][Self_Employed]
            Property_Area = CATEGORY_MAPPINGS["Property_Area"][Property_Area]

            # Convert `3+` to numeric value 3
            Dependents = 3 if Dependents == "3+" else int(Dependents)

            ApplicantIncome = float(ApplicantIncome)
            CoapplicantIncome = float(CoapplicantIncome)
            LoanAmount = float(LoanAmount)
            Loan_Amount_Term = float(Loan_Amount_Term)
            Credit_History = float(Credit_History)

            # Prepare the input for the model
            input_features = [[
                Gender, Married, Dependents,Education, Self_Employed, ApplicantIncome,
                CoapplicantIncome, LoanAmount, Loan_Amount_Term, Credit_History, Property_Area
            ]]

            # Make a prediction
            prediction = model.predict(input_features)[0]  # Model outputs 0 or 1
            prediction_text = "Approved" if prediction == 1 else "Rejected"

        except Exception as e:
            # Handle errors in input or prediction
            prediction_text = f"Error: {str(e)}"

        return render_template('predict.html', prediction=prediction_text)

if __name__ == '__main__':
    app.run(debug=True)
