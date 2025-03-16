import joblib
import numpy as np
from config.paths_config import MODEL_OUTPUT_PATH
from flask import Flask, render_template,request

app = Flask(__name__)

loaded_model = joblib.load(MODEL_OUTPUT_PATH)

@app.route('/',methods=['GET','POST'])
def index():
    if request.method=='POST':
        patient_age = int(request.form["patient_age"])

        cancelled_slots = int(request.form["cancelled_slots"])
        scheduled_slots = int(request.form["scheduled_slots"])

        appt_type = int(request.form["appt_type"])
        appt_day = int(request.form["appt_day"])
        race = int(request.form["race"])
        ethnicity = int(request.form["ethnicity"])
        marital_status = int(request.form["marital_status"])
        patient_lang = int(request.form["patient_lang"])
        pblchouspat = int(request.form["pblchouspat"])

        # Prepare the feature array for prediction
        features = np.array([[cancelled_slots, patient_age, appt_type, appt_day, race, scheduled_slots, ethnicity, marital_status, patient_lang, pblchouspat]])
        #features = np.array([[0,2,2,2,5,1,1,4,1,0]])
        print(f"{features=}")

        # Get the prediction
        prediction = loaded_model.predict(features)
        return render_template('index.html', prediction=prediction[0])
    
    return render_template("index.html" , prediction=None)

if __name__=="__main__":
    port = int(os.environ.get("PORT", 8080))  # Default to 8080 if PORT is not set
    app.run(host="0.0.0.0", port=port)