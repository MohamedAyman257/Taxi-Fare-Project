from django.shortcuts import render
import joblib
import pandas as pd
import numpy as np

model = joblib.load("taxi_pipeline.pkl")


def predict_fare(request):

    prediction = None

    if request.method == "POST":

        data = {
            "passenger_count": int(request.POST.get("passenger_count")),
            "hour": int(request.POST.get("hour")),
            "day": int(request.POST.get("day")),
            "month": int(request.POST.get("month")),
            "year": int(request.POST.get("year")),
            "weekday": int(request.POST.get("weekday")),

            "distance": float(request.POST.get("distance")),
            "bearing": float(request.POST.get("bearing")),

            "jfk_dist": float(request.POST.get("jfk_dist")),
            "ewr_dist": float(request.POST.get("ewr_dist")),
            "lga_dist": float(request.POST.get("lga_dist")),
            "sol_dist": float(request.POST.get("sol_dist")),
            "nyc_dist": float(request.POST.get("nyc_dist")),
        }

        df = pd.DataFrame([data])

        pred = model.predict(df)
        prediction = np.expm1(pred[0])

    return render(request, "index.html", {"prediction": prediction})