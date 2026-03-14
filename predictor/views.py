from django.shortcuts import render
import joblib
import pandas as pd
import numpy as np
 
model = joblib.load("taxi_pipeline.pkl")
 
 
def predict_fare(request):
 
    prediction = None
 
    if request.method == "POST":
 
        data = {
            "user_id":           request.POST.get("user_id"),
            "user_name":         request.POST.get("user_name"),
            "driver_name":       request.POST.get("driver_name"),
            "car_condition":     request.POST.get("car_condition"),
            "weather":           request.POST.get("weather"),
            "traffic_condition": request.POST.get("traffic_condition"),
            "key":               request.POST.get("key"),
            "pickup_datetime":   request.POST.get("pickup_datetime"),
            "passenger_count":   int(request.POST.get("passenger_count")),
            "hour":              int(request.POST.get("hour")),
            "day":               int(request.POST.get("day")),
            "month":             int(request.POST.get("month")),
            "year":              int(request.POST.get("year")),
            "weekday":           int(request.POST.get("weekday")),
            "pickup_latitude":   float(request.POST.get("pickup_latitude")),
            "pickup_longitude":  float(request.POST.get("pickup_longitude")),
            "dropoff_latitude":  float(request.POST.get("dropoff_latitude")),
            "dropoff_longitude": float(request.POST.get("dropoff_longitude")),
            "distance":          float(request.POST.get("distance")),
            "bearing":           float(request.POST.get("bearing")),
            "jfk_dist":          float(request.POST.get("jfk_dist")),
            "ewr_dist":          float(request.POST.get("ewr_dist")),
            "lga_dist":          float(request.POST.get("lga_dist")),
            "sol_dist":          float(request.POST.get("sol_dist")),
            "nyc_dist":          float(request.POST.get("nyc_dist")),
        }
 
        df = pd.DataFrame([data])
 
        pred = model.predict(df)
        prediction = round(np.expm1(pred[0]), 2)
 
    return render(request, "index.html", {"prediction": prediction})