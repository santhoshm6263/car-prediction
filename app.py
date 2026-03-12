from flask import Flask, request, render_template
import pickle
import numpy as np
import os
import requests

app = Flask(__name__)

# Load the trained model
model_path = 'model.pkl'

def get_car_image(car_name):
    """
    Dynamically fetch a representative image URL for a car from Wikipedia.
    """
    try:
        # Step 1: Search for the most relevant Wikipedia page title
        headers = {"User-Agent": "CarPricePredictor/1.0 (santho@example.com)"}
        search_url = "https://en.wikipedia.org/w/api.php"
        search_params = {
            "action": "query",
            "list": "search",
            "srsearch": car_name + " car",
            "format": "json",
            "srlimit": 1
        }
        res = requests.get(search_url, params=search_params, headers=headers, timeout=5)
        search_res = res.json()
        
        if not search_res.get("query", {}).get("search"):
            return "https://images.unsplash.com/photo-1533473359331-0135ef1b58bf?auto=format&fit=crop&q=80&w=400"
            
        page_title = search_res["query"]["search"][0]["title"]
        
        # Step 2: Get the lead image (thumbnail) for that page
        img_params = {
            "action": "query",
            "titles": page_title,
            "prop": "pageimages",
            "format": "json",
            "pithumbsize": 500
        }
        img_res = requests.get(search_url, params=img_params, headers=headers, timeout=5).json()
        pages = img_res.get("query", {}).get("pages", {})
        
        for page_id in pages:
            if "thumbnail" in pages[page_id]:
                return pages[page_id]["thumbnail"]["source"]
                
    except Exception as e:
        print(f"Error fetching image for {car_name}: {e}")
        
    # Fallback to a high-quality generic car image if anything fails
    return "https://images.unsplash.com/photo-1533473359331-0135ef1b58bf?auto=format&fit=crop&q=80&w=400"

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    # Attempt to load model at prediction time in case it was created after server start
    if not os.path.exists(model_path):
        import train_model
        train_model.main()
    
    try:
        with open(model_path, 'rb') as file:
            model = pickle.load(file)
            
        features = [
            float(request.form['wheelbase']),
            float(request.form['carlength']),
            float(request.form['carwidth']),
            float(request.form['curbweight']),
            float(request.form['enginesize']),
            float(request.form['horsepower'])
        ]
        
        final_features = [np.array(features)]
        prediction = model.predict(final_features)
        
        price_usd = prediction[0]
        price_inr = price_usd * 83
        formatted_output = f"₹{price_inr:,.2f}"
        
        # Car catalog with optimized names for Wikipedia searching
        car_catalog = [
            {"name": "Maruti Alto 800", "price": 354000, "specs": "Petrol • Manual • 5 Seater"},
            {"name": "Renault Kwid", "price": 470000, "specs": "Petrol • Manual/Auto • 5 Seater"},
            {"name": "Tata Tiago", "price": 565000, "specs": "Petrol/CNG • Manual/Auto • 5 Seater"},
            {"name": "Maruti Swift", "price": 649000, "specs": "Petrol • Manual/Auto • 5 Seater"},
            {"name": "Hyundai i20", "price": 704000, "specs": "Petrol • Manual/Auto • 5 Seater"},
            {"name": "Tata Nexon", "price": 815000, "specs": "Petrol/Diesel • Manual/Auto • 5 Seater"},
            {"name": "Hyundai Venue", "price": 800000, "specs": "Petrol/Diesel • Manual/Auto • 5 Seater"},
            {"name": "Mahindra XUV300", "price": 749000, "specs": "Petrol/Diesel • Manual/Auto • 5 Seater"},
            {"name": "Hyundai Creta", "price": 1100000, "specs": "Petrol/Diesel • Manual/Auto • 5 Seater"},
            {"name": "Kia Seltos", "price": 1090000, "specs": "Petrol/Diesel • Manual/Auto • 5 Seater"},
            {"name": "Mahindra Scorpio-N", "price": 1385000, "specs": "Petrol/Diesel • Manual/Auto • 7 Seater"},
            {"name": "Tata Harrier", "price": 1549000, "specs": "Diesel • Manual/Auto • 5 Seater"},
            {"name": "Mahindra XUV700", "price": 1400000, "specs": "Petrol/Diesel • Manual/Auto • 5/7 Seater"},
            {"name": "Toyota Innova Crysta", "price": 1999000, "specs": "Diesel • Manual • 7/8 Seater"},
            {"name": "Jeep Compass", "price": 2069000, "specs": "Petrol/Diesel • Manual/Auto • 5 Seater"},
            {"name": "Toyota Fortuner", "price": 3343000, "specs": "Petrol/Diesel • Manual/Auto • 7 Seater"},
            {"name": "BYD Seal", "price": 4100000, "specs": "Electric • Automatic • 5 Seater"},
            {"name": "BMW X1", "price": 4950000, "specs": "Petrol/Diesel • Automatic • 5 Seater"},
            {"name": "Audi Q3", "price": 5370000, "specs": "Petrol • Automatic • 5 Seater"},
            {"name": "Mercedes-Benz C-Class", "price": 6185000, "specs": "Petrol/Diesel • Automatic • 5 Seater"},
            {"name": "Volvo XC60", "price": 6890000, "specs": "Petrol Mild Hybrid • Automatic • 5 Seater"},
            {"name": "BMW X5", "price": 9600000, "specs": "Petrol/Diesel • Automatic • 5 Seater"},
            {"name": "Mercedes-Benz S-Class", "price": 17700000, "specs": "Petrol/Diesel • Automatic • 5 Seater"},
            {"name": "Porsche 911", "price": 19000000, "specs": "Petrol • Automatic • 4 Seater"},
            {"name": "Land Rover Defender", "price": 10400000, "specs": "Petrol/Diesel • Automatic • 5/7 Seater"},
        ]
        
        # Filter and Sort
        lower_bound = price_inr * 0.6
        upper_bound = price_inr * 1.4
        suggestions = [car for car in car_catalog if lower_bound <= car["price"] <= upper_bound]
        suggestions.sort(key=lambda x: abs(x["price"] - price_inr))
        top_suggestions = suggestions[:3]
        
        # Dynamically fetch accurate images for the results
        for car in top_suggestions:
            car["image"] = get_car_image(car["name"])
        
        return render_template('index.html', prediction_text=f'Predicted Car Price: {formatted_output}', suggestions=top_suggestions)
    
    except Exception as e:
        return render_template('index.html', prediction_text=f'Error occurred: {str(e)}')

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=10000)
