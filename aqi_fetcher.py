import requests

API_TOKEN = "3a2aa2c4954e9a7cf7fc7e952a3fd96662775a32"


def fetch_aqi(city):
    city = city.strip().lower()

    url = f"https://api.waqi.info/feed/{city}/?token={API_TOKEN}"
    

    try:
        response = requests.get(url, timeout=10)
        data = response.json()

        # SAFE CHECK (prevents crash)
        if "status" not in data:
            raise Exception("Invalid API response")

        if data["status"] != "ok":
            raise Exception(f"API Error: {data.get('data')}")

        return data["data"]

    except requests.exceptions.Timeout:
        raise Exception("Request timeout — try again")

    except Exception as e:
        raise Exception(str(e))