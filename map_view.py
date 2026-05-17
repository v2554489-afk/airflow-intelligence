import folium

def get_color(aqi):
    if aqi <= 50:
        return "green"
    elif aqi <= 100:
        return "yellow"
    elif aqi <= 150:
        return "orange"
    elif aqi <= 200:
        return "red"
    else:
        return "darkred"


def create_map(lat, lon, aqi, city):

    #  fallback center
    if lat is None or lon is None:
        lat, lon = 20.5937, 78.9629  # India center

    #  LOCKED MAP (no zoom, no drag, no scroll)
    m = folium.Map(
        location=[lat, lon],
        zoom_start=10,
        control_scale=False,
        zoom_control=False,
        scrollWheelZoom=False,
        doubleClickZoom=False,
        dragging=False,
        touchZoom=False,
        boxZoom=False,
        keyboard=False
    )

    #  AQI glowing marker
    folium.CircleMarker(
        location=[lat, lon],
        radius=14,
        popup=f"{city} AQI: {aqi}",
        color=get_color(aqi),
        fill=True,
        fill_opacity=0.75
    ).add_to(m)

    return m