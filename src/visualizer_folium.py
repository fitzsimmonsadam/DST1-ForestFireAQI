import os
import logging
import pandas as pd
import folium
import webbrowser
from folium.plugins import HeatMap, TimestampedGeoJson
import matplotlib.pyplot as plt
from io import BytesIO
import base64
import json

class Visualizer:
    def __init__(self, aqi_pm25_path, aqi_ozone_path, wildfire_data_path, output_dir='visuals'):
        self.aqi_pm25_path = aqi_pm25_path
        self.aqi_ozone_path = aqi_ozone_path
        self.wildfire_data_path = wildfire_data_path
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)

        # Logging setup
        logging.basicConfig(
            filename='data/logs/visualizer.log',
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s'
        )
        self.logger = logging.getLogger(__name__)
        self.logger.info("Visualizer initialized.")

        # Load Data
        self.aqi_pm25 = pd.read_csv(aqi_pm25_path)
        self.aqi_ozone = pd.read_csv(aqi_ozone_path)
        self.wildfire_data = pd.read_csv(wildfire_data_path)

        # Ensure all column names are strings
        self.aqi_pm25.columns = self.aqi_pm25.columns.astype(str)
        self.aqi_ozone.columns = self.aqi_ozone.columns.astype(str)
        self.wildfire_data.columns = self.wildfire_data.columns.astype(str)

        # Convert 'Date' to string format, handling errors gracefully
        self.wildfire_data['Date'] = pd.to_datetime(self.wildfire_data['Date'], errors='coerce').dt.strftime('%Y-%m-%d')
        self.aqi_pm25["Date"] = pd.to_datetime(self.aqi_pm25["Date"], errors='coerce').dt.strftime('%Y-%m-%d')
        self.aqi_ozone["Date"] = pd.to_datetime(self.aqi_ozone["Date"], errors='coerce').dt.strftime('%Y-%m-%d')

        # Drop NaN values in essential columns
        self.wildfire_data.dropna(subset=['latitude', 'longitude', 'Date'], inplace=True)
        self.aqi_pm25.dropna(subset=['Latitude', 'Longitude', 'AQI'], inplace=True)
        self.aqi_ozone.dropna(subset=['Latitude', 'Longitude', 'AQI'], inplace=True)

    def generate_time_series_html(self, data, station_name):
        """
        Generate a time series plot of AQI data for a station, encode as a base64 PNG, and return as HTML.
        """
        # Convert Date to datetime and sort
        data['Date'] = pd.to_datetime(data['Date'])
        data = data.sort_values('Date')
        
        fig, ax = plt.subplots(figsize=(4, 3))
        ax.plot(data['Date'], data['AQI'], marker='o', linestyle='-')
        ax.set_title(station_name)
        ax.set_xlabel("Date")
        ax.set_ylabel("AQI")
        fig.autofmt_xdate()
        
        # Save figure to a bytes buffer
        buf = BytesIO()
        plt.savefig(buf, format='png', bbox_inches='tight')
        buf.seek(0)
        image_base64 = base64.b64encode(buf.getvalue()).decode('utf-8')
        plt.close(fig)
        # Return HTML string embedding the image
        html = f'<img src="data:image/png;base64,{image_base64}" style="width:100%; height:auto;">'
        return html

    def create_interactive_map(self, year_filter=None):
        """
        Creates an interactive Folium map with a combined TimestampedGeoJson layer 
        for both AQI (PM2.5 & Ozone) and wildfire events. Each event is represented 
        as a circle marker with styling defined in the 'iconstyle' property.
        
        Args:
            year_filter (int, optional): If provided, filters data to include only that year.
        """
        try:
            self.logger.info("Creating combined time slider map for AQI and Wildfire events.")

            # Initialize map centered on Colorado
            m = folium.Map(location=[39.5501, -105.7821], zoom_start=6, tiles='cartodbpositron')

            # Filter data by year if provided
            if year_filter:
                filtered_wildfires = self.wildfire_data[self.wildfire_data["Date"].str.startswith(str(year_filter))]
                filtered_pm25 = self.aqi_pm25[self.aqi_pm25["Date"].str.startswith(str(year_filter))]
                filtered_ozone = self.aqi_ozone[self.aqi_ozone["Date"].str.startswith(str(year_filter))]
            else:
                filtered_wildfires = self.wildfire_data
                filtered_pm25 = self.aqi_pm25
                filtered_ozone = self.aqi_ozone

            # Optionally sample if datasets are very large
            MAX_WILDFIRE_POINTS = 1000
            MAX_AQI_POINTS = 1000
            sampled_wildfires = filtered_wildfires.sample(n=min(len(filtered_wildfires), MAX_WILDFIRE_POINTS), random_state=42)
            sampled_pm25 = filtered_pm25.sample(n=min(len(filtered_pm25), MAX_AQI_POINTS), random_state=42)
            sampled_ozone = filtered_ozone.sample(n=min(len(filtered_ozone), MAX_AQI_POINTS), random_state=42)

            features = []

            # --- Wildfire features ---
            for idx, row in sampled_wildfires.iterrows():
                # Convert the Date to ISO format
                date_iso = pd.to_datetime(row['Date']).isoformat() if pd.notnull(row['Date']) else ""
                feature = {
                    "type": "Feature",
                    "geometry": {
                        "type": "Point",
                        "coordinates": [float(row['longitude']), float(row['latitude'])]
                    },
                    "properties": {
                        "time": date_iso,
                        "popup": f"Wildfire on {date_iso}",
                        "layer": "wildfire",
                        "iconstyle": {
                            "fillColor": "red",
                            "fillOpacity": 0.7,
                            "stroke": False,
                            "radius": 6
                        }
                    }
                }
                features.append(feature)

            # --- PM2.5 features ---
            for idx, row in sampled_pm25.iterrows():
                date_iso = pd.to_datetime(row['Date']).isoformat() if pd.notnull(row['Date']) else ""
                try:
                    aqi_val = float(row['AQI'])
                except Exception:
                    continue
                # Determine EPA color for PM2.5 based on AQI value
                if aqi_val < 51:
                    color = "#00e400"
                elif aqi_val < 101:
                    color = "#ffff00"
                elif aqi_val < 151:
                    color = "#ff7e00"
                elif aqi_val < 201:
                    color = "#ff0000"
                elif aqi_val < 301:
                    color = "#8f3f97"
                else:
                    color = "#7e0023"
                feature = {
                    "type": "Feature",
                    "geometry": {
                        "type": "Point",
                        "coordinates": [float(row['Longitude']), float(row['Latitude'])]
                    },
                    "properties": {
                        "time": date_iso,
                        "popup": f"PM2.5 AQI: {aqi_val} on {date_iso}",
                        "layer": "aqi_pm25",
                        "AQI": aqi_val,
                        "iconstyle": {
                            "fillColor": color,
                            "fillOpacity": 0.7,
                            "stroke": False,
                            "radius": 5
                        }
                    }
                }
                features.append(feature)

            # --- Ozone features ---
            for idx, row in sampled_ozone.iterrows():
                date_iso = pd.to_datetime(row['Date']).isoformat() if pd.notnull(row['Date']) else ""
                try:
                    aqi_val = float(row['AQI'])
                except Exception:
                    continue
                if aqi_val < 51:
                    color = "#00e400"
                elif aqi_val < 101:
                    color = "#ffff00"
                elif aqi_val < 151:
                    color = "#ff7e00"
                elif aqi_val < 201:
                    color = "#ff0000"
                elif aqi_val < 301:
                    color = "#8f3f97"
                else:
                    color = "#7e0023"
                feature = {
                    "type": "Feature",
                    "geometry": {
                        "type": "Point",
                        "coordinates": [float(row['Longitude']), float(row['Latitude'])]
                    },
                    "properties": {
                        "time": date_iso,
                        "popup": f"Ozone AQI: {aqi_val} on {date_iso}",
                        "layer": "aqi_ozone",
                        "AQI": aqi_val,
                        "iconstyle": {
                            "fillColor": color,
                            "fillOpacity": 0.7,
                            "stroke": False,
                            "radius": 5
                        }
                    }
                }
                features.append(feature)

            # Combine all features into one FeatureCollection
            combined_geojson = {
                "type": "FeatureCollection",
                "features": features
            }

            # Create a single TimestampedGeoJson layer that will control both wildfire and AQI features
            ts = TimestampedGeoJson(
                combined_geojson,
                transition_time=200,
                period="P1D",
                add_last_point=True,
                auto_play=False,
                loop=False,
                max_speed=1,
                loop_button=True,
                date_options='YYYY-MM-DD',
                time_slider_drag_update=True
            )
            ts.add_to(m)

            # Add layer control (if additional layers are added later)
            folium.LayerControl(collapsed=False).add_to(m)

            year_suffix = f"_{year_filter}" if year_filter else ""
            map_path = os.path.join(self.output_dir, f"combined_time_slider_map{year_suffix}.html")
            m.save(map_path)
            self.logger.info(f"Map saved to {map_path}. Open manually: file://{os.path.abspath(map_path)}")

        except Exception as e:
            self.logger.error(f"Error creating combined time slider map: {e}")
            raise

if __name__ == "__main__":
    ozone_dp = "data/aqi_data/aqi_processed/ozone_aqi_2019_2024_30.csv"
    pm25_dp = "data/aqi_data/aqi_processed/pm25_aqi_2019_2024_30.csv"
    wildfire_dp = "data/wildfire_data/wildfire_processed/wildfire_processed_2019_2024_n.csv"

    visualizer = Visualizer(aqi_pm25_path=pm25_dp, aqi_ozone_path=ozone_dp, wildfire_data_path=wildfire_dp)
    visualizer.create_interactive_map(year_filter=2020)