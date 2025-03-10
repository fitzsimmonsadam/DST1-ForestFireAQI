import os
import logging
import pandas as pd
import folium
import webbrowser
from folium.plugins import HeatMap, FastMarkerCluster

class Visualizer:
    def __init__(self, aqi_pm25_path, aqi_ozone_path, wildfire_data_path, output_dir='visuals'):
        self.aqi_pm25_path = aqi_pm25_path
        self.aqi_ozone_path = aqi_ozone_path
        self.wildfire_data_path = wildfire_data_path
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)

        # Setup logging
        logging.basicConfig(
            filename='data/logs/visualizer.log',
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s'
        )
        self.logger = logging.getLogger(__name__)
        self.logger.info("Visualizer initialized.")

        # Load aqi and wildfire data
        self.aqi_pm25 = pd.read_csv(aqi_pm25_path)
        self.aqi_ozone = pd.read_csv(aqi_ozone_path)
        self.wildfire_data = pd.read_csv(wildfire_data_path)

        # Set column names as strings
        self.aqi_pm25.columns = self.aqi_pm25.columns.astype(str)
        self.aqi_ozone.columns = self.aqi_ozone.columns.astype(str)
        self.wildfire_data.columns = self.wildfire_data.columns.astype(str)

        # Convert dates to datetime
        self.wildfire_data['Date'] = pd.to_datetime(self.wildfire_data['Date'], errors='coerce').dt.strftime('%Y-%m-%d')
        self.aqi_pm25["Date"] = pd.to_datetime(self.aqi_pm25["Date"], errors='coerce').dt.strftime('%Y-%m-%d')
        self.aqi_ozone["Date"] = pd.to_datetime(self.aqi_ozone["Date"], errors='coerce').dt.strftime('%Y-%m-%d')

        # Drop rows missing values
        self.wildfire_data.dropna(subset=['latitude', 'longitude', 'Date'], inplace=True)
        self.aqi_pm25.dropna(subset=['Latitude', 'Longitude', 'AQI'], inplace=True)
        self.aqi_ozone.dropna(subset=['Latitude', 'Longitude', 'AQI'], inplace=True)

    def create_interactive_map(self, year_filter=None):
        """
        Creates an interactive Folium map with AQI stations, wildfire markers, and heatmaps.
        
        Args:
            year_filter (int, optional): If provided, filters data to include only that year.
        """
        try:
            self.logger.info("Creating interactive Folium map.")

            # Initialize Map Centered on Colorado
            m = folium.Map(location=[39.5501, -105.7821], zoom_start=6, tiles='cartodbpositron')

            # Feature Groups for Layer Control
            wildfire_layer = folium.FeatureGroup(name="Wildfires", overlay=True)
            aqi_pm25_layer = folium.FeatureGroup(name="PM2.5 AQI Stations", overlay=True)
            aqi_ozone_layer = folium.FeatureGroup(name="Ozone AQI Stations", overlay=True)
            wildfire_heatmap = folium.FeatureGroup(name="Wildfire Heatmap", overlay=True)
            aqi_pm25_heatmap = folium.FeatureGroup(name="PM2.5 Heatmap", overlay=True)
            aqi_ozone_heatmap = folium.FeatureGroup(name="Ozone Heatmap", overlay=True)

            # Apply year filter if provided
            if year_filter:
                filtered_wildfires = self.wildfire_data[self.wildfire_data["Date"].str.startswith(str(year_filter))]
                filtered_pm25 = self.aqi_pm25[self.aqi_pm25["Date"].str.startswith(str(year_filter))]
                filtered_ozone = self.aqi_ozone[self.aqi_ozone["Date"].str.startswith(str(year_filter))]
            else:
                filtered_wildfires = self.wildfire_data
                filtered_pm25 = self.aqi_pm25
                filtered_ozone = self.aqi_ozone

            # Limit markers to speed up rendering
            MAX_MARKERS = 1000
            sampled_wildfires = filtered_wildfires.sample(n=min(len(filtered_wildfires), MAX_MARKERS), random_state=42)
            sampled_pm25 = filtered_pm25.sample(n=min(len(filtered_pm25), MAX_MARKERS), random_state=42)
            sampled_ozone = filtered_ozone.sample(n=min(len(filtered_ozone), MAX_MARKERS), random_state=42)

            # Wildfire Marker Cluster
            FastMarkerCluster(sampled_wildfires[['latitude', 'longitude']].values.tolist()).add_to(wildfire_layer)

            # Wildfire Heatmap
            wildfire_heat_data = sampled_wildfires[['latitude', 'longitude']].values.tolist()
            if wildfire_heat_data:
                HeatMap(wildfire_heat_data, radius=15, blur=10, gradient={str(0.2): "yellow", str(0.4): "orange", str(0.6): "red"}).add_to(wildfire_heatmap)

            # PM2.5 AQI Stations
            FastMarkerCluster(sampled_pm25[['Latitude', 'Longitude']].values.tolist()).add_to(aqi_pm25_layer)

            # Ozone AQI Stations
            FastMarkerCluster(sampled_ozone[['Latitude', 'Longitude']].values.tolist()).add_to(aqi_ozone_layer)

            # PM2.5 Heatmap
            pm25_heat_data = sampled_pm25[['Latitude', 'Longitude', 'AQI']].dropna().values.tolist()
            if pm25_heat_data:
                HeatMap(pm25_heat_data, radius=12, blur=8, gradient={str(0.2): "blue", str(0.4): "cyan", str(0.6): "purple"}).add_to(aqi_pm25_heatmap)

            # Ozone Heatmap (Fix NaN Values)
            ozone_heat_data = sampled_ozone[['Latitude', 'Longitude', 'AQI']].dropna().values.tolist()
            if ozone_heat_data:
                HeatMap(ozone_heat_data, radius=12, blur=8, gradient={str(0.2): "green", str(0.4): "yellow", str(0.6): "red"}).add_to(aqi_ozone_heatmap)
            
            # Add Layers to Map
            wildfire_layer.add_to(m)
            aqi_pm25_layer.add_to(m)
            aqi_ozone_layer.add_to(m)
            wildfire_heatmap.add_to(m)
            aqi_pm25_heatmap.add_to(m)
            aqi_ozone_heatmap.add_to(m)

            # Layer Control
            folium.LayerControl(collapsed=False).add_to(m)

            year_suffix = f"_{year_filter}" if year_filter else ""
            map_path = os.path.join(self.output_dir, f"filtered_wildfire_aqi_map{year_suffix}.html")
            m.save(map_path)
            self.logger.info(f"Map saved to {map_path}. Open manually: file://{os.path.abspath(map_path)}")
        except Exception as e:
            self.logger.error(f"Error creating Folium map: {e}")
            raise

if __name__ == "__main__":
    aqi_color_map = {
        "Good": "#00e400",
        "Moderate": "#ffff00",
        "Unhealthy for Sensitive Groups": "#ff7e00",
        "Unhealthy": "#ff0000",
        "Very Unhealthy": "#8f3f97",
        "Hazardous": "#7e0023",
        "Unknown": "#000000"
        }
    
    ozone_dp = "data/aqi_data/aqi_processed/ozone_aqi_2019_2024.csv"
    pm25_dp = "data/aqi_data/aqi_processed/pm25_aqi_2019_2024.csv"
    wildfire_dp = "data/wildfire_data/wildfire_processed/wildfire_processed_2019_2024_n.csv"

    visualizer = Visualizer(aqi_pm25_path=pm25_dp, aqi_ozone_path=ozone_dp, wildfire_data_path=wildfire_dp)
    visualizer.create_interactive_map(year_filter=2020)