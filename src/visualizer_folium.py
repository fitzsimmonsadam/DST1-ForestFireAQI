import os
import logging
import pandas as pd
import geopandas as gpd
import folium
import webbrowser
from folium.plugins import HeatMap, MarkerCluster

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

        # Ensure date format consistency
        self.wildfire_data['Date'] = pd.to_datetime(self.wildfire_data['Date']).dt.strftime('%Y-%m-%d')
        self.aqi_pm25["Date"] = pd.to_datetime(self.aqi_pm25["Date"]).dt.strftime('%Y-%m-%d')
        self.aqi_ozone["Date"] = pd.to_datetime(self.aqi_ozone["Date"]).dt.strftime('%Y-%m-%d')
    
    def create_interactive_map(self):
        """Creates an interactive Folium map with AQI stations, wildfire markers, and heatmaps."""
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

            # Wildfire Marker Cluster
            wildfire_cluster = MarkerCluster(name="Wildfire Clusters").add_to(wildfire_layer)
            for _, row in self.wildfire_data.iterrows():
                folium.CircleMarker(
                    location=[row['latitude'], row['longitude']],
                    radius=6,
                    color="red",
                    fill=True,
                    fill_color="red",
                    fill_opacity=0.7,
                    popup=folium.Popup(
                        f"Date: {row['Date']}<br>FRP: {row.get('frp', 'N/A')}", max_width=250),
                    tooltip=f"Wildfire: {row['Date']}"
                ).add_to(wildfire_cluster)

            # Wildfire Heatmap
            wildfire_heat_data = self.wildfire_data[['latitude', 'longitude']].values.tolist()
            if wildfire_heat_data:
                HeatMap(wildfire_heat_data, radius=15, blur=10, gradient={0.2: "yellow", 0.4: "orange", 0.6: "red"}).add_to(wildfire_heatmap)

            # PM2.5 AQI Station Markers
            aqi_pm25_cluster = MarkerCluster(name="PM2.5 AQI Clusters").add_to(aqi_pm25_layer)
            for _, row in self.aqi_pm25.iterrows():
                folium.CircleMarker(
                    location=[row['Latitude'], row['Longitude']],
                    radius=6,
                    color="blue",
                    fill=True,
                    fill_color="blue",
                    fill_opacity=0.7,
                    popup=folium.Popup(
                        f"Date: {row['Date']}<br>AQI: {row['AQI']}", max_width=250),
                    tooltip=f"PM2.5 AQI: {row['AQI']}"
                ).add_to(aqi_pm25_cluster)

            # Ozone AQI Station Markers
            aqi_ozone_cluster = MarkerCluster(name="Ozone AQI Clusters").add_to(aqi_ozone_layer)
            for _, row in self.aqi_ozone.iterrows():
                folium.CircleMarker(
                    location=[row['Latitude'], row['Longitude']],
                    radius=6,
                    color="green",
                    fill=True,
                    fill_color="green",
                    fill_opacity=0.7,
                    popup=folium.Popup(
                        f"Date: {row['Date']}<br>AQI: {row['AQI']}", max_width=250),
                    tooltip=f"Ozone AQI: {row['AQI']}"
                ).add_to(aqi_ozone_cluster)

            # Heatmaps for AQI Density
            pm25_heat_data = self.aqi_pm25[['Latitude', 'Longitude', 'AQI']].values.tolist()
            if pm25_heat_data:
                HeatMap(pm25_heat_data, radius=12, blur=8, gradient={0.2: "blue", 0.4: "cyan", 0.6: "purple"}).add_to(aqi_pm25_heatmap)

            ozone_heat_data = self.aqi_ozone[['Latitude', 'Longitude', 'AQI']].values.tolist()
            if ozone_heat_data:
                HeatMap(ozone_heat_data, radius=12, blur=8, gradient={0.2: "green", 0.4: "yellow", 0.6: "red"}).add_to(aqi_ozone_heatmap)

            # Add Layers to Map
            wildfire_layer.add_to(m)
            aqi_pm25_layer.add_to(m)
            aqi_ozone_layer.add_to(m)
            wildfire_heatmap.add_to(m)
            aqi_pm25_heatmap.add_to(m)
            aqi_ozone_heatmap.add_to(m)

            # Layer Control
            folium.LayerControl(collapsed=False).add_to(m)

            # Save Map
            map_path = os.path.join(self.output_dir, "enhanced_wildfire_aqi_map.html")
            m.save(map_path)
            self.logger.info(f"Map saved to {map_path}")
            webbrowser.open_new_tab(os.path.abspath(map_path))

        except Exception as e:
            self.logger.error(f"Error creating Folium map: {e}")
            raise

if __name__ == "__main__":
    # File paths
    ozone_dp = "data/aqi_data/aqi_processed/ozone_aqi_2019_2024_30.csv"
    pm25_dp = "data/aqi_data/aqi_processed/pm25_aqi_2019_2024_30.csv"
    wildfire_dp = "data/wildfire_data/wildfire_processed/wildfire_processed_2019_2024_n.csv"

    # Initialize Visualizer
    visualizer = Visualizer(
        aqi_pm25_path=pm25_dp,
        aqi_ozone_path=ozone_dp,
        wildfire_data_path=wildfire_dp
    )

    # Generate Folium Map
    visualizer.create_interactive_map()