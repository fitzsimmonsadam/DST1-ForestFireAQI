import os
import logging
import pandas as pd
import folium
import webbrowser
from folium.plugins import HeatMap, HeatMapWithTime, TimestampedGeoJson
import matplotlib.pyplot as plt
from io import BytesIO
import base64

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

    def add_wildfire_full_year_heatmap(self, m, filtered_wildfires):
        """This layer shows aggregated wildfire data over entire year."""
        wf_coords = filtered_wildfires[['latitude', 'longitude']].values.tolist()
        wf_coords = [[float(lat), float(lon)] for lat, lon in wf_coords]
        HeatMap(wf_coords, radius=15, blur=10, 
                gradient={"0.2": "yellow", "0.4": "orange", "0.6": "red"}).add_to(m)
    
    def add_wildfire_animated_heatmap(self, m, filtered_wildfires):
        """Group the wildfire data by month and create an animated heatmap."""
        wf = filtered_wildfires.copy()
        wf['Date'] = pd.to_datetime(wf['Date'])
        wf['Month'] = wf['Date'].dt.strftime('%Y-%m')
        unique_months = sorted(wf['Month'].unique())
        wildfire_data_by_month = []
        for month in unique_months:
            coords = wf[wf['Month'] == month][['latitude', 'longitude']].values.tolist()
            # Ensure values are floats
            coords = [[float(lat), float(lon)] for lat, lon in coords]
            wildfire_data_by_month.append(coords)
        # Animated heatmap layer for wildfires
        hm_time = HeatMapWithTime(
            data=wildfire_data_by_month,
            index=unique_months,
            radius=15,
            auto_play=False,
            max_opacity=0.8,
            gradient={"0.2": "yellow", "0.4": "orange", "0.6": "red"},
            name="Wildfire Animated Heatmap"
        )
        hm_time.add_to(m)
    
    def generate_time_series_html(self, data, station_name):
        """Generate a time series plot of AQI data by station."""
        data['Date'] = pd.to_datetime(data['Date'])
        data = data.sort_values('Date')
        fig, ax = plt.subplots(figsize=(4, 3))
        ax.plot(data['Date'], data['AQI'], marker='o', linestyle='-')
        ax.set_title(station_name)
        ax.set_xlabel("Date")
        ax.set_ylabel("AQI")
        fig.autofmt_xdate()
        buf = BytesIO()
        plt.savefig(buf, format='png', bbox_inches='tight')
        buf.seek(0)
        image_base64 = base64.b64encode(buf.getvalue()).decode('utf-8')
        plt.close(fig)
        return f'<img src="data:image/png;base64,{image_base64}" style="width:100%; height:auto;">'

    def add_static_aqi_station_markers(self, m, filtered_pm25, filtered_ozone):
        """Add circle markers for the fixed AQI station locations."""
        # For PM2.5 stations:
        grouped_pm25 = filtered_pm25.groupby(['SiteName', 'Latitude', 'Longitude'])
        for (site, lat, lon), group in grouped_pm25:
            popup_html = self.generate_time_series_html(group.copy(), site)
            #Circle markers for pm2.5 stations black with blue outline
            folium.CircleMarker(
                location=[float(lat), float(lon)],
                radius=6,
                color="blue",
                fill=True,
                fill_color="black",
                fill_opacity=0.8,
                popup=popup_html
            ).add_to(m)
        # For Ozone stations
        grouped_ozone = filtered_ozone.groupby(['SiteName', 'Latitude', 'Longitude'])
        for (site, lat, lon), group in grouped_ozone:
            popup_html = self.generate_time_series_html(group.copy(), site)
            #Circle markers for ozone stations black with green outline
            folium.CircleMarker(
                location=[float(lat), float(lon)],
                radius=6,
                color="green",
                fill=True,
                fill_color="black",
                fill_opacity=0.8,
                popup=popup_html
            ).add_to(m)
        
    def add_animated_aqi_markers(self, m, filtered_pm25, filtered_ozone):
        """Creates feature for each AQI reading that belongs to a given month. 
            The markers color is determined by set EPA thresholds."""
        features = []
        def get_station_name(row):
            return row.get('SiteName', f"Station at ({row['Latitude']}, {row['Longitude']})")
    
        # Process PM2.5 data
        for idx, row in filtered_pm25.iterrows():
            # Create a month-based timestamp
            try:
                date_obj = pd.to_datetime(row['Date']).replace(day=1)
            except Exception:
                continue
            time_str = date_obj.isoformat()
            color = aqi_color_map.get(row.get('AQI_Category', "Unknown"), "#000000")
            station_name = get_station_name(row)
            feature = {
                "type": "Feature",
                "geometry": {
                    "type": "Point",
                    "coordinates": [float(row['Longitude']), float(row['Latitude'])]
                },
                "properties": {
                    "time": time_str,
                    "popup": f"{station_name}<br>PM2.5 AQI: {row['AQI']} ({row.get('AQI_Category', 'Unknown')}) on {time_str}",
                    "icon": "circle",
                    "iconstyle": {
                        "fillColor": color,
                        "fillOpacity": 0.8,
                        "stroke": False,
                        "radius": 5
                    }
                }
            }
            features.append(feature)

        # Process Ozone data
        for idx, row in filtered_ozone.iterrows():
            try:
                date_obj = pd.to_datetime(row['Date']).replace(day=1)
            except Exception:
                continue
            time_str = date_obj.isoformat()
            color = aqi_color_map.get(row.get('AQI_Category', "Unknown"), "#000000")
            station_name = get_station_name(row)
            feature = {
                "type": "Feature",
                "geometry": {
                    "type": "Point",
                    "coordinates": [float(row['Longitude']), float(row['Latitude'])]
                },
                "properties": {
                    "time": time_str,
                    "popup": f"{station_name}<br>Ozone AQI: {row['AQI']} ({row.get('AQI_Category', 'Unknown')}) on {time_str}",
                    "icon": "circle",
                    "iconstyle": {
                        "fillColor": color,
                        "fillOpacity": 0.8,
                        "stroke": False,
                        "radius": 5
                    }
                }
            }
            features.append(feature)

        aqi_geojson = {
            "type": "FeatureCollection",
            "features": features
        }
        ts_aqi = TimestampedGeoJson(
            aqi_geojson,
            transition_time=200,
            period="P1M",  # Monthly period
            add_last_point=True,
            auto_play=False,
            loop=False,
            max_speed=1,
            loop_button=True,
            date_options='YYYY-MM',
            time_slider_drag_update=True
        )
        ts_aqi.add_to(m)
    
    def create_interactive_map(self, year_filter=None):
        """Creates the final interactive map with all layers."""
        try:
            self.logger.info("Creating the final interactive map.")

            # Initialize the map centered on Colorado
            m = folium.Map(location=[39.5501, -105.7821], zoom_start=6, tiles='cartodbpositron')

            # Filter data by year if specified
            if year_filter:
                filtered_wildfires = self.wildfire_data[self.wildfire_data["Date"].str.startswith(str(year_filter))]
                filtered_pm25 = self.aqi_pm25[self.aqi_pm25["Date"].str.startswith(str(year_filter))]
                filtered_ozone = self.aqi_ozone[self.aqi_ozone["Date"].str.startswith(str(year_filter))]
            else:
                filtered_wildfires = self.wildfire_data
                filtered_pm25 = self.aqi_pm25
                filtered_ozone = self.aqi_ozone

            # --- Wildfire Layers ---
            # Full-year wildfire heatmap (static)
            wf_heatmap_layer = folium.FeatureGroup(name="Wildfire Full-Year Heatmap", overlay=True)
            wf_coords = filtered_wildfires[['latitude', 'longitude']].values.tolist()
            wf_coords = [[float(lat), float(lon)] for lat, lon in wf_coords]
            HeatMap(wf_coords, radius=15, blur=10, 
                    gradient={"0.2": "yellow", "0.4": "orange", "0.6": "red"}).add_to(wf_heatmap_layer)
            wf_heatmap_layer.add_to(m)

            # Animated wildfire heatmap (monthly)
            self.add_wildfire_animated_heatmap(m, filtered_wildfires)

            # --- AQI Layers ---
            # Static AQI station markers (for both PM2.5 and Ozone)
            aqi_station_layer = folium.FeatureGroup(name="Static AQI Station Markers", overlay=True)
            self.add_static_aqi_station_markers(aqi_station_layer, filtered_pm25, filtered_ozone)
            aqi_station_layer.add_to(m)

            # Animated AQI markers (monthly)
            self.add_animated_aqi_markers(m, filtered_pm25, filtered_ozone)

            # --- Layer Control ---
            folium.LayerControl(collapsed=False).add_to(m)

            year_suffix = f"_{year_filter}" if year_filter else ""
            map_path = os.path.join(self.output_dir, f"final_interactive_map{year_suffix}.html")
            m.save(map_path)
            self.logger.info(f"Map saved to {map_path}. Open manually: file://{os.path.abspath(map_path)}")
        except Exception as e:
            self.logger.error(f"Error creating interactive map: {e}")
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
