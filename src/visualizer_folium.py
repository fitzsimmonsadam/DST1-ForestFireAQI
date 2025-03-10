import os
import logging
import pandas as pd
import folium
import webbrowser
from folium.plugins import HeatMap, HeatMapWithTime, TimestampedGeoJson, FastMarkerCluster
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

        # Convert dates to datetime then format as YYYY-MM-DD strings
        self.wildfire_data['Date'] = pd.to_datetime(self.wildfire_data['Date'], errors='coerce').dt.strftime('%Y-%m-%d')
        self.aqi_pm25["Date"] = pd.to_datetime(self.aqi_pm25["Date"], errors='coerce').dt.strftime('%Y-%m-%d')
        self.aqi_ozone["Date"] = pd.to_datetime(self.aqi_ozone["Date"], errors='coerce').dt.strftime('%Y-%m-%d')

        # Drop rows missing essential values
        self.wildfire_data.dropna(subset=['latitude', 'longitude', 'Date'], inplace=True)
        self.aqi_pm25.dropna(subset=['Latitude', 'Longitude', 'AQI'], inplace=True)
        self.aqi_ozone.dropna(subset=['Latitude', 'Longitude', 'AQI'], inplace=True)

    def add_wildfire_full_year_heatmap(self, m, filtered_wildfires):
        """Add a static full-year wildfire heatmap to the map."""
        wf_coords = filtered_wildfires[['latitude', 'longitude']].values.tolist()
        wf_coords = [[float(lat), float(lon)] for lat, lon in wf_coords]
        HeatMap(wf_coords, radius=15, blur=10, 
                gradient={"0.2": "yellow", "0.4": "orange", "0.6": "red"}).add_to(m)
    
    def add_wildfire_animated_heatmap(self, m, filtered_wildfires):
        """Group wildfire data by month and add an animated heatmap."""
        wf = filtered_wildfires.copy()
        wf['Date'] = pd.to_datetime(wf['Date'])
        wf['Month'] = wf['Date'].dt.strftime('%Y-%m')
        unique_months = sorted(wf['Month'].unique())
        # Convert each month into a ISO timestamp
        unique_months_iso = [pd.to_datetime(month + "-01").isoformat() for month in unique_months]
        
        wildfire_data_by_month = []
        for month in unique_months:
            coords = wf[wf['Month'] == month][['latitude', 'longitude']].values.tolist()
            coords = [[float(lat), float(lon)] for lat, lon in coords]
            wildfire_data_by_month.append(coords)
        hm_time = HeatMapWithTime(
            data=wildfire_data_by_month,
            index=unique_months_iso,
            radius=15,
            auto_play=False,
            max_opacity=0.8,
            gradient={"0.2": "yellow", "0.4": "orange", "0.6": "red"}
            # Note: No extra parameters causing recursion are added here.
        )
        hm_time.add_to(m)

    def add_wildfire_cluster(self, m, filtered_wildfires):
        """Add wildfire markers as clusters using FastMarkerCluster."""
        coords = filtered_wildfires[['latitude', 'longitude']].values.tolist()
        coords = [[float(lat), float(lon)] for lat, lon in coords]
        FastMarkerCluster(coords).add_to(m)

    def generate_time_series_html(self, data, station_name):
        """Generate a time series plot for a station as a base64 PNG embedded in HTML."""
        data['Date'] = pd.to_datetime(data['Date'])
        data = data.sort_values('Date')
        fig, ax = plt.subplots(figsize=(8, 6))
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
        """Add static circle markers for AQI stations with time series popups."""
        # For PM2.5 stations, group by SiteName, Latitude, and Longitude
        grouped_pm25 = filtered_pm25.groupby(['SiteName', 'Latitude', 'Longitude'])
        for (site, lat, lon), group in grouped_pm25:
            popup_html = self.generate_time_series_html(group.copy(), site)
            popup = folium.Popup(popup_html, max_width=500)
            folium.CircleMarker(
                location=[float(lat), float(lon)],
                radius=6,
                color="blue",
                fill=True,
                fill_color="black",
                fill_opacity=0.5,
                popup=popup
            ).add_to(m)
        # For Ozone stations, group by SiteName, Latitude, and Longitude
        grouped_ozone = filtered_ozone.groupby(['SiteName', 'Latitude', 'Longitude'])
        for (site, lat, lon), group in grouped_ozone:
            popup_html = self.generate_time_series_html(group.copy(), site)
            popup = folium.Popup(popup_html, max_width=500)
            folium.CircleMarker(
                location=[float(lat), float(lon)],
                radius=6,
                color="green",
                fill=True,
                fill_color="black",
                fill_opacity=0.5,
                popup=popup
            ).add_to(m)
        
    def add_animated_aqi_markers(self, m, filtered_pm25, filtered_ozone):
        """(Optional) Add animated AQI markers using TimestampedGeoJson.
           This layer uses monthly timestamps and AQI_Category for colors."""
        features = []
        def get_station_name(row):
            return row.get('SiteName', f"Station at ({row['Latitude']}, {row['Longitude']})")
    
        # Process PM2.5 data
        for idx, row in filtered_pm25.iterrows():
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
            period="P1M",  # Monthly period
            transition_time=200,
            auto_play=False,
            loop=False,
            time_slider_drag_update=False,
            date_options='YYYY-MM'
        )
        ts_aqi.add_to(m)
    
    def create_static_map(self, year_filter=None):
        """Creates a static map with full-year wildfire heatmap and static AQI station markers."""
        try:
            self.logger.info("Creating the static map.")
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

            # Add static wildfire full-year heatmap
            wf_heatmap_layer = folium.FeatureGroup(name="Wildfire Full-Year Heatmap", overlay=True)
            wf_coords = filtered_wildfires[['latitude', 'longitude']].values.tolist()
            wf_coords = [[float(lat), float(lon)] for lat, lon in wf_coords]
            HeatMap(wf_coords, radius=15, blur=10, 
                    gradient={"0.2": "yellow", "0.4": "orange", "0.6": "red"}).add_to(wf_heatmap_layer)
            wf_heatmap_layer.add_to(m)

            # Add static AQI station markers
            aqi_station_layer = folium.FeatureGroup(name="Static AQI Station Markers", overlay=True)
            self.add_static_aqi_station_markers(aqi_station_layer, filtered_pm25, filtered_ozone)
            aqi_station_layer.add_to(m)

            folium.LayerControl(collapsed=False).add_to(m)
            year_suffix = f"_{year_filter}" if year_filter else ""
            map_path = os.path.join(self.output_dir, f"static_map{year_suffix}.html")
            m.save(map_path)
            self.logger.info(f"Static map saved to {map_path}.")
        except Exception as e:
            self.logger.error(f"Error creating static map: {e}")
            raise

    def create_animated_map(self, year_filter=None):
        """Creates an animated map with a monthly wildfire heatmap.
           (Animated AQI markers can be added if desired.)"""
        try:
            self.logger.info("Creating the animated map.")
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

            # Add animated wildfire heatmap
            self.add_wildfire_animated_heatmap(m, filtered_wildfires)

            # (Optional) Add animated AQI markers if needed:
            self.add_animated_aqi_markers(m, filtered_pm25, filtered_ozone)

            folium.LayerControl(collapsed=False).add_to(m)
            year_suffix = f"_{year_filter}" if year_filter else ""
            map_path = os.path.join(self.output_dir, f"animated_map{year_suffix}.html")
            m.save(map_path)
            self.logger.info(f"Animated map saved to {map_path}.")
        except Exception as e:
            self.logger.error(f"Error creating animated map: {e}")
            raise

if __name__ == "__main__":
    # Global AQI color mapping using AQI_Category values
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
    # Create a static map (with full-year wildfire heatmap and static station markers)
    visualizer.create_static_map(year_filter=2020)
    # Create an animated map (with monthly animated wildfire heatmap)
    #visualizer.create_animated_map(year_filter=2020)