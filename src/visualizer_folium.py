import os
import logging
import pandas as pd
import folium
import webbrowser
from folium.plugins import HeatMap, HeatMapWithTime, TimestampedGeoJson, MarkerCluster
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

        # Load Data
        self.aqi_pm25 = pd.read_csv(aqi_pm25_path)
        self.aqi_ozone = pd.read_csv(aqi_ozone_path)
        self.wildfire_data = pd.read_csv(wildfire_data_path)

        # Ensure column names are strings
        self.aqi_pm25.columns = self.aqi_pm25.columns.astype(str)
        self.aqi_ozone.columns = self.aqi_ozone.columns.astype(str)
        self.wildfire_data.columns = self.wildfire_data.columns.astype(str)

        # Convert 'Date' to string format (YYYY-MM-DD)
        self.wildfire_data['Date'] = pd.to_datetime(self.wildfire_data['Date'], errors='coerce').dt.strftime('%Y-%m-%d')
        self.aqi_pm25["Date"] = pd.to_datetime(self.aqi_pm25["Date"], errors='coerce').dt.strftime('%Y-%m-%d')
        self.aqi_ozone["Date"] = pd.to_datetime(self.aqi_ozone["Date"], errors='coerce').dt.strftime('%Y-%m-%d')

        # Drop rows missing essential values
        self.wildfire_data.dropna(subset=['latitude', 'longitude', 'Date'], inplace=True)
        self.aqi_pm25.dropna(subset=['Latitude', 'Longitude', 'AQI'], inplace=True)
        self.aqi_ozone.dropna(subset=['Latitude', 'Longitude', 'AQI'], inplace=True)

    # -------------------------------------------------------------------------
    # Existing Methods
    # -------------------------------------------------------------------------
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
        )
        hm_time.add_to(m)

    def add_wildfire_cluster_numbered(self, m, filtered_wildfires):
        """Add wildfire markers as a numbered cluster using MarkerCluster."""
        coords = filtered_wildfires[['latitude', 'longitude']].values.tolist()
        coords = [[float(lat), float(lon)] for lat, lon in coords]
        marker_cluster = MarkerCluster(name="Wildfire Cluster")
        for lat, lon in coords:
            folium.Marker(location=[lat, lon]).add_to(marker_cluster)
        marker_cluster.add_to(m)

    def generate_time_series_html(self, data, station_name):
        """Generate a time series plot for a station as a base64 PNG embedded in HTML."""
        data['Date'] = pd.to_datetime(data['Date'])
        data = data.sort_values('Date')

        fig, ax = plt.subplots(figsize=(8, 6), dpi=100)

        # Add AQI threshold bands
        ax.axhspan(0, 50, facecolor="#00e400", alpha=0.3)     
        ax.axhspan(50, 100, facecolor="#ffff00", alpha=0.3)    
        ax.axhspan(100, 150, facecolor="#ff7e00", alpha=0.3)   
        ax.axhspan(150, 200, facecolor="#ff0000", alpha=0.3)   
        ax.axhspan(200, 300, facecolor="#8f3f97", alpha=0.3)   
        ax.axhspan(300, 500, facecolor="#7e0023", alpha=0.3)   
        
        ax.plot(data['Date'], data['AQI'], marker='o', linestyle='-', color="black")
        ax.set_title(station_name, fontsize=18)
        ax.set_xlabel("Date", fontsize=14)
        ax.set_ylabel("AQI", fontsize=14)
        ax.grid(True)
        ax.set_ylim(0,250)
        fig.tight_layout()

        buf = BytesIO()
        plt.savefig(buf, format='png', bbox_inches='tight')
        buf.seek(0)
        image_base64 = base64.b64encode(buf.getvalue()).decode('utf-8')
        plt.close(fig)
        return f'<img src="data:image/png;base64,{image_base64}" style="width:100%; height:auto;">'

    def add_static_aqi_station_markers(self, m, filtered_pm25, filtered_ozone):
        """Add static circle markers for AQI stations with time series popups."""
        grouped_pm25 = filtered_pm25.groupby(['SiteName', 'Latitude', 'Longitude'])
        for (site, lat, lon), group in grouped_pm25:
            popup_html = self.generate_time_series_html(group.copy(), site)
            iframe = folium.IFrame(html=popup_html, width=500, height=400)
            popup = folium.Popup(iframe, max_width=600)
            folium.CircleMarker(
                location=[float(lat), float(lon)],
                radius=6,
                color="blue",
                fill=True,
                fill_color="black",
                fill_opacity=0.5,
                popup=popup
            ).add_to(m)

        grouped_ozone = filtered_ozone.groupby(['SiteName', 'Latitude', 'Longitude'])
        for (site, lat, lon), group in grouped_ozone:
            popup_html = self.generate_time_series_html(group.copy(), site)
            iframe = folium.IFrame(html=popup_html, width=500, height=400)
            popup = folium.Popup(iframe, max_width=600)
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
        """Add animated AQI markers using TimestampedGeoJson (monthly)."""
        features = []
        def get_station_name(row):
            return row.get('SiteName', f"Station at ({row['Latitude']}, {row['Longitude']})")
    
        # PM2.5
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

        # Ozone
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
            period="P1M", 
            transition_time=200,
            auto_play=False,
            loop=False,
            time_slider_drag_update=False,
            date_options='YYYY-MM'
        )
        ts_aqi.add_to(m)

    def create_static_map(self, year_filter=None):
        """Existing static map example with a full-year heatmap + static stations."""
        try:
            self.logger.info("Creating the static map.")
            m = folium.Map(location=[39.5501, -105.7821], zoom_start=6, tiles='cartodbpositron')

            if year_filter:
                filtered_wildfires = self.wildfire_data[self.wildfire_data["Date"].str.startswith(str(year_filter))]
                filtered_pm25 = self.aqi_pm25[self.aqi_pm25["Date"].str.startswith(str(year_filter))]
                filtered_ozone = self.aqi_ozone[self.aqi_ozone["Date"].str.startswith(str(year_filter))]
            else:
                filtered_wildfires = self.wildfire_data
                filtered_pm25 = self.aqi_pm25
                filtered_ozone = self.aqi_ozone

            # Full-year wildfire heatmap
            wf_heatmap_layer = folium.FeatureGroup(name="Full Year Wildfire Heatmap", overlay=True)
            self.add_wildfire_full_year_heatmap(m, filtered_wildfires)
            wf_heatmap_layer.add_to(m)

            # Static AQI stations
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
        """Existing map with monthly wildfire heatmap + optional monthly AQI markers."""
        try:
            self.logger.info("Creating the animated map.")
            m = folium.Map(location=[39.5501, -105.7821], zoom_start=6, tiles='cartodbpositron')

            if year_filter:
                filtered_wildfires = self.wildfire_data[self.wildfire_data["Date"].str.startswith(str(year_filter))]
                filtered_pm25 = self.aqi_pm25[self.aqi_pm25["Date"].str.startswith(str(year_filter))]
                filtered_ozone = self.aqi_ozone[self.aqi_ozone["Date"].str.startswith(str(year_filter))]
            else:
                filtered_wildfires = self.wildfire_data
                filtered_pm25 = self.aqi_pm25
                filtered_ozone = self.aqi_ozone

            # Animated wildfire heatmap
            self.add_wildfire_animated_heatmap(m, filtered_wildfires)

            # Animated AQI (monthly)
            self.add_animated_aqi_markers(m, filtered_pm25, filtered_ozone)

            folium.LayerControl(collapsed=False).add_to(m)
            year_suffix = f"_{year_filter}" if year_filter else ""
            map_path = os.path.join(self.output_dir, f"animated_map{year_suffix}.html")
            m.save(map_path)
            self.logger.info(f"Animated map saved to {map_path}.")
        except Exception as e:
            self.logger.error(f"Error creating animated map: {e}")
            raise

    def create_monthly_map(self, year_filter=None):
        """
        Unifies wildfires, PM2.5, and Ozone data into a single TimestampedGeoJson
        so that one time slider controls all monthly data (Approach One).
        
        - Wildfire data is shown as red circle markers (monthly).
        - PM2.5 and Ozone data are shown as colored markers based on EPA categories (monthly).
        No heatmap is generated to avoid recursion issues.
        """
        try:
            self.logger.info("Creating a single monthly time slider map (unified approach).")
            m = folium.Map(location=[39.5501, -105.7821], zoom_start=6, tiles='cartodbpositron')

            # 1) Filter data by year if provided
            if year_filter:
                wf_df = self.wildfire_data[self.wildfire_data["Date"].str.startswith(str(year_filter))]
                pm25_df = self.aqi_pm25[self.aqi_pm25["Date"].str.startswith(str(year_filter))]
                ozone_df = self.aqi_ozone[self.aqi_ozone["Date"].str.startswith(str(year_filter))]
            else:
                wf_df = self.wildfire_data
                pm25_df = self.aqi_pm25
                ozone_df = self.aqi_ozone

            # 2) Convert each DataFrame to monthly point features with a "time" property
            #    We'll unify them in a single list of features.
            features = []

            # 2a) Wildfire features (no heatmap, just points). We'll color them red.
            #     Month is the first day of that month for the time slider.
            for idx, row in wf_df.iterrows():
                try:
                    date_obj = pd.to_datetime(row['Date']).replace(day=1)
                except Exception:
                    continue
                time_str = date_obj.isoformat()
                lat = float(row['latitude'])
                lon = float(row['longitude'])
                # Build the feature
                feat = {
                    "type": "Feature",
                    "geometry": {
                        "type": "Point",
                        "coordinates": [lon, lat]
                    },
                    "properties": {
                        "time": time_str,
                        "popup": f"Wildfire on {row['Date']}",
                        # We can style wildfires as red circle markers
                        "icon": "circle",
                        "iconstyle": {
                            "fillColor": "red",
                            "fillOpacity": 0.7,
                            "stroke": False,
                            "radius": 5
                        }
                    }
                }
                features.append(feat)

            # 2b) PM2.5 features
            def get_station_name(row):
                return row.get('SiteName', f"Station at ({row['Latitude']}, {row['Longitude']})")

            for idx, row in pm25_df.iterrows():
                try:
                    date_obj = pd.to_datetime(row['Date']).replace(day=1)
                except Exception:
                    continue
                time_str = date_obj.isoformat()
                color = aqi_color_map.get(row.get('AQI_Category', "Unknown"), "#000000")
                station_name = get_station_name(row)
                lat = float(row['Latitude'])
                lon = float(row['Longitude'])
                feat = {
                    "type": "Feature",
                    "geometry": {
                        "type": "Point",
                        "coordinates": [lon, lat]
                    },
                    "properties": {
                        "time": time_str,
                        "popup": f"{station_name}<br>PM2.5 AQI: {row['AQI']} "
                                f"({row.get('AQI_Category', 'Unknown')}) on {row['Date']}",
                        "icon": "circle",
                        "iconstyle": {
                            "fillColor": color,
                            "fillOpacity": 0.8,
                            "stroke": False,
                            "radius": 5
                        }
                    }
                }
                features.append(feat)

            # 2c) Ozone features
            for idx, row in ozone_df.iterrows():
                try:
                    date_obj = pd.to_datetime(row['Date']).replace(day=1)
                except Exception:
                    continue
                time_str = date_obj.isoformat()
                color = aqi_color_map.get(row.get('AQI_Category', "Unknown"), "#000000")
                station_name = get_station_name(row)
                lat = float(row['Latitude'])
                lon = float(row['Longitude'])
                feat = {
                    "type": "Feature",
                    "geometry": {
                        "type": "Point",
                        "coordinates": [lon, lat]
                    },
                    "properties": {
                        "time": time_str,
                        "popup": f"{station_name}<br>Ozone AQI: {row['AQI']} "
                                f"({row.get('AQI_Category', 'Unknown')}) on {row['Date']}",
                        "icon": "circle",
                        "iconstyle": {
                            "fillColor": color,
                            "fillOpacity": 0.8,
                            "stroke": False,
                            "radius": 5
                        }
                    }
                }
                features.append(feat)

            # 3) Build a single FeatureCollection
            all_geojson = {
                "type": "FeatureCollection",
                "features": features
            }

            # 4) Create the TimestampedGeoJson layer
            ts_layer = TimestampedGeoJson(
                data=all_geojson,
                period="P1M",          # Step by month
                transition_time=200,   # Transition speed
                auto_play=False,
                loop=False,
                time_slider_drag_update=False,
                date_options='YYYY-MM'  # Display "2020-01" type labels
            )
            ts_layer.add_to(m)

            # 5) Add a layer control if you want (optional).
            #    Typically, there's only one time layer, so this is up to you.
            # folium.LayerControl(collapsed=False).add_to(m)

            # 6) Save the map
            year_suffix = f"_{year_filter}" if year_filter else ""
            map_path = os.path.join(self.output_dir, f"unified_monthly_map{year_suffix}.html")
            m.save(map_path)
            self.logger.info(f"Monthly unified map saved to {map_path}.")
        except Exception as e:
            self.logger.error(f"Error creating monthly map: {e}")
            raise

    def create_seasonal_map(self, year_filter=None):
        """
        Creates a map with one FeatureGroup per season, each containing:
        - A wildfire heatmap (filtered by that season)
        - AQI station markers (filtered by that season)
            * Each station's popup time series only includes data from that season
        The user can toggle each season in a LayerControl.
        """
        try:
            self.logger.info("Creating a seasonal toggle map (heatmap + station markers).")
            m = folium.Map(location=[39.5501, -105.7821], zoom_start=6, tiles='cartodbpositron')

            # -------------------------------------------------------
            # 1) Filter Data by Year (Optional)
            # -------------------------------------------------------
            if year_filter:
                wf_df = self.wildfire_data[
                    self.wildfire_data["Date"].str.startswith(str(year_filter))
                ]
                pm25_df = self.aqi_pm25[
                    self.aqi_pm25["Date"].str.startswith(str(year_filter))
                ]
                ozone_df = self.aqi_ozone[
                    self.aqi_ozone["Date"].str.startswith(str(year_filter))
                ]
            else:
                wf_df = self.wildfire_data
                pm25_df = self.aqi_pm25
                ozone_df = self.aqi_ozone

            # -------------------------------------------------------
            # 2) Filter by Season - We assume a "Season" column exists
            # -------------------------------------------------------
            # If needed, convert "Season" to a string. E.g. df["Season"] = df["Season"].astype(str)
            wf_df = wf_df.copy()
            pm25_df = pm25_df.copy()
            ozone_df = ozone_df.copy()

            # Drop rows missing the Season column if that can happen
            wf_df.dropna(subset=["Season"], inplace=True)
            pm25_df.dropna(subset=["Season"], inplace=True)
            ozone_df.dropna(subset=["Season"], inplace=True)

            # Get unique seasons across all data
            all_seasons = pd.concat([wf_df["Season"], pm25_df["Season"], ozone_df["Season"]]).unique()
            # If you want a specific order, define a list like:
            # season_order = ["Winter", "Spring", "Summer", "Fall"]
            # Then sort based on that. Otherwise, just sort alphabetically:
            sorted_seasons = sorted(all_seasons)

            # -------------------------------------------------------
            # 3) For Each Season, Create a FeatureGroup
            # -------------------------------------------------------
            for season_name in sorted_seasons:
                # Filter each DF to this season
                wf_season = wf_df[wf_df["Season"] == season_name]
                pm25_season = pm25_df[pm25_df["Season"] == season_name]
                ozone_season = ozone_df[ozone_df["Season"] == season_name]

                # Create a FeatureGroup for that season
                fg_season = folium.FeatureGroup(name=f"{season_name} Season", overlay=True)

                # 3a) Wildfire Heatmap
                coords = wf_season[["latitude","longitude"]].dropna().values.tolist()
                coords = [[float(lat), float(lon)] for lat, lon in coords]
                if coords:
                    HeatMap(
                        coords,
                        radius=15,
                        blur=10,
                        gradient={"0.2": "yellow", "0.4": "orange", "0.6": "red"}
                    ).add_to(fg_season)

                # 3b) PM2.5 Station Markers (Seasonal)
                grouped_pm25 = pm25_season.groupby(["SiteName","Latitude","Longitude"])
                for (site, lat, lon), group in grouped_pm25:
                    # We pass only that season's data to generate_time_series_html
                    popup_html = self.generate_time_series_html(group.copy(), station_name=site)
                    iframe = folium.IFrame(html=popup_html, width=500, height=400)
                    popup = folium.Popup(iframe, max_width=600)
                    folium.CircleMarker(
                        location=[float(lat), float(lon)],
                        radius=6,
                        color="blue",
                        fill=True,
                        fill_color="black",
                        fill_opacity=0.5,
                        popup=popup
                    ).add_to(fg_season)

                # 3c) Ozone Station Markers (Seasonal)
                grouped_ozone = ozone_season.groupby(["SiteName","Latitude","Longitude"])
                for (site, lat, lon), group in grouped_ozone:
                    popup_html = self.generate_time_series_html(group.copy(), station_name=site)
                    iframe = folium.IFrame(html=popup_html, width=500, height=400)
                    popup = folium.Popup(iframe, max_width=600)
                    folium.CircleMarker(
                        location=[float(lat), float(lon)],
                        radius=6,
                        color="green",
                        fill=True,
                        fill_color="black",
                        fill_opacity=0.5,
                        popup=popup
                    ).add_to(fg_season)

                # Add the season FeatureGroup to the map
                fg_season.add_to(m)

            # -------------------------------------------------------
            # 4) Add a LayerControl so user can toggle each season
            # -------------------------------------------------------
            folium.LayerControl(collapsed=False).add_to(m)

            # -------------------------------------------------------
            # 5) Save the map
            # -------------------------------------------------------
            year_suffix = f"_{year_filter}" if year_filter else ""
            map_path = os.path.join(self.output_dir, f"seasonal_map{year_suffix}.html")
            m.save(map_path)
            self.logger.info(f"Seasonal map saved to {map_path}.")

        except Exception as e:
            self.logger.error(f"Error creating seasonal map: {e}")
            raise

    def create_animated_wf_map(self, year_filter=None):
        """
        Creates a single animated wildfire heatmap for the specified year (monthly).
        Uses the existing add_wildfire_animated_heatmap() method.
        """
        try:
            self.logger.info("Creating an animated wildfire heatmap map.")

            # 1) Initialize the Folium map
            m = folium.Map(location=[39.5501, -105.7821], zoom_start=6, tiles='cartodbpositron')

            # 2) If user provided a year filter, reduce the wildfire data
            if year_filter:
                filtered_wildfires = self.wildfire_data[self.wildfire_data["Date"].str.startswith(str(year_filter))]
            else:
                filtered_wildfires = self.wildfire_data

            # 3) Add the monthly animated wildfire heatmap to the map
            self.add_wildfire_animated_heatmap(m, filtered_wildfires)

            # 4) Optional: If you want a layer control (in case you add more layers)
            folium.LayerControl(collapsed=True).add_to(m)

            # 5) Save the map
            year_suffix = f"_{year_filter}" if year_filter else ""
            map_path = os.path.join(self.output_dir, f"animated_wildfire_heatmap{year_suffix}.html")
            m.save(map_path)
            self.logger.info(f"Animated wildfire heatmap map saved to {map_path}.")
        except Exception as e:
            self.logger.error(f"Error creating animated wildfire map: {e}")
            raise

if __name__ == "__main__":
    # Define color map for your AQI_Category
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
    # Just call create_monthly_map with a year filter to see combined monthly data.
    #visualizer.create_seasonal_map(year_filter=2020)
    visualizer.create_animated_wf_map(year_filter=2020)
