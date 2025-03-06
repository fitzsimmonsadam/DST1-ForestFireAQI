import os
import pandas as pd
import geopandas as gpd
import logging
import matplotlib.pyplot as plt
import folium
import webbrowser
import numpy as np
import matplotlib.animation as animation

class GeoPlots:
    def __init__(self, ozone_data_path, pm25_data_path, wildfire_data_path, state_shapefile, start_year=None, end_year=None):

        self.aqi_pm25_path = pm25_data_path
        self.aqi_ozone_path = ozone_data_path
        self.wildfire_data_path = wildfire_data_path
        self.state_shapefile_path = state_shapefile

        logging.basicConfig(
            filename=os.path.join(os.path.dirname(__file__), '../logs/geo_plots.log'),
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s'
        )
        self.logger = logging.getLogger(__name__)
        self.logger.info("GeoPlots initialized.")
        self.logger.info(f"PM2.5 data path: {pm25_data_path}")
        self.logger.info(f"Ozone data path: {ozone_data_path}")
        self.logger.info(f"Wildfire data path: {wildfire_data_path}")
        self.aqi_pm25 = pd.read_csv(pm25_data_path)
        self.aqi_ozone = pd.read_csv(ozone_data_path)
        self.wildfire_data = pd.read_csv(wildfire_data_path)
        self.wildfire_data['acq_date'] = pd.to_datetime(self.wildfire_data['acq_date']).dt.strftime('%Y-%m-%d')
        if start_year and end_year:
            self.aqi_pm25 = self.aqi_pm25[self.aqi_pm25['Year'].between(start_year, end_year)]
            self.aqi_ozone = self.aqi_ozone[self.aqi_ozone['Year'].between(start_year, end_year)]
            self.wildfire_data = self.wildfire_data[self.wildfire_data['Year'].between(start_year, end_year)]

    def plot_stations(self):
        try:
            self.logger.info("Plotting air quality monitoring stations.")
            gdf = gpd.read_file(self.state_shapefile_path)
            gdf = gdf.to_crs(epsg=4326)  # Convert to WGS84
            m = folium.Map(location=[39.5501, -105.7821], zoom_start=6)
            folium.GeoJson(gdf).add_to(m)
            unique_stations = self.aqi_pm25[['Latitude', 'Longitude']].drop_duplicates()
            for _, row in unique_stations.iterrows():
                folium.Marker(
                    location=[row['Latitude'], row['Longitude']],
                    popup='AQI Station',
                    icon=folium.Icon(color='blue')
                ).add_to(m)
            bounds = gdf.total_bounds
            m.fit_bounds([[bounds[1], bounds[0]], [bounds[3], bounds[2]]])
            title_html = '''
            <h3 align="center" style="font-size:20px"><b>Air Quality Monitoring Stations</b></h3>
            '''
            m.get_root().html.add_child(folium.Element(title_html))
            m.save('../visuals/aqi_stations_map.html')
            self.logger.info(f"Map saved to ../visuals/aqi_stations_map.html")
            plt.show()
            self.logger.info(f"Opening map in local browser")
            webbrowser.open_new_tab(os.getcwd() + '/../visuals/aqi_stations_map.html')
        except Exception as e:
            self.logger.error(f"Error plotting stations: {e}")
            raise

    def plot_wildfires(self):
        try:
            self.logger.info("Plotting wildfires.")
            gdf = gpd.read_file(self.state_shapefile_path)
            gdf = gdf.to_crs(epsg=4326)  # Convert to WGS84
            m = folium.Map(location=[39.5501, -105.7821], zoom_start=6)
            folium.GeoJson(gdf).add_to(m)
            for _, row in self.wildfire_data.iterrows():
                folium.Marker(
                    location=[row['latitude'], row['longitude']],
                    popup='Fire on ' + str(row['acq_date']),
                    icon=folium.Icon(icon='fire', prefix='fa', color='red')
                ).add_to(m)
            bounds = gdf.total_bounds
            m.fit_bounds([[bounds[1], bounds[0]], [bounds[3], bounds[2]]])
            title_html = f"'<h3 align='center' style='font-size:20px'><b>Wildfires in Colorado ({self.wildfire_data['Year'].min()} - {self.wildfire_data['Year'].max()})</b></h3>'"
            m.get_root().html.add_child(folium.Element(title_html))
            m.save('../visuals/wildfires_map.html')
            self.logger.info(f"Map saved to ../visuals/wildfires_map.html")
            plt.show()
            self.logger.info(f"Opening map in local browser")
            webbrowser.open_new_tab(os.getcwd() + '/../visuals/wildfires_map.html')
        except Exception as e:
            self.logger.error(f"Error plotting wildfires: {e}")
            raise

    def plot_timeline(self):
        try:
            self.logger.info("Plotting timeline.")
            fig, ax = plt.subplots()
            gdf = gpd.read_file(self.state_shapefile_path)
            gdf = gdf.to_crs(epsg=4326)  # Convert to WGS84
            gdf.plot(ax=ax, color='white', edgecolor='black')
            unique_stations = self.aqi_pm25[['Latitude', 'Longitude']].drop_duplicates()
            for _, row in unique_stations.iterrows():
                ax.plot(row['Longitude'], row['Latitude'], 'bo')
            bounds = gdf.total_bounds
            ax.set_xlim(bounds[0], bounds[2])
            ax.set_ylim(bounds[1], bounds[3])
            ax.set_title('Colorado AQI Stations and Wildfires')
            ax.set_xlabel('Longitude')
            ax.set_ylabel('Latitude')
            dates = self.wildfire_data['acq_date'].unique()
            for date in dates:
                print(date)
                ax.set_title(f'Colorado AQI Stations and Wildfires on {date}')
                wildfires = self.wildfire_data[self.wildfire_data['acq_date'] == date]
                for _, row in wildfires.iterrows():
                    ax.plot(row['longitude'], row['latitude'], 'ro')
                plt.pause(0.5)
                ax.clear()
                gdf.plot(ax=ax, color='white', edgecolor='black')
                for _, row in unique_stations.iterrows():
                    ax.plot(row['Longitude'], row['Latitude'], 'bo')
                ax.set_xlim(bounds[0], bounds[2])
                ax.set_ylim(bounds[1], bounds[3])
            plt.show()
        except Exception as e:
            self.logger.error(f"Error plotting timeline: {e}")
            raise

if __name__ == "__main__":
    ozone_dp = '../data/aqi_data/aqi_processed/ozone_aqi_2019_2024.csv'
    pm25_dp = '../data/aqi_data/aqi_processed/pm25_aqi_2019_2024.csv'
    wildfire_dp = '../data/wildfire_data/wildfire_processed/wildfire_processed_2019_2024.csv'
    state_shapefile = '../data/co_shapefile/counties/counties_19.shp'
    geo_plots = GeoPlots(ozone_dp, pm25_dp, wildfire_dp, state_shapefile, 2023, 2024)
    gdf = gpd.read_file(geo_plots.state_shapefile_path)
    print(gdf.columns)
    geo_plots.plot_stations()
    geo_plots.plot_wildfires()
    geo_plots.plot_timeline()

