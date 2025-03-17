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
    """
    A class to plot goegraphical views of air quality monitoring stations and wildfires in Colorado

    Attributes
    ----------
    ozone_data_path : str
        Path to the ozone AQI data
    pm25_data_path : str
        Path to the PM2.5 AQI data
    wildfire_data_path : str
        Path to the wildfire data
    state_shapefile_path : str
        Path to the shapefile of Colorado
    start_year : int
        Start year for data filtering
    end_year : int
        End year for data filtering
    """

    def __init__(self, ozone_data_path, pm25_data_path, wildfire_data_path, state_shapefile, visuals_path, start_year=None, end_year=None):
        """
        Initialize the GeoPlots class with the given data paths and shapefile path

        Parameters
        ----------
        ozone_data_path : str
            Path to the ozone AQI data
        pm25_data_path : str
            Path to the PM2.5 AQI data
        wildfire_data_path : str
            Path to the wildfire data
        state_shapefile : str
            Path to the shapefile of Colorado
        start_year : int
            Start year for data filtering
        end_year : int
            End year for data filtering
        visual_path : str
            Path to save the visual
        """
        # Save filepaths
        self.aqi_pm25_path = pm25_data_path
        self.aqi_ozone_path = ozone_data_path
        self.wildfire_data_path = wildfire_data_path
        self.state_shapefile_path = state_shapefile
        self.visuals_path = visuals_path
        # Setup logging
        logging.basicConfig(
            filename = 'data/logs/geo_plots.log',
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s'
            )
        
        self.logger = logging.getLogger(__name__)
        self.logger.info("GeoPlots initialized.")
        self.logger.info(f"PM2.5 data path: {pm25_data_path}")
        self.logger.info(f"Ozone data path: {ozone_data_path}")
        self.logger.info(f"Wildfire data path: {wildfire_data_path}")
        # Load in data
        self.aqi_pm25 = pd.read_csv(pm25_data_path)
        self.aqi_ozone = pd.read_csv(ozone_data_path)
        self.wildfire_data = pd.read_csv(wildfire_data_path)
        # Date handling
        self.wildfire_data['acq_date'] = pd.to_datetime(self.wildfire_data['acq_date']).dt.strftime('%Y-%m-%d')
        # Year filtering
        if start_year and end_year:
            self.aqi_pm25 = self.aqi_pm25[self.aqi_pm25['Year'].between(start_year, end_year)]
            self.aqi_ozone = self.aqi_ozone[self.aqi_ozone['Year'].between(start_year, end_year)]
            self.wildfire_data = self.wildfire_data[self.wildfire_data['Year'].between(start_year, end_year)]

    def plot_stations(self):
        """
        Plot the air quality monitoring stations on a map of Colorado
        """
        self.logger.info("Plotting air quality monitoring stations.")
        # Load in state shapefile
        gdf = gpd.read_file(self.state_shapefile_path)
        gdf = gdf.to_crs(epsg=4326)
        # Start folium map centered on CO
        m = folium.Map(location=[39.5501, -105.7821], zoom_start=6)
        folium.GeoJson(gdf).add_to(m)
        # Plot stations only once
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
        # Save figures
        m.save(self.visuals_path + '/aqi_stations_map.html')
        self.logger.info(f"Map saved to ../visuals/aqi_stations_map.html")
        return m
        self.logger.info(f"Opening map in local browser")
        #webbrowser.open_new_tab(self.visuals_path + '/aqi_stations_map.html')

    def plot_wildfires(self):
        """
        Plot the wildfires on a map of Colorado
        """
        self.logger.info("Plotting wildfires.")
        # Load shapefile
        gdf = gpd.read_file(self.state_shapefile_path)
        gdf = gdf.to_crs(epsg=4326)  # Convert to WGS84
        # Folium map centered on CO
        m = folium.Map(location=[39.5501, -105.7821], zoom_start=6)
        folium.GeoJson(gdf).add_to(m)
        # Plot fires
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
        # Save map
        m.save(self.visuals_path + '/wildfires_map.html')
        self.logger.info(f"Map saved to visuals/wildfires_map.html")
        return m
        #self.logger.info(f"Opening map in local browser")
        #webbrowser.open_new_tab(self.visuals_path + '/aqi_stations_map.html')

    def plot_timeline(self):
        """
        Plot a timeline of the wildfires and air quality monitoring stations for every wildfire date
        """
        self.logger.info("Plotting timeline.")
        fig, ax = plt.subplots()
        # Load shapefile
        gdf = gpd.read_file(self.state_shapefile_path)
        gdf = gdf.to_crs(epsg=4326)  # Convert to WGS84
        gdf.plot(ax=ax, color='white', edgecolor='black')
        # Only plot stations once
        unique_stations = self.aqi_pm25[['Latitude', 'Longitude']].drop_duplicates()
        for _, row in unique_stations.iterrows():
            ax.plot(row['Longitude'], row['Latitude'], 'bo')
        bounds = gdf.total_bounds
        ax.set_xlim(bounds[0], bounds[2])
        ax.set_ylim(bounds[1], bounds[3])
        ax.set_title('Colorado AQI Stations and Wildfires')
        ax.set_xlabel('Longitude')
        ax.set_ylabel('Latitude')
        # Animate wildfires only using dates with fires
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

if __name__ == "__main__":#
    # sed 30 day rolling average for AQI
    ozone_dp = 'data/aqi_data/aqi_processed/ozone_aqi_2019_2024_30.csv'
    pm25_dp = 'data/aqi_data/aqi_processed/pm25_aqi_2019_2024_30.csv'
    wildfire_dp = 'data/wildfire_data/wildfire_processed/wildfire_processed_2019_2024.csv'
    state_shapefile = 'data/co_shapefile/counties/counties_19.shp'
    # Example usage with data between 2023 and 2024
    geo_plots = GeoPlots(ozone_dp, pm25_dp, wildfire_dp, state_shapefile, 2023, 2024)
    geo_plots.plot_stations()
    geo_plots.plot_wildfires()
    geo_plots.plot_timeline()

