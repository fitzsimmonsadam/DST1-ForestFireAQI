import os
import pandas as pd
import logging
import matplotlib.pyplot as plt

class StatPlots:
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
        print(self.wildfire_data[['acq_date', 'latitude', 'longitude']].head())


    def aqi_timeseries_plots(self):
        fig, ax = plt.subplots(2, 1, figsize=(12, 8))
        self.aqi_pm25.groupby('Date')['AQI'].mean().plot(ax=ax[0], title='PM2.5 AQI Time Series')
        self.aqi_pm25.groupby('Date')['AQI'].mean().plot(ax=ax[0], title='PM2.5 AQI Time Series')
        self.aqi_ozone.groupby('Date')['AQI'].mean().plot(ax=ax[1], title='Ozone AQI Time Series')
        self.aqi_ozone.groupby('Date')['AQI'].mean().plot(ax=ax[1], title='Ozone AQI Time Series')

        plt.tight_layout()
        plt.show()

    def processed_timeseries(self):



if __name__ == "__main__":
    ozone_dp = '../data/aqi_data/aqi_processed/ozone_aqi_2019_2024.csv'
    pm25_dp = '../data/aqi_data/aqi_processed/pm25_aqi_2019_2024.csv'
    wildfire_dp = '../data/wildfire_data/wildfire_processed/wildfire_processed_2019_2024.csv'
    state_shapefile = '../data/co_shapefile/counties/counties_19.shp'
    stat_plots = StatPlots(ozone_dp, pm25_dp, wildfire_dp, state_shapefile, 2023, 2023)
    stat_plots.aqi_timeseries_plots()
