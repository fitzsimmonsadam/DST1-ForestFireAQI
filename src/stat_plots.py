import os
import pandas as pd
import logging
import matplotlib.pyplot as plt
import geopandas as gpd
import plotly.graph_objects as go
from shapely.geometry import Point
from plotly.subplots import make_subplots

class StatPlots:
    def __init__(self, ozone_data_path, pm25_data_path, wildfire_data_path, state_shapefile, start_year=None, end_year=None, conf_level=None, frp_thresh=None):

        self.aqi_pm25_path = pm25_data_path
        self.aqi_ozone_path = ozone_data_path
        self.wildfire_data_path = wildfire_data_path
        self.state_shapefile_path = state_shapefile
        self.conf_level = conf_level
        self.frp_thresh = frp_thresh
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
        self.aqi_pm25['Date'] = pd.to_datetime(self.aqi_pm25['Date'])
        self.aqi_ozone['Date'] = pd.to_datetime(self.aqi_ozone['Date'])
        self.wildfire_data['acq_date'] = pd.to_datetime(self.wildfire_data['acq_date'])
        if start_year and end_year:
            self.aqi_pm25 = self.aqi_pm25[self.aqi_pm25['Year'].between(start_year, end_year)]
            self.aqi_ozone = self.aqi_ozone[self.aqi_ozone['Year'].between(start_year, end_year)]
            self.wildfire_data = self.wildfire_data[self.wildfire_data['Year'].between(start_year, end_year)]
        if self.conf_level:
            self.wildfire_data = self.wildfire_data[self.wildfire_data['confidence'] >= self.conf_level]
        if self.frp_thresh:
            self.wildfire_data = self.wildfire_data[self.wildfire_data['frp'] >= self.frp_thresh]
        print(self.wildfire_data[['acq_date', 'latitude', 'longitude']].head())


    def avg_timeseries_plots(self):
        fig, ax = plt.subplots(2, 1, figsize=(12, 8))
        self.aqi_pm25.groupby('Date')['AQI'].mean().plot(ax=ax[0], title='Average AQI$_{PM2.5}$, All Stations')
        self.aqi_ozone.groupby('Date')['AQI'].mean().plot(ax=ax[1], title='Average AQI$_{Ozone}$, All Stations')
        pm25_avg = self.aqi_pm25.groupby('Date')['AQI'].mean()
        ozone_avg = self.aqi_ozone.groupby('Date')['AQI'].mean()
        for _, row in self.wildfire_data.iterrows():
            if row['acq_date'] in self.aqi_pm25['Date'].values:
                ax[0].scatter(row['acq_date'], pm25_avg[pm25_avg.index == row['acq_date']].values[0], color='red')
            if row['acq_date'] in self.aqi_ozone['Date'].values:
                ax[1].scatter(row['acq_date'], ozone_avg[ozone_avg.index == row['acq_date']].values[0], color='red')
        ax[0].scatter([], [], color='red', label='Fire Incident')
        ax[0].legend()
        ax[1].scatter([], [], color='red', label='Fire Incident')
        ax[1].legend()
        ax[0].set_ylabel('AQI$_{PM2.5}$')
        ax[1].set_ylabel('AQI$_{Ozone}$')
        plt.tight_layout()
        plt.show()

    def station_timeseries_plots(self):
        counties = self.aqi_ozone['County'].unique()
        fig = make_subplots(rows=1, cols=1)

        for idx, county in enumerate(counties):
            county_pm25 = self.aqi_pm25[self.aqi_pm25['County'] == county].groupby('Date')['AQI'].max().reset_index()
            county_ozone = self.aqi_ozone[self.aqi_ozone['County'] == county].groupby('Date')['AQI'].max().reset_index()
            county_wildfires = self.wildfire_data[self.wildfire_data['County'] == county]

            if not county_pm25.empty:
                fig.add_trace(
                    go.Scatter(x=county_pm25['Date'], y=county_pm25['AQI'], mode='lines', name=f'{county} PM2.5',
                               visible=(idx == 0), fill='tozeroy'))
            else:
                fig.add_trace(go.Scatter(x=[], y=[], mode='lines', name=f'{county} PM2.5', visible=(idx == 0)))

            if not county_ozone.empty:
                fig.add_trace(
                    go.Scatter(x=county_ozone['Date'], y=county_ozone['AQI'], mode='lines', name=f'{county} Ozone',
                               visible=(idx == 0), fill='tozeroy'))
            else:
                fig.add_trace(go.Scatter(x=[], y=[], mode='lines', name=f'{county} Ozone', visible=(idx == 0)))

            if not county_wildfires.empty:
                wildfire_dates = county_wildfires['acq_date']
                max_aqi = []
                for date in wildfire_dates:
                    pm25_aqi = county_pm25[county_pm25['Date'] == date]['AQI'].max() if not county_pm25.empty else 0
                    ozone_aqi = county_ozone[county_ozone['Date'] == date]['AQI'].max() if not county_ozone.empty else 0
                    max_aqi.append(max(pm25_aqi, ozone_aqi))
                fig.add_trace(go.Scatter(x=wildfire_dates, y=max_aqi, mode='markers',
                                         marker=dict(color='black', size=10), name=f'{county} Wildfire',
                                         visible=(idx == 0)))
            else:
                fig.add_trace(go.Scatter(x=[], y=[], mode='markers', name=f'{county} Wildfire', visible=(idx == 0)))

        dropdown_buttons = [
            dict(
                label=county,
                method='update',
                args=[{'visible': [i // 3 == idx for i in range(len(fig.data))],
                       'showlegend': [i // 3 == idx for i in range(len(fig.data))]},
                      {'title': {
                          'text': f'AQI Time Series for {county}, {self.aqi_pm25["Date"].min().year} to {self.aqi_pm25["Date"].max().year}'}}]
            ) for idx, county in enumerate(counties)
        ]
        fig.update_layout(
            updatemenus=[
                dict(
                    active=0,
                    buttons=dropdown_buttons,
                    x=0,
                    xanchor='left',
                    y=1,
                    yanchor='top')],
            title=f'AQI between {self.aqi_pm25["Date"].min().year} and {self.aqi_pm25["Date"].max().year}',
            xaxis_title='Date',
            yaxis_title='AQI',
            width=1000,
            height=800,
            xaxis=dict(
                range=[self.aqi_ozone["Date"].min(), self.aqi_ozone["Date"].max()]

            ),
            yaxis=dict(
                range=[0, 250]))
        fig.show()

    def assign_county(self):
        gdf = gpd.read_file(self.state_shapefile_path)
        gdf = gdf.to_crs(epsg=4326)
        self.wildfire_data['County'] = None
        for idx, row in self.wildfire_data.iterrows():
            point = Point(row['longitude'], row['latitude'])
            for _, county in gdf.iterrows():
                if county['geometry'].contains(point):
                    self.wildfire_data.at[idx, 'County'] = county['NAME']
                    break
        self.wildfire_data = self.wildfire_data.dropna(subset=['County'])

    #def timeseries_processing(self):



if __name__ == "__main__":
    ozone_dp = '../data/aqi_data/aqi_processed/ozone_aqi_2019_2024.csv'
    pm25_dp = '../data/aqi_data/aqi_processed/pm25_aqi_2019_2024.csv'
    wildfire_dp = '../data/wildfire_data/wildfire_processed/wildfire_processed_2019_2024.csv'
    state_shapefile = '../data/co_shapefile/counties/counties_19.shp'
    stat_plots = StatPlots(ozone_dp, pm25_dp, wildfire_dp, state_shapefile, 2019, 2024, conf_level = 100)
    stat_plots.assign_county()
    stat_plots.station_timeseries_plots()
    # save plot

