import os
import pandas as pd
import logging
import matplotlib.pyplot as plt
import geopandas as gpd
import plotly.graph_objects as go
from shapely.geometry import Point
from plotly.subplots import make_subplots
import statsmodels.api as sm

class StatPlots:
    """
    A class to create statistical plots for AQI data and wildfire data

    Parameters
    ----------
    ozone_data_path : str
        Path to processed ozone AQI data
    pm25_data_path : str
        Path to processed PM2.5 AQI data
    wildfire_data_path : str
        Path to processed wildfire data
    state_shapefile : str
        Path to state shapefile
    start_year : int
        Start year for data filtering
    end_year : int
        End year for data filtering
    conf_level : int
        Confidence level for wildfire data
    frp_thresh : int
        FRP threshold for wildfire data
    """
            
    def __init__(self, ozone_data_path, pm25_data_path, wildfire_data_path, state_shapefile, start_year=None, end_year=None, conf_level=None, frp_thresh=None):

        """
        Initializes the StatPlots class with data paths and potential filters
        
        Parameters
        ----------
        ozone_data_path : str
            Path to processed ozone AQI data
        pm25_data_path : str
            Path to processed PM2.5 AQI data
        wildfire_data_path : str
            Path to processed wildfire data
        state_shapefile : str
            Path to state shapefile
        start_year : int
            Start year for data filtering
        end_year : int
            End year for data filtering
        conf_level : int
            Confidence level for wildfire data
        frp_thresh : int
            FRP threshold for wildfire
        """
        
        # Setting parameters
        self.aqi_pm25_path = pm25_data_path
        self.aqi_ozone_path = ozone_data_path
        self.wildfire_data_path = wildfire_data_path
        self.state_shapefile_path = state_shapefile
        self.conf_level = conf_level
        self.frp_thresh = frp_thresh
        # Start logging
        logging.basicConfig(
            filename='data/logs/geo_plots.log',
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s'
        )
        self.logger = logging.getLogger(__name__)
        self.logger.info("StatPlots initialized.")
        self.logger.info(f"PM2.5 data path: {pm25_data_path}")
        self.logger.info(f"Ozone data path: {ozone_data_path}")
        self.logger.info(f"Wildfire data path: {wildfire_data_path}")
        # Load in data
        self.aqi_pm25 = pd.read_csv(pm25_data_path)
        self.aqi_ozone = pd.read_csv(ozone_data_path)
        self.wildfire_data = pd.read_csv(wildfire_data_path)
        # Date standardization
        self.wildfire_data['acq_date'] = pd.to_datetime(self.wildfire_data['Date']).dt.strftime('%Y-%m-%d')
        self.aqi_pm25['Date'] = pd.to_datetime(self.aqi_pm25['Date'])
        self.aqi_ozone['Date'] = pd.to_datetime(self.aqi_ozone['Date'])
        self.wildfire_data['acq_date'] = pd.to_datetime(self.wildfire_data['Date'])
        # Filter with year, conf level or frp threshold if needed
        if start_year and end_year:
            self.aqi_pm25 = self.aqi_pm25[self.aqi_pm25['Year'].between(start_year, end_year)]
            self.aqi_ozone = self.aqi_ozone[self.aqi_ozone['Year'].between(start_year, end_year)]
            self.wildfire_data = self.wildfire_data[self.wildfire_data['Year'].between(start_year, end_year)]
        if self.conf_level:
            self.wildfire_data = self.wildfire_data[self.wildfire_data['confidence'] >= self.conf_level]
        if self.frp_thresh:
            self.wildfire_data = self.wildfire_data[self.wildfire_data['frp'] >= self.frp_thresh]


    def avg_timeseries_plots(self):
        """
        Plots the average AQI for PM2.5 and Ozone over time, with wildfire incidents marked
        """
        fig, ax = plt.subplots(2, 1, figsize=(12, 8))
        # Plot AQI from each parameter
        self.aqi_pm25.groupby('Date')['AQI'].mean().plot(ax=ax[0], title='Average AQI$_{PM2.5}$, All Stations')
        self.aqi_ozone.groupby('Date')['AQI'].mean().plot(ax=ax[1], title='Average AQI$_{Ozone}$, All Stations')
        pm25_avg = self.aqi_pm25.groupby('Date')['Rolling_AQI'].mean()
        ozone_avg = self.aqi_ozone.groupby('Date')['Rolling_AQI'].mean()
        # Overlay wildfires
        for _, row in self.wildfire_data.iterrows():
            if row['acq_date'] in self.aqi_pm25['Date'].values:
                ax[0].scatter(row['acq_date'], pm25_avg[pm25_avg.index == row['acq_date']].values[0], color='red')
            if row['acq_date'] in self.aqi_ozone['Date'].values:
                ax[1].scatter(row['acq_date'], ozone_avg[ozone_avg.index == row['acq_date']].values[0], color='red')
        # Labeling
        ax[0].scatter([], [], color='red', label='Fire Incident')
        ax[0].legend()
        ax[1].scatter([], [], color='red', label='Fire Incident')
        ax[1].legend()
        ax[0].set_ylabel('AQI$_{PM2.5}$')
        ax[1].set_ylabel('AQI$_{Ozone}$')
        plt.tight_layout()
        plt.show()

    def station_timeseries_plots(self):
        """
        Plots the AQI for PM2.5 and Ozone over time for each station, with wildfire incidents marked
        """
        # Get unique stations
        counties = self.aqi_ozone['County'].unique()
        fig = make_subplots(rows=1, cols=1)
        # Plot AQI from each parameter, uses plotly interactive plots
        for idx, county in enumerate(counties):
            # Pull daily max values
            county_pm25 = self.aqi_pm25[self.aqi_pm25['County'] == county].groupby('Date')['Rolling_AQI'].max().reset_index()
            county_ozone = self.aqi_ozone[self.aqi_ozone['County'] == county].groupby('Date')['Rolling_AQI'].max().reset_index()
            county_wildfires = self.wildfire_data[self.wildfire_data['County'] == county]
            # Add plots for each parameter on each unique station, only if the data exists
            if not county_pm25.empty:
                fig.add_trace(
                    go.Scatter(x=county_pm25['Date'], y=county_pm25['Rolling_AQI'], mode='lines', name=f'{county} PM2.5',
                               visible=(idx == 0), fill='tozeroy'))
            else:
                fig.add_trace(go.Scatter(x=[], y=[], mode='lines', name=f'{county} PM2.5', visible=(idx == 0)))

            if not county_ozone.empty:
                fig.add_trace(
                    go.Scatter(x=county_ozone['Date'], y=county_ozone['Rolling_AQI'], mode='lines', name=f'{county} Ozone',
                               visible=(idx == 0), fill='tozeroy'))
            else:
                fig.add_trace(go.Scatter(x=[], y=[], mode='lines', name=f'{county} Ozone', visible=(idx == 0)))

            if not county_wildfires.empty:
                wildfire_dates = county_wildfires['acq_date']
                max_aqi = []
                for date in wildfire_dates:
                    pm25_aqi = county_pm25[county_pm25['Date'] == date]['Rolling_AQI'].max() if not county_pm25.empty else 0
                    ozone_aqi = county_ozone[county_ozone['Date'] == date]['Rolling_AQI'].max() if not county_ozone.empty else 0
                    max_aqi.append(max(pm25_aqi, ozone_aqi))
                fig.add_trace(go.Scatter(x=wildfire_dates, y=max_aqi, mode='markers',
                                         marker=dict(color='black', size=10), name=f'{county} Wildfire',
                                         visible=(idx == 0)))
            else:
                fig.add_trace(go.Scatter(x=[], y=[], mode='markers', name=f'{county} Wildfire', visible=(idx == 0)))
        # Dropdown buttons for county selection
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

    def timeseries_processing(self, county=None, year=None):
        """
        Decomposes the time series data for PM2.5 and Ozone AQI for a given county and year
        """
        # Filtering by year and county if needed
        if county:
            pm25_data = self.aqi_pm25[self.aqi_pm25['County'] == county]
            ozone_data = self.aqi_ozone[self.aqi_ozone['County'] == county]
        else:
            pm25_data = self.aqi_pm25
            ozone_data = self.aqi_ozone
        if year:
            pm25_data = pm25_data[pm25_data['Date'].dt.year == year]
            ozone_data = ozone_data[ozone_data['Date'].dt.year == year]
        pm25_max = pm25_data.groupby('Date')['Rolling_AQI'].max().fillna(method='ffill')
        ozone_max = ozone_data.groupby('Date')['Rolling_AQI'].max().fillna(method='ffill')
        # Decomposotion of pm25, if available
        if pm25_max.empty:
            print(f"No PM2.5 data available for {county} in {year}")
        else:
            pm25_decomposition = sm.tsa.seasonal_decompose(pm25_max, model='additive', period=128)
            fig, (ax1, ax2, ax3, ax4) = plt.subplots(4, 1, figsize=(12, 10))
            pm25_decomposition.observed.plot(ax=ax1, title='Observed PM2.5 AQI')
            pm25_decomposition.trend.plot(ax=ax2, title='Trend')
            pm25_decomposition.seasonal.plot(ax=ax3, title='Seasonal')
            pm25_decomposition.resid.plot(ax=ax4, title='Residual')
            plt.tight_layout()
            plt.show()
        # Decomposition of ozone, if available
        if ozone_max.empty:
            print(f"No Ozone data available for {county} in {year}")
        else:
            ozone_decomposition = sm.tsa.seasonal_decompose(ozone_max, model='additive', period=128)
            fig, (ax1, ax2, ax3, ax4) = plt.subplots(4, 1, figsize=(12, 10))
            ozone_decomposition.observed.plot(ax=ax1, title='Observed Ozone AQI')
            ozone_decomposition.trend.plot(ax=ax2, title='Trend')
            ozone_decomposition.seasonal.plot(ax=ax3, title='Seasonal')
            ozone_decomposition.resid.plot(ax=ax4, title='Residual')
            plt.tight_layout()
            plt.show()

if __name__ == "__main__":
    ozone_dp = 'data/aqi_data/aqi_processed/ozone_aqi_2019_2024.csv'
    pm25_dp = 'data/aqi_data/aqi_processed/pm25_aqi_2019_2024.csv'
    wildfire_dp = 'data/wildfire_data/wildfire_processed/wildfire_processed_2019_2024_n.csv'
    state_shapefile = 'data/co_shapefile/counties/counties_19.shSp'
    # Example usage for 2019-2024 data, with a focus on the Larmier county wildfires in 2020
    stat_plots = StatPlots(ozone_dp, pm25_dp, wildfire_dp, state_shapefile, 2019, 2024, frp_thresh=40)
    stat_plots.station_timeseries_plots()
    stat_plots.timeseries_processing(county = 'Larimer', year = 2020)
