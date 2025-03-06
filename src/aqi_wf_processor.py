import os
import pandas as pd
import numpy as np
import geopandas as gpd
import logging
from typing import Optional, List
from shapely.geometry import Point
from datetime import date
import matplotlib.pyplot as plt


class BaseProcessor:
    def __init__(self, output_dir: str):
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)
        os.makedirs("data/logs", exist_ok=True)
        self.logger = self.setup_logger()

    def setup_logger(self):
        logger = logging.getLogger(f"{__name__}")
        logger.setLevel(logging.INFO)
        if logger.hasHandlers():
            logger.handlers.clear()
        fh = logging.FileHandler("data/logs/processing.log", mode="a")
        fh.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s - %(message)s"))
        logger.addHandler(fh)
        sh = logging.StreamHandler()
        sh.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s - %(message)s"))
        logger.addHandler(sh)
        return logger


class WildfireProcessor(BaseProcessor):
    def __init__(self, wildfire_filepath: str, start_year: int, end_year: int, output_dir: Optional[str] = "data/wildfire_data/wildfire_processed", county_shapefile: str = "data/co_shapefile/counties/counties_19.shp"):
        super().__init__(output_dir)
        self.wildfire_filepath = wildfire_filepath
        self.start_year = start_year
        self.end_year = end_year
        self.county_shapefile = county_shapefile
        self.logger.info("Initializing WildfireProcessor.")
        self.wildfire_df = pd.read_csv(self.wildfire_filepath)

    def clean_dataframe(self, df):
        self.logger.info("Cleaning wildfire data.")
        df.replace(-999, np.nan, inplace=True)
        desired_columns = ["latitude", "longitude", "acq_date", "frp", "confidence", "type"]
        df = df[desired_columns]
        # Force confidence to string and filter
        df["confidence"] = df["confidence"].astype(str).str.lower()
        initial_count = len(df)
        df = df[df["confidence"] != 'n']
        removed_count = initial_count - len(df)
        self.logger.info(f"Removed {removed_count} rows with 'n' confidence.")
        return df

    def assign_season(self, df):
        df['Month'] = pd.to_datetime(df['acq_date']).dt.month
        df['Season'] = df['Month'].map({
            12: 'Winter', 1: 'Winter', 2: 'Winter',
            3: 'Spring', 4: 'Spring', 5: 'Spring',
            6: 'Summer', 7: 'Summer', 8: 'Summer',
            9: 'Fall', 10: 'Fall', 11: 'Fall'
        })
        return df

    def filter_to_colorado(self, df):
        self.logger.info("Filtering wildfire records to Colorado boundary.")
        counties_gdf = gpd.read_file(self.county_shapefile)
        if counties_gdf.crs is None:
            counties_gdf.set_crs("EPSG:4269", inplace=True)
        counties_gdf = counties_gdf.to_crs("EPSG:4326")

        gdf = gpd.GeoDataFrame(
            df,
            geometry=gpd.points_from_xy(df["longitude"], df["latitude"]),
            crs="EPSG:4326"
        )

        filtered_gdf = gpd.sjoin(gdf, counties_gdf, how="inner", predicate="within")

        self.logger.info(f"Filtered from {len(df)} to {len(filtered_gdf)} records within Colorado.")
        return filtered_gdf.drop(columns="geometry")

    def derive_county(self, df):
        self.logger.info("Starting county derivation.")

        if not os.path.exists(self.county_shapefile):
            self.logger.error(f"County shapefile not found at: {self.county_shapefile}")
            df["County"] = np.nan
            return df

        # Prepare wildfire GeoDataFrame
        df = df.copy()
        df["geometry"] = gpd.points_from_xy(df["longitude"], df["latitude"])
        wildfire_gdf = gpd.GeoDataFrame(df, geometry="geometry", crs="EPSG:4326")

        # Load counties shapefile
        counties_gdf = gpd.read_file(self.county_shapefile)
        if counties_gdf.crs is None:
            counties_gdf.set_crs("EPSG:4269", inplace=True)
        counties_gdf = counties_gdf.to_crs(wildfire_gdf.crs)

        # Drop 'index_right' if exists before join
        if 'index_right' in wildfire_gdf.columns:
            wildfire_gdf.drop(columns=['index_right'], inplace=True)

        # Perform spatial join
        joined = gpd.sjoin(wildfire_gdf, counties_gdf[['NAME', 'geometry']], how="left", predicate="within")

        # Get county name from correct column
        county_name_col = next((col for col in joined.columns if col.endswith('NAME')), None)
        if county_name_col:
            joined["County"] = joined[county_name_col]
        else:
            self.logger.error("County name column not found after spatial join.")
            joined["County"] = np.nan

        # Report missing counties
        missing_counties = joined["County"].isna().sum()
        self.logger.info(f"County derivation complete. Missing counties: {missing_counties}")

        # Select only desired columns
        final_columns = [
            "latitude", "longitude", "acq_date", "frp", "confidence", "type",
            "Year", "Month", "Season", "County"
        ]
        result_df = joined[final_columns].copy()

        return result_df

    def process_wildfire(self, year_range: Optional[tuple] = None):
        self.logger.info("Starting wildfire processing.")
        self.wildfire_df = self.clean_dataframe(self.wildfire_df)
        self.wildfire_df['acq_date'] = pd.to_datetime(self.wildfire_df['acq_date'])
        self.wildfire_df['Year'] = self.wildfire_df['acq_date'].dt.year

        if year_range:
            start_year, end_year = year_range
            self.wildfire_df = self.wildfire_df[
                (self.wildfire_df['Year'] >= start_year) &
                (self.wildfire_df['Year'] <= end_year)
            ]

        self.wildfire_df = self.filter_to_colorado(self.wildfire_df)

        combined_df = []
        for year in sorted(self.wildfire_df['Year'].unique()):
            year_df = self.wildfire_df[self.wildfire_df['Year'] == year].copy()
            year_df = self.assign_season(year_df)
            year_df = self.derive_county(year_df)
            year_output_path = os.path.join(self.output_dir, f"wildfire_processed_{year}.csv")
            year_df.to_csv(year_output_path, index=False)
            self.logger.info(f"Saved wildfire data for {year} to {year_output_path}.")
            combined_df.append(year_df)

        combined_df = pd.concat(combined_df)
        combined_output_path = os.path.join(self.output_dir, f"wildfire_processed_{self.start_year}_{self.end_year}.csv")
        combined_df.to_csv(combined_output_path, index=False)
        self.logger.info(f"Saved combined wildfire data to {combined_output_path}.")

class AQIProcessor(BaseProcessor):
    def __init__(self, aqi_filepath: str, wildfire_filepath: str, start_year: int, end_year: int, output_dir: str, county_shapefile: str):
        super().__init__(output_dir)
        self.aqi_filepath = aqi_filepath
        self.wildfire_filepath = wildfire_filepath
        self.start_year = start_year
        self.end_year = end_year
        self.county_shapefile = county_shapefile
        self.logger.info("Initializing AQIProcessor.")
        self.aq_df = pd.read_csv(self.aqi_filepath)
        self.wildfire_df = pd.read_csv(self.wildfire_filepath)

    def clean_dataframe(self, df):
        self.logger.info("Cleaning AQI data.")
        df.replace(-999, np.nan, inplace=True)
        return df[["Latitude", "Longitude", "UTC", "Parameter", "AQI", "Category"]]
    
    def categorize_aqi(self, df):
        self.logger.info("Categorizing AQI values.")
        def category(aqi):
            if pd.isna(aqi):
                return "Unknown"
            elif aqi <= 50:
                return "Good"
            elif aqi <= 100:
                return "Moderate"
            elif aqi <= 150:
                return "Unhealthy for Sensitive Groups"
            elif aqi <= 200:
                return "Unhealthy"
            elif aqi <= 300:
                return "Very Unhealthy"
            else:
                return "Hazardous"
        df["AQI_Category"] = df["AQI"].apply(category)
        return df

    def assign_season(self, df):
        self.logger.info("Assigning seasons.")
        df["Season"] = df["Month"].map({
            12: 'Winter', 1: 'Winter', 2: 'Winter',
            3: 'Spring', 4: 'Spring', 5: 'Spring',
            6: 'Summer', 7: 'Summer', 8: 'Summer',
            9: 'Fall', 10: 'Fall', 11: 'Fall'
        })
        return df

    def compute_rolling_averages(self, df, window_days=7):
        self.logger.info(f"Computing {window_days}-day rolling averages.")
        df['Date'] = pd.to_datetime(df['Date'])
        df.sort_values('Date', inplace=True)
        df['Rolling_AQI'] = df.groupby('County')['AQI'].transform(
            lambda x: x.rolling(window=window_days, min_periods=1).mean()
        )
        return df

    def derive_county(self, df):
        gdf = gpd.GeoDataFrame(df, geometry=gpd.points_from_xy(df["Longitude"], df["Latitude"]), crs="EPSG:4269")
        counties = gpd.read_file(self.county_shapefile).to_crs("EPSG:4269")
        joined = gpd.sjoin(gdf, counties, how="left", predicate="within")
        df["County"] = joined["NAME"].values
        return df

    def wildfire_in_county(self, df):
        self.logger.info("Flagging wildfires in the same county and date.")
        wildfire_dates_counties = self.wildfire_df[['acq_date', 'County']].drop_duplicates()
        wildfire_dates_counties['acq_date'] = pd.to_datetime(wildfire_dates_counties['acq_date']).dt.date
        df["Wildfire_In_County"] = df.apply(
            lambda row: ((wildfire_dates_counties['acq_date'] == row['Date']).any() and
                         (wildfire_dates_counties['County'] == row['County']).any()),
            axis=1)
        return df

    def flag_proximity(self, df, distance_km):
        self.logger.info(f"Flagging proximity within {distance_km} km of wildfires.")
        
        # Convert AQI and wildfire data to GeoDataFrames
        gdf_aqi = gpd.GeoDataFrame(
            df.copy(),
            geometry=gpd.points_from_xy(df["Longitude"], df["Latitude"]),
            crs="EPSG:4326")
        gdf_wf = gpd.GeoDataFrame(
            self.wildfire_df.copy(),
            geometry=gpd.points_from_xy(self.wildfire_df["longitude"], self.wildfire_df["latitude"]),
            crs="EPSG:4326")

        gdf_aqi = gdf_aqi.to_crs("EPSG:3857")
        gdf_wf = gdf_wf.to_crs("EPSG:3857")
        gdf_wf["geometry"] = gdf_wf.geometry.buffer(distance_km * 1000)  # Convert km to meters

        gdf_aqi["WithinWildfireDistance"] = gdf_aqi.geometry.apply(
            lambda point: gdf_wf.geometry.intersects(point).any())
        
        gdf_aqi = gdf_aqi.to_crs("EPSG:4326")
        
        df["WithinWildfireDistance"] = gdf_aqi["WithinWildfireDistance"]
        self.logger.info(f"Completed proximity flagging for {len(df)} records.")
        return df

    def process_aqi(self, years_to_process: Optional[List[int]] = None):
        self.logger.info("Starting AQI processing.")
        df = self.clean_dataframe(self.aq_df)
        df['Date'] = pd.to_datetime(df['UTC']).dt.date
        df['Year'] = pd.to_datetime(df['Date']).dt.year
        df['Month'] = pd.to_datetime(df['Date']).dt.month
        years = years_to_process or sorted(df['Year'].unique())
        combined = []
        for year in years:
            year_df = df[df['Year'] == year].copy()
            year_df = self.derive_county(year_df)
            year_df = self.assign_season(year_df)
            year_df = self.categorize_aqi(year_df)
            year_df = self.wildfire_in_county(year_df)
            year_df = self.flag_proximity(year_df, distance_km=50)
            year_df = self.compute_rolling_averages(year_df, window_days=7)
            year_path = os.path.join(self.output_dir, f"aqi_processed_{year}.csv")
            year_df.to_csv(year_path, index=False)
            self.logger.info(f"Saved AQI data for {year} to {year_path}.")
            combined.append(year_df)
        combined_df = pd.concat(combined)
        combined_path = os.path.join(self.output_dir, f"aqi_final_{self.start_year}_{self.end_year}.csv")
        combined_df.to_csv(combined_path, index=False)
        self.logger.info(f"Saved combined AQI data to {combined_path}.")

if __name__ == "__main__":

    # Paths and settings
    wildfire_csv = "../data/wildfire_data/FIRMS_data/wildfire_data_sv_2019_2024.csv"
    aqi_csv = "../data/aqi_data/Colorado_AQI_2019_2024.csv"
    wildfire_output_dir = "../data/wildfire_data/wildfire_processed/"
    aqi_output_dir = "../data/aqi_data/aqi_processed/"
    county_shapefile = "../data/co_shapefile/counties/counties_19.shp"

    start_year = 2019
    end_year = 2024

    # Process Wildfire Data
    wildfire_processor = WildfireProcessor(
        wildfire_filepath=wildfire_csv,
        start_year=start_year,
        end_year=end_year,
        output_dir=wildfire_output_dir,
        county_shapefile=county_shapefile
    )
    wildfire_processor.process_wildfire(year_range=(start_year, end_year))

    # Load processed wildfire data for AQI processing
    processed_wildfire_csv = os.path.join(
        wildfire_output_dir, f"wildfire_processed_{start_year}_{end_year}.csv")

    # Process AQI Data
    aqi_processor = AQIProcessor(
        aqi_filepath=aqi_csv,
        wildfire_filepath=processed_wildfire_csv,
        start_year=start_year,
        end_year=end_year,
        output_dir=aqi_output_dir,
        county_shapefile=county_shapefile
    )
    aqi_processor.process_aqi(years_to_process=list(range(start_year, end_year + 1)))
    
    #save df by pollutant
    df = pd.read_csv(f"../data/aqi_data/aqi_processed/aqi_final_{start_year}_{end_year}.csv")
    pm25_df = df[df["Parameter"].str.upper() == "PM2.5"]
    ozone_df = df[df["Parameter"].str.upper() == "OZONE"]
    pm25_df.to_csv(f"../data/aqi_data/aqi_processed/pm25_aqi_{start_year}_{end_year}.csv", index=False)
    ozone_df.to_csv(f"../data/aqi_data/aqi_processed/ozone_aqi_{start_year}_{end_year}.csv", index=False)
    # Load the PM2.5 AQI data
    df = pd.read_csv("../data/aqi_data/aqi_processed/pm25_aqi_2019_2024.csv")

    # Ensure Date is datetime
    df['Date'] = pd.to_datetime(df['Date'])

    # Filter for 2023
    df_2023 = df[df['Date'].dt.year == 2023].copy()
    df_2023 = df_2023.sort_values('Date')

    # Rolling 7-day average
    df_2023['Rolling_AQI'] = df_2023['AQI'].rolling(window=7, min_periods=1).mean()

    # Group by week and month
    df_2023['Week'] = df_2023['Date'].dt.isocalendar().week
    df_2023['Month'] = df_2023['Date'].dt.month

    # Map month numbers to names
    month_map = {
        1: 'Jan', 2: 'Feb', 3: 'Mar', 4: 'Apr',
        5: 'May', 6: 'Jun', 7: 'Jul', 8: 'Aug',
        9: 'Sep', 10: 'Oct', 11: 'Nov', 12: 'Dec'
    }
    df_2023['Month_Name'] = df_2023['Month'].map(month_map)

    # Get weekly averages
    weekly_avg = df_2023.groupby(['Week', 'Month_Name'])['Rolling_AQI'].mean().reset_index()

    # Plot
    plt.figure(figsize=(14, 6))
    plt.plot(weekly_avg['Week'], weekly_avg['Rolling_AQI'], marker='o')
    plt.title('Weekly 7-Day Rolling Average AQI for PM2.5 in 2023')
    plt.xlabel('Week Number')
    plt.ylabel('Rolling 7-Day Average AQI')

    # Map the x-ticks (weeks) to the corresponding month names
    week_labels = weekly_avg['Month_Name']
    plt.xticks(ticks=weekly_avg['Week'], labels=week_labels, rotation=45)

    plt.grid(True)
    plt.tight_layout()
    plt.show()