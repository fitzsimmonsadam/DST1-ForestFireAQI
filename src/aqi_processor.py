import os
import logging
import pandas as pd
import numpy as np
import functools
from typing import Optional, List
import geopandas as gpd
from shapely.geometry import Point
from datetime import date

def skip_if_exists(file_names: List[str]):
    """
    Decorator that skips the decorated function if specified output files exist in self.output_dir,
    unless force_run=True is passed.
    Args:
        file_names: List of file names to check in self.output_dir.
    """
    def decorator(func):
        @functools.wraps(func)
        def wrapper(self, *args, force_run=False, **kwargs):
            if force_run:
                self.logger.info(f"Force run enabled: Executing {func.__name__} despite existing files.")
                return func(self, *args, **kwargs)
            all_exist = all(os.path.exists(os.path.join(self.output_dir, f)) for f in file_names)
            if all_exist:
                self.logger.info(f"Skipping {func.__name__}: {', '.join(file_names)} already exists.")
                return
            return func(self, *args, **kwargs)
        return wrapper
    return decorator

class AQIProcessor:
    def __init__(
        self,
        aqi_filepath: str,
        wildfire_filepath: str,
        start_year: int,
        end_year: int,
        output_dir: Optional[str] = None,
        county_shapefile: str = "data/co_shapefile/counties/counties_19.shp"
        ):

        self.aqi_filepath = aqi_filepath
        self.wildfire_filepath = wildfire_filepath
        self.start_year = start_year
        self.end_year = end_year
        self.output_dir = output_dir or "AQIProcessed"
        self.county_shapefile = county_shapefile
        os.makedirs(self.output_dir, exist_ok=True)

        self.logger = logging.getLogger(f"{__name__}.AQIProcessor")
        self.logger.setLevel(logging.INFO)
        if self.logger.hasHandlers():
            self.logger.handlers.clear()
        fh = logging.FileHandler(os.path.join(self.output_dir, "aqi_processor.log"), mode="w")
        fh.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s - %(message)s"))
        self.logger.addHandler(fh)
        sh = logging.StreamHandler()
        sh.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s - %(message)s"))
        self.logger.addHandler(sh)
        self.logger.info("AQIProcessor initialized.")

        self.logger.info(f"Loading AQI data from {self.aqi_filepath}.")
        self.aq_df = pd.read_csv(self.aqi_filepath)
        self.wildfire_df = pd.read_csv(self.wildfire_filepath)

    @staticmethod
    def clean_dataframe(df: pd.DataFrame) -> None:
        """Replace invalid values (e.g., -999) with NaN."""
        df.replace(-999, np.nan, inplace=True)
    
    def categorize_aqi(self, aqi_value):
        """Categorizes AQI value into EPA-defined categories."""
        if pd.isna(aqi_value):
            return "Unknown"
        if aqi_value <= 50:
            return "Good"  
        elif aqi_value <= 100:
            return "Moderate"  
        elif aqi_value <= 150:
            return "Unhealthy for Sensitive Groups"
        elif aqi_value <= 200:
            return "Unhealthy"
        elif aqi_value <= 300:
            return "Very Unhealthy"
        elif aqi_value <= 500:
            return "Hazardous"
        else:
            return "Unknown"

    def derive_county(self):
        """Assigns county names to AQI records based on latitude/longitude."""
        if not os.path.exists(self.county_shapefile):
            self.logger.error(f"County shapefile not found at: {self.county_shapefile}")
            return
        
        if {"Latitude", "Longitude"}.issubset(self.aq_df.columns):
            try:
                gdf = gpd.GeoDataFrame(
                    self.aq_df.copy(),
                    geometry=gpd.points_from_xy(self.aq_df["Longitude"], self.aq_df["Latitude"]),
                    crs="EPSG:4326"
                )
                counties = gpd.read_file(self.county_shapefile).to_crs(gdf.crs)
                joined = gpd.sjoin(gdf, counties, how="left", predicate="within")
                county_name_col = "NAME" if "NAME" in counties.columns else counties.columns[0]
                self.aq_df["County"] = joined[county_name_col].values
                self.logger.info(f"Derived 'County' using shapefile column '{county_name_col}'.")
            except Exception as e:
                self.logger.error(f"Error during county derivation: {e}")
        else:
            self.logger.warning("Latitude/Longitude columns missing. Cannot derive counties.")

    def wildfire_in_county(self):
        wildfire_dates_counties = self.wildfire_df[['StartDate', 'County']].drop_duplicates()
        self.aq_df["Wildfire_In_County"] = self.aq_df.apply(
            lambda row: ((wildfire_dates_counties['StartDate'] == pd.to_datetime(row['Date'])).any() and
                         (wildfire_dates_counties['County'] == row['County']).any()), axis=1)

    def wildfire_within_distance(self, distance_km: float):
        fire_gdf = gpd.GeoDataFrame(
            self.wildfire_df,
            geometry=gpd.points_from_xy(self.wildfire_df["longitude"], self.wildfire_df["latitude"]),
            crs="EPSG:4326"
        )
        aqi_gdf = gpd.GeoDataFrame(
            self.aq_df,
            geometry=gpd.points_from_xy(self.aq_df["Longitude"], self.aq_df["Latitude"]),
            crs="EPSG:4326"
        )
        fire_gdf = fire_gdf.to_crs(epsg=3857)
        aqi_gdf = aqi_gdf.to_crs(epsg=3857)
        buffer = fire_gdf.copy()
        buffer["geometry"] = buffer.geometry.buffer(distance_km * 1000)

        self.aq_df["Wildfire_Within_Distance"] = aqi_gdf.geometry.apply(
            lambda point: buffer.intersects(point).any()
        )

    def compute_rolling_averages(self, window_days=7):
        self.aq_df['Date'] = pd.to_datetime(self.aq_df['Date'])
        self.aq_df.sort_values('Date', inplace=True)
        self.aq_df['Rolling_AQI'] = self.aq_df.groupby('County')['AQI'].transform(
            lambda x: x.rolling(window=window_days, min_periods=1).mean()
        )

    @skip_if_exists([f"aqi_preprocessed_{'{start_year}'}_{'{end_year}'}.csv"])
    def preprocess_aqi(self, force_run=False, date_range: Optional[tuple] = None) -> None:
        """Preprocesses AQI data: cleaning, categorization, season assignment, county derivation."""
        self.logger.info("Starting AQI preprocessing.")
        self.clean_dataframe(self.aq_df)

        #add dates
        if 'UTC' in self.aq_df.columns:
            self.aq_df['UTC'] = pd.to_datetime(self.aq_df['UTC'], errors='coerce')
            self.aq_df['Date'] = self.aq_df['UTC'].dt.date
            self.aq_df['Month'] = self.aq_df['UTC'].dt.month
        elif 'Date' in self.aq_df.columns:
            self.aq_df['Date'] = pd.to_datetime(self.aq_df['Date'], errors='coerce').dt.date
        else:
            self.logger.warning("No date column found in AQI data.")
        
        #filter date range
        if date_range:
            start_date, end_date = date_range
            self.aq_df['Date'] = pd.to_datetime(self.aq_df.get('UTC', self.aq_df.get('Date')), errors='coerce').dt.date
            self.aq_df = self.aq_df[(self.aq_df['Date'] >= start_date) & (self.aq_df['Date'] <= end_date)]

        #categorize AQI
        if 'AQI' in self.aq_df.columns:
            self.aq_df['AQI_Category'] = self.aq_df['AQI'].apply(self.categorize_aqi)
        
        #add seasons
        if 'Month' in self.aq_df.columns:
            self.aq_df['Season'] = self.aq_df['Month'].map({
                12: 'Winter', 1: 'Winter', 2: 'Winter',
                3: 'Spring', 4: 'Spring', 5: 'Spring',
                6: 'Summer', 7: 'Summer', 8: 'Summer',
                9: 'Fall', 10: 'Fall', 11: 'Fall'
            })
        else:
            self.logger.warning("No 'Month' column found; cannot compute season.")
        
        #add counties
        if 'County' not in self.aq_df.columns:
            self.logger.info("Deriving 'County' via spatial join.")
            self.derive_county()
        else:
            self.aq_df['County'] = self.aq_df['County'].astype(str).str.strip()
            
        # Add wildfire tags
        self.wildfire_in_county()
        self.wildfire_within_distance(distance_km=50)

        # Add rolling averages
        self.compute_rolling_averages(window_days=7)

        #save preprocessed
        output_path = os.path.join(self.output_dir, f"aqi_preprocessed_{self.start_year}_{self.end_year}.csv")
        self.aq_df.to_csv(output_path, index=False)
        self.logger.info(f"Preprocessed AQI data saved to {output_path}.")
        print(f"Preprocessed AQI data saved to {output_path}.")

    def get_final_df(self, dataset:str="aqi", date_range: Optional[tuple] = None) -> pd.DataFrame:
        """Returns the specified dataset of interest.
        Args:
            dataset (str): One of:
                - "aqi" -> Full AQI data.
                - "pm25" -> PM2.5 data only.
                - "ozone" -> OZONE data only.
            date_range (tuple, optional): A tuple of (start_date, end_date) as datetime.date objects
                eg. date_range=(date(2021, 6, 1), date(2021, 9, 30))
        Returns:
            pd.DataFrame: Requested dataset.
        """
        if self.aq_df.empty:
            preprocessed_path = os.path.join(self.output_dir, f"aqi_preprocessed_{self.start_year}_{self.end_year}.csv")
            if os.path.exists(preprocessed_path):
                self.logger.info(f"Loading preprocessed AQI data from {preprocessed_path}.")
                self.aq_df = pd.read_csv(preprocessed_path)
            else:
                self.logger.error("No AQI data available.")
                return pd.DataFrame()
        df = self.aq_df.copy()

        if dataset == "pm25":
            return self.aq_df[self.aq_df["Parameter"].str.upper() == "PM2.5"]

        elif dataset == "ozone":
            return self.aq_df[self.aq_df["Parameter"].str.upper() == "OZONE"]

        elif dataset != "aqi":
            self.logger.warning(f"Invalid dataset option '{dataset}'. Returning full AQI dataset.")

        if date_range:
            start_date, end_date = date_range
            df['Date'] = pd.to_datetime(df['Date'], errors='coerce').dt.date
            df = df[(df['Date'] >= start_date) & (df['Date'] <= end_date)]
            self.logger.info(f"Filtered dataset from {start_date} to {end_date}.")

        return df

if __name__ == "__main__":
    
    AQI_FILEPATH = "/Users/alexvalone/Desktop/DS_Q2/DS_Tools1/Final_Project/DST1-ForestFireAQI/data/aqi_data/Colorado_AQI_2019_2024.csv"
    WILDFIRE_FILEPATH = "/Users/alexvalone/Desktop/DS_Q2/DS_Tools1/Final_Project/DST1-ForestFireAQI/data/wildfire_data/colorado_wildfires_2019_2024.csv"
    COUNTY_SHAPEFILE = "/Users/alexvalone/Desktop/DS_Q2/DS_Tools1/Final_Project/DST1-ForestFireAQI/data/co_shapefile/counties/counties_19.shp"
    OUTPUT_DIR = "/Users/alexvalone/Desktop/DS_Q2/DS_Tools1/Final_Project/DST1-ForestFireAQI/data/aqi_processed"

    START_YEAR = 2019
    END_YEAR = 2024

    processor = AQIProcessor(
        aqi_filepath=AQI_FILEPATH,
        wildfire_filepath=WILDFIRE_FILEPATH,
        start_year=START_YEAR,
        end_year=END_YEAR,
        output_dir=OUTPUT_DIR,
        county_shapefile=COUNTY_SHAPEFILE
    )

    final_output_path = os.path.join(
        OUTPUT_DIR, f"aqi_final_{START_YEAR}_{END_YEAR}.csv"
    )
    processor.aq_df.to_csv(final_output_path, index=False)
    processor.logger.info(f"Final AQI dataset saved to {final_output_path}")
    processor.get_final_df("aqi", (date(2024, 1, 1), date(2024, 12, 31)))