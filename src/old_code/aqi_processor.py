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
    """
    Class for processing Air Quality Index (AQI) data.

    Parameters:
        aqi_filepath (str): The file path to the AQI data file.
        wildfire_filepath (str): The file path to the wildfire data file.
        start_year (int): The starting year for data processing.
        end_year (int): The ending year for data processing.
        output_dir (Optional[str]): The output directory for saving processed data. If not provided, a default directory will be used.
        county_shapefile (str): The file path to the county shapefile for deriving county names based on latitude/longitude.

    Attributes:
        aqi_filepath (str): The file path to the AQI data file.
        wildfire_filepath (str): The file path to the wildfire data file.
        start_year (int): The starting year for data processing.
        end_year (int): The ending year for data processing.
        output_dir (str): The output directory for saving processed data.
        county_shapefile (str): The file path to the county shapefile for deriving county names based on latitude/longitude.
        logger (logging.Logger): The logger for logging messages.
        aq_df (pd.DataFrame): The DataFrame for storing the AQI data.
        wildfire_df (pd.DataFrame): The DataFrame for storing the wildfire data.

    Methods:
        setup_logger(): Set up the logger for logging messages.
        clean_dataframe(df: pd.DataFrame) -> pd.DataFrame: Clean the AQI DataFrame by replacing invalid values and keeping only specified columns.
        categorize_aqi(aqi_value) -> str: Categorize the AQI value into EPA-defined categories.
        derive_county(df: pd.DataFrame) -> pd.DataFrame: Assign county names to a given DataFrame based on latitude/longitude.
        wildfire_in_county(df: pd.DataFrame) -> pd.DataFrame: Tag the DataFrame rows with whether there was a wildfire in the county on the given date.
        wildfire_within_distance(df: pd.DataFrame, distance_km: float) -> pd.DataFrame: Tag the DataFrame rows with whether there was a wildfire within a certain distance of the location.
        compute_rolling_averages(df: pd.DataFrame, window_days=7) -> pd.DataFrame: Compute rolling averages of AQI values for each county.
        combine_yearly_csvs(): Combine all yearly AQI CSV files into one final CSV.
        preprocess_aqi(force_run=False, date_range: Optional[tuple] = None) -> None: Preprocess the AQI data year-by-year.

    """
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

        self.logger = self.setup_logger()
        self.logger.info("AQIProcessor initialized.")

        self.logger.info(f"Loading AQI data from {self.aqi_filepath}.")
        self.aq_df = pd.read_csv(self.aqi_filepath)
        self.wildfire_df = pd.read_csv(self.wildfire_filepath)

    def setup_logger(self):
        logger = logging.getLogger(f"{__name__}.AQIProcessor")
        logger.setLevel(logging.INFO)
        if logger.hasHandlers():
            logger.handlers.clear()
        fh = logging.FileHandler(os.path.join(self.output_dir, "aqi_processor.log"), mode="w")
        fh.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s - %(message)s"))
        logger.addHandler(fh)
        sh = logging.StreamHandler()
        sh.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s - %(message)s"))
        logger.addHandler(sh)
        return logger

    @staticmethod
    def clean_dataframe(df: pd.DataFrame) -> pd.DataFrame:
        """
        Replace invalid values (e.g., -999) with NaN.
        Keep only specified columns:
        - Latitude
        - Longitude
        - UTC
        - Parameter
        - AQI
        - Category
        """
        desired_columns = ["Latitude", "Longitude", "UTC", "Parameter", "AQI", "Category"]
        df.replace(-999, np.nan, inplace=True)
        existing_columns = [col for col in desired_columns if col in df.columns]
        if not existing_columns:
            raise ValueError("None of the specified columns were found in the AQI data.")
        df = df[existing_columns]
        return df
    
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

    def derive_county(self, df: pd.DataFrame) -> pd.DataFrame:
        """Assigns county names to a given dataframe based on latitude/longitude."""
        if not os.path.exists(self.county_shapefile):
            self.logger.error(f"County shapefile not found at: {self.county_shapefile}")
            df["County"] = np.nan
            return df

        if {"Latitude", "Longitude"}.issubset(df.columns):
            try:
                gdf = gpd.GeoDataFrame(
                    df.copy(),
                    geometry=gpd.points_from_xy(df["Longitude"], df["Latitude"]),
                    crs="EPSG:4326"
                )
                counties = gpd.read_file(self.county_shapefile).to_crs(gdf.crs)
                joined = gpd.sjoin(gdf, counties, how="left", predicate="within")
                county_name_col = "NAME" if "NAME" in counties.columns else counties.columns[0]
                df["County"] = joined[county_name_col].values
                self.logger.info(f"Derived 'County' using shapefile column '{county_name_col}'.")
            except Exception as e:
                df["County"] = np.nan
                self.logger.error(f"Error during county derivation: {e}")
        else:
            self.logger.warning("Latitude/Longitude columns missing. Cannot derive counties.")
            df["County"] = np.nan

        return df
        
    def wildfire_in_county(self, df: pd.DataFrame) -> pd.DataFrame:
        """Tags the DataFrame rows with whether there was a wildfire in the county on the given date."""
        wildfire_dates_counties = self.wildfire_df[['acq_date', 'County']].drop_duplicates()
        wildfire_dates_counties['acq_date'] = pd.to_datetime(wildfire_dates_counties['acq_date']).dt.date
        df["Wildfire_In_County"] = df.apply(
            lambda row: (
                ((wildfire_dates_counties['acq_date'] == row['Date']).any()) and
                ((wildfire_dates_counties['County'] == row['County']).any())
                ), axis=1)
        self.logger.info(f"Completed wildfire in county tagging for {len(df)} records.")
        return df

    def wildfire_within_distance(self, df: pd.DataFrame, distance_km: float) -> pd.DataFrame:
        """Tags the DataFrame rows with whether there was a wildfire within a certain distance of the monitor location."""
        fire_gdf = gpd.GeoDataFrame(
            self.wildfire_df,
            geometry=gpd.points_from_xy(self.wildfire_df["longitude"], self.wildfire_df["latitude"]),
            crs="EPSG:4326"
        )
        aqi_gdf = gpd.GeoDataFrame(
            df,
            geometry=gpd.points_from_xy(df["Longitude"], df["Latitude"]),
            crs="EPSG:4326"
        )
        fire_gdf = fire_gdf.to_crs(epsg=3857)
        aqi_gdf = aqi_gdf.to_crs(epsg=3857)
        buffer = fire_gdf.copy()
        buffer["geometry"] = buffer.geometry.buffer(distance_km * 1000)

        df["Wildfire_Within_Distance"] = aqi_gdf.geometry.apply(
            lambda point: buffer.intersects(point).any()
        )
        self.logger.info("Completed wildfire proximity tagging.")
        return df

    def compute_rolling_averages(self, df: pd.DataFrame, window_days=7) -> pd.DataFrame:
        df['Date'] = pd.to_datetime(df['Date'])
        df.sort_values('Date', inplace=True)
        df['Rolling_AQI'] = df.groupby('County')['AQI'].transform(
            lambda x: x.rolling(window=window_days, min_periods=1).mean())
        self.logger.info("Rolling averages computed.")
        return df

    def combine_yearly_csvs(self):
        """
        Combines all yearly AQI CSV files into one final CSV.
        """
        self.logger.info("Combining yearly AQI CSVs into one final dataset.")
        csv_files = [
            os.path.join(self.output_dir, f)
            for f in os.listdir(self.output_dir)
            if f.startswith("aqi_preprocessed_") and f.endswith(".csv")]
        if not csv_files:
            self.logger.error("No yearly AQI CSV files found to combine.")
            return
        combined_df = pd.concat(
            (pd.read_csv(f) for f in csv_files),
            ignore_index=True)
        combined_path = os.path.join(
            self.output_dir, f"aqi_final_{self.start_year}_{self.end_year}.csv")
        combined_df.to_csv(combined_path, index=False)
        self.logger.info(f"Final AQI dataset saved to {combined_path}.")

    #@skip_if_exists([f"aqi_preprocessed_{'{start_year}'}_{'{end_year}'}.csv"])
    def preprocess_aqi(self, force_run=False, date_range: Optional[tuple] = None, years_to_process: Optional[List[int]]=None) -> pd.DataFrame:
        """Preprocesses AQI data year-by-year: cleaning, date filtering, categorization, county derivation, wildfire tagging, and rolling averages."""
        
        self.logger.info("Starting AQI preprocessing.")
        # Clean and filter columns
        self.aq_df = self.clean_dataframe(self.aq_df)

        #get dates
        if 'UTC' in self.aq_df.columns:
            self.aq_df['UTC'] = pd.to_datetime(self.aq_df['UTC'], errors='coerce')
            self.aq_df['Date'] = self.aq_df['UTC'].dt.date
            self.aq_df['Month'] = self.aq_df['UTC'].dt.month
        elif 'Date' in self.aq_df.columns:
            self.aq_df['Date'] = pd.to_datetime(self.aq_df['Date'], errors='coerce').dt.date
            self.aq_df['Month'] = pd.to_datetime(self.aq_df['Date'], errors='coerce').dt.month
        else:
            self.logger.error("No valid date column ('UTC' or 'Date') found. Cannot preprocess.")
            return
        #get year
        self.aq_df['Year'] = pd.to_datetime(self.aq_df['Date'], errors='coerce').dt.year

        # apply date range if provided
        if date_range:
            start_date, end_date = date_range
            self.aq_df = self.aq_df[(self.aq_df['Date'] >= start_date) & (self.aq_df['Date'] <= end_date)]
            self.logger.info(f"Filtered data to date range: {start_date} - {end_date}.")

        # get years
        available_years = sorted(self.aq_df['Year'].dropna().unique())
        if years_to_process:
            years = [year for year in years_to_process if year in available_years]
            if not years:
                self.logger.error(f"No matching years found in the dataset from {years_to_process}. Aborting preprocessing.")
                return
            self.logger.info(f"Processing specified years: {years}.")
        else:
            years = available_years
            self.logger.info(f"Processing all available years: {years}.")
        
        processed_dfs = []
        #get each year df
        for year in years:
            year_df = self.aq_df[self.aq_df['Year'] == year].copy()

            if 'AQI' in year_df.columns:
                year_df['AQI_Category'] = year_df['AQI'].apply(self.categorize_aqi)

            year_df['Season'] = year_df['Month'].map({
                12: 'Winter', 1: 'Winter', 2: 'Winter',
                3: 'Spring', 4: 'Spring', 5: 'Spring',
                6: 'Summer', 7: 'Summer', 8: 'Summer',
                9: 'Fall', 10: 'Fall', 11: 'Fall'
            })

            year_df = self.derive_county(year_df)
            year_df = self.wildfire_in_county(year_df)
            year_df = self.wildfire_within_distance(year_df, distance_km=50)
            year_df = self.compute_rolling_averages(year_df, window_days=7)

            year_output_path = os.path.join(self.output_dir, f"aqi_preprocessed_{int(year)}.csv")
            year_df.to_csv(year_output_path, index=False)
            self.logger.info(f"Processed year {int(year)} and saved to {year_output_path}.")
            
            processed_dfs.append(year_df)
        combined_df = pd.concat(processed_dfs, ignore_index=True)
        self.logger.info(f"Processed AQI data for {years} saved to .")
        return combined_df

if __name__ == "__main__":
 
    AQI_FILEPATH = "data/aqi_data/Colorado_AQI_2019_2024.csv"
    WILDFIRE_FILEPATH = "data/wildfire_data/wildfire_preprocessed_2019_2024.csv"
    COUNTY_SHAPEFILE = "data/co_shapefile/counties/counties_19.shp"
    OUTPUT_DIR = "data/aqi_processed"
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

    processor.preprocess_aqi(years_to_process=[2020, 2021])
    processor.combine_yearly_csvs()

    final_output_path = os.path.join(OUTPUT_DIR, f"aqi_final_{START_YEAR}_{END_YEAR}.csv")
    processor.aq_df = pd.read_csv(final_output_path)
    processor.logger.info(f"Final AQI dataset loaded from {final_output_path}")

    #save df by pollutant
    df = pd.read_csv("aqi_final_2023_2024.csv")
    pm25_df = df[df["Parameter"].str.upper() == "PM2.5"]
    ozone_df = df[df["Parameter"].str.upper() == "OZONE"]
    pm25_df.to_csv("pm25_aqi_final_2023_2024.csv", index=False)
    ozone_df.to_csv("ozone_aqi_final_2023_2024.csv", index=False)