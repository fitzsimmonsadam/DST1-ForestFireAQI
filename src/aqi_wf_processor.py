import os
import pandas as pd
import numpy as np
import geopandas as gpd
import logging
from typing import Optional, List
from shapely.geometry import Point


def derive_county(df, lon_col, lat_col, county_shapefile, final_columns=None):
    """
    Assigns county names based on latitude and longitude.
    
    Parameters:
        df (pd.DataFrame): DataFrame with coordinates.
        lon_col (str): Longitude column name.
        lat_col (str): Latitude column name.
        county_shapefile (str): Path to county shapefile.

    Returns:
        pd.DataFrame: The input DataFrame with a 'County' column added.
    """
    
    # Convert df to GeoDataFrame
    gdf = gpd.GeoDataFrame(
        df,
        geometry=gpd.points_from_xy(df[lon_col], df[lat_col]),
        crs="EPSG:4269"
    )

    # Load and prepare counties shapefile
    counties = gpd.read_file(county_shapefile).to_crs(gdf.crs)
    # Drop 'index_right' from previous joins if it exists
    counties = counties.drop(columns=["index_right"], errors="ignore")
    gdf = gdf.drop(columns=["index_right"], errors="ignore")

    # Perform spatial join
    joined = gpd.sjoin(gdf, counties[["geometry", "NAME"]], how="left", predicate="within")

    # Ensure 'County' column is created correctly
    if "NAME" in joined.columns:
        joined.rename(columns={"NAME": "County"}, inplace=True)
    elif "NAME_left" in joined.columns:
        joined.rename(columns={"NAME_left": "County"}, inplace=True)
    elif "NAME_right" in joined.columns:
        joined.rename(columns={"NAME_right": "County"}, inplace=True)
    else:
        joined["County"] = np.nan 

    joined.drop(columns=["geometry", "index_right"], errors="ignore", inplace=True)

    if final_columns:
        return joined[final_columns]
    else:
        return joined

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
        df = df[desired_columns].copy()
        df["confidence"] = df["confidence"].astype(str).str.lower()
        return df

    def filter_confidence_level(self, df, confidence_level: Optional[str] = None):
        """
        Filters the dataframe based on confidence level.
        Args:
            df (pd.DataFrame): The wildfire dataframe.
            confidence_level (str, optional): Specify 'n' (nominal), 'l' (low), or 'h' (high) 
                                              to filter. If None, retains all levels.
                                            "nominal" represents the most reliable detection
        Returns:
            pd.DataFrame: The filtered dataframe.
        """
        if confidence_level and confidence_level in ['n', 'l', 'h']:
            filtered_df = df[df["confidence"] == confidence_level]
            self.logger.info(f"Filtered wildfire data to only include confidence level '{confidence_level}'. Remaining records: {len(filtered_df)}")
            return filtered_df
        return df
    
    def assign_season(self, df):
        df['Month'] = pd.to_datetime(df['Date']).dt.month
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

    def process_wildfire(self, year_range: Optional[tuple] = None):
        self.logger.info("Starting wildfire processing.")
        self.wildfire_df = self.clean_dataframe(self.wildfire_df)
        self.wildfire_df.rename(columns={"acq_date": "Date"}, inplace=True)
        self.wildfire_df['Date'] = pd.to_datetime(self.wildfire_df['Date'])
        self.wildfire_df['Year'] = self.wildfire_df['Date'].dt.year

        if year_range:
            start_year, end_year = year_range
            self.wildfire_df = self.wildfire_df[
                (self.wildfire_df['Year'] >= start_year) &
                (self.wildfire_df['Year'] <= end_year)
            ]

        self.wildfire_df = self.filter_to_colorado(self.wildfire_df)
        final_columns = [
        "latitude", "longitude", "Date", "frp", "confidence", "type",
        "Year", "Month", "Season", "County"]   
        confidence_filter = "n"
        self.wildfire_df = self.filter_confidence_level(self.wildfire_df, confidence_filter)
        combined_df = []
        for year in sorted(self.wildfire_df['Year'].unique()):
            year_df = self.wildfire_df[self.wildfire_df['Year'] == year].copy()
            year_df = self.assign_season(year_df)
            year_df = derive_county(year_df, "longitude", "latitude", self.county_shapefile, final_columns=final_columns)
            year_output_path = os.path.join(self.output_dir, f"wildfire_processed_{year}.csv")
            year_df.to_csv(year_output_path, index=False)
            self.logger.info(f"Saved wildfire data for {year} to {year_output_path}.")
            combined_df.append(year_df)

        combined_df = pd.concat(combined_df)
        combined_output_path = os.path.join(self.output_dir, f"wildfire_processed_{self.start_year}_{self.end_year}_{confidence_filter}.csv")
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
        df = df.rename(columns={"UTC": "Date"})
        # Ensure Date is in datetime format
        df["Date"] = pd.to_datetime(df["Date"])
        df["Year"] = pd.to_datetime(df["Date"]).dt.year
        df["Month"] = pd.to_datetime(df["Date"]).dt.month
        return df[["Latitude", "Longitude", "SiteName", "Date", "Month", "Year", "Parameter", "AQI"]]
    
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
        """
        Computes rolling averages for AQI values based on Latitude and Longitude.
        Args:
            df (pd.DataFrame): The AQI DataFrame.
            window_days (int, optional): The rolling window in days. Defaults to 7.
        Returns:
            pd.DataFrame: DataFrame with a new column 'Rolling_AQI'.
        """
        self.logger.info(f"Computing {window_days}-day rolling averages.")

        # Sort values by Latitude, Longitude, and Date for correct rolling calculations
        df = df.sort_values(by=["SiteName", "Date"])

        # Compute rolling averages by Latitude and Longitude
        df["Rolling_AQI"] = df.groupby(["SiteName"])["AQI"].transform(
            lambda x: x.rolling(window=window_days, min_periods=1).mean()
        )

        self.logger.info("Rolling averages computation complete.")
        return df

    def wildfire_in_county(self, df):
        """Flags AQI records that fall within the same county and date as wildfires."""
        self.logger.info("Flagging wildfires in the same county and date.")
        wildfire_dates_counties = self.wildfire_df[['Date', 'County']].drop_duplicates()
        wildfire_dates_counties['Date'] = pd.to_datetime(wildfire_dates_counties['Date']).dt.date
        df["Wildfire_In_County"] = df.apply(
            lambda row: ((wildfire_dates_counties['Date'] == row['Date']).any() and
                         (wildfire_dates_counties['County'] == row['County']).any()),
            axis=1)
        return df

    def process_aqi(self, years_to_process: Optional[List[int]] = None):
        """Processes AQI data by year, applying various transformations and saving results."""
        self.logger.info("Starting AQI processing.")
        df = self.clean_dataframe(self.aq_df)

        final_columns = ["Latitude", "Longitude", "SiteName", "Date", "Month", "Year", "Parameter", "AQI", "County"]
        df= derive_county(df, "Longitude", "Latitude", self.county_shapefile, final_columns=final_columns)

        # Filter by year range if specified
        years = years_to_process or sorted(df['Year'].unique())
        combined = []
        window_days=30
        for year in years:
            year_path = os.path.join(self.output_dir, f"aqi_processed_{year}.csv")
            self.logger.info(f"Processing AQI data for year: {year}")   
            year_df = df[df['Year'] == year].copy()
            #Apply processing
            year_df = self.assign_season(year_df)
            year_df = self.categorize_aqi(year_df)
            year_df=  self.wildfire_in_county(year_df)
            year_df = self.compute_rolling_averages(year_df, window_days=window_days)
            # Save processed data
            year_df.to_csv(year_path, index=False)
            self.logger.info(f"Saved AQI data for {year} to {year_path}.")
            combined.append(year_df)
        combined_df = pd.concat(combined)
        print("Final AQI DataFrame columns:", combined_df.columns.tolist())
        combined_path = os.path.join(self.output_dir, f"aqi_final_{self.start_year}_{self.end_year}_{window_days}.csv")
        combined_df.to_csv(combined_path, index=False)
        self.logger.info(f"Saved combined AQI data to {combined_path}.")

if __name__ == "__main__":

    # Paths and settings
    wildfire_csv = "data/large_data/fire_archive_SV-C2_584955.csv"
    aqi_csv = "data/large_data/Colorado_AQI_2019_2024.csv"
    wildfire_output_dir = "data/wildfire_data/wildfire_processed/"
    aqi_output_dir = "data/aqi_data/aqi_processed/"
    county_shapefile = "data/co_shapefile/counties/counties_19.shp"

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
    #processed_wildfire_csv = os.path.join(
    #    wildfire_output_dir, f"wildfire_processed_{start_year}_{end_year}.csv")
    processed_wildfire_csv = "data/wildfire_data/wildfire_processed/wildfire_processed_2019_2024_n.csv"
    # Process AQI Data
    aqi_processor = AQIProcessor(
        aqi_filepath=aqi_csv,
        wildfire_filepath=processed_wildfire_csv,
        start_year=start_year,
        end_year=end_year,
        output_dir=aqi_output_dir,
        county_shapefile=county_shapefile)
    
    aqi_processor.process_aqi(years_to_process=list(range(start_year, end_year+1)))
    
    #save df by pollutant
    df = pd.read_csv(f"data/aqi_data/aqi_processed/aqi_final_{start_year}_{end_year}_30.csv")
    pm25_df = df[df["Parameter"].str.upper() == "PM2.5"]
    ozone_df = df[df["Parameter"].str.upper() == "OZONE"]
    pm25_df.to_csv(f"data/aqi_data/aqi_processed/pm25_aqi_{start_year}_{end_year}.csv", index=False)
    ozone_df.to_csv(f"data/aqi_data/aqi_processed/ozone_aqi_{start_year}_{end_year}.csv", index=False)