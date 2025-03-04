import os
import pandas as pd
import numpy as np
import geopandas as gpd
import logging
from typing import Optional


class WildfireProcessor:
    """
    Processes wildfire detection data for Colorado wildfire analysis.
    Includes cleaning, date processing, county assignment, seasonal categorization, and confidence classification.
    """
    def __init__(
        self,
        wildfire_filepath: str,
        start_year: int,
        end_year: int,
        output_dir: Optional[str] = None,
        county_shapefile: str = "data/co_shapefile/counties/counties_19.shp"
    ):
        self.wildfire_filepath = wildfire_filepath
        self.start_year = start_year
        self.end_year = end_year
        self.output_dir = output_dir or f"Analysis_Output_{start_year}_{end_year}"
        self.county_shapefile = county_shapefile
        os.makedirs(self.output_dir, exist_ok=True)

        self.logger = self.setup_logger()
        self.logger.info("WildfireProcessor initialized.")

        self.logger.info(f"Loading wildfire data from {self.wildfire_filepath}.")
        self.wildfire_df = pd.read_csv(self.wildfire_filepath)

    def setup_logger(self):
        logger = logging.getLogger(f"{__name__}.WildfireProcessor")
        logger.setLevel(logging.INFO)
        if logger.hasHandlers():
            logger.handlers.clear()
        fh = logging.FileHandler(os.path.join(self.output_dir, "wildfire_processor.log"), mode="w")
        fh.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s - %(message)s"))
        logger.addHandler(fh)
        sh = logging.StreamHandler()
        sh.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s - %(message)s"))
        logger.addHandler(sh)
        return logger

    def clean_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        """Replace invalid values and keep only desired columns."""
        self.logger.info("Cleaning dataframe and selecting desired columns.")
        df.replace(-999, np.nan, inplace=True)
        desired_columns = ["latitude", "longitude", "acq_date", "frp", "confidence", "type"]
        existing_columns = [col for col in desired_columns if col in df.columns]
        if not existing_columns:
            self.logger.error("None of the desired columns were found.")
            return pd.DataFrame()
        return df[existing_columns]

    def categorize_confidence(self, df: pd.DataFrame) -> pd.DataFrame:
        self.logger.info("Categorizing confidence levels.")
        bins = [-np.inf, 30, 80, np.inf]
        labels = ["Low", "Medium", "High"]
        df["Confidence_Category"] = pd.cut(df["confidence"], bins=bins, labels=labels)
        return df

    def assign_season(self, df: pd.DataFrame) -> pd.DataFrame:
        self.logger.info("Assigning seasons based on month.")
        df['Month'] = pd.to_datetime(df['acq_date'], errors='coerce').dt.month
        df['Season'] = df['Month'].map({
            12: 'Winter', 1: 'Winter', 2: 'Winter',
            3: 'Spring', 4: 'Spring', 5: 'Spring',
            6: 'Summer', 7: 'Summer', 8: 'Summer',
            9: 'Fall', 10: 'Fall', 11: 'Fall'
        })
        return df

    def derive_county(self, df: pd.DataFrame) -> pd.DataFrame:
        if not os.path.exists(self.county_shapefile):
            self.logger.error(f"County shapefile not found: {self.county_shapefile}")
            df["County"] = np.nan
            return df

        self.logger.info("Deriving county for wildfire detections.")
        gdf = gpd.GeoDataFrame(
            df.copy(),
            geometry=gpd.points_from_xy(df["longitude"], df["latitude"]),
            crs="EPSG:4326"
        )
        counties = gpd.read_file(self.county_shapefile).to_crs(gdf.crs)
        joined = gpd.sjoin(gdf, counties, how="left", predicate="within")
        county_name_col = "NAME" if "NAME" in counties.columns else counties.columns[0]
        df["County"] = joined[county_name_col].values
        return df

    def preprocess(self, year_range: Optional[tuple] = None) -> None:
        """
        Full wildfire preprocessing pipeline with yearly splitting and saving.
        """
        self.logger.info("Starting wildfire preprocessing.")
        self.wildfire_df = self.clean_dataframe(self.wildfire_df)
        self.wildfire_df['acq_date'] = pd.to_datetime(self.wildfire_df['acq_date'], errors='coerce')
        self.wildfire_df['Year'] = self.wildfire_df['acq_date'].dt.year

        if year_range:
            start_year, end_year = year_range
            self.wildfire_df = self.wildfire_df[
                (self.wildfire_df['Year'] >= start_year) &
                (self.wildfire_df['Year'] <= end_year)
            ]
            self.logger.info(f"Filtered wildfire data to years {start_year}-{end_year}.")

        years = sorted(self.wildfire_df['Year'].dropna().unique())
        if not years:
            self.logger.error("No valid years found after date parsing. Aborting preprocessing.")
            return

        combined_df = []

        for year in years:
            self.logger.info(f"Processing year {int(year)}.")
            year_df = self.wildfire_df[self.wildfire_df['Year'] == year].copy()
            year_df = self.categorize_confidence(year_df)
            year_df = self.assign_season(year_df)
            year_df = self.derive_county(year_df)

            year_output_path = os.path.join(self.output_dir, f"wildfire_preprocessed_{int(year)}.csv")
            year_df.to_csv(year_output_path, index=False)
            self.logger.info(f"Saved preprocessed data for {int(year)} to {year_output_path}.")

            combined_df.append(year_df)

        if combined_df:
            combined_df = pd.concat(combined_df, ignore_index=True)
            combined_output_path = os.path.join(
                self.output_dir, f"wildfire_preprocessed_{self.start_year}_{self.end_year}.csv"
            )
            combined_df.to_csv(combined_output_path, index=False)
            self.logger.info(f"Saved combined wildfire data to {combined_output_path}.")

if __name__ == "__main__":
    
    wildfire_csv = "data/wildfire_data/FIRMS_data/wildfire_data_sv_2019_2024.csv"
    county_shapefile = "data/co_shapefile/counties/counties_19.shp"
    output_dir = "data/wildfire_data/wildfire_processed"
    start_year = 2019
    end_year = 2020

    processor = WildfireProcessor(
        wildfire_filepath=wildfire_csv,
        start_year=start_year,
        end_year=end_year,
        output_dir=output_dir,
        county_shapefile=county_shapefile
    )
    processor.preprocess(year_range=(start_year, end_year))