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
        colorado_shapefile: str = "data/co_shapefile/co_boundary/co_boundary.shp"
        ):
        self.wildfire_filepath = wildfire_filepath
        self.start_year = start_year
        self.end_year = end_year
        self.output_dir = output_dir or f"Analysis_Output_{start_year}_{end_year}"
        self.colorado_shapefile = colorado_shapefile
        os.makedirs(self.output_dir, exist_ok=True)

        self.logger = logging.getLogger(f"{__name__}.WildfireProcessor")
        self.logger.setLevel(logging.INFO)
        if self.logger.hasHandlers():
            self.logger.handlers.clear()
        fh = logging.FileHandler(os.path.join(self.output_dir, "wildfire_processor.log"), mode="w")
        fh.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s - %(message)s"))
        self.logger.addHandler(fh)
        sh = logging.StreamHandler()
        sh.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s - %(message)s"))
        self.logger.addHandler(sh)
        self.logger.info("WildfireProcessor initialized.")

        # Load wildfire data
        self.logger.info(f"Loading wildfire data from {self.wildfire_filepath}.")
        self.wildfire_df = pd.read_csv(self.wildfire_filepath)

    def merge_wildfire_data(
        self,
        archive_path: str,
        nrt_path: str,
        output_filename: str = "wildfire_data_2019-2024.csv"
        ) -> None:
        """
        Merges the archive and near-real-time wildfire data into one CSV.

        Args:
            archive_path (str): Path to the archive CSV.
            nrt_path (str): Path to the NRT CSV.
            output_filename (str): Name of the merged output CSV.
        """
        self.logger.info(f"Loading wildfire archive from {archive_path}.")
        archive_df = pd.read_csv(archive_path)
        
        self.logger.info(f"Loading NRT wildfire data from {nrt_path}.")
        nrt_df = pd.read_csv(nrt_path)
        
        self.logger.info("Combining archive and NRT wildfire data.")
        combined_df = pd.concat([archive_df, nrt_df], ignore_index=True)
        
        combined_df['acq_date'] = pd.to_datetime(combined_df['acq_date'], errors='coerce')
        combined_df.sort_values(by='acq_date', inplace=True)
        
        output_path = os.path.join(self.output_dir, output_filename)
        combined_df.to_csv(output_path, index=False)
        
        self.logger.info(f"Merged wildfire data saved to {output_path}.")

    @staticmethod
    def clean_dataframe(df: pd.DataFrame) -> None:
        df.replace(-999, np.nan, inplace=True)

    def categorize_confidence(self):
        """
        Categorizes confidence values into Low (<30), Medium (30-80), High (>80).
        """
        if "confidence" in self.wildfire_df.columns:
            self.logger.info("Categorizing MODIS confidence.")
            bins = [-np.inf, 30, 80, np.inf]
            labels = ["Low", "Medium", "High"]
            self.wildfire_df["Confidence_Category"] = pd.cut(self.wildfire_df["confidence"], bins=bins, labels=labels)
        else:
            self.logger.warning("'confidence' column missing. Skipping confidence categorization.")

    def assign_season(self):
        """
        Assigns a season based on the month of detection.
        """
        self.wildfire_df['Month'] = self.wildfire_df['acq_date'].dt.month
        self.wildfire_df['Season'] = self.wildfire_df['Month'].map({
            12: 'Winter', 1: 'Winter', 2: 'Winter',
            3: 'Spring', 4: 'Spring', 5: 'Spring',
            6: 'Summer', 7: 'Summer', 8: 'Summer',
            9: 'Fall', 10: 'Fall', 11: 'Fall'
        })

    def filter_to_colorado(self):
        """
        Filters wildfire detections to only those within Colorado boundaries.
        """
        self.logger.info(f"Filtering wildfire detections to Colorado using {self.colorado_shapefile}.")
        colorado_boundary = gpd.read_file(self.colorado_shapefile).to_crs("EPSG:4326")
        wildfire_gdf = gpd.GeoDataFrame(
            self.wildfire_df,
            geometry=gpd.points_from_xy(self.wildfire_df.longitude, self.wildfire_df.latitude),
            crs="EPSG:4326"
        )
        filtered_gdf = gpd.sjoin(wildfire_gdf, colorado_boundary, predicate="within")
        self.wildfire_df = pd.DataFrame(filtered_gdf.drop(columns=["geometry", "index_right"]))
        self.logger.info(f"Filtered wildfire data to {len(self.wildfire_df)} records within Colorado.")

    def derive_county(self):
        """
        Assigns county names to wildfire detections using the same Colorado shapefile.
        """
        self.logger.info("Assigning county names to wildfire detections.")
        counties = gpd.read_file(self.colorado_shapefile).to_crs("EPSG:4326")
        wildfire_gdf = gpd.GeoDataFrame(
            self.wildfire_df,
            geometry=gpd.points_from_xy(self.wildfire_df.longitude, self.wildfire_df.latitude),
            crs="EPSG:4326"
        )
        joined = gpd.sjoin(wildfire_gdf, counties, predicate="within")
        county_name_col = "NAME" if "NAME" in counties.columns else counties.columns[0]
        self.wildfire_df["County"] = joined[county_name_col].values
        self.logger.info("County assignment complete.")

    def preprocess(self):
        """
        Full wildfire preprocessing pipeline.
        """
        self.logger.info("Starting wildfire preprocessing.")
        self.filter_to_colorado()
        self.clean_dataframe(self.wildfire_df)

        for date_col in ["StartDate", "EndDate", "acq_date"]:
            if date_col in self.wildfire_df.columns:
                self.wildfire_df[date_col] = pd.to_datetime(self.wildfire_df[date_col], errors="coerce")

        self.assign_season()
        self.categorize_confidence()
        self.derive_county()

        output_path = os.path.join(self.output_dir, f"wildfire_preprocessed_{self.start_year}_{self.end_year}.csv")
        self.wildfire_df.to_csv(output_path, index=False)
        self.logger.info(f"Preprocessed wildfire data saved to {output_path}.")
        print(f"Preprocessed wildfire data saved to {output_path}.")
    
if __name__ == "__main__":
    
    wildfire_csv_path = (
        "/Users/alexvalone/Desktop/DS_Q2/DS_Tools1/Final_Project/"
        "DST1-ForestFireAQI/data/wildfire_data/"
    )
    colorado_shapefile = (
        "/Users/alexvalone/Desktop/DS_Q2/DS_Tools1/Final_Project/"
        "DST1-ForestFireAQI/data/co_shapefile/co_boundary/co_boundary.shp"
    )
    output_dir = "/Users/alexvalone/Desktop/DS_Q2/DS_Tools1/Final_Project/DST1-ForestFireAQI/data/wildfire_data"

    start_year = 2019
    end_year = 2024
    
    processor = WildfireProcessor(
        wildfire_filepath=wildfire_csv_path,
        start_year=start_year,
        end_year=end_year,
        output_dir=output_dir,
        colorado_shapefile=colorado_shapefile
    )
    processor.preprocess()