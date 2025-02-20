#!/usr/bin/env python
import os
import requests
import zipfile
import json
import pandas as pd

class WildfireDataDownloader:
    """
    Downloads and extracts wildfire data files (any data source).
    Checks for existing files before re-downloading or re-extracting.
    """

    def __init__(self, download_info, output_dir="wildfire_data"):
        """
        :param download_info: List of dicts, each with:
                              - 'download_id': Download identifier.
                              - 'data_source': Source (e.g., 'modis-c6.1', 'jpss1-viirs-c2', etc.).
                              - 'url': URL for the ZIP file.
        :param output_dir: Directory to store downloads and extracted files.
        """
        self.download_info = download_info
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)

    def download_and_extract(self, info):
        """
        Download the ZIP file from the URL and extract its contents.
        Checks if the file already exists before downloading.
        """
        zip_filename = f"{info['download_id']}_{info['data_source']}.zip"
        zip_filepath = os.path.join(self.output_dir, zip_filename)
        
        # Check if the ZIP file is already downloaded
        if os.path.exists(zip_filepath):
            print(f"ZIP file '{zip_filepath}' already exists. Skipping download.")
        else:
            print(f"Downloading data for {info['data_source']} (ID: {info['download_id']})")
            response = requests.get(info['url'])
            if response.status_code == 200:
                with open(zip_filepath, "wb") as f:
                    f.write(response.content)
                print(f"Saved ZIP file as '{zip_filepath}'")
            else:
                print(f"Failed to download data for ID {info['download_id']}. Status code: {response.status_code}")
                return

        # Check if the JSON data has already been extracted
        extracted_files = [
            f for f in os.listdir(self.output_dir)
            if f.endswith(".json") and f.startswith(f"{info['download_id']}_{info['data_source']}")
        ]
        if extracted_files:
            print(f"Data already extracted for {info['download_id']} {info['data_source']}. Skipping extraction.")
            return

        # Extract the ZIP file
        try:
            with zipfile.ZipFile(zip_filepath, "r") as zip_ref:
                zip_ref.extractall(self.output_dir)
            print(f"Extracted ZIP file to '{self.output_dir}'")
        except zipfile.BadZipFile as e:
            print(f"Error extracting ZIP file '{zip_filepath}': {e}")

    def run(self):
        """
        Download and extract files for all entries in download_info.
        (No longer filtering on 'modis'—any data source can be used.)
        """
        for info in self.download_info:
            self.download_and_extract(info)

class WildfireDataConverter:
    """
    Converts the extracted wildfire JSON data into a CSV file that retains latitude/longitude
    and date for each detection. The CSV name includes the specified year range.
    """

    def __init__(
        self,
        extracted_dir="wildfire_data",
        start_year=2019,
        end_year=2024,
        output_csv=None
    ):
        """
        :param extracted_dir: Directory where the extracted JSON file(s) are located.
        :param start_year: Earliest year in the data (used for naming the final CSV).
        :param end_year: Latest year in the data (used for naming the final CSV).
        :param output_csv: Optional custom name for the final CSV file.
                           If None, defaults to "colorado_wildfires_{start_year}_{end_year}.csv".
        """
        self.extracted_dir = extracted_dir
        self.start_year = start_year
        self.end_year = end_year
        
        if output_csv is None:
            output_csv = f"colorado_wildfires_{self.start_year}_{self.end_year}.csv"
        self.output_csv = output_csv

    def convert_to_csv(self):
        """
        Reads all JSON files in extracted_dir, concatenates them,
        and outputs a CSV that retains geographic location (lat/lon)
        and acquisition date per detection.
        """
        json_files = [f for f in os.listdir(self.extracted_dir) if f.endswith(".json")]
        if not json_files:
            print("No JSON file found in the extracted directory.")
            return
        
        all_dfs = []
        for jf in json_files:
            json_path = os.path.join(self.extracted_dir, jf)
            print(f"Reading wildfire JSON data from '{json_path}'")
            with open(json_path, "r") as f:
                data = json.load(f)
            
            if not data:
                print(f"{jf} contains no data or is not formatted as expected. Skipping.")
                continue

            df = pd.DataFrame(data)
            if df.empty:
                print(f"{jf} is empty. Skipping.")
                continue

            # Convert acq_date to datetime
            df['acq_date'] = pd.to_datetime(df['acq_date'], errors='coerce')
            
            # For compatibility with AQIAnalysis, set StartDate = EndDate = the detection date
            df['StartDate'] = df['acq_date']
            df['EndDate'] = df['acq_date']
            
            # Keep columns of interest. Adjust as needed:
            # e.g. 'latitude', 'longitude', 'brightness', 'confidence', 'daynight', etc.
            columns_to_keep = [
                'StartDate', 'EndDate', 'latitude', 'longitude',
                'acq_date', 'confidence', 'brightness', 'frp'
            ]
            existing_columns = [col for col in columns_to_keep if col in df.columns]
            df = df[existing_columns]
            
            all_dfs.append(df)
        
        if not all_dfs:
            print("No valid data found after reading JSON files.")
            return

        final_df = pd.concat(all_dfs, ignore_index=True)
        
        # Sort by acquisition date, if desired
        final_df.sort_values(by='acq_date', inplace=True, ignore_index=True)
        
        output_path = os.path.join(self.extracted_dir, self.output_csv)
        final_df.to_csv(output_path, index=False)
        print(f"Saved final wildfire data (with lat/lon) to '{output_path}'")

if __name__ == "__main__":
    # Example download info. Replace with your actual data sources:
    download_info = [
        {
            "download_id": "575994",
            "data_source": "modis-c6.1",
            "url": ("https://urldefense.com/v3/__https://firms.modaps.eosdis.nasa.gov/data/download/"
                    "DL_FIRE_M-C61_575994.zip__;!!NCZxaNi9jForCP_SxBKJCA!V6g2hfAXPmWgf7I5lH9wj4Mfl9l-9NzD5-"
                    "Xw7_9qSGknhHOT0__q1KLHus-P_CPWdE-tAgEfdHGVrXmHVmZofQ$")
        }
        # Add more entries if desired, for different data sources or download IDs
    ]
    
    # Step 1: Download and extract wildfire data.
    #downloader = WildfireDataDownloader(download_info=download_info)
    #downloader.run()
    
    converter = WildfireDataConverter(
        extracted_dir="wildfire_data",
        start_year=2023,
        end_year=2024,
        output_csv=None  # Will default to "colorado_wildfires_{start_year}_{end_year}.csv"
    )
    converter.convert_to_csv()