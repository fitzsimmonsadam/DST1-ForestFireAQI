import requests
import pandas as pd
import time
import logging
import os
import json
from datetime import datetime, timedelta

class AirQualityCollector:
    """
    A class for collecting air quality data from the AirNow API. 
    """
    def __init__(
        self,
        config_path,
        start_date,
        end_date,
        output_file=None,
        base_url="https://www.airnowapi.org/aq/data/",
        bbox="-109.060253,36.992424,-102.041524,41.003444", #geographic boundary for CO, “minLongitude,minLatitude,maxLongitude,maxLatitude”
        parameters="PM25,OZONE",        # parameters of interest
        monitor_type="0",               # 0 = Permanent monitors, 1 = Temporary, 2 = All
        verbose="1",                    # 1 = Enable detailed metadata, 0=
        data_format="application/json", # text/csv or application/json format
        include_raw_concentrations="1", # 1 = Include raw data, 0 = AQI only
        data_type="A",                  # "A" (AQI only), "C" (concentration only), "B" (both)
        batch_days=10,                  # Retrieve data by day increments (suggested less than 30)
        retry_limit=3                   # Number of retries per request
    ):

        """
        Initialize the collector with API details and collection parameters.
        """
        self.api_key = self.load_api_key(config_path)
        self.base_url = base_url
        self.bbox = bbox
        self.parameters = parameters
        self.monitor_type = monitor_type
        self.verbose = verbose
        self.data_format = data_format
        self.include_raw_concentrations = include_raw_concentrations
        self.data_type = data_type
        self.start_date = start_date
        self.end_date = end_date
        self.batch_days = batch_days
        self.retry_limit = retry_limit
        self.all_data = []

        # setup directories
        script_dir = os.path.dirname(os.path.abspath(__file__))
        self.data_dir = os.path.abspath(os.path.join(script_dir, "../data/aqi_data"))
        self.log_dir = os.path.abspath(os.path.join(script_dir, "../logs"))
        os.makedirs(self.data_dir, exist_ok=True)
        os.makedirs(self.log_dir, exist_ok=True)

        # Output filename
        if output_file is None:
            self.output_file = os.path.join(
                self.data_dir, f"Colorado_AQI_{start_date.strftime('%Y%m%d')}_{end_date.strftime('%Y%m%d')}.csv"
            )
        else:
            self.output_file = os.path.join(self.data_dir, output_file)
        # Setup logging
        logging.basicConfig(
            filename=os.path.join(self.log_dir, "air_quality_data.log"),
            level=logging.INFO,
            format="%(asctime)s - %(levelname)s - %(message)s"
        )
        self.logger = logging.getLogger(__name__)
        self.logger.info("AirQualityCollector initialized.")
    
    @staticmethod
    def load_api_key(config_path):
        if not os.path.exists(config_path):
            raise FileNotFoundError(f"Config file not found at {config_path}")
        with open(config_path, "r") as file:
            config = json.load(file)
        return config.get("API_KEY")

    def _construct_url(self, start_date, end_date):
        return (
            f"{self.base_url}?startDate={start_date.strftime('%Y-%m-%dT00')}"
            f"&endDate={end_date.strftime('%Y-%m-%dT23')}"
            f"&parameters={self.parameters}"
            f"&BBOX={self.bbox}"
            f"&dataType={self.data_type}"
            f"&format={self.data_format}"
            f"&verbose={self.verbose}"
            f"&monitorType={self.monitor_type}"
            f"&includerawconcentrations={self.include_raw_concentrations}"
            f"&API_KEY={self.api_key}"
        )
    
    def fetch_aqs_data(self, start_date, end_date):
        """
        Fetches AQI data within specified date range from AirNow API. 
        Args:
            start_date (_type_): Start Date
            end_date (_type_): End Date
        Returns:
            list: aqi data json
        """        
        url = self._construct_url(start_date, end_date)
        self.logger.info(f"Requesting data from {start_date.date()} to {end_date.date()}")
        for attempt in range(1, self.retry_limit + 1):
            try:
                response = requests.get(url)
                if response.status_code == 200:
                    data = response.json()
                    if isinstance(data, list) and data:
                        self.logger.info(f"Retrieved {len(data)} records.")
                        return data
                    else:
                        self.logger.warning(f"No data for {start_date.date()} to {end_date.date()}")
                        return []
                else:
                    self.logger.error(f"HTTP {response.status_code}: {response.text}")
            except Exception as e:
                self.logger.error(f"Error fetching data: {e}")
            self.logger.info(f"Retrying... Attempt {attempt}/{self.retry_limit}")
            time.sleep(2)
        self.logger.error(f"Failed to retrieve data after {self.retry_limit} attempts.")
        return []
    
    def save_data(self):
        """
        Save the collected data to a CSV file.
        """
        if self.all_data:
            df = pd.DataFrame(self.all_data)
            df.to_csv(self.output_file, index=False)
            self.logger.info(f"Data saved to {self.output_file}")
        else:
            self.logger.warning("No data to save.")
    
    def collect_data(self):
        """
        Collect data over the specified date range in batches.
        """
        self.logger.info(f"Starting data collection from {self.start_date.date()} to {self.end_date.date()}")
        current_date = self.start_date
        while current_date <= self.end_date:
            batch_end_date = min(current_date + timedelta(days=self.batch_days - 1), self.end_date)
            batch_data = self.fetch_aqs_data(current_date, batch_end_date)
            if batch_data:
                self.all_data.extend(batch_data)
                self.save_data()
                self.logger.info(f"Saved batch: {current_date.date()} to {batch_end_date.date()}")
            current_date = batch_end_date + timedelta(days=1)
        self.logger.info("Data collection complete.")
        print(f"Data collection complete. Check logs at {self.log_dir}.")


if __name__ == "__main__":
    CONFIG_PATH = "config.json"
    START_DATE = datetime(2019, 1, 1)
    END_DATE = datetime(2024, 12, 1)
    
    collector = AirQualityCollector(
        config_path=CONFIG_PATH, 
        start_date=START_DATE, 
        end_date=END_DATE
    )
    collector.collect_data()