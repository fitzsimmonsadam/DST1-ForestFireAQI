import requests
import pandas as pd
import time
import logging
from datetime import datetime, timedelta

class AirQualityCollector:
    """
    A class for collecting air quality data from the AirNow API.
    """
    
    def __init__(
        self,
        api_key,
        start_date,
        end_date,
        output_file=None,
        base_url="https://www.airnowapi.org/aq/data/",
        bbox="-109.087288,37.072380,-102.099983,40.996171",
        parameters="OZONE,PM25",
        monitor_type="2",               # 0 = Permanent monitors, 1 = Temporary, 2 = All
        verbose="1",                    # 1 = Enable detailed metadata
        data_format="application/json", # Use JSON format
        include_raw_concentrations="1", # 1 = Include raw data, 0 = AQI only
        data_type="B",                  # B = Both raw & AQI
        batch_days=10,                  # Retrieve data in 10 day increments, not sure of max batch size limit but must be <30
        retry_limit=3                   # Number of retries per request
    ):
        """
        Initialize the collector with API details and collection parameters.
        """
        self.api_key = api_key
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
        
        # If output_file is not provided, create one using the specified date range
        if output_file is None:
            self.output_file = (
                f"Colorado_AQI_{start_date.strftime('%Y%m')}_"    #Used yearmonth for example but %Y%m%d gives full date range 
                f"{end_date.strftime('%Y%m')}.csv"               # Can reconfigure based on preference
            )
        else:
            self.output_file = output_file

        self.all_data = []
        

        logging.basicConfig(
            filename="../../PythonProject/.venv/air_quality_data.log",
            level=logging.INFO,
            format="%(asctime)s - %(levelname)s - %(message)s"
        )
    
    def _construct_url(self, start_date, end_date):
        """
        Construct the API URL for the given date range.
        """
        url = (
            f"{self.base_url}?startDate={start_date.strftime('%Y-%m-%dT00')}"
            f"&endDate={end_date.strftime('%Y-%m-%dT23')}"
            f"&parameters={self.parameters}&BBOX={self.bbox}&dataType={self.data_type}"
            f"&format={self.data_format}&verbose={self.verbose}&monitorType={self.monitor_type}"
            f"&includerawconcentrations={self.include_raw_concentrations}&API_KEY={self.api_key}"
        )
        return url
    
    def fetch_aqs_data(self, start_date, end_date):
        """
        Fetch air quality data for a given date range.

        Returns:
            A list of data dictionaries if available, or an empty list.
        """
        url = self._construct_url(start_date, end_date)
        logging.info(f"Fetching data from {start_date.date()} to {end_date.date()}")
        
        for attempt in range(self.retry_limit):
            response = requests.get(url)
            
            if response.status_code == 200:
                try:
                    data = response.json()
                    if isinstance(data, list) and data:
                        return data
                    else:
                        logging.warning(f"No data available for {start_date.date()} to {end_date.date()}")
                        return []
                except Exception as e:
                    logging.error(f"Error parsing JSON response: {str(e)}")
            else:
                logging.error(f"Error {response.status_code}: {response.text}. Retrying {attempt + 1}/{self.retry_limit}...")
                time.sleep(2)  # Delay before retrying

        logging.error(f"Failed to retrieve data for {start_date.date()} to {end_date.date()} after {self.retry_limit} attempts.")
        return []
    
    def save_data(self):
        """
        Save the collected data to a CSV file.
        """
        if self.all_data:
            df = pd.DataFrame(self.all_data)
            df.to_csv(self.output_file, index=False)
            logging.info(f"Data saved to {self.output_file}")
        else:
            logging.warning("No data to save.")
    
    def collect_data(self):
        """
        Collect data over the specified date range in batches.
        """
        current_date = self.start_date
        
        while current_date <= self.end_date:
            batch_end_date = min(current_date + timedelta(days=self.batch_days - 1), self.end_date)
            batch_data = self.fetch_aqs_data(current_date, batch_end_date)
            
            if batch_data:
                self.all_data.extend(batch_data)
                self.save_data()  # Save data incrementally after each batch
                logging.info(f"Saved batch {current_date.date()} to {batch_end_date.date()} to {self.output_file}")
            
            # Move to the next batch
            current_date = batch_end_date + timedelta(days=1)
        
        logging.info("Data collection complete.")
        print("Data collection finished. Check air_quality_data.log for details.")


if __name__ == "__main__":
    API_KEY = "544ED264-55E3-4422-94CA-406B625CFF54"
    START_DATE = datetime(2023, 3, 1)
    END_DATE = datetime(2023, 9, 1)
    
    collector = AirQualityCollector(api_key=API_KEY, start_date=START_DATE, end_date=END_DATE)
    collector.collect_data()