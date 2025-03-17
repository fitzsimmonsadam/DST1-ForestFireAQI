# DST1-ForestFireAQI
### Analysis of EPA's AQI data in relation to Colorado forest fires

 by Alex Valone, Adam Fitzsimmons, and Lin Ma

#### Research Questions: 
- Visualize patterns of air quality over time, 2019 – 2024 

- Identify the patterns of changes in daily air quality by pollutant
   + Two most significant pollutants: Ozone and PM2.5

- Compare values of air quality index (AQI) 
  + By year 
  + By season

- Visualize patterns of air quality during wildfire events
  + Investigate the impact of wildfires on the air quality in Colorado



#### Datasets: 
1. The AQI data from AirNow provided by U.S. Environmental Protection Agency (EPA)
   https://www.epa.gov/outdoor-air-quality-data
3. The wildfire data from VIIRS provided by NASA's Fire Information for Resource Management System (FIRMS)
   https://firms.modaps.eosdis.nasa.gov/


#### Directory Structure:
1. data: Include the downloaded data files – AQI and wildfires and the analysis data files
2. notebooks: the Jupyter Notebook files for data cleaning, exploration, and analysis
3. src: The python scripts and the json scripts for exploratory data analysis (EDA) and data analysis
4. visuals: All visual maps and analysis outputs

#### Files Included:
1. aqi_collector.py - Downloads yearly AQI data from EPA source. Not necessary to run since Github LFS is hosting the file used for this analysis.
2. aqi_wf_processor.py - Segments and processes both data sources (AQI and wildfire) to produce properly formatted dataframes containing dates and measurements.
3. geo_plots.py - Geographic data exploration of the datasets (state-wide stations, wildfires, etc.)
4. stat_plots.py - Statistical data exploration of the datasets (Time-series exploration)
5. visualizer_folium.py - Heatmap chart of the dataset, interactive
6. DataExploration.ipynb - Jupyter notebook stepping through the download, processing, and geo/temporal exploratory anaylsis.
7. DataAnalysis.ipynb - Jupyter notbeook containing the seasonal and trend analyses of the dataset

#### Main Findings: 
- PM2.5 is more sensitive to fire events than Ozone, likely better feature
- AQI followed seasonal trends of being higher in the summer than winter




 





