import os
import logging
import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
from typing import Optional, List
from shapely.strtree import STRtree
import functools
import folium
from datetime import datetime

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

class AQIVisualizer:
    def __init__(self, processed_df: pd.DataFrame, wildfire_df: pd.DataFrame, start_year: int, end_year: int, output_dir: Optional[str] = None):
        self.df = processed_df.copy()
        self.wildfire_df = wildfire_df.copy()
        self.start_year = start_year
        self.end_year = end_year
        self.output_dir = output_dir or "Analysis_Output"
        os.makedirs(self.output_dir, exist_ok=True)
        
        self.logger = logging.getLogger(f"{__name__}.AQIVisualizer")
        self.logger.setLevel(logging.INFO)
        if self.logger.hasHandlers():
            self.logger.handlers.clear()
        fh = logging.FileHandler(os.path.join(self.output_dir, "aqi_visualizer.log"), mode="w")
        fh.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s - %(message)s"))
        self.logger.addHandler(fh)
        sh = logging.StreamHandler()
        sh.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s - %(message)s"))
        self.logger.addHandler(sh)
        self.logger.info("AQIVisualizer initialized.")

    @skip_if_exists([f"county_data_{'{user_input}'}.png"])
    def visualize_county_data(self, merged_df: pd.DataFrame, user_input: str) -> None:
        """
        Plots aggregated data for either a single day (YYYY-MM-DD) or an entire month (YYYY-MM).
        
        Args:
            merged_df (pd.DataFrame): DataFrame aggregated by county and day (e.g., from compare_wildfire_aqi_by_county_day).
            user_input (str): Date string in 'YYYY-MM-DD' or 'YYYY-MM' format.
        
        Produces a bar chart showing, for each county on that day or month:
            - Wildfire_Count
            - Avg_AQI (or other columns)
        """
        self.logger.info(f"Visualizing county-level data for input: {user_input}.")

        # Step 1: Attempt to parse user_input to see if it's YYYY-MM or YYYY-MM-DD
        # Simple approach: check length of the string
        if len(user_input) == 7:
            # e.g. '2023-08' => monthly
            try:
                year_str, month_str = user_input.split("-")
                year_val = int(year_str)
                month_val = int(month_str)
                # Filter the merged_df for this entire year+month
                # Ensure 'Date' column is datetime, if not already
                if not pd.api.types.is_datetime64_any_dtype(merged_df["Date"]):
                    merged_df["Date"] = pd.to_datetime(merged_df["Date"], errors="coerce")
                
                # Create new columns for year and month
                merged_df["Year"] = merged_df["Date"].dt.year
                merged_df["Month"] = merged_df["Date"].dt.month
                
                day_df = merged_df[
                    (merged_df["Year"] == year_val) & (merged_df["Month"] == month_val)
                ]
                plot_title = f"Aggregated Data for {user_input}"
            except ValueError:
                self.logger.error(f"Invalid month string: {user_input}. Format should be 'YYYY-MM'.")
                return

        elif len(user_input) == 10:
            try:
                date_obj = pd.to_datetime(user_input).date()
                if not pd.api.types.is_datetime64_any_dtype(merged_df["Date"]):
                    merged_df["Date"] = pd.to_datetime(merged_df["Date"], errors="coerce").dt.date
                
                day_df = merged_df[merged_df["Date"] == date_obj]
                plot_title = f"Aggregated Data for {user_input}"
            except Exception as e:
                self.logger.error(f"Error parsing user date '{user_input}': {e}")
                return
        else:
            self.logger.warning(f"Unrecognized format for user_input '{user_input}'. Expected 'YYYY-MM' or 'YYYY-MM-DD'.")
            return

        # Step 2: Check if day_df is empty
        if day_df.empty:
            self.logger.warning(f"No aggregated data found for {user_input}.")
            return

        # Step 3: Plot the aggregated data
        fig, ax1 = plt.subplots(figsize=(10, 6))
        x_vals = range(len(day_df))
        counties = day_df["County"].astype(str).tolist()

        # Example: Plot Wildfire_Count as bars
        if "Wildfire_Count" in day_df.columns:
            ax1.bar(x_vals, day_df["Wildfire_Count"], color="red", alpha=0.6, label="Wildfire Count")
        ax1.set_ylabel("Wildfire Count", color="red")
        ax1.set_xticks(x_vals)
        ax1.set_xticklabels(counties, rotation=45, ha="right")
        ax1.tick_params(axis="y", labelcolor="red")

        # Create a second y-axis to plot average AQI
        ax2 = ax1.twinx()
        if "Avg_AQI" in day_df.columns:
            ax2.plot(x_vals, day_df["Avg_AQI"], color="blue", marker="o", label="Avg AQI")
            ax2.set_ylabel("Average AQI", color="blue")
            ax2.tick_params(axis="y", labelcolor="blue")

        # Add more lines or bars if you want (e.g. PM2.5, etc.)

        plt.title(plot_title)
        plt.tight_layout()
        # Save figure
        plot_path = os.path.join(self.output_dir, f"county_data_{user_input}.png")
        plt.savefig(plot_path)
        plt.close()
        self.logger.info(f"Plot for {user_input} saved to {plot_path}.")

    @skip_if_exists([f"county_day_{'{selected_date}'}.png"])
    def visualize_county_day(self, merged_df: pd.DataFrame, selected_date: str) -> None:
        """
        Plots the aggregated data for a specific day from the merged_df.
        
        Args:
            merged_df (pd.DataFrame): The aggregated DataFrame from compare_wildfire_aqi_by_county_day.
            selected_date (str): The date to plot, in 'YYYY-MM-DD' format.
        """
        self.logger.info(f"Visualizing county-level data for {selected_date}.")

        # Filter the merged_df for the user-specified day
        date_obj = pd.to_datetime(selected_date).date()
        day_df = merged_df[merged_df["Date"] == date_obj]
        if day_df.empty:
            self.logger.warning(f"No data found for {selected_date}.")
            return

        # Example: plot a dual-axis bar/line chart
        fig, ax1 = plt.subplots(figsize=(10, 6))
        x_vals = range(len(day_df))
        counties = day_df["County"].tolist()

        # Plot Wildfire_Count as bars
        ax1.bar(x_vals, day_df["Wildfire_Count"], color="red", alpha=0.6, label="Wildfire Count")
        ax1.set_ylabel("Wildfire Count", color="red")
        ax1.set_xticks(x_vals)
        ax1.set_xticklabels(counties, rotation=45, ha="right")
        ax1.tick_params(axis="y", labelcolor="red")

        # Create a second y-axis to plot Avg_AQI
        ax2 = ax1.twinx()
        ax2.plot(x_vals, day_df["Avg_AQI"], color="blue", marker="o", label="Avg AQI")
        ax2.set_ylabel("Average AQI (>= threshold)", color="blue")
        ax2.tick_params(axis="y", labelcolor="blue")

        plt.title(f"County-level Data for {selected_date}")
        plt.tight_layout()
        plot_path = os.path.join(self.output_dir, f"county_day_{selected_date}.png")
        plt.savefig(plot_path)
        plt.close()
        self.logger.info(f"Plot for {selected_date} saved to {plot_path}.")
    
    @skip_if_exists([f"aqi_category_distribution_{'{start_year}'}_{'{end_year}'}.png"])
    def plot_aqi_categories(self, selected_levels: Optional[List[str]] = None, force_run=False) -> None:
        """
        Plots the distribution of AQI categories.

        Args:
            selected_levels (Optional[List[str]]): List of AQI categories to include in the plot.
                                                If None, all categories will be shown.
            force_run (bool): Whether to force execution even if the file exists.
        """
        self.logger.info("Plotting AQI category distribution.")
        
        if 'AQI_Category' not in self.df.columns:
            self.logger.warning("AQI_Category column not found.")
            return
        
        # Filter the dataframe based on selected AQI levels (if provided)
        if selected_levels:
            filtered_df = self.df[self.df["AQI_Category"].isin(selected_levels)]
        else:
            filtered_df = self.df

        # Count occurrences of each category
        counts = filtered_df["AQI_Category"].value_counts()

        if counts.empty:
            self.logger.warning(f"No data found for selected AQI categories: {selected_levels}")
            return
        
        # Plot
        plt.figure(figsize=(8, 6))
        counts.plot(kind='bar', color='skyblue', edgecolor='k')
        plt.title("AQI Category Distribution")
        plt.xlabel("AQI Category")
        plt.ylabel("Count")
        plt.xticks(rotation=45)
        plt.tight_layout()
        
        # Save the plot
        selected_str = "_".join(selected_levels) if selected_levels else "all"
        plot_path = os.path.join(self.output_dir, f"aqi_category_distribution_{self.start_year}_{self.end_year}_{selected_str}.png")
        plt.savefig(plot_path)
        plt.close()

        self.logger.info(f"AQI category distribution plot saved to {plot_path}.")

    @skip_if_exists([f"geospatial_aqi_{'{start_year}'}_{'{end_year}'}.png"])
    def plot_geospatial_aqi(self, force_run=False) -> None:
        self.logger.info("Plotting geospatial AQI data.")
        if not {'Latitude', 'Longitude'}.issubset(self.df.columns):
            self.logger.warning("Missing Latitude/Longitude columns for geospatial plot.")
            return
        gdf = gpd.GeoDataFrame(
            self.df, geometry=gpd.points_from_xy(self.df['Longitude'], self.df['Latitude']),
            crs="EPSG:4326"
        )
        ax = gdf.plot(figsize=(10, 8), alpha=0.5, edgecolor='k')
        ax.set_title("Geospatial Distribution of AQI Data")
        plot_path = os.path.join(self.output_dir, f"geospatial_aqi_{self.start_year}_{self.end_year}.png")
        plt.savefig(plot_path)
        plt.close()
        self.logger.info(f"Geospatial AQI plot saved to {plot_path}.")

    @skip_if_exists([f"wildfire_aqi_map_{'{start_year}'}_{'{end_year}'}.png"])
    def plot_wildfire_aqi_map(self) -> None:
        """
        Produces a visual map overlaying wildfire events and AQI points.
        Wildfire events are colored by Confidence_Category.
        """
        self.logger.info("Plotting wildfire and AQI overlay map.")
        # Convert wildfire data to GeoDataFrame
        if not {'latitude', 'longitude'}.issubset(self.wildfire_df.columns):
            self.logger.warning("Wildfire data missing coordinate columns; cannot plot map.")
            return
        wf_gdf = gpd.GeoDataFrame(
            self.wildfire_df,
            geometry=gpd.points_from_xy(self.wildfire_df['longitude'], self.wildfire_df['latitude']),
            crs="EPSG:4326"
        )
        # Convert AQI data to GeoDataFrame
        if not {'Latitude', 'Longitude'}.issubset(self.df.columns):
            self.logger.warning("AQI data missing coordinate columns; cannot plot map.")
            return
        aq_gdf = gpd.GeoDataFrame(
            self.df,
            geometry=gpd.points_from_xy(self.df['Longitude'], self.df['Latitude']),
            crs="EPSG:4326"
        )
        fig, ax = plt.subplots(figsize=(12, 10))
        # Plot AQI points, colored by AQI_Category
        if 'AQI_Category' in aq_gdf.columns:
            aq_gdf.plot(ax=ax, marker='o', color='blue', alpha=0.5, label="AQI")
        else:
            aq_gdf.plot(ax=ax, marker='o', color='blue', alpha=0.5, label="AQI")
        # Plot wildfire points, colored by Confidence_Category
        if 'Confidence_Category' in wf_gdf.columns:
            colors = {'Low': 'green', 'Medium': 'orange', 'High': 'red', 'Unknown': 'grey'}
            for conf, group in wf_gdf.groupby('Confidence_Category'):
                group.plot(ax=ax, marker='x', color=colors.get(conf, 'grey'), label=f"Wildfire ({conf})")
        else:
            wf_gdf.plot(ax=ax, marker='x', color='red', label="Wildfire")
        ax.set_title("Wildfire Events & AQI Data")
        ax.legend()
        plot_path = os.path.join(self.output_dir, f"wildfire_aqi_map_{self.start_year}_{self.end_year}.png")
        plt.savefig(plot_path)
        plt.close()
        self.logger.info(f"Wildfire-AQI map saved to {plot_path}.")

    #@skip_if_exists([f"folium_wildfire_aqi_map_{'{start_year}'}_{'{end_year}'}.html"])
    def create_folium_map(self, wf_gdf: gpd.GeoDataFrame, aqi_gdf: gpd.GeoDataFrame, center=[39.0, -105.5], zoom_start=6):
        """
        Creates an interactive Folium map centered on Colorado, with layers for wildfires,
        AQI monitoring stations, and heatmaps.
        
        wf_gdf columns might include:
            - geometry (Point)
            - Date
            - confidence
            - Avg_AQI_WithinDist
            - other attributes
        Args:
            wf_gdf (gpd.GeoDataFrame): Filtered wildfire data (already in EPSG:4326).
            aqi_gdf (gpd.GeoDataFrame): AQI monitoring station data.
            center (list): Latitude/Longitude center for initial map.
            zoom_start (int): Initial zoom level.
            force_run (bool): Whether to skip checking file_exists or not.
        """
        self.logger.info("Creating Folium map of wildfire data.")
        # 1) Create base map
        m = folium.Map(location=center, zoom_start=zoom_start, tiles='cartodbpositron')
        # Feature groups for toggle control
        wildfire_layer = folium.FeatureGroup(name="Wildfires", overlay=True)
        aqi_layer = folium.FeatureGroup(name="AQI Stations", overlay=True)
        wildfire_heatmap = folium.FeatureGroup(name="Wildfire Heatmap", overlay=True)
        aqi_heatmap = folium.FeatureGroup(name="AQI Heatmap", overlay=True)

        if not wf_gdf.empty:
            wildfire_cluster = folium.MarkerCluster(name="Wildfire Clusters").add_to(wildfire_layer)
            
            for idx, row in wf_gdf.iterrows():
                lat, lon = row.geometry.y, row.geometry.x
                frp = row.get("frp", 1)
                marker_size = min(max(frp / 50, 3), 10)

                popup_html = f"""
                <strong>County:</strong> {row.get('County', 'N/A')}<br>
                <strong>Date:</strong> {row.get('acq_date', 'N/A')}<br>
                <strong>Confidence:</strong> {row.get('confidence', 'N/A')}<br>
                <strong>Avg AQI (Within Dist):</strong> {row.get('Avg_AQI_WithinDist', 'N/A')}
                """
                folium.CircleMarker(
                location=[lat, lon],
                radius=marker_size,
                color="red",
                fill=True,
                fill_color="red",
                fill_opacity=0.7,
                popup=popup_html,
                tooltip=f"Wildfire: {row.get('Date', 'N/A')}").add_to(wildfire_cluster)
            
            wildfire_heat_data = [[row.geometry.y, row.geometry.x] for idx, row in wf_gdf.iterrows()]
            if wildfire_heat_data:
                folium.HeatMap(wildfire_heat_data, radius=15, blur=10, gradient={0.2: "yellow", 0.4: "orange", 0.6: "red"}).add_to(wildfire_heatmap)


        if not aqi_gdf.empty:
            aqi_cluster = folium.MarkerCluster(name="AQI Clusters").add_to(aqi_layer)

            for idx, row in aqi_gdf.iterrows():
                lat, lon = row.geometry.y, row.geometry.x
                aqi_val = row.get("AQI", "N/A")

                # AQI color coding
                aqi_color = "green" if aqi_val <= 50 else "yellow" if aqi_val <= 100 else \
                            "orange" if aqi_val <= 150 else "red" if aqi_val <= 200 else "purple"

                popup_html = f"""
                <strong>Site Name:</strong> {row.get('SiteName', 'N/A')}<br>
                <strong>Date:</strong> {row.get('Date', 'N/A')}<br>
                <strong>AQI Value:</strong> {aqi_val}<br>
                <strong>Category:</strong> {row.get('AQI_Category', 'N/A')}<br>
                <strong>Within Wildfire Distance:</strong> {row.get('WithinWildfireDistance', 'N/A')}
                """

                folium.CircleMarker(
                    location=[lat, lon],
                    radius=6,
                    color=aqi_color,
                    fill=True,
                    fill_color=aqi_color,
                    fill_opacity=0.7,
                    popup=popup_html,
                    tooltip=f"AQI: {aqi_val}"
                ).add_to(aqi_cluster)

            #  Heatmap for AQI Density
            aqi_heat_data = [[row.geometry.y, row.geometry.x, row.get("AQI", 0)] for idx, row in aqi_gdf.iterrows()]
            if aqi_heat_data:
                folium.HeatMap(aqi_heat_data, radius=12, blur=8, gradient={0.2: "blue", 0.4: "cyan", 0.6: "purple"}).add_to(aqi_heatmap)

        #  Add layers to the map
        wildfire_layer.add_to(m)
        aqi_layer.add_to(m)
        wildfire_heatmap.add_to(m)
        aqi_heatmap.add_to(m)

        #  Layer Control
        folium.LayerControl().add_to(m)

        #  Save Map
        map_path = os.path.join(self.output_dir, f"enhanced_wildfire_aqi_map_{self.start_year}_{self.end_year}.html")
        m.save(map_path)
        self.logger.info(f"Enhanced Folium wildfire-AQI map saved to {map_path}.")
