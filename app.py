import streamlit as st
import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
from shapely.geometry import Point, LineString
import matplotlib.cm as cm
import matplotlib.colors as mcolors
from matplotlib.lines import Line2D
import os
import numpy as np

# =============================================================================
# Placeholder Functions for Your Code Blocks
# -----------------------------------------------------------------------------
# Instructions:
# 1. Replace the content of each function with your corresponding code block.
# 2. Ensure each function returns a Matplotlib Figure object (`fig`).
# 3. If your code already displays a plot (e.g., using `plt.show()`),
#    you should remove that line and return the figure object instead.
# =============================================================================


# =============================================================================
# Streamlit Page Configuration & Title
# =============================================================================
st.set_page_config(layout="wide")
st.title("North Sea Infrastructure Analysis")


# =============================================================================
# DATA LOADING FUNCTION (CACHED FOR EFFICIENCY)
# =============================================================================
@st.cache_data
def load_all_data():
    """Loads and cleans all necessary dataframes once."""
    SENSITIVITY_FILE = "Post-Process manual sensitivity"
    COLUMN_HEADERS = [
        'Baseline', 'HVDC-Min', 'HVDC-Max', 'OWF-C-Min', 'OWF-C-Max', 'ED-Min', 'ED-Max',
        'EP-Min', 'EP-Max', 'WACC-Min', 'WACC-Max', 'OWF-S-Min', 'OWF-S-Max'
    ]
    excel_file_locations = "manual input/Hub locations input.xlsx"
    excel_file_path = f"manual input/{SENSITIVITY_FILE}.xlsx"

    # Load raw data
    gdf_locations = gpd.GeoDataFrame(
        pd.read_excel(excel_file_locations, sheet_name="NSG"),
        geometry=gpd.points_from_xy(pd.read_excel(excel_file_locations, sheet_name="NSG").longitude,
                                    pd.read_excel(excel_file_locations, sheet_name="NSG").latitude),
        crs="EPSG:4326"
    )
    gdf_locations['label'] = gdf_locations['label'].astype(str).str.strip()

    df_line_capacity_all = pd.read_excel(excel_file_path, sheet_name="XC Trade Stock")
    df_point_stock_all = pd.read_excel(excel_file_path, sheet_name="Offshore Power Stock")
    df_line_use_all = pd.read_excel(excel_file_path, sheet_name="XC Trade Use")

    # Clean all dataframes to ensure numeric types
    for col in COLUMN_HEADERS:
        for df in [df_line_capacity_all, df_point_stock_all, df_line_use_all]:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)

    return gdf_locations, df_line_capacity_all, df_point_stock_all, df_line_use_all


# =============================================================================
# PLOTTING FUNCTIONS
# Each function generates and returns a Matplotlib Figure object.
# ================================================================

# --- New helper function for the requested sensitivity calculation ---
def calculate_avg_parameter_sensitivity(df, baseline_col, param_pairs):

    baseline = df[baseline_col]
    epsilon = 1e-9  # To prevent division by zero

    all_param_sensitivities = pd.DataFrame()

    for param, (min_col, max_col) in param_pairs.items():
        if min_col in df.columns and max_col in df.columns:
            param_range = abs((df[max_col] - df[baseline_col])/df[baseline_col]) + abs((df[min_col] - df[baseline_col])/df[baseline_col])
            all_param_sensitivities[param] = param_range / 2

    return all_param_sensitivities.mean(axis=1)

def find_disappearing_scenario(row, sensitivity_cols):
        for col in sensitivity_cols:
            if row[col] < 0.1:
                return "IC disappears in one or more cases"
        return None


def create_use_sensitivity_figure():
    # SENSITIVITY ANALYSIS VISUALIZATION - USE
    #
    # --- DESCRIPTION ---
    # This script visualizes the results of a sensitivity analysis for the North Sea
    # energy infrastructure. 
    # --- MODIFIED: Sensitivity is the average of individual parameter sensitivities, where
    #   each is calculated as (Param_Max - Param_Min) / Baseline.

    # --- Configuration ---
    SENSITIVITY_FILE = "Post-Process manual sensitivity"
    COLUMN_HEADERS = [
        'Baseline', 'HVDC-Min', 'HVDC-Max', 'OWF-C-Min', 'OWF-C-Max', 'ED-Min', 'ED-Max',
        'EP-Min', 'EP-Max', 'WACC-Min', 'WACC-Max', 'OWF-S-Min', 'OWF-S-Max'
    ]
    SENSITIVITY_COLUMNS = COLUMN_HEADERS[1:]
    
    # *** Define parameter pairs for the new calculation ***
    PARAMETER_PAIRS = {
        'HVDC': ('HVDC-Min', 'HVDC-Max'),
        'OWF-C': ('OWF-C-Min', 'OWF-C-Max'),
        'ED': ('ED-Min', 'ED-Max'),
        'EP': ('EP-Min', 'EP-Max'),
        'WACC': ('WACC-Min', 'WACC-Max'),
        'OWF-S': ('OWF-S-Min', 'OWF-S-Max')
    }

    # Common file paths and settings
    excel_file_locations = "manual input/Hub locations input.xlsx"
    sheet_name_line_capacity = "XC Trade Use"
    sheet_name_point_size = "Offshore Power Use"
    sheet_name_NSGorDirect = "NSG"

    eez_shapefile = "data/eez/eez_v12.shp"
    land_shapefile = "data/naturalearth/ne_10m_admin_0_countries/ne_10m_admin_0_countries.shp"
    bbox = (-2, 51, 10, 60)
    output_directory = "output/visualisation/"

    os.makedirs(output_directory, exist_ok=True)

    # --- Scaling and other Helper Functions ---
    def scale_line_thickness(capacity_gw, data_min=0, data_max=1000, viz_min=5, viz_max=60.0):
        if pd.isna(capacity_gw) or capacity_gw <= data_min: return 0.0
        if capacity_gw >= data_max: return viz_max
        return viz_min + ((capacity_gw - data_min) / (data_max - data_min)) * (viz_max - viz_min)

    def scale_point_size(power_stock_gw, data_min=0, data_max=1000, viz_min=50, viz_max=4000):
        if pd.isna(power_stock_gw) or power_stock_gw <= data_min: return 0.0
        if power_stock_gw >= data_max: return viz_max
        return viz_min + ((power_stock_gw - data_min) / (data_max - data_min)) * (viz_max - viz_min)

    def draw_compass(ax, x_pos=0.97, y_pos=0.97, size_val=0.05):
        ax.annotate('N', xy=(x_pos, y_pos), xytext=(x_pos, y_pos - size_val * 1.5),
                    arrowprops=dict(facecolor='black', width=1, headwidth=6, shrink=0.1),
                    ha='center', va='center', fontsize=12, xycoords='axes fraction')
        ax.plot(x_pos, y_pos - size_val * 0.75, 'o', color='black', markersize=size_val * 100,
                transform=ax.transAxes, fillstyle='none')

    def find_disappearing_scenario(row, sensitivity_cols):
        for col in sensitivity_cols:
            if row[col] < 0.1:
                return "IC disappears in one or more cases"
        return None

    def generate_sensitivity_map(ax):
        excel_file_path = f"manual input/{SENSITIVITY_FILE}.xlsx"
        
        df_locations_all = pd.read_excel(excel_file_locations, sheet_name=sheet_name_NSGorDirect)
        gdf_all_points = gpd.GeoDataFrame(
            df_locations_all,
            geometry=gpd.points_from_xy(df_locations_all.longitude, df_locations_all.latitude),
            crs="EPSG:4326"
        )
        gdf_all_points['label'] = gdf_all_points['label'].astype(str).str.strip()

        # --- Process Line Data (Interconnectors) ---
        df_lines_raw = pd.read_excel(excel_file_path, sheet_name=sheet_name_line_capacity)
        for col in COLUMN_HEADERS:
            if col in df_lines_raw.columns:
                df_lines_raw[col] = pd.to_numeric(df_lines_raw[col], errors='coerce')
        df_lines_raw[COLUMN_HEADERS] = df_lines_raw[COLUMN_HEADERS].fillna(0)
        df_lines_raw['pair_key'] = df_lines_raw.apply(
            lambda row: tuple(sorted([str(row['DistPointA']).strip(), str(row['DistPointB']).strip()])), axis=1
        )
        df_lines_agg = df_lines_raw.groupby('pair_key', as_index=False)[COLUMN_HEADERS].max()

        # *** MODIFIED SENSITIVITY CALCULATION ***
        df_lines_agg['sensitivity'] = calculate_avg_parameter_sensitivity(df_lines_agg, 'Baseline', PARAMETER_PAIRS)
        df_lines_agg['thickness'] = df_lines_agg['Baseline'].apply(scale_line_thickness)
        df_lines_agg[['DistPointA', 'DistPointB']] = pd.DataFrame(df_lines_agg['pair_key'].tolist(),
                                                                  index=df_lines_agg.index)
        df_lines_agg['disappearance_label'] = df_lines_agg.apply(
            lambda row: find_disappearing_scenario(row, SENSITIVITY_COLUMNS),
            axis=1
        )

        # --- Process Point Data (Hubs) ---
        df_points_raw = pd.read_excel(excel_file_path, sheet_name=sheet_name_point_size)
        for col in COLUMN_HEADERS:
            if col in df_points_raw.columns:
                df_points_raw[col] = pd.to_numeric(df_points_raw[col], errors='coerce')
        df_points_raw[COLUMN_HEADERS] = df_points_raw[COLUMN_HEADERS].fillna(0)
        df_points_raw = df_points_raw.rename(columns={'DistPointA': 'label'})
        df_points_raw['label'] = df_points_raw['label'].astype(str).str.strip()
        
        # *** MODIFIED SENSITIVITY CALCULATION ***
        df_points_raw['sensitivity'] = calculate_avg_parameter_sensitivity(df_points_raw, 'Baseline', PARAMETER_PAIRS)
        df_points_raw['size'] = df_points_raw['Baseline'].apply(scale_point_size)

        # --- Prepare Geodataframes for Plotting ---
        lines_data, connected_points_labels = [], set()
        for _, row in df_lines_agg.iterrows():
            point_a = gdf_all_points[gdf_all_points['label'] == row['DistPointA']]
            point_b = gdf_all_points[gdf_all_points['label'] == row['DistPointB']]
            if not point_a.empty and not point_b.empty and row['thickness'] > 0:
                lines_data.append({
                    'geometry': LineString([point_a.geometry.iloc[0], point_b.geometry.iloc[0]]),
                    'thickness': row['thickness'],
                    'sensitivity': row['sensitivity'],
                    'disappearance_label': row['disappearance_label'],
                    'Baseline': row['Baseline']
                })
                connected_points_labels.update([row['DistPointA'], row['DistPointB']])
        gdf_lines = gpd.GeoDataFrame(lines_data, crs="EPSG:4326")

        gdf_points_to_plot = gdf_all_points[gdf_all_points['label'].isin(connected_points_labels)].copy()
        gdf_points_to_plot = gdf_points_to_plot.merge(df_points_raw[['label', 'Baseline', 'sensitivity', 'size']],
                                                      on='label', how='left')
        gdf_points_to_plot.fillna(0, inplace=True)

        # --- Plotting ---
        cmap = plt.get_cmap('viridis')
        norm = mcolors.Normalize(vmin=0, vmax=1.0) # Adjust vmax as needed
        land = gpd.read_file(land_shapefile).to_crs("EPSG:4326").cx[bbox[0]:bbox[2], bbox[1]:bbox[3]]
        eez = gpd.read_file(eez_shapefile).to_crs("EPSG:4326").cx[bbox[0]:bbox[2], bbox[1]:bbox[3]]
        ax.set_facecolor("#aadaff")
        land.plot(ax=ax, color='lightgrey', edgecolor='black', linewidth=0.5)
        eez.boundary.plot(ax=ax, color='black', linewidth=0.6, linestyle='--')

        if not gdf_lines.empty:
            gdf_lines.plot(ax=ax, column='sensitivity', cmap=cmap, norm=norm, linewidth=gdf_lines['thickness'], zorder=4)
            for _, row in gdf_lines.iterrows():
                if row['geometry']:
                    midpoint = row.geometry.centroid
                    fontsize = 10
                    baseline_value = row['Baseline']
                    if pd.notna(row['disappearance_label']):
                        label_text = row['disappearance_label']
                        fontsize = 10
                    else:
                        label_text = f"sens: {row['sensitivity']:.2f}\n({baseline_value:.0f} PJ)"

                    ax.text(midpoint.x, midpoint.y + 0.05, label_text, fontsize=fontsize, ha='center', va='bottom',
                            bbox=dict(facecolor='white', alpha=0.7, edgecolor='none', pad=0.1), zorder=7)

        if not gdf_points_to_plot.empty:
            gdf_points_to_plot.plot(ax=ax, column='sensitivity', cmap=cmap, norm=norm,
                                    markersize=gdf_points_to_plot['size'], zorder=5,
                                    edgecolor='black', linewidth=0.8)
            for _, row in gdf_points_to_plot.iterrows():
                label_text = f"{row['label']}\n({row['Baseline']:.0f} PJ)" if row['Baseline'] > 0 else row['label']
                ax.text(row.geometry.x, row.geometry.y + 0.1, label_text, fontsize=11, ha='center',
                        bbox=dict(facecolor='white', alpha=0.6, edgecolor='none', pad=0.1), zorder=6)

        draw_compass(ax)
        ax.set_xlim(bbox[0], bbox[2]); ax.set_ylim(bbox[1], bbox[3])
        ax.set_title("Sensitivity of North Sea Infrastructure Use", fontsize=20, pad=20)
        ax.set_xlabel("Longitude"); ax.set_ylabel("Latitude")
        return cmap, norm

    # --- Main Script Execution ---
    fig, ax = plt.subplots(figsize=(15, 15))
    plt.subplots_adjust(left=0.05, right=0.85, top=0.95, bottom=0.05)
    cmap, norm = generate_sensitivity_map(ax)

    # --- Create Legends ---
    cbar_ax = fig.add_axes([0.87, 0.25, 0.03, 0.5])
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    cbar = fig.colorbar(sm, cax=cbar_ax)

    # *** MODIFIED LABEL ***
    cbar.set_label('Average Parameter Sensitivity', fontsize=12, rotation=270, labelpad=25)
    
    legend_elements = [
        Line2D([0], [0], color='grey', lw=scale_line_thickness(100), label='100PJ Interconnection'),
        Line2D([0], [0], color='grey', lw=scale_line_thickness(300), label='300PJ Interconnection'),
        Line2D([0], [0], linestyle='--', color='black', linewidth=0.6, label='EEZ Boundaries')
    ]
    ax.legend(handles=legend_elements, loc='lower left', fontsize=12, title="Baseline Use Legend")

    return fig


def create_stock_sensitivity_figure():
    """
    Generates and returns the complete sensitivity map figure for 'Stock'.
    This version uses the average parameter sensitivity calculation and advanced line labeling.
    """
    # --- Configuration ---
    SENSITIVITY_FILE = "Post-Process manual sensitivity"
    COLUMN_HEADERS = [
        'Baseline', 'HVDC-Min', 'HVDC-Max', 'OWF-C-Min', 'OWF-C-Max', 'ED-Min', 'ED-Max',
        'EP-Min', 'EP-Max', 'WACC-Min', 'WACC-Max', 'OWF-S-Min', 'OWF-S-Max'
    ]
    SENSITIVITY_COLUMNS = COLUMN_HEADERS[1:]

    # Define parameter pairs for the sensitivity calculation
    PARAMETER_PAIRS = {
        'HVDC': ('HVDC-Min', 'HVDC-Max'),
        'OWF-C': ('OWF-C-Min', 'OWF-C-Max'),
        'ED': ('ED-Min', 'ED-Max'),
        'EP': ('EP-Min', 'EP-Max'),
        'WACC': ('WACC-Min', 'WACC-Max'),
        'OWF-S': ('OWF-S-Min', 'OWF-S-Max')
    }

    # Common file paths and settings
    excel_file_locations = "manual input/Hub locations input.xlsx"
    sheet_name_line_capacity = "XC Trade Stock"
    sheet_name_point_size = "Offshore Power Stock"
    sheet_name_NSGorDirect = "NSG"

    eez_shapefile = "data/eez/eez_v12.shp"
    land_shapefile = "data/naturalearth/ne_10m_admin_0_countries/ne_10m_admin_0_countries.shp"
    bbox = (-2, 51, 10, 60)
    output_directory = "output/visualisation/"
    os.makedirs(output_directory, exist_ok=True)

    # --- Scaling and other Helper Functions ---
    def scale_line_thickness(capacity_gw, data_min=0, data_max=20, viz_min=5, viz_max=35.0):
        if pd.isna(capacity_gw) or capacity_gw <= data_min: return 0.0
        if capacity_gw >= data_max: return viz_max
        return viz_min + ((capacity_gw - data_min) / (data_max - data_min)) * (viz_max - viz_min)

    def scale_point_size(power_stock_gw, data_min=0, data_max=80, viz_min=50, viz_max=4000):
        if pd.isna(power_stock_gw) or power_stock_gw <= data_min: return 0.0
        if power_stock_gw >= data_max: return viz_max
        return viz_min + ((power_stock_gw - data_min) / (data_max - data_min)) * (viz_max - viz_min)

    def draw_compass(ax, x_pos=0.97, y_pos=0.97, size_val=0.05):
        ax.annotate('N', xy=(x_pos, y_pos), xytext=(x_pos, y_pos - size_val * 1.5),
                    arrowprops=dict(facecolor='black', width=1, headwidth=6, shrink=0.1),
                    ha='center', va='center', fontsize=12, xycoords='axes fraction')
        ax.plot(x_pos, y_pos - size_val * 0.75, 'o', color='black', markersize=size_val * 100,
                transform=ax.transAxes, fillstyle='none')

    # This part was your 'generate_sensitivity_map' function, now fully updated
    def build_map_on_axes(ax):
        excel_file_path = f"manual input/{SENSITIVITY_FILE}.xlsx"
        
        # --- Read Location and Base Data ---
        df_locations_all = pd.read_excel(excel_file_locations, sheet_name=sheet_name_NSGorDirect)
        gdf_all_points = gpd.GeoDataFrame(
            df_locations_all,
            geometry=gpd.points_from_xy(df_locations_all.longitude, df_locations_all.latitude),
            crs="EPSG:4326"
        )
        gdf_all_points['label'] = gdf_all_points['label'].astype(str).str.strip()

        # --- Process Line Data (Interconnectors) ---
        df_lines_raw = pd.read_excel(excel_file_path, sheet_name=sheet_name_line_capacity)
        for col in COLUMN_HEADERS:
            if col in df_lines_raw.columns:
                df_lines_raw[col] = pd.to_numeric(df_lines_raw[col], errors='coerce')
        df_lines_raw[COLUMN_HEADERS] = df_lines_raw[COLUMN_HEADERS].fillna(0)
        df_lines_raw['pair_key'] = df_lines_raw.apply(
            lambda row: tuple(sorted([str(row['DistPointA']).strip(), str(row['DistPointB']).strip()])), axis=1
        )
        df_lines_agg = df_lines_raw.groupby('pair_key', as_index=False)[COLUMN_HEADERS].max()
        
        # *** MODIFIED SENSITIVITY CALCULATION & LABELING ***
        df_lines_agg['sensitivity'] = calculate_avg_parameter_sensitivity(df_lines_agg, 'Baseline', PARAMETER_PAIRS)
        df_lines_agg['thickness'] = df_lines_agg['Baseline'].apply(scale_line_thickness)
        df_lines_agg[['DistPointA', 'DistPointB']] = pd.DataFrame(df_lines_agg['pair_key'].tolist(), index=df_lines_agg.index)
        df_lines_agg['disappearance_label'] = df_lines_agg.apply(
            lambda row: find_disappearing_scenario(row, SENSITIVITY_COLUMNS), axis=1
        )

        # --- Process Point Data (Hubs) ---
        df_points_raw = pd.read_excel(excel_file_path, sheet_name=sheet_name_point_size)
        for col in COLUMN_HEADERS:
            if col in df_points_raw.columns:
                df_points_raw[col] = pd.to_numeric(df_points_raw[col], errors='coerce')
        df_points_raw[COLUMN_HEADERS] = df_points_raw[COLUMN_HEADERS].fillna(0)
        df_points_raw = df_points_raw.rename(columns={'DistPointA': 'label'})
        df_points_raw['label'] = df_points_raw['label'].astype(str).str.strip()
        
        # *** MODIFIED SENSITIVITY CALCULATION ***
        df_points_raw['sensitivity'] = calculate_avg_parameter_sensitivity(df_points_raw, 'Baseline', PARAMETER_PAIRS)
        df_points_raw['size'] = df_points_raw['Baseline'].apply(scale_point_size)

        # --- Prepare Geodataframes for Plotting ---
        lines_data, connected_points_labels = [], set()
        for _, row in df_lines_agg.iterrows():
            point_a = gdf_all_points[gdf_all_points['label'] == row['DistPointA']]
            point_b = gdf_all_points[gdf_all_points['label'] == row['DistPointB']]
            if not point_a.empty and not point_b.empty and row['thickness'] > 0:
                lines_data.append({
                    'geometry': LineString([point_a.geometry.iloc[0], point_b.geometry.iloc[0]]),
                    'thickness': row['thickness'],
                    'sensitivity': row['sensitivity'],
                    'disappearance_label': row['disappearance_label'], 
                    'Baseline': row['Baseline']# Pass the new label
                })
                connected_points_labels.update([row['DistPointA'], row['DistPointB']])
        gdf_lines = gpd.GeoDataFrame(lines_data, crs="EPSG:4326")
        
        gdf_points_to_plot = gdf_all_points[gdf_all_points['label'].isin(connected_points_labels)].copy()
        gdf_points_to_plot = gdf_points_to_plot.merge(df_points_raw[['label', 'Baseline', 'sensitivity', 'size']], on='label', how='left')
        gdf_points_to_plot.fillna(0, inplace=True)

        # --- Plotting ---
        cmap = plt.get_cmap('viridis')
        norm = mcolors.Normalize(vmin=0, vmax=1.0) # Adjust vmax as needed based on data
        land = gpd.read_file(land_shapefile).to_crs("EPSG:4326").cx[bbox[0]:bbox[2], bbox[1]:bbox[3]]
        eez = gpd.read_file(eez_shapefile).to_crs("EPSG:4326").cx[bbox[0]:bbox[2], bbox[1]:bbox[3]]
        ax.set_facecolor("#aadaff")
        land.plot(ax=ax, color='lightgrey', edgecolor='black', linewidth=0.5)
        eez.boundary.plot(ax=ax, color='black', linewidth=0.6, linestyle='--')
        
        if not gdf_lines.empty:
            gdf_lines.plot(ax=ax, column='sensitivity', cmap=cmap, norm=norm, linewidth=gdf_lines['thickness'], zorder=4)
            # *** ADDED: Loop for line labels ***
            for _, row in gdf_lines.iterrows():
                if row['geometry']:
                    midpoint = row.geometry.centroid
                    fontsize = 10
                    baseline_value = row['Baseline']
                    # Check if the special label exists, otherwise use the numeric sensitivity
                    if pd.notna(row['disappearance_label']):
                        label_text = row['disappearance_label']
                        fontsize = 10  # Use a specific font size for the descriptive text
                    else:
                        label_text = f"sens: {row['sensitivity']:.2f}\n({baseline_value:.0f} GW"
                        
                    
                    ax.text(midpoint.x, midpoint.y + 0.05, label_text, fontsize=fontsize, ha='center', va='bottom',
                            bbox=dict(facecolor='white', alpha=0.7, edgecolor='none', pad=0.1), zorder=7)

        if not gdf_points_to_plot.empty:
            gdf_points_to_plot.plot(ax=ax, column='sensitivity', cmap=cmap, norm=norm,
                                    markersize=gdf_points_to_plot['size'], zorder=5,
                                    edgecolor='black', linewidth=0.8)
            for _, row in gdf_points_to_plot.iterrows():
                label_text = f"{row['label']}\n({row['Baseline']:.0f} GW)" if row['Baseline'] > 0 else row['label']
                ax.text(row.geometry.x, row.geometry.y + 0.1, label_text, fontsize=11, ha='center',
                        bbox=dict(facecolor='white', alpha=0.6, edgecolor='none', pad=0.1), zorder=6)
        
        draw_compass(ax)
        ax.set_xlim(bbox[0], bbox[2]); ax.set_ylim(bbox[1], bbox[3])
        ax.set_title("Sensitivity of North Sea Infrastructure Capacity (Stock)", fontsize=20, pad=20)
        ax.set_xlabel("Longitude"); ax.set_ylabel("Latitude")
        return cmap, norm

    # --- MAIN EXECUTION PART OF THE FUNCTION ---
    fig, ax = plt.subplots(figsize=(15, 15))
    plt.subplots_adjust(left=0.05, right=0.85, top=0.95, bottom=0.05)

    cmap, norm = build_map_on_axes(ax)

    # --- Create Legends ---
    cbar_ax = fig.add_axes([0.87, 0.25, 0.03, 0.5])
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    cbar = fig.colorbar(sm, cax=cbar_ax)
    
    # *** MODIFIED: Updated color bar label ***
    cbar.set_label('Average Parameter Sensitivity', fontsize=12, rotation=270, labelpad=25)

    legend_elements = [
        Line2D([0], [0], color='grey', lw=scale_line_thickness(5), label='5GW Interconnection'),
        Line2D([0], [0], color='grey', lw=scale_line_thickness(15), label='15GW Interconnection'),
        Line2D([0], [0], linestyle='--', color='black', linewidth=0.6, label='EEZ Boundaries')
    ]
    ax.legend(handles=legend_elements, loc='lower left', fontsize=12, title="Baseline Capacity Legend")

    return fig


def create_stock_sensitivity_per_case():
    # --- Configuration ---
    SENSITIVITY_FILE = "Post-Process manual sensitivity"
    COLUMN_HEADERS = [
        'Baseline', 'HVDC-Min', 'HVDC-Max', 'ED-Min', 'ED-Max',
        'EP-Min', 'EP-Max', 'WACC-Min', 'WACC-Max', 'OWF-S-Min', 'OWF-S-Max'
    ]

    # Define the parameter cases for each subplot
    PARAMETER_CASES = {
        "Baseline": ['Baseline', 'Baseline'],
        "HVDC Parameter": ['HVDC-Min', 'HVDC-Max'],
        "Energy Demand Parameter": ['ED-Min', 'ED-Max'],
        "Electricity Price Parameter": ['EP-Min', 'EP-Max'],
        "OWF Stock Parameter": ['OWF-S-Min', 'OWF-S-Max'],
        "WACC Parameter": ['WACC-Min', 'WACC-Max']
    }

    # Common file paths and settings
    excel_file_locations = "manual input/Hub locations input.xlsx"
    sheet_name_line_capacity = "XC Trade Stock"
    sheet_name_point_size = "Offshore Power Stock"
    sheet_name_NSGorDirect = "NSG"

    eez_shapefile = "data/eez/eez_v12.shp"
    land_shapefile = "data/naturalearth/ne_10m_admin_0_countries/ne_10m_admin_0_countries.shp"
    bbox = (-2, 51, 10, 60)
    output_directory = "output/visualisation/"

    os.makedirs(output_directory, exist_ok=True)

    # --- Scaling Functions ---
    def scale_line_thickness(capacity_gw, data_min=0, data_max=20, viz_min=0.5, viz_max=15.0):
        if pd.isna(capacity_gw) or capacity_gw <= data_min: return 0.0
        if capacity_gw >= data_max: return viz_max
        return viz_min + ((capacity_gw - data_min) / (data_max - data_min)) * (viz_max - viz_min)

    def scale_point_size(power_stock_gw, data_min=0, data_max=80, viz_min=50, viz_max=1000):
        if pd.isna(power_stock_gw) or power_stock_gw <= data_min: return 0.0
        if power_stock_gw >= data_max: return viz_max
        return viz_min + ((power_stock_gw - data_min) / (data_max - data_min)) * (viz_max - viz_min)

    def draw_compass(ax, x_pos=0.9, y_pos=0.95, size_val=0.08):
        ax.annotate('N', xy=(x_pos, y_pos), xytext=(x_pos, y_pos - size_val * 1.5),
                    arrowprops=dict(facecolor='black', width=0.5, headwidth=4, shrink=0.1),
                    ha='center', va='center', fontsize=9, xycoords='axes fraction')

    def calculate_sensitivity(df, case_cols):
        """Calculates sensitivity for a specific parameter case."""
        epsilon = 1e-9
        # Consider the min/max of the specific case columns AND the baseline
        all_case_cols = case_cols + ['Baseline']
        min_vals = df[all_case_cols].min(axis=1)
        max_vals = df[all_case_cols].max(axis=1)
        sensitivity = (abs((max_vals - df['Baseline'])/ df['Baseline'] ) + abs((min_vals - df['Baseline'])/ df['Baseline'] )) / 2
        return sensitivity.clip(0)  # Sensitivity cannot be negative

    def generate_map_for_case(ax, title, case_cols, df_lines_agg, df_points_raw, gdf_all_points, cmap, norm,
                              show_y_label=False):
        """Generates a single map for a specific sensitivity case."""
        # --- Calculate Sensitivity for this specific case ---
        df_lines_agg['sensitivity'] = calculate_sensitivity(df_lines_agg, case_cols)
        df_points_raw['sensitivity'] = calculate_sensitivity(df_points_raw, case_cols)

        # --- Prepare Geodataframes for Plotting ---
        lines_data, connected_points_labels = [], set()
        for _, row in df_lines_agg.iterrows():
            point_a = gdf_all_points[gdf_all_points['label'] == row['DistPointA']]
            point_b = gdf_all_points[gdf_all_points['label'] == row['DistPointB']]
            if not point_a.empty and not point_b.empty and row['thickness'] > 0:
                lines_data.append({
                    'geometry': LineString([point_a.geometry.iloc[0], point_b.geometry.iloc[0]]),
                    'thickness': row['thickness'],
                    'sensitivity': row['sensitivity']
                })
                connected_points_labels.update([row['DistPointA'], row['DistPointB']])
        gdf_lines = gpd.GeoDataFrame(lines_data, crs="EPSG:4326") if lines_data else gpd.GeoDataFrame()

        gdf_points_to_plot = gdf_all_points[gdf_all_points['label'].isin(connected_points_labels)].copy()
        gdf_points_to_plot = gdf_points_to_plot.merge(df_points_raw[['label', 'Baseline', 'sensitivity', 'size']],
                                                      on='label', how='left')
        gdf_points_to_plot.fillna(0, inplace=True)

        # --- Plotting ---
        land = gpd.read_file(land_shapefile).to_crs("EPSG:4326").cx[bbox[0]:bbox[2], bbox[1]:bbox[3]]
        eez = gpd.read_file(eez_shapefile).to_crs("EPSG:4326").cx[bbox[0]:bbox[2], bbox[1]:bbox[3]]
        ax.set_facecolor("#aadaff")
        land.plot(ax=ax, color='lightgrey', edgecolor='black', linewidth=0.5)
        eez.boundary.plot(ax=ax, color='black', linewidth=0.6, linestyle='--')

        if not gdf_lines.empty:
            gdf_lines.plot(ax=ax, column='sensitivity', cmap=cmap, norm=norm, linewidth=gdf_lines['thickness'],
                           zorder=4)
        if not gdf_points_to_plot.empty:
            gdf_points_to_plot.plot(ax=ax, column='sensitivity', cmap=cmap, norm=norm,
                                    markersize=gdf_points_to_plot['size'], zorder=5,
                                    edgecolor='white', linewidth=0.8)

        # Final map settings
        draw_compass(ax)
        ax.set_xlim(bbox[0], bbox[2]);
        ax.set_ylim(bbox[1], bbox[3])
        ax.set_title(title, fontsize=14)
        ax.set_xlabel("Longitude")
        if show_y_label:
            ax.set_ylabel("Latitude")
        else:
            ax.tick_params(axis='y', which='both', left=False, labelleft=False)

    # --- Main Script Execution ---
    # Load and process data ONCE
    excel_file_path = f"manual input/{SENSITIVITY_FILE}.xlsx"
    gdf_all_points = gpd.GeoDataFrame(
        pd.read_excel(excel_file_locations, sheet_name=sheet_name_NSGorDirect),
        geometry=gpd.points_from_xy(pd.read_excel(excel_file_locations, sheet_name=sheet_name_NSGorDirect).longitude,
                                    pd.read_excel(excel_file_locations, sheet_name=sheet_name_NSGorDirect).latitude),
        crs="EPSG:4326"
    )
    gdf_all_points['label'] = gdf_all_points['label'].astype(str).str.strip()

    # Load and clean line data
    df_lines_raw = pd.read_excel(excel_file_path, sheet_name=sheet_name_line_capacity)
    for col in COLUMN_HEADERS:
        if col in df_lines_raw.columns:
            df_lines_raw[col] = pd.to_numeric(df_lines_raw[col], errors='coerce')
    df_lines_raw[COLUMN_HEADERS] = df_lines_raw[COLUMN_HEADERS].fillna(0)
    df_lines_raw['pair_key'] = df_lines_raw.apply(lambda r: tuple(sorted([str(r['DistPointA']), str(r['DistPointB'])])),
                                                  axis=1)
    df_lines_agg = df_lines_raw.groupby('pair_key', as_index=False)[COLUMN_HEADERS].max()
    df_lines_agg['thickness'] = df_lines_agg['Baseline'].apply(scale_line_thickness)
    df_lines_agg[['DistPointA', 'DistPointB']] = pd.DataFrame(df_lines_agg['pair_key'].tolist(),
                                                              index=df_lines_agg.index)

    # Load and clean point data
    df_points_raw = pd.read_excel(excel_file_path, sheet_name=sheet_name_point_size)
    for col in COLUMN_HEADERS:
        if col in df_points_raw.columns:
            df_points_raw[col] = pd.to_numeric(df_points_raw[col], errors='coerce')
    df_points_raw[COLUMN_HEADERS] = df_points_raw[COLUMN_HEADERS].fillna(0)
    df_points_raw = df_points_raw.rename(columns={'DistPointA': 'label'})
    df_points_raw['label'] = df_points_raw['label'].astype(str).str.strip()
    df_points_raw['size'] = df_points_raw['Baseline'].apply(scale_point_size)

    # --- Create the Figure and Subplots ---
    fig, axes = plt.subplots(2, 3, figsize=(20, 14), sharex=True, sharey=True)
    axes_flat = axes.flatten()
    cmap = plt.get_cmap('viridis')
    norm = mcolors.Normalize(vmin=0, vmax=1.0)  # Sensitivity from 0% to 100%

    # Generate each map
    for i, (title, case_cols) in enumerate(PARAMETER_CASES.items()):
        print(f"Generating map for: {title}...")
        show_y = (i % 3 == 0)  # Show Y-label only for the first column
        generate_map_for_case(axes_flat[i], title, case_cols, df_lines_agg.copy(), df_points_raw.copy(), gdf_all_points,
                              cmap, norm, show_y_label=show_y)

    # --- Final Figure Adjustments ---
    fig.suptitle("Sensitivity of North Sea Infrastructure Capacity to Key Parameters", fontsize=24)
    plt.subplots_adjust(left=0.05, right=0.85, top=0.9, bottom=0.1, wspace=0.05, hspace=0.2)

    # --- Create Unified Legend and Colorbar ---
    cbar_ax = fig.add_axes([0.87, 0.25, 0.02, 0.5])
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    cbar = fig.colorbar(sm, cax=cbar_ax)
    cbar.set_label('Sensitivity\n(Parameter Range / Baseline)', fontsize=12, rotation=270, labelpad=25)

    legend_elements = [
        Line2D([0], [0], color='grey', lw=scale_line_thickness(5), label='5GW Interconnection'),
        Line2D([0], [0], color='grey', lw=scale_line_thickness(15), label='15GW Interconnection'),
        Line2D([0], [0], marker='o', color='grey', label='20GW OWF', markersize=np.sqrt(scale_point_size(20)),
               linestyle='None'),
        Line2D([0], [0], marker='o', color='grey', label='60GW OWF', markersize=np.sqrt(scale_point_size(60)),
               linestyle='None'),
        Line2D([0], [0], linestyle='--', color='black', linewidth=0.6, label='EEZ Boundaries')
    ]
    fig.legend(handles=legend_elements, loc='lower center', ncol=5, bbox_to_anchor=(0.45, 0.02), fontsize=12,
               title="Baseline Capacity Legend")

    return fig

def create_use_sensitivity_per_case():
    # --- Configuration ---
    SENSITIVITY_FILE = "Post-Process manual sensitivity"
    COLUMN_HEADERS = [
        'Baseline', 'HVDC-Min', 'HVDC-Max', 'ED-Min', 'ED-Max',
        'EP-Min', 'EP-Max', 'WACC-Min', 'WACC-Max', 'OWF-S-Min', 'OWF-S-Max'
    ]

    # Define the parameter cases for each subplot
    PARAMETER_CASES = {
        "Baseline": ['Baseline', 'Baseline'],
        "HVDC Parameter": ['HVDC-Min', 'HVDC-Max'],
        "Energy Demand Parameter": ['ED-Min', 'ED-Max'],
        "Electricity Price Parameter": ['EP-Min', 'EP-Max'],
        "OWF Stock Parameter": ['OWF-S-Min', 'OWF-S-Max'],
        "WACC Parameter": ['WACC-Min', 'WACC-Max']
    }

    # Common file paths and settings
    excel_file_locations = "manual input/Hub locations input.xlsx"
    sheet_name_line_capacity = "XC Trade Use"
    sheet_name_point_size = "Offshore Power Use"
    sheet_name_NSGorDirect = "NSG"

    eez_shapefile = "data/eez/eez_v12.shp"
    land_shapefile = "data/naturalearth/ne_10m_admin_0_countries/ne_10m_admin_0_countries.shp"
    bbox = (-2, 51, 10, 60)
    output_directory = "output/visualisation/"

    os.makedirs(output_directory, exist_ok=True)

    # --- Scaling Functions ---
    def scale_line_thickness(capacity_gw, data_min=0.1, data_max=1000, viz_min=5, viz_max=25.0):
        """Scales line thickness based on Baseline Use in PJ."""
        if pd.isna(capacity_gw) or capacity_gw <= data_min: return 0.0
        if capacity_gw >= data_max: return viz_max
        return viz_min + ((capacity_gw - data_min) / (data_max - data_min)) * (viz_max - viz_min)

    def scale_point_size(power_stock_gw, data_min=0, data_max=1000, viz_min=50, viz_max=1000):
        """Scales point size based on Baseline Use in PJ."""
        if pd.isna(power_stock_gw) or power_stock_gw <= data_min: return 0.0
        if power_stock_gw >= data_max: return viz_max
        return viz_min + ((power_stock_gw - data_min) / (data_max - data_min)) * (viz_max - viz_min)

    def draw_compass(ax, x_pos=0.9, y_pos=0.95, size_val=0.08):
        ax.annotate('N', xy=(x_pos, y_pos), xytext=(x_pos, y_pos - size_val * 1.5),
                    arrowprops=dict(facecolor='black', width=0.5, headwidth=4, shrink=0.1),
                    ha='center', va='center', fontsize=9, xycoords='axes fraction')

    def calculate_sensitivity(df, case_cols):
        """Calculates sensitivity for a specific parameter case."""
        epsilon = 1e-9
        # Consider the min/max of the specific case columns AND the baseline
        all_case_cols = case_cols + ['Baseline']
        min_vals = df[all_case_cols].min(axis=1)
        max_vals = df[all_case_cols].max(axis=1)
        sensitivity = (abs((max_vals - df['Baseline'])/ df['Baseline'] ) + abs((min_vals - df['Baseline'])/ df['Baseline'] )) / 2
        return sensitivity.clip(0)  # Sensitivity cannot be negative

    def generate_map_for_case(ax, title, case_cols, df_lines_agg, df_points_raw, gdf_all_points, cmap, norm,
                              show_y_label=False):
        """Generates a single map for a specific sensitivity case."""
        # --- Calculate Sensitivity for this specific case ---
        df_lines_agg['sensitivity'] = calculate_sensitivity(df_lines_agg, case_cols)
        df_points_raw['sensitivity'] = calculate_sensitivity(df_points_raw, case_cols)

        # --- Prepare Geodataframes for Plotting ---
        lines_data, connected_points_labels = [], set()
        for _, row in df_lines_agg.iterrows():
            point_a = gdf_all_points[gdf_all_points['label'] == row['DistPointA']]
            point_b = gdf_all_points[gdf_all_points['label'] == row['DistPointB']]
            if not point_a.empty and not point_b.empty and row['thickness'] > 0:
                lines_data.append({
                    'geometry': LineString([point_a.geometry.iloc[0], point_b.geometry.iloc[0]]),
                    'thickness': row['thickness'],
                    'sensitivity': row['sensitivity']
                })
                connected_points_labels.update([row['DistPointA'], row['DistPointB']])
        gdf_lines = gpd.GeoDataFrame(lines_data, crs="EPSG:4326") if lines_data else gpd.GeoDataFrame()

        gdf_points_to_plot = gdf_all_points[gdf_all_points['label'].isin(connected_points_labels)].copy()
        gdf_points_to_plot = gdf_points_to_plot.merge(df_points_raw[['label', 'Baseline', 'sensitivity', 'size']],
                                                      on='label', how='left')
        gdf_points_to_plot.fillna(0, inplace=True)

        # --- Plotting ---
        land = gpd.read_file(land_shapefile).to_crs("EPSG:4326").cx[bbox[0]:bbox[2], bbox[1]:bbox[3]]
        eez = gpd.read_file(eez_shapefile).to_crs("EPSG:4326").cx[bbox[0]:bbox[2], bbox[1]:bbox[3]]
        ax.set_facecolor("#aadaff")
        land.plot(ax=ax, color='lightgrey', edgecolor='black', linewidth=0.5)
        eez.boundary.plot(ax=ax, color='black', linewidth=0.6, linestyle='--')

        if not gdf_lines.empty:
            gdf_lines.plot(ax=ax, column='sensitivity', cmap=cmap, norm=norm, linewidth=gdf_lines['thickness'],
                           zorder=4)
        if not gdf_points_to_plot.empty:
            gdf_points_to_plot.plot(ax=ax, column='sensitivity', cmap=cmap, norm=norm,
                                    markersize=gdf_points_to_plot['size'], zorder=5,
                                    edgecolor='white', linewidth=0.8)

        # Final map settings
        draw_compass(ax)
        ax.set_xlim(bbox[0], bbox[2]);
        ax.set_ylim(bbox[1], bbox[3])
        ax.set_title(title, fontsize=14)
        ax.set_xlabel("Longitude")
        if show_y_label:
            ax.set_ylabel("Latitude")
        else:
            ax.tick_params(axis='y', which='both', left=False, labelleft=False)

    # --- Main Script Execution ---
    # Load and process data ONCE
    excel_file_path = f"manual input/{SENSITIVITY_FILE}.xlsx"
    gdf_all_points = gpd.GeoDataFrame(
        pd.read_excel(excel_file_locations, sheet_name=sheet_name_NSGorDirect),
        geometry=gpd.points_from_xy(pd.read_excel(excel_file_locations, sheet_name=sheet_name_NSGorDirect).longitude,
                                    pd.read_excel(excel_file_locations, sheet_name=sheet_name_NSGorDirect).latitude),
        crs="EPSG:4326"
    )
    gdf_all_points['label'] = gdf_all_points['label'].astype(str).str.strip()

    # Load and clean line data
    df_lines_raw = pd.read_excel(excel_file_path, sheet_name=sheet_name_line_capacity)
    for col in COLUMN_HEADERS:
        if col in df_lines_raw.columns:
            df_lines_raw[col] = pd.to_numeric(df_lines_raw[col], errors='coerce')
    df_lines_raw[COLUMN_HEADERS] = df_lines_raw[COLUMN_HEADERS].fillna(0)
    df_lines_raw['pair_key'] = df_lines_raw.apply(lambda r: tuple(sorted([str(r['DistPointA']), str(r['DistPointB'])])),
                                                  axis=1)
    df_lines_agg = df_lines_raw.groupby('pair_key', as_index=False)[COLUMN_HEADERS].max()
    df_lines_agg['thickness'] = df_lines_agg['Baseline'].apply(scale_line_thickness)
    df_lines_agg[['DistPointA', 'DistPointB']] = pd.DataFrame(df_lines_agg['pair_key'].tolist(),
                                                              index=df_lines_agg.index)

    # Load and clean point data
    df_points_raw = pd.read_excel(excel_file_path, sheet_name=sheet_name_point_size)
    for col in COLUMN_HEADERS:
        if col in df_points_raw.columns:
            df_points_raw[col] = pd.to_numeric(df_points_raw[col], errors='coerce')
    df_points_raw[COLUMN_HEADERS] = df_points_raw[COLUMN_HEADERS].fillna(0)
    df_points_raw = df_points_raw.rename(columns={'DistPointA': 'label'})
    df_points_raw['label'] = df_points_raw['label'].astype(str).str.strip()
    df_points_raw['size'] = df_points_raw['Baseline'].apply(scale_point_size)

    # --- Create the Figure and Subplots ---
    fig, axes = plt.subplots(2, 3, figsize=(20, 14), sharex=True, sharey=True)
    axes_flat = axes.flatten()
    cmap = plt.get_cmap('viridis')
    norm = mcolors.Normalize(vmin=0, vmax=1.0)  # Sensitivity from 0% to 100%

    # Generate each map
    for i, (title, case_cols) in enumerate(PARAMETER_CASES.items()):
        print(f"Generating map for: {title}...")
        show_y = (i % 3 == 0)  # Show Y-label only for the first column
        generate_map_for_case(axes_flat[i], title, case_cols, df_lines_agg.copy(), df_points_raw.copy(), gdf_all_points,
                              cmap, norm, show_y_label=show_y)

    # --- Final Figure Adjustments ---
    fig.suptitle("Sensitivity of North Sea Infrastructure Use to Key Parameters", fontsize=24)
    plt.subplots_adjust(left=0.05, right=0.85, top=0.9, bottom=0.1, wspace=0.05, hspace=0.2)

    # --- Create Unified Legend and Colorbar ---
    cbar_ax = fig.add_axes([0.87, 0.25, 0.02, 0.5])
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    cbar = fig.colorbar(sm, cax=cbar_ax)
    cbar.set_label('Sensitivity\n(Parameter Range / Baseline)', fontsize=12, rotation=270, labelpad=25)

    legend_elements = [
        Line2D([0], [0], color='grey', lw=scale_line_thickness(50), label='50PJ Interconnection'),
        Line2D([0], [0], color='grey', lw=scale_line_thickness(350), label='350PJ Interconnection'),
        Line2D([0], [0], marker='o', color='grey', label='200PJ OWF', markersize=np.sqrt(scale_point_size(200)),
               linestyle='None'),
        Line2D([0], [0], marker='o', color='grey', label='600PJ OWF', markersize=np.sqrt(scale_point_size(600)),
               linestyle='None'),
        Line2D([0], [0], linestyle='--', color='black', linewidth=0.6, label='EEZ Boundaries')
    ]
    fig.legend(handles=legend_elements, loc='lower center', ncol=5, bbox_to_anchor=(0.45, 0.02), fontsize=12,
               title="Baseline Capacity Legend")

    return fig

def create_individual_case_figure(case_name):
    """
    Builds a single, detailed map for a specific case, showing capacity and capacity factor.
    It uses pre-loaded data for efficiency.
    """
    # --- Get the pre-loaded data ---
    df_locations, df_line_capacity_all, df_point_stock_all, df_line_use_all = load_all_data()

    # --- Constants and Helper Functions ---
    PJ_TO_TWH = 1 / 3.6
    GW_TO_TWH_PER_YEAR = 8.76
    epsilon = 1e-9
    bbox = (-2, 51, 10, 60)
    eez_shapefile = "data/eez/eez_v12.shp"
    land_shapefile = "data/naturalearth/ne_10m_admin_0_countries/ne_10m_admin_0_countries.shp"

    def scale_line_thickness(capacity_gw, data_min=0.1, data_max=25, viz_min=0.5, viz_max=18.0):
        if pd.isna(capacity_gw) or capacity_gw <= data_min: return 0.0
        if capacity_gw >= data_max: return viz_max
        return viz_min + ((capacity_gw - data_min) / (data_max - data_min)) * (viz_max - viz_min)

    def scale_point_size(power_stock_gw, data_min=0, data_max=90, viz_min=50, viz_max=1200):
        if pd.isna(power_stock_gw) or power_stock_gw <= data_min: return 0.0
        if power_stock_gw >= data_max: return viz_max
        return viz_min + ((power_stock_gw - data_min) / (data_max - data_min)) * (viz_max - viz_min)

    def draw_compass(ax, x_pos=0.9, y_pos=0.95, size_val=0.08):
        ax.annotate('N', xy=(x_pos, y_pos), xytext=(x_pos, y_pos - size_val * 1.5),
                    arrowprops=dict(facecolor='black', width=0.5, headwidth=4, shrink=0.1),
                    ha='center', va='center', fontsize=9, xycoords='axes fraction')

    # --- Main Plotting Logic ---
    fig, ax = plt.subplots(figsize=(12, 12))
    plt.subplots_adjust(left=0.05, right=0.8, top=0.9, bottom=0.05)

    # 1. Process Line Capacity Data for the selected case
    df_lines = df_line_capacity_all[['DistPointA', 'DistPointB', case_name]].copy()
    df_lines.rename(columns={case_name: 'capacity_gw'}, inplace=True)
    df_lines['pair_key'] = df_lines.apply(lambda r: tuple(sorted([str(r['DistPointA']), str(r['DistPointB'])])), axis=1)
    df_lines_agg = df_lines.groupby('pair_key').agg(
        {'capacity_gw': 'max', 'DistPointA': 'first', 'DistPointB': 'first'}).reset_index()

    # 2. Process Line Usage Data for the selected case
    df_use = df_line_use_all[['DistPointA', 'DistPointB', case_name]].copy()
    df_use.rename(columns={case_name: 'use_pj'}, inplace=True)
    df_use['pair_key'] = df_use.apply(lambda r: tuple(sorted([str(r['DistPointA']), str(r['DistPointB'])])), axis=1)
    df_use_agg = df_use.groupby('pair_key')['use_pj'].sum().reset_index()

    # 3. Merge and calculate capacity factor
    df_lines_final = pd.merge(df_lines_agg, df_use_agg, on='pair_key', how='left').fillna(0)
    df_lines_final['capacity_factor'] = (df_lines_final['use_pj'] * PJ_TO_TWH) / (
                (df_lines_final['capacity_gw'] * GW_TO_TWH_PER_YEAR) + epsilon)
    df_lines_final['capacity_factor'] = df_lines_final['capacity_factor'].clip(0, 1)
    df_lines_final['thickness'] = df_lines_final['capacity_gw'].apply(scale_line_thickness)

    # 4. Process Point Data for the selected case
    df_points = df_point_stock_all[['DistPointA', case_name]].copy()
    df_points.rename(columns={case_name: 'power_stock_gw', 'DistPointA': 'label'}, inplace=True)
    df_points['size'] = df_points['power_stock_gw'].apply(scale_point_size)

    # Prepare GeoDataFrames for plotting
    lines_data, connected_points_labels = [], set()
    for _, row in df_lines_final.iterrows():
        point_a = df_locations[df_locations['label'] == row['DistPointA']]
        point_b = df_locations[df_locations['label'] == row['DistPointB']]
        if not point_a.empty and not point_b.empty and row['thickness'] > 0:
            lines_data.append({'geometry': LineString([point_a.geometry.iloc[0], point_b.geometry.iloc[0]]),
                               'thickness': row['thickness'], 'capacity_factor': row['capacity_factor'],
                               'label': f"{row['capacity_gw']:.1f} GW"})
            connected_points_labels.update([row['DistPointA'], row['DistPointB']])
    gdf_lines = gpd.GeoDataFrame(lines_data, crs="EPSG:4326")
    gdf_points_to_plot = df_locations[df_locations['label'].isin(connected_points_labels)].copy()
    gdf_points_to_plot = gdf_points_to_plot.merge(df_points, on='label', how='left').fillna(0)

    # Plotting basemap and data
    cmap = plt.get_cmap('plasma')
    norm = mcolors.Normalize(vmin=0, vmax=1.0)
    land = gpd.read_file(land_shapefile).to_crs("EPSG:4326").cx[bbox[0]:bbox[2], bbox[1]:bbox[3]]
    eez = gpd.read_file(eez_shapefile).to_crs("EPSG:4326").cx[bbox[0]:bbox[2], bbox[1]:bbox[3]]
    ax.set_facecolor("#aadaff")
    land.plot(ax=ax, color='lightgrey', edgecolor='black', linewidth=0.5)
    eez.boundary.plot(ax=ax, color='black', linewidth=0.6, linestyle='--')

    if not gdf_lines.empty:
        gdf_lines.plot(ax=ax, column='capacity_factor', cmap=cmap, norm=norm, linewidth=gdf_lines['thickness'],
                       zorder=4)
        for _, row in gdf_lines.iterrows():
            midpoint = row.geometry.interpolate(0.5, normalized=True)
            ax.text(midpoint.x, midpoint.y, row['label'], ha='center', va='center', fontsize=9,
                    bbox=dict(facecolor='white', alpha=0.6, edgecolor='none', pad=0.1), zorder=7)

    if not gdf_points_to_plot.empty:
        ax.scatter(gdf_points_to_plot.geometry.x, gdf_points_to_plot.geometry.y, s=gdf_points_to_plot['size'],
                   facecolors='none', edgecolors='black', linewidths=1.2, zorder=4.9)
        ax.scatter(gdf_points_to_plot.geometry.x, gdf_points_to_plot.geometry.y, s=gdf_points_to_plot['size'] * 0.85,
                   c='orange', zorder=5)
        for _, row in gdf_points_to_plot.iterrows():
            if row['power_stock_gw'] > 0.1:
                ax.text(row.geometry.x, row.geometry.y + 0.12, f"{row['label']}: {row['power_stock_gw']:.1f} GW",
                        ha='center', fontsize=9, bbox=dict(facecolor='white', alpha=0.5, edgecolor='none', pad=0.1),
                        zorder=6)

    # Final Touches
    draw_compass(ax)
    ax.set_xlim(bbox[0], bbox[2]);
    ax.set_ylim(bbox[1], bbox[3])
    ax.set_title(f"Infrastructure Map for Case: {case_name}", fontsize=16, pad=20)
    ax.set_xlabel("Longitude");
    ax.set_ylabel("Latitude")

    cbar_ax = fig.add_axes([0.82, 0.25, 0.03, 0.5])
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    cbar = fig.colorbar(sm, cax=cbar_ax)
    cbar.set_label('Interconnector Capacity Factor', fontsize=12)

    return fig



def build_interconnection_summary_table(df_stock, df_use):
    """
    Constructs a summary table of interconnections, labeling disappearing cases and 
    excluding entries with missing data.
    
    Args:
        df_stock (pd.DataFrame): Stock interconnection data with 'pair_key', 'sensitivity', and 'Baseline'.
        df_use (pd.DataFrame): Use interconnection data with 'pair_key', 'sensitivity', and 'Baseline'.
    
    Returns:
        pd.DataFrame: Cleaned and labeled summary table.
    """
    # Prepare stock summary
    stock_summary = df_stock[['pair_key', 'sensitivity', 'Baseline']].copy()
    stock_summary.rename(columns={
        'sensitivity': 'Avg Stock Sensitivity',
        'Baseline': 'Stock Baseline'
    }, inplace=True)

    # Prepare use summary
    use_summary = df_use[['pair_key', 'sensitivity', 'Baseline']].copy()
    use_summary.rename(columns={
        'sensitivity': 'Avg Use Sensitivity',
        'Baseline': 'Use Baseline'
    }, inplace=True)

    # Merge on pair_key
    merged = pd.merge(stock_summary, use_summary, on='pair_key', how='outer')

    # Label high sensitivities as disappearing

def label_disappearance(value):
        if pd.isna(value):
            return value
        return "IC disappears in one or more cases" if value > 5 else round(value, 2)

    merged['Avg Stock Sensitivity'] = merged['Avg Stock Sensitivity'].apply(label_disappearance)
    merged['Avg Use Sensitivity'] = merged['Avg Use Sensitivity'].apply(label_disappearance)

    # Drop rows where either sensitivity is missing
    merged = merged.dropna(subset=['Avg Stock Sensitivity', 'Avg Use Sensitivity'])

    # Drop rows where either sensitivity became 'None' due to NaN (e.g. before labeling)
    merged = merged[
        ~merged['Avg Stock Sensitivity'].astype(str).str.lower().eq('none') &
        ~merged['Avg Use Sensitivity'].astype(str).str.lower().eq('none')
    ]

    # Sort and set index
    merged.sort_values(by='pair_key', inplace=True)
    merged.set_index('pair_key', inplace=True)

    return merged


def compute_stock_sensitivity_lines():
    """
    Computes stock sensitivity and baseline values for interconnections where both points are in NSG.
    
    Returns:
        pd.DataFrame: Filtered DataFrame with 'pair_key', 'sensitivity', and 'Baseline'.
    """
    # Load data
    gdf_locations, df_line_capacity_all, _, _ = load_all_data()
    valid_labels = set(gdf_locations['label'].astype(str).str.strip())

    PARAMETER_PAIRS = {
        'HVDC': ('HVDC-Min', 'HVDC-Max'),
        'OWF-C': ('OWF-C-Min', 'OWF-C-Max'),
        'ED': ('ED-Min', 'ED-Max'),
        'EP': ('EP-Min', 'EP-Max'),
        'WACC': ('WACC-Min', 'WACC-Max'),
        'OWF-S': ('OWF-S-Min', 'OWF-S-Max')
    }

    df_line_capacity_all['DistPointA'] = df_line_capacity_all['DistPointA'].astype(str).str.strip()
    df_line_capacity_all['DistPointB'] = df_line_capacity_all['DistPointB'].astype(str).str.strip()

    df_line_capacity_all = df_line_capacity_all[
        df_line_capacity_all['DistPointA'].isin(valid_labels) &
        df_line_capacity_all['DistPointB'].isin(valid_labels)
    ]

    df_line_capacity_all['pair_key'] = df_line_capacity_all.apply(
        lambda row: tuple(sorted([row['DistPointA'], row['DistPointB']])), axis=1
    )

    df_lines_stock = df_line_capacity_all.groupby('pair_key', as_index=False).agg({
        **{col: 'max' for col in ['Baseline'] + [c for pair in PARAMETER_PAIRS.values() for c in pair]}
    })

    df_lines_stock['sensitivity'] = calculate_avg_parameter_sensitivity(df_lines_stock, 'Baseline', PARAMETER_PAIRS)
    return df_lines_stock[['pair_key', 'sensitivity', 'Baseline']]


def compute_use_sensitivity_lines():
    """
    Computes use sensitivity and baseline values for interconnections where both points are in NSG.
    
    Returns:
        pd.DataFrame: Filtered DataFrame with 'pair_key', 'sensitivity', and 'Baseline'.
    """
    # Load data
    gdf_locations, _, _, df_line_use_all = load_all_data()
    valid_labels = set(gdf_locations['label'].astype(str).str.strip())

    PARAMETER_PAIRS = {
        'HVDC': ('HVDC-Min', 'HVDC-Max'),
        'OWF-C': ('OWF-C-Min', 'OWF-C-Max'),
        'ED': ('ED-Min', 'ED-Max'),
        'EP': ('EP-Min', 'EP-Max'),
        'WACC': ('WACC-Min', 'WACC-Max'),
        'OWF-S': ('OWF-S-Min', 'OWF-S-Max')
    }

    df_line_use_all['DistPointA'] = df_line_use_all['DistPointA'].astype(str).str.strip()
    df_line_use_all['DistPointB'] = df_line_use_all['DistPointB'].astype(str).str.strip()

    df_line_use_all = df_line_use_all[
        df_line_use_all['DistPointA'].isin(valid_labels) &
        df_line_use_all['DistPointB'].isin(valid_labels)
    ]

    df_line_use_all['pair_key'] = df_line_use_all.apply(
        lambda row: tuple(sorted([row['DistPointA'], row['DistPointB']])), axis=1
    )

    df_lines_use = df_line_use_all.groupby('pair_key', as_index=False).agg({
        **{col: 'max' for col in ['Baseline'] + [c for pair in PARAMETER_PAIRS.values() for c in pair]}
    })

    df_lines_use['sensitivity'] = calculate_avg_parameter_sensitivity(df_lines_use, 'Baseline', PARAMETER_PAIRS)
    return df_lines_use[['pair_key', 'sensitivity', 'Baseline']]




# =============================================================================
# Streamlit App Main Interface
# =============================================================================

# --- Sidebar for selecting the plot ---
st.sidebar.header("Select Visualization")
# You can add more options here as you fill in the other functions
vis_options = [
    "Select a visualization...",
    "Overall Sensitivity of Capacity (Stock)",
    "Overall Sensitivity of System Use (Use)",
    "Capacity Sensitivity per Parameter",
    "Use Sensitivity per Parameter",
    "Detailed Map per Case (Stock & Capacity Factor)",
    "Table: IC Sensitivity Overview"
]
plot_choice = st.sidebar.selectbox(
    "Choose which analysis to display:",
    vis_options
)

# --- Display the selected plot ---
if plot_choice == "Overall Sensitivity of Capacity (Stock)":
    st.header("Overall Sensitivity of Infrastructure Capacity (Stock)")
    with st.spinner('Generating map...'):
        fig = create_stock_sensitivity_figure()
        st.pyplot(fig)

elif plot_choice == "Overall Sensitivity of System Use (Use)":
    st.header("Overall Sensitivity of System Use (Use)")
    with st.spinner('Generating map...'):
        fig = create_use_sensitivity_figure()
        st.pyplot(fig)

elif plot_choice == "Capacity Sensitivity per Parameter":
    st.header("Capacity Sensitivity per Parameter")
    with st.spinner('Generating all 6 maps...'):
        fig = create_stock_sensitivity_per_case()
        st.pyplot(fig)

elif plot_choice == "Use Sensitivity per Parameter":
    st.header("Use Sensitivity per Parameter")
    with st.spinner('Generating all 6 maps...'):
        fig = create_use_sensitivity_per_case()
        st.pyplot(fig)


elif plot_choice == "Detailed Map per Case (Stock & Capacity Factor)":
    st.header("Detailed Infrastructure Map for a Single Case")
    st.write(
        "This map shows the detailed results for a single sensitivity case, including installed capacity (GW) and the resulting capacity factor.\n\n"
        "The respective scenarios are:\n\n"
        "Baseline: Baseline results (AVG scenario)\n\n"
        "HVDC-Min/Max: minimum and maximum HVDC connection cost\n\n"
        "OWF-C-Min/Max: minimum and maximum OWF construction cost\n\n"
        "ED-Min/Max: minimum and maximum changes in energy demand\n\n"
        "EP-Min/Max: minimum and maximum electricity prices\n\n"
        "WACC-Min/Max: minimum and maximum WACC value\n\n"
        "OWF-S-Min/Max: minimum and maximum installed OWF capacity\n\n")

    COLUMN_HEADERS = [
        'Baseline', 'HVDC-Min', 'HVDC-Max', 'OWF-C-Min', 'OWF-C-Max', 'ED-Min', 'ED-Max',
        'EP-Min', 'EP-Max', 'WACC-Min', 'WACC-Max', 'OWF-S-Min', 'OWF-S-Max'
    ]

    # This is where the user selects which of the 13 maps to see
    case_selection = st.selectbox("Select a case to visualize:", COLUMN_HEADERS)

    with st.spinner(f'Generating map for {case_selection}...'):
        # This corresponds to your "code_block_5_stock_simple_vis"
        fig = create_individual_case_figure(case_selection)
        st.pyplot(fig)

elif plot_choice == "Table: IC Sensitivity Overview":

    # Compute data
    df_lines_stock = compute_stock_sensitivity_lines()
    df_lines_use = compute_use_sensitivity_lines()
    
    # Build and display summary
    summary_table = build_interconnection_summary_table(df_lines_stock, df_lines_use)
    st.dataframe(summary_table.style.format("{:.2f}"), use_container_width=True)

else:
    st.write("Please select a visualization from the sidebar to begin.")
