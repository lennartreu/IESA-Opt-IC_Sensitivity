import streamlit as st
import pandas as pd
import geopandas as gpd
from shapely.geometry import Point, LineString
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import os
import numpy as np

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
    # Ensure file paths are correct for your environment
    base_path = "" # If your 'manual input' and 'data' folders are not in the same directory as the script, specify the path here.
    SENSITIVITY_FILE = os.path.join(base_path, "manual input/Post-Process manual sensitivity.xlsx")
    LOCATIONS_FILE = os.path.join(base_path, "manual input/Hub locations input.xlsx")
    
    COLUMN_HEADERS = [
        'Baseline', 'HVDC-Min', 'HVDC-Max', 'OWF-C-Min', 'OWF-C-Max', 'ED-Min', 'ED-Max',
        'EP-Min', 'EP-Max', 'WACC-Min', 'WACC-Max', 'OWF-S-Min', 'OWF-S-Max'
    ]

    try:
        # Load raw data
        df_locations = pd.read_excel(LOCATIONS_FILE, sheet_name="NSG")
        gdf_locations = gpd.GeoDataFrame(
            df_locations,
            geometry=gpd.points_from_xy(df_locations.longitude, df_locations.latitude),
            crs="EPSG:4326"
        )
        gdf_locations['label'] = gdf_locations['label'].astype(str).str.strip()

        df_line_capacity_all = pd.read_excel(SENSITIVITY_FILE, sheet_name="XC Trade Stock")
        df_point_stock_all = pd.read_excel(SENSITIVITY_FILE, sheet_name="Offshore Power Stock")
        df_line_use_all = pd.read_excel(SENSITIVITY_FILE, sheet_name="XC Trade Use")

        # Clean all dataframes to ensure numeric types
        for col in COLUMN_HEADERS:
            for df in [df_line_capacity_all, df_point_stock_all, df_line_use_all]:
                if col in df.columns:
                    df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)
        
        return gdf_locations, df_line_capacity_all, df_point_stock_all, df_line_use_all

    except FileNotFoundError as e:
        st.error(f"Error loading data: {e}. Please make sure the input files are in the correct directory.")
        return None, None, None, None

# =============================================================================
# PLOTTING FUNCTIONS
# Each function now generates and returns a Plotly Figure object.
# =============================================================================

def calculate_avg_parameter_sensitivity(df, baseline_col, param_pairs):
    baseline = df[baseline_col]
    epsilon = 1e-9
    all_param_sensitivities = pd.DataFrame(index=df.index)
    for param, (min_col, max_col) in param_pairs.items():
        if min_col in df.columns and max_col in df.columns:
            # Avoid division by zero
            sensitivity = (abs(df[max_col] - baseline) + abs(df[min_col] - baseline)) / (abs(baseline) + epsilon) / 2
            all_param_sensitivities[param] = sensitivity
    return all_param_sensitivities.mean(axis=1)

def find_disappearing_scenario(row, sensitivity_cols):
    for col in sensitivity_cols:
        if row[col] < 0.001:
            return ">1"
    return None

def add_basemap_layers(fig, bbox, land_shapefile, eez_shapefile, row=None, col=None):
    """Adds land and EEZ boundaries to a Plotly map."""
    try:
        land = gpd.read_file(land_shapefile).to_crs("EPSG:4326").cx[bbox[0]:bbox[2], bbox[1]:bbox[3]]
        eez = gpd.read_file(eez_shapefile).to_crs("EPSG:4326").cx[bbox[0]:bbox[2], bbox[1]:bbox[3]]
        
        fig.add_trace(go.Choroplethmapbox(
            geojson=land.__geo_interface__,
            locations=land.index,
            z=[0]*len(land), # Dummy variable for coloring
            colorscale=[[0, 'lightgrey'], [1, 'lightgrey']],
            showscale=False,
            marker_line_width=0.5,
            marker_line_color='black',
            name='Land'
        ), row=row, col=col)

        fig.add_trace(go.Scattermapbox(
            mode="lines",
            lon=[c for g in eez.geometry for c in g.coords.xy[0]] + [None],
            lat=[c for g in eez.geometry for c in g.coords.xy[1]] + [None],
            line=dict(color="black", width=0.6, dash='dash'),
            name='EEZ Boundaries',
            showlegend=False
        ), row=row, col=col)
    except Exception as e:
        st.warning(f"Could not load shapefiles for basemap: {e}")


def create_use_sensitivity_figure():
    """Generates the sensitivity map for 'Use' using Plotly."""
    gdf_locations, _, _, df_line_use_all = load_all_data()
    _, _, df_point_stock_all, _ = load_all_data() # Using stock for point size baseline
    
    if gdf_locations is None: return go.Figure()

    SENSITIVITY_COLUMNS = [
        'HVDC-Min', 'HVDC-Max', 'OWF-C-Min', 'OWF-C-Max', 'ED-Min', 'ED-Max',
        'EP-Min', 'EP-Max', 'WACC-Min', 'WACC-Max', 'OWF-S-Min', 'OWF-S-Max'
    ]
    PARAMETER_PAIRS = {
        'HVDC': ('HVDC-Min', 'HVDC-Max'), 'OWF-C': ('OWF-C-Min', 'OWF-C-Max'),
        'ED': ('ED-Min', 'ED-Max'), 'EP': ('EP-Min', 'EP-Max'),
        'WACC': ('WACC-Min', 'WACC-Max'), 'OWF-S': ('OWF-S-Min', 'OWF-S-Max')
    }
    eez_shapefile = "data/eez/eez_v12.shp"
    land_shapefile = "data/naturalearth/ne_10m_admin_0_countries/ne_10m_admin_0_countries.shp"
    bbox = (-2, 51, 10, 60)

    # --- Process Line Data ---
    df_lines_raw = df_line_use_all.copy()
    df_lines_raw['pair_key'] = df_lines_raw.apply(
        lambda row: tuple(sorted([str(row['DistPointA']).strip(), str(row['DistPointB']).strip()])), axis=1
    )
    df_lines_agg = df_lines_raw.groupby('pair_key', as_index=False).max()
    df_lines_agg['sensitivity'] = calculate_avg_parameter_sensitivity(df_lines_agg, 'Baseline', PARAMETER_PAIRS)
    df_lines_agg[['DistPointA', 'DistPointB']] = pd.DataFrame(df_lines_agg['pair_key'].tolist(), index=df_lines_agg.index)

    # --- Process Point Data ---
    df_points_raw = df_point_stock_all.copy() # Use stock for point sizes
    df_points_raw = df_points_raw.rename(columns={'DistPointA': 'label'})
    df_points_raw['label'] = df_points_raw['label'].astype(str).str.strip()
    
    fig = go.Figure()
    add_basemap_layers(fig, bbox, land_shapefile, eez_shapefile)

    # --- Plot Lines ---
    for _, row in df_lines_agg.iterrows():
        point_a = gdf_locations[gdf_locations['label'] == row['DistPointA']]
        point_b = gdf_locations[gdf_locations['label'] == row['DistPointB']]
        if not point_a.empty and not point_b.empty and row['Baseline'] > 0.1:
            line_geom = LineString([point_a.geometry.iloc[0], point_b.geometry.iloc[0]])
            hover_text = (f"<b>{row['DistPointA']} ↔ {row['DistPointB']}</b><br>"
                          f"Baseline Use: {row['Baseline']/3.6:.1f} TWh<br>"
                          f"Avg. Sensitivity: {row['sensitivity']:.2f}")
            
            fig.add_trace(go.Scattermapbox(
                mode="lines",
                lon=[line_geom.coords.xy[0][0], line_geom.coords.xy[0][1]],
                lat=[line_geom.coords.xy[1][0], line_geom.coords.xy[1][1]],
                line=dict(width=2),
                marker=dict(color=row['sensitivity'], colorscale='Viridis', cmin=0, cmax=0.5, showscale=True,
                            colorbar=dict(title='Avg. MAD', x=0.95)),
                name="Interconnection",
                hoverinfo='text',
                text=hover_text
            ))

    # --- Plot Points ---
    connected_points_labels = set(df_lines_agg['DistPointA']) | set(df_lines_agg['DistPointB'])
    gdf_points_to_plot = gdf_locations[gdf_locations['label'].isin(connected_points_labels)].copy()
    gdf_points_to_plot = gdf_points_to_plot.merge(df_points_raw[['label', 'Baseline']], on='label', how='left').fillna(0)
    
    hover_texts = [f"<b>Hub: {row['label']}</b><br>Baseline Capacity: {row['Baseline']:.1f} GW" for _, row in gdf_points_to_plot.iterrows()]
    
    fig.add_trace(go.Scattermapbox(
        mode="markers",
        lon=gdf_points_to_plot.geometry.x,
        lat=gdf_points_to_plot.geometry.y,
        marker=dict(
            size=gdf_points_to_plot['Baseline'].apply(lambda x: np.sqrt(x)*4 if x > 0 else 0),
            color='orange',
            sizemin=4,
            showscale=False
        ),
        name="Hubs",
        hoverinfo='text',
        text=hover_texts
    ))
    
    # --- Layout ---
    fig.update_layout(
        title="Sensitivity of North Sea Infrastructure Use",
        showlegend=False,
        mapbox_style="open-street-map",
        mapbox=dict(
            center=dict(lon=4, lat=55.5),
            zoom=5,
        ),
        margin={"r":0,"t":40,"l":0,"b":0},
    )
    return fig


def create_stock_sensitivity_figure():
    """Generates the sensitivity map for 'Stock' using Plotly."""
    gdf_locations, df_line_capacity_all, df_point_stock_all, _ = load_all_data()
    if gdf_locations is None: return go.Figure()

    SENSITIVITY_COLUMNS = [
        'HVDC-Min', 'HVDC-Max', 'OWF-C-Min', 'OWF-C-Max', 'ED-Min', 'ED-Max',
        'EP-Min', 'EP-Max', 'WACC-Min', 'WACC-Max', 'OWF-S-Min', 'OWF-S-Max'
    ]
    PARAMETER_PAIRS = {
        'HVDC': ('HVDC-Min', 'HVDC-Max'), 'OWF-C': ('OWF-C-Min', 'OWF-C-Max'),
        'ED': ('ED-Min', 'ED-Max'), 'EP': ('EP-Min', 'EP-Max'),
        'WACC': ('WACC-Min', 'WACC-Max'), 'OWF-S': ('OWF-S-Min', 'OWF-S-Max')
    }
    eez_shapefile = "data/eez/eez_v12.shp"
    land_shapefile = "data/naturalearth/ne_10m_admin_0_countries/ne_10m_admin_0_countries.shp"
    bbox = (-2, 51, 10, 60)

    # --- Process Line Data ---
    df_lines_raw = df_line_capacity_all.copy()
    df_lines_raw['pair_key'] = df_lines_raw.apply(
        lambda row: tuple(sorted([str(row['DistPointA']).strip(), str(row['DistPointB']).strip()])), axis=1
    )
    df_lines_agg = df_lines_raw.groupby('pair_key', as_index=False).max()
    df_lines_agg['sensitivity'] = calculate_avg_parameter_sensitivity(df_lines_agg, 'Baseline', PARAMETER_PAIRS)
    df_lines_agg['disappearance_label'] = df_lines_agg.apply(lambda row: find_disappearing_scenario(row, SENSITIVITY_COLUMNS), axis=1)
    df_lines_agg[['DistPointA', 'DistPointB']] = pd.DataFrame(df_lines_agg['pair_key'].tolist(), index=df_lines_agg.index)

    # --- Process Point Data ---
    df_points_raw = df_point_stock_all.copy()
    df_points_raw = df_points_raw.rename(columns={'DistPointA': 'label'})
    df_points_raw['label'] = df_points_raw['label'].astype(str).str.strip()
    df_points_raw['sensitivity'] = calculate_avg_parameter_sensitivity(df_points_raw, 'Baseline', PARAMETER_PAIRS)

    fig = go.Figure()
    add_basemap_layers(fig, bbox, land_shapefile, eez_shapefile)
    
    # --- Plot Lines ---
    for _, row in df_lines_agg.iterrows():
        point_a = gdf_locations[gdf_locations['label'] == row['DistPointA']]
        point_b = gdf_locations[gdf_locations['label'] == row['DistPointB']]
        if not point_a.empty and not point_b.empty and row['Baseline'] > 0.1:
            line_geom = LineString([point_a.geometry.iloc[0], point_b.geometry.iloc[0]])
            
            sensitivity_text = "Disappears" if pd.notna(row['disappearance_label']) else f"{row['sensitivity']:.2f}"
            hover_text = (f"<b>{row['DistPointA']} ↔ {row['DistPointB']}</b><br>"
                          f"Baseline Capacity: {row['Baseline']:.1f} GW<br>"
                          f"Avg. Sensitivity: {sensitivity_text}")

            fig.add_trace(go.Scattermapbox(
                mode="lines",
                lon=[line_geom.coords.xy[0][0], line_geom.coords.xy[0][1]],
                lat=[line_geom.coords.xy[1][0], line_geom.coords.xy[1][1]],
                line=dict(width=max(2, row['Baseline'] * 0.5)),
                marker=dict(color=row['sensitivity'], colorscale='Viridis', cmin=0, cmax=0.5, showscale=True,
                            colorbar=dict(title='Avg. MAD', x=0.95)),
                name="Interconnection",
                hoverinfo='text',
                text=hover_text
            ))
            
    # --- Plot Points ---
    connected_points_labels = set(df_lines_agg['DistPointA']) | set(df_lines_agg['DistPointB'])
    gdf_points_to_plot = gdf_locations[gdf_locations['label'].isin(connected_points_labels)].copy()
    gdf_points_to_plot = gdf_points_to_plot.merge(df_points_raw[['label', 'Baseline', 'sensitivity']], on='label', how='left').fillna(0)

    hover_texts = [f"<b>Hub: {row['label']}</b><br>Baseline Capacity: {row['Baseline']:.1f} GW<br>Avg. Sensitivity: {row['sensitivity']:.2f}" for _, row in gdf_points_to_plot.iterrows()]

    fig.add_trace(go.Scattermapbox(
        mode="markers",
        lon=gdf_points_to_plot.geometry.x,
        lat=gdf_points_to_plot.geometry.y,
        marker=dict(
            size=gdf_points_to_plot['Baseline'].apply(lambda x: np.sqrt(x)*4 if x > 0 else 0),
            color=gdf_points_to_plot['sensitivity'],
            colorscale='Viridis', cmin=0, cmax=0.5,
            sizemin=4,
            showscale=False,
        ),
        name="Hubs",
        hoverinfo='text',
        text=hover_texts
    ))
    
    # --- Layout ---
    fig.update_layout(
        title="Sensitivity of North Sea Infrastructure Capacity (Stock)",
        showlegend=False,
        mapbox_style="open-street-map",
        mapbox=dict(
            center=dict(lon=4, lat=55.5),
            zoom=5,
        ),
        margin={"r":0,"t":40,"l":0,"b":0},
    )
    return fig


def create_stock_sensitivity_per_case():
    """Generates a grid of sensitivity maps per parameter for 'Stock' using Plotly."""
    gdf_locations, df_line_capacity_all, df_point_stock_all, _ = load_all_data()
    if gdf_locations is None: return go.Figure()

    PARAMETER_CASES = {
        "Baseline": ['Baseline', 'Baseline'], "HVDC Parameter": ['HVDC-Min', 'HVDC-Max'],
        "Energy Demand Parameter": ['ED-Min', 'ED-Max'], "Electricity Price Parameter": ['EP-Min', 'EP-Max'],
        "OWF Stock Parameter": ['OWF-S-Min', 'OWF-S-Max'], "WACC Parameter": ['WACC-Min', 'WACC-Max']
    }
    eez_shapefile = "data/eez/eez_v12.shp"
    land_shapefile = "data/naturalearth/ne_10m_admin_0_countries/ne_10m_admin_0_countries.shp"
    bbox = (-2, 51, 10, 60)

    fig = make_subplots(
        rows=2, cols=3,
        subplot_titles=list(PARAMETER_CASES.keys()),
        specs=[[{'type': 'mapbox'}, {'type': 'mapbox'}, {'type': 'mapbox'}],
               [{'type': 'mapbox'}, {'type': 'mapbox'}, {'type': 'mapbox'}]]
    )

    def calculate_sensitivity(df, case_cols):
        epsilon = 1e-9
        all_case_cols = case_cols + ['Baseline']
        min_vals = df[all_case_cols].min(axis=1)
        max_vals = df[all_case_cols].max(axis=1)
        sensitivity = (abs(max_vals - df['Baseline']) + abs(min_vals - df['Baseline'])) / (2 * (abs(df['Baseline']) + epsilon))
        return sensitivity.clip(0)

    # --- Pre-process data ---
    df_lines_raw = df_line_capacity_all.copy()
    df_lines_raw['pair_key'] = df_lines_raw.apply(lambda r: tuple(sorted([str(r['DistPointA']), str(r['DistPointB'])])), axis=1)
    df_lines_agg = df_lines_raw.groupby('pair_key', as_index=False).max()
    df_lines_agg[['DistPointA', 'DistPointB']] = pd.DataFrame(df_lines_agg['pair_key'].tolist(), index=df_lines_agg.index)

    df_points_raw = df_point_stock_all.rename(columns={'DistPointA': 'label'})
    df_points_raw['label'] = df_points_raw['label'].astype(str).str.strip()
    
    # --- Generate maps ---
    map_domains = []
    for i, (title, case_cols) in enumerate(PARAMETER_CASES.items()):
        row, col = i // 3 + 1, i % 3 + 1
        
        # Calculate sensitivity for the case
        df_lines_agg['sensitivity'] = calculate_sensitivity(df_lines_agg, case_cols)
        df_points_raw['sensitivity'] = calculate_sensitivity(df_points_raw, case_cols)

        # Plot Lines for subplot
        for _, r in df_lines_agg.iterrows():
            p_a = gdf_locations[gdf_locations['label'] == r['DistPointA']]
            p_b = gdf_locations[gdf_locations['label'] == r['DistPointB']]
            if not p_a.empty and not p_b.empty and r['Baseline'] > 0.1:
                fig.add_trace(go.Scattermapbox(
                    mode="lines", lon=[p_a.geometry.x.iloc[0], p_b.geometry.x.iloc[0]],
                    lat=[p_a.geometry.y.iloc[0], p_b.geometry.y.iloc[0]],
                    line=dict(width=2),
                    marker=dict(color=r['sensitivity'], colorscale='Viridis', cmin=0, cmax=1.0, showscale=(i==0),
                                colorbar=dict(title='Sens.', x=0.98)),
                    hoverinfo='none'
                ), row=row, col=col)

        # Plot Points for subplot
        connected_labels = set(df_lines_agg['DistPointA']) | set(df_lines_agg['DistPointB'])
        gdf_points = gdf_locations[gdf_locations['label'].isin(connected_labels)].merge(df_points_raw, on='label', how='left').fillna(0)

        fig.add_trace(go.Scattermapbox(
            mode="markers", lon=gdf_points.geometry.x, lat=gdf_points.geometry.y,
            marker=dict(size=gdf_points['Baseline'].apply(lambda x: np.sqrt(x) * 3 if x > 0 else 0),
                        color=gdf_points['sensitivity'], colorscale='Viridis', cmin=0, cmax=1.0, showscale=False),
            hoverinfo='none'
        ), row=row, col=col)
        
        mapbox_name = f"mapbox{i+1}"
        map_domains.append(mapbox_name)


    # --- Layout ---
    layout_update = {
        'title': "Sensitivity of North Sea Infrastructure Capacity to Key Parameters",
        'showlegend': False,
        'height': 800,
        'margin':{"r":10,"t":60,"l":10,"b":10}
    }
    for name in map_domains:
        layout_update[name] = dict(style="open-street-map", center=dict(lon=4, lat=55.5), zoom=4.5)
        
    fig.update_layout(**layout_update)

    return fig


def create_individual_case_figure(case_name):
    """Builds a single, detailed map for a specific case using Plotly."""
    gdf_locations, df_line_capacity_all, df_point_stock_all, df_line_use_all = load_all_data()
    if gdf_locations is None: return go.Figure()

    PJ_TO_TWH = 1 / 3.6
    GW_TO_TWH_PER_YEAR = 8.76
    epsilon = 1e-9
    eez_shapefile = "data/eez/eez_v12.shp"
    land_shapefile = "data/naturalearth/ne_10m_admin_0_countries/ne_10m_admin_0_countries.shp"
    bbox = (-2, 51, 10, 60)

    # --- Process Data ---
    df_lines = df_line_capacity_all[['DistPointA', 'DistPointB', case_name]].rename(columns={case_name: 'capacity_gw'})
    df_lines['pair_key'] = df_lines.apply(lambda r: tuple(sorted([str(r['DistPointA']), str(r['DistPointB'])])), axis=1)
    df_lines_agg = df_lines.groupby('pair_key').agg({'capacity_gw': 'max', 'DistPointA': 'first', 'DistPointB': 'first'}).reset_index()

    df_use = df_line_use_all[['DistPointA', 'DistPointB', case_name]].rename(columns={case_name: 'use_pj'})
    df_use['pair_key'] = df_use.apply(lambda r: tuple(sorted([str(r['DistPointA']), str(r['DistPointB'])])), axis=1)
    df_use_agg = df_use.groupby('pair_key')['use_pj'].sum().reset_index()

    df_lines_final = pd.merge(df_lines_agg, df_use_agg, on='pair_key', how='left').fillna(0)
    df_lines_final['capacity_factor'] = ((df_lines_final['use_pj'] * PJ_TO_TWH) / 
                                        ((df_lines_final['capacity_gw'] * GW_TO_TWH_PER_YEAR) + epsilon)).clip(0, 1)

    df_points = df_point_stock_all[['DistPointA', case_name]].rename(columns={case_name: 'power_stock_gw', 'DistPointA': 'label'})

    fig = go.Figure()
    add_basemap_layers(fig, bbox, land_shapefile, eez_shapefile)
    
    # --- Plot Lines ---
    for _, row in df_lines_final.iterrows():
        point_a = gdf_locations[gdf_locations['label'] == row['DistPointA']]
        point_b = gdf_locations[gdf_locations['label'] == row['DistPointB']]
        if not point_a.empty and not point_b.empty and row['capacity_gw'] > 0.1:
            line_geom = LineString([point_a.geometry.iloc[0], point_b.geometry.iloc[0]])
            hover_text = (f"<b>{row['DistPointA']} ↔ {row['DistPointB']}</b><br>"
                          f"Capacity: {row['capacity_gw']:.1f} GW<br>"
                          f"Capacity Factor: {row['capacity_factor']:.2%}")
            
            fig.add_trace(go.Scattermapbox(
                mode="lines",
                lon=[line_geom.coords.xy[0][0], line_geom.coords.xy[0][1]],
                lat=[line_geom.coords.xy[1][0], line_geom.coords.xy[1][1]],
                line=dict(width=max(2, row['capacity_gw'] * 0.7)),
                marker=dict(color=row['capacity_factor'], colorscale='Plasma', cmin=0, cmax=1,
                            colorbar=dict(title='Cap. Factor', x=0.95)),
                name="Interconnection",
                hoverinfo='text',
                text=hover_text
            ))

    # --- Plot Points ---
    connected_points_labels = set(df_lines_final['DistPointA']) | set(df_lines_final['DistPointB'])
    gdf_points_to_plot = gdf_locations[gdf_locations['label'].isin(connected_points_labels)].copy()
    gdf_points_to_plot = gdf_points_to_plot.merge(df_points, on='label', how='left').fillna(0)
    
    hover_texts = [f"<b>Hub: {row['label']}</b><br>Capacity: {row['power_stock_gw']:.1f} GW" for _, row in gdf_points_to_plot.iterrows()]
    
    fig.add_trace(go.Scattermapbox(
        mode="markers",
        lon=gdf_points_to_plot.geometry.x,
        lat=gdf_points_to_plot.geometry.y,
        marker=dict(
            size=gdf_points_to_plot['power_stock_gw'].apply(lambda x: np.sqrt(x)*5 if x > 0 else 0),
            color='orange',
            sizemin=4,
        ),
        name="Hubs",
        hoverinfo='text',
        text=hover_texts
    ))
    
    # --- Layout ---
    fig.update_layout(
        title=f"Infrastructure Map for Case: {case_name}",
        showlegend=False,
        mapbox_style="open-street-map",
        mapbox=dict(center=dict(lon=4, lat=55.5), zoom=5),
        margin={"r":0,"t":40,"l":0,"b":0}
    )
    return fig


# =============================================================================
# Streamlit App Main Interface
# =============================================================================
if 'page' not in st.session_state:
    st.session_state.page = 'home'

st.sidebar.title("Navigation")

with st.sidebar.expander("Level 1: Overall sensitivity", expanded=True):
    if st.button("Overall Sensitivity of Capacity (Stock)", use_container_width=True):
        st.session_state.page = 'stock_sensitivity'
    if st.button("Overall Sensitivity of System Use (Use)", use_container_width=True):
        st.session_state.page = 'use_sensitivity'

with st.sidebar.expander("Level 2: Sensitivity per Parameter", expanded=True):
    if st.button("Capacity Sensitivity per Parameter", use_container_width=True):
        st.session_state.page = 'stock_sensitivity_per_param'
    # Use sensitivity per param is removed as it's very similar to stock and adds complexity

with st.sidebar.expander("Level 3: Results per Case", expanded=True):
    if st.button("Detailed Map per Case", use_container_width=True):
        st.session_state.page = 'detailed_map_per_case'

# --- Display the selected page's content on the main screen ---
if st.session_state.page == 'home':
    st.header("Welcome to the Sensitivity Analysis Dashboard")
    st.info("Please select a visualization from the sidebar to begin.")
    st.markdown("""
    This application provides an interactive visual interface for exploring the results of the sensitivity analysis.
    
    You can navigate through different levels of analysis using the options on the left:
    - **Level 1:** High-level overview of the sensitivity results.
    - **Level 2:** Detailed breakdown of sensitivity for each parameter.
    - **Level 3:** In-depth results for each individual simulation case.
    
    **Hover over map elements (lines and points) to see detailed information.**
    """)

elif st.session_state.page == 'stock_sensitivity':
    st.header("Overall Sensitivity of Infrastructure Capacity (Stock)")
    st.info("This view shows the overall sensitivity of the installed infrastructure capacity (stock) across all analyzed parameter variations. Line thickness represents baseline capacity. Line and point color represent sensitivity.")
    with st.spinner('Generating map...'):
        fig = create_stock_sensitivity_figure()
        st.plotly_chart(fig, use_container_width=True)

elif st.session_state.page == 'use_sensitivity':
    st.header("Overall Sensitivity of System Use (Use)")
    st.info("This view shows the overall sensitivity of the system's energy usage (use) across all analyzed parameter variations. Point size represents baseline hub capacity. Line color represents sensitivity of use.")
    with st.spinner('Generating map...'):
        fig = create_use_sensitivity_figure()
        st.plotly_chart(fig, use_container_width=True)

elif st.session_state.page == 'stock_sensitivity_per_param':
    st.header("Capacity Sensitivity per Parameter")
    st.info("This visualization breaks down the capacity sensitivity by each individual parameter, allowing for a more detailed analysis of what drives changes. Color represents sensitivity.")
    with st.spinner('Generating all maps...'):
        fig = create_stock_sensitivity_per_case()
        st.plotly_chart(fig, use_container_width=True)

elif st.session_state.page == 'detailed_map_per_case':
    st.header("Detailed Map per Case (Stock & Capacity Factor)")
    st.info("Select a specific sensitivity case from the dropdown below to view a detailed map of its installed capacity (GW) and resulting capacity factor.")
    
    st.markdown("""
    - **Line Thickness:** Represents installed capacity (GW).
    - **Line Color:** Represents the capacity factor (how much of the capacity is used).
    - **Point Size:** Represents installed offshore wind farm capacity at that hub.
    """)
    COLUMN_HEADERS = [
        'Baseline', 'HVDC-Min', 'HVDC-Max', 'OWF-C-Min', 'OWF-C-Max', 'ED-Min', 'ED-Max',
        'EP-Min', 'EP-Max', 'WACC-Min', 'WACC-Max', 'OWF-S-Min', 'OWF-S-Max'
    ]
    case_selection = st.selectbox("Select a case to visualize:", COLUMN_HEADERS)
    
    with st.spinner(f'Generating map for {case_selection}...'):
        fig = create_individual_case_figure(case_selection)
        st.plotly_chart(fig, use_container_width=True)
