import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# Constants
LIGHTSPEED = 299792458.0  
EARTH_RADIUS = 6371000.0  
OMEGA_E = 7.2921151467e-5

# TIME MATCHING FUNCTIONS

def parse_rtk_time(time_str):
    try:
        if len(time_str) > 10:
            return pd.to_datetime(time_str)
        else:
            time_obj = datetime.strptime(time_str, "%H:%M:%S").time()
            return datetime.combine(datetime(2024, 1, 1), time_obj)
    except:
        return pd.NaT

def convert_utc_millis_to_utc_datetime(utc_time_millis):
    if pd.isna(utc_time_millis):
        return None
    utc_time_sec = utc_time_millis / 1000.0
    dt_utc = datetime.utcfromtimestamp(utc_time_sec)
    return dt_utc

def convert_utc_to_thai_time(utc_time_millis):
    if pd.isna(utc_time_millis):
        return None
    utc_time_sec = utc_time_millis / 1000.0
    dt_utc = datetime.utcfromtimestamp(utc_time_sec)
    dt_thai = dt_utc + timedelta(hours=7)
    return dt_thai

def find_nearest_rtk(gnss_time, rtk_times, max_time_diff_sec=1.0):
    if pd.isna(gnss_time):
        return None, np.nan
    time_diffs = np.abs([(t - gnss_time).total_seconds() for t in rtk_times])
    min_idx = np.argmin(time_diffs)
    min_diff = time_diffs[min_idx]
    if min_diff <= max_time_diff_sec:
        return min_idx, min_diff
    else:
        return None, min_diff


# GEOMETRY FUNCTIONS


def calculate_elevation_angle(sat_pos, rcv_pos):
    los_vector = sat_pos - rcv_pos
    up_vector = rcv_pos / np.linalg.norm(rcv_pos)
    cos_angle = np.dot(los_vector, up_vector) / (np.linalg.norm(los_vector) * np.linalg.norm(up_vector))
    elevation = np.pi/2 - np.arccos(np.clip(cos_angle, -1, 1))
    return elevation

def ecef_to_lla(x, y, z):
    if np.isnan(x) or np.isnan(y) or np.isnan(z):
        return np.nan, np.nan, np.nan
    a = 6378137.0
    e2 = 6.6943799901377997e-3
    lon = np.arctan2(y, x)
    p = np.sqrt(x**2 + y**2)
    lat = np.arctan2(z, p * (1 - e2))
    for _ in range(10):
        N = a / np.sqrt(1 - e2 * np.sin(lat)**2)
        alt = p / np.cos(lat) - N
        lat = np.arctan2(z, p * (1 - e2 * N / (N + alt)))
    return lat, lon, alt

def lla_to_ecef(lat, lon, alt):
    a = 6378137.0
    e2 = 6.6943799901377997e-3
    lat_rad = np.radians(lat)
    lon_rad = np.radians(lon)
    N = a / np.sqrt(1 - e2 * np.sin(lat_rad)**2)
    x = (N + alt) * np.cos(lat_rad) * np.cos(lon_rad)
    y = (N + alt) * np.cos(lat_rad) * np.sin(lon_rad)
    z = (N * (1 - e2) + alt) * np.sin(lat_rad)
    return np.array([x, y, z])

def calculate_2d_error(lat1, lon1, lat2, lon2):
    R_earth = 6371000
    error_lat = (lat1 - lat2) * (np.pi/180) * R_earth
    error_lon = (lon1 - lon2) * (np.pi/180) * R_earth * np.cos(np.radians((lat1 + lat2)/2))
    error_2d = np.sqrt(error_lat**2 + error_lon**2)
    return error_2d


# TROPOSPHERIC CORRECTION 


def tropospheric_correction_hopfield(elevation, height_m=0):
    if elevation < np.radians(5):
        elevation = np.radians(5)
    T0 = 306.15
    T = T0 - 0.0065 * height_m
    T = max(T, 200.0)
    h_d = 40136.0 + 148.72 * (T - 243.04)
    h_d0 = 44600.0
    scale = h_d / h_d0
    ZD_nominal = 2.3
    sin_elev = np.sin(elevation)
    M = 1.0 / sin_elev
    correction = ZD_nominal * scale * M
    if not np.isfinite(correction) or correction < 0:
        correction = 2.3 / sin_elev
    correction = min(correction, 50.0)
    return correction


# DATA PREPARATION


def prepare_kinematic_data(df):
    print("="*80)
    print("KINEMATIC DATA PREPARATION")
    print("="*80)
    
    df_clean = df.copy()
    print(f"\nInitial: {len(df_clean)} observations")
    
    df_clean = df_clean[df_clean['PseudorangeCorrected_m'].notna()]
    df_clean = df_clean[(df_clean['SvPositionEcefXMeters'].notna()) & 
                        (df_clean['SvPositionEcefYMeters'].notna()) & 
                        (df_clean['SvPositionEcefZMeters'].notna())]
    
    ranges = np.sqrt(df_clean['SvPositionEcefXMeters']**2 + df_clean['SvPositionEcefYMeters']**2 + df_clean['SvPositionEcefZMeters']**2)
    df_clean = df_clean[(ranges > 2e7) & (ranges < 3e7)]
    
    df_clean = df_clean[(df_clean['PseudorangeCorrected_m'] > 1e7) & (df_clean['PseudorangeCorrected_m'] < 3e7)]
    
    df_clean = df_clean[(df_clean['Cn0DbHz'].notna()) & (df_clean['Cn0DbHz'] >= 26)]
    print(f"After C/N0 >= 26: {len(df_clean)}")
    
    if 'State' in df_clean.columns:
        code_lock = (df_clean['State'].astype(int) & 1) == 1
        tow_decoded = (df_clean['State'].astype(int) & 8) == 8
        df_clean = df_clean[code_lock & tow_decoded]
    
    if 'MultipathIndicator' in df_clean.columns:
        mp_dist = df_clean['MultipathIndicator'].value_counts().sort_index()
        print(f"\nMultipath: {dict(mp_dist)}")
    
    df_clean['Epoch'] = pd.factorize(df_clean['utcTimeMillis'])[0]
    
    epoch_counts = df_clean.groupby('Epoch').size()
    valid_epochs = epoch_counts[epoch_counts >= 6].index
    df_clean = df_clean[df_clean['Epoch'].isin(valid_epochs)]
    
    print(f"\nFinal: {len(df_clean)} obs, {df_clean['Epoch'].nunique()} epochs")
    
    return df_clean


# SINGLE-DIFFERENCE POSITIONING


def get_initial_guess(xs):
    sat_center = np.mean(xs, axis=0)
    sat_distance = np.linalg.norm(sat_center)
    x0 = sat_center * (EARTH_RADIUS / sat_distance)
    return x0

def single_diff_wls_kinematic(xs, measured_pr, weights, x0, max_iterations=30):
    n_sats = len(xs)
    if n_sats < 6:
        raise ValueError("Need >=6 satellites")
    
    elevations = np.array([calculate_elevation_angle(xs[i], x0) for i in range(n_sats)])
    ref_idx = np.argmax(elevations)
    
    mask = np.arange(n_sats) != ref_idx
    xs_diff = xs[mask]
    
    pr_ref = measured_pr[ref_idx]
    pr_diff = measured_pr[mask] - pr_ref
    
    weights_diff = weights[mask] * weights[ref_idx]
    weights_diff = weights_diff / np.mean(weights_diff)
    
    dx = 100 * np.ones(3)
    iterations = 0
    W = np.diag(weights_diff)
    
    while np.linalg.norm(dx) > 1e-5 and iterations < max_iterations:
        r_ref = np.linalg.norm(xs[ref_idx] - x0)
        r_diff = np.linalg.norm(xs_diff - x0, axis=1)
        
        elev_ref = calculate_elevation_angle(xs[ref_idx], x0)
        tropo_ref = tropospheric_correction_hopfield(elev_ref)
        
        tropo_diff = np.zeros(len(xs_diff))
        for i in range(len(xs_diff)):
            elev_i = calculate_elevation_angle(xs_diff[i], x0)
            tropo_diff[i] = tropospheric_correction_hopfield(elev_i)
        
        pr_ref_corr = pr_ref + tropo_ref
        pr_diff_corr = pr_diff + (tropo_diff - tropo_ref)
        
        rho_diff = r_diff - r_ref
        deltaP = pr_diff_corr - rho_diff
        
        los_ref = (xs[ref_idx] - x0) / r_ref
        G = np.zeros((len(xs_diff), 3))
        for i in range(len(xs_diff)):
            los_i = (xs_diff[i] - x0) / r_diff[i]
            G[i, :] = los_ref - los_i
        
        GTW = G.T @ W
        GTWG = GTW @ G + 1e-8 * np.eye(3)
        GTWdeltaP = GTW @ deltaP
        
        try:
            dx = np.linalg.solve(GTWG, GTWdeltaP)
        except:
            dx = np.linalg.lstsq(GTWG, GTWdeltaP, rcond=None)[0]
        
        if iterations > 20:
            dx *= 0.5
        
        x0 = x0 + dx
        iterations += 1
    
    r_ref = np.linalg.norm(xs[ref_idx] - x0)
    r_diff = np.linalg.norm(xs_diff - x0, axis=1)
    rho_diff = r_diff - r_ref
    
    elev_ref = calculate_elevation_angle(xs[ref_idx], x0)
    tropo_ref = tropospheric_correction_hopfield(elev_ref)
    
    tropo_diff = np.zeros(len(xs_diff))
    for i in range(len(xs_diff)):
        elev_i = calculate_elevation_angle(xs_diff[i], x0)
        tropo_diff[i] = tropospheric_correction_hopfield(elev_i)
    
    pr_ref_corr = pr_ref + tropo_ref
    pr_diff_corr = pr_diff + (tropo_diff - tropo_ref)
    
    residuals = pr_diff_corr - rho_diff
    rms_residual = np.sqrt(np.mean(residuals**2))
    
    try:
        Q = np.linalg.inv(G.T @ W @ G)
        hdop = np.sqrt(Q[0,0] + Q[1,1])
        vdop = np.sqrt(Q[2,2])
        pdop = np.sqrt(np.trace(Q))
    except:
        hdop = vdop = pdop = np.nan
    
    return x0, rms_residual, iterations, pdop, hdop, vdop, residuals, ref_idx


# STAGE 1: POSITIONING


def process_kinematic_positions(df_gnss, verbose=False):
    print("\n" + "="*80)
    print("STAGE 1: KINEMATIC POSITIONING")
    print("="*80)
    
    if 'utcTimeMillis' in df_gnss.columns:
        df_gnss['UTC_Time'] = df_gnss['utcTimeMillis'].apply(convert_utc_millis_to_utc_datetime)
        df_gnss['Thai_Time'] = df_gnss['utcTimeMillis'].apply(convert_utc_to_thai_time)
        print(f"\nTime conversion:")
        print(f"  UTC  : {df_gnss['UTC_Time'].iloc[0]}")
        print(f"  Thai : {df_gnss['Thai_Time'].iloc[0]}")
    
    epochs = sorted(df_gnss['Epoch'].unique())
    results = []
    
    print(f"\nProcessing {len(epochs)} epochs...")
    
    for idx, epoch in enumerate(epochs):
        epoch_data = df_gnss[df_gnss['Epoch'] == epoch].copy()
        
        if verbose and idx < 3:
            print(f"\nEpoch {epoch} ({idx+1}/{len(epochs)}): {len(epoch_data)} sats")
        
        utc_time  = epoch_data['UTC_Time'].iloc[0]  if 'UTC_Time'  in epoch_data.columns else None
        thai_time = epoch_data['Thai_Time'].iloc[0] if 'Thai_Time' in epoch_data.columns else None
        
        xs = epoch_data[['SvPositionEcefXMeters', 'SvPositionEcefYMeters', 'SvPositionEcefZMeters']].values.astype(float)
        measured_pr = epoch_data['PseudorangeCorrected_m'].values.astype(float)
        
        valid_mask = ~(np.isnan(xs).any(axis=1) | np.isnan(measured_pr))
        xs_clean = xs[valid_mask]
        measured_pr_clean = measured_pr[valid_mask]
        epoch_data_clean = epoch_data[valid_mask].copy()
        
        if len(xs_clean) < 6:
            continue
        
        x0 = get_initial_guess(xs_clean)
        elevation_angles = np.array([calculate_elevation_angle(xs_clean[i], x0)
                                     for i in range(len(xs_clean))])
        
        w_elev = np.sin(elevation_angles)**4
        
        cn0 = epoch_data_clean['Cn0DbHz'].values
        cn0_norm = np.clip((cn0 - 26) / 15, 0, 1)
        w_cn0 = np.exp(2.5 * cn0_norm)
        
        if 'MultipathIndicator' in epoch_data_clean.columns:
            mp = epoch_data_clean['MultipathIndicator'].values
            w_mp = np.ones(len(mp))
            w_mp[mp == 0] = 10.0
            w_mp[mp == 1] = 3.0
            w_mp[mp == 2] = 0.1
        else:
            w_mp = np.ones(len(xs_clean))
        
        weights = w_elev * w_cn0 * w_mp
        weights = weights / np.mean(weights)
        weights = np.maximum(weights, 0.01)
        
        try:
            pos_ecef, rms_res, iters, pdop, hdop, vdop, residuals, ref_idx = \
                single_diff_wls_kinematic(xs_clean, measured_pr_clean, weights, x0)
            
            median_res = np.median(residuals)
            mad = np.median(np.abs(residuals - median_res))
            
            if mad > 1e-6:
                z_scores = 0.6745 * (residuals - median_res) / mad
                outlier_mask = np.abs(z_scores) < 2.5
                
                full_mask = np.ones(len(xs_clean), dtype=bool)
                full_mask[ref_idx] = True
                non_ref_indices = np.where(np.arange(len(xs_clean)) != ref_idx)[0]
                full_mask[non_ref_indices] = outlier_mask
                
                n_outliers = np.sum(~full_mask)
                
                if n_outliers > 0 and np.sum(full_mask) >= 6:
                    xs_final = xs_clean[full_mask]
                    pr_final = measured_pr_clean[full_mask]
                    weights_final = weights[full_mask] / np.mean(weights[full_mask])
                    
                    x0 = get_initial_guess(xs_final)
                    pos_ecef, rms_res, iters, pdop, hdop, vdop, _, _ = \
                        single_diff_wls_kinematic(xs_final, pr_final, weights_final, x0)
                    
                    xs_clean = xs_final
                    elevation_angles = np.array([calculate_elevation_angle(xs_clean[i], pos_ecef)
                                               for i in range(len(xs_clean))])
                    cn0 = cn0[full_mask]
                    
                    if verbose and idx < 3:
                        print(f"  Removed {n_outliers} outliers")
            
            if pdop > 4.0 or rms_res > 25:
                if verbose and idx < 3:
                    print(f"  Rejected: PDOP={pdop:.2f}, RMS={rms_res:.2f}m")
                continue
            
            lat_rad, lon_rad, alt = ecef_to_lla(pos_ecef[0], pos_ecef[1], pos_ecef[2])
            lat = np.degrees(lat_rad)
            lon = np.degrees(lon_rad)
            
            if np.isnan(lat) or np.isnan(lon):
                continue
            
            elev_ref = calculate_elevation_angle(xs_clean[0], pos_ecef)
            tropo = tropospheric_correction_hopfield(elev_ref)
            
            result = {
                'Epoch': epoch,
                'UTC_Time': utc_time,
                'Thai_Time': thai_time,
                'X_ECEF': pos_ecef[0],
                'Y_ECEF': pos_ecef[1],
                'Z_ECEF': pos_ecef[2],
                'Latitude': lat,
                'Longitude': lon,
                'Altitude': alt,
                'RMS_Residual': rms_res,
                'PDOP': pdop,
                'HDOP': hdop,
                'VDOP': vdop,
                'NumSatellites': len(xs_clean),
                'MeanCNo': np.mean(cn0),
                'MeanElevation': np.degrees(np.mean(elevation_angles)),
                'Iterations': iters,
                'Tropo_m': tropo
            }
            
            results.append(result)
            
            if verbose and idx < 3:
                print(f"  {lat:.6f}, {lon:.6f}, {alt:.1f}m")
                print(f"    UTC: {utc_time}  Thai: {thai_time}")
                print(f"    HDOP={hdop:.2f}, Sats={len(xs_clean)}")
        
        except Exception as e:
            if verbose and idx < 3:
                print(f"  Error: {e}")
            continue
    
    results_df = pd.DataFrame(results)
    print(f"\nComputed positions for {len(results)} epochs")
    
    return results_df


# STAGE 2: MATCH WITH RTK


def match_with_rtk_and_compute_errors(gnss_results, df_rtk, verbose=False):
    print("\n" + "="*80)
    print("STAGE 2: MATCHING WITH RTK GROUND TRUTH")
    print("="*80)
    
    print(f"\nRTK Ground Truth:")
    print(f"  Total points: {len(df_rtk)}")
    
    if 'UTC_Time' in df_rtk.columns:
        df_rtk['RTK_Time'] = df_rtk['UTC_Time'].apply(parse_rtk_time)
        print(f"  Time range: {df_rtk['RTK_Time'].min()} to {df_rtk['RTK_Time'].max()}")
    
    gnss_utc_times = gnss_results['UTC_Time'].dropna()
    if len(gnss_utc_times) > 0:
        print(f"\nGNSS Data (UTC):")
        print(f"  Time range: {gnss_utc_times.min()} to {gnss_utc_times.max()}")
        print(f"  Total epochs: {len(gnss_results)}")
        
        rtk_min = df_rtk['RTK_Time'].min()
        rtk_max = df_rtk['RTK_Time'].max()
        gnss_min = gnss_utc_times.min()
        gnss_max = gnss_utc_times.max()
        
        overlap = (gnss_min <= rtk_max) and (gnss_max >= rtk_min)
        print(f"\nTime overlap: {'YES' if overlap else 'NO - Times do not overlap!'}")
        
        if not overlap:
            print(f"\nWARNING: Time ranges do not overlap!")
            print(f"  RTK:  {rtk_min} to {rtk_max}")
            print(f"  GNSS: {gnss_min} to {gnss_max}")
    
    matched_count = 0
    errors_2d = []
    errors_lat = []
    errors_lon = []
    errors_alt = []
    errors_3d = []
    rtk_lats = []
    rtk_lons = []
    rtk_alts = []
    time_diffs = []
    
    print(f"\nMatching {len(gnss_results)} GNSS positions with RTK...")
    
    for idx, row in gnss_results.iterrows():
        gnss_time = row['UTC_Time']
        gnss_lat  = row['Latitude']
        gnss_lon  = row['Longitude']
        gnss_alt  = row['Altitude']
        
        rtk_idx  = None
        time_diff = np.nan
        
        if pd.notna(gnss_time) and 'RTK_Time' in df_rtk.columns:
            rtk_idx, time_diff = find_nearest_rtk(gnss_time, df_rtk['RTK_Time'].values, max_time_diff_sec=2.0)
        
        if rtk_idx is not None:
            rtk_lat = df_rtk.iloc[rtk_idx]['Latitude']
            rtk_lon = df_rtk.iloc[rtk_idx]['Longitude']
            rtk_alt = df_rtk.iloc[rtk_idx]['Altitude_m']
            
            error_2d  = calculate_2d_error(gnss_lat, gnss_lon, rtk_lat, rtk_lon)
            R_earth   = 6371000
            error_lat = (gnss_lat - rtk_lat) * (np.pi/180) * R_earth
            error_lon = (gnss_lon - rtk_lon) * (np.pi/180) * R_earth * np.cos(np.radians(rtk_lat))
            error_alt = gnss_alt - rtk_alt
            
            gnss_ecef = lla_to_ecef(gnss_lat, gnss_lon, gnss_alt)
            rtk_ecef  = lla_to_ecef(rtk_lat,  rtk_lon,  rtk_alt)
            error_3d  = np.linalg.norm(gnss_ecef - rtk_ecef)
            
            errors_2d.append(error_2d)
            errors_lat.append(error_lat)
            errors_lon.append(error_lon)
            errors_alt.append(error_alt)
            errors_3d.append(error_3d)
            rtk_lats.append(rtk_lat)
            rtk_lons.append(rtk_lon)
            rtk_alts.append(rtk_alt)
            time_diffs.append(time_diff)
            matched_count += 1
            
            if verbose and idx < 3:
                print(f"\nEpoch {row['Epoch']}:")
                print(f"  GNSS: {gnss_lat:.6f}, {gnss_lon:.6f}")
                print(f"  RTK:  {rtk_lat:.6f}, {rtk_lon:.6f}")
                print(f"  Error 2D: {error_2d:.2f}m  Time diff: {time_diff:.2f}s")
        else:
            errors_2d.append(np.nan)
            errors_lat.append(np.nan)
            errors_lon.append(np.nan)
            errors_alt.append(np.nan)
            errors_3d.append(np.nan)
            rtk_lats.append(np.nan)
            rtk_lons.append(np.nan)
            rtk_alts.append(np.nan)
            time_diffs.append(np.nan)
    
    gnss_results['RTK_Latitude']  = rtk_lats
    gnss_results['RTK_Longitude'] = rtk_lons
    gnss_results['RTK_Altitude']  = rtk_alts
    gnss_results['Time_Diff_sec'] = time_diffs
    gnss_results['Error_3D']  = errors_3d
    gnss_results['Error_Lat'] = errors_lat
    gnss_results['Error_Lon'] = errors_lon
    gnss_results['Error_2D']  = errors_2d
    gnss_results['Error_Alt'] = errors_alt
    
    print(f"\nMatched {matched_count}/{len(gnss_results)} epochs ({100*matched_count/len(gnss_results):.1f}%)")
    
    return gnss_results


# MAP PLOTTING  


def plot_gnss_map_kinematic(results_df, output_file='gnss_kinematic_map.html'):
    import folium
    import webbrowser
    import os

    
    matched   = results_df[results_df['Error_2D'].notna()].copy()
    unmatched = results_df[results_df['Error_2D'].isna()].copy()

    print(f"\nMap: {len(matched)} matched | {len(unmatched)} unmatched epochs")

    if len(matched) == 0:
        print("No matched epochs to plot.")
        return

    mean_err = matched['Error_2D'].mean()
    med_err  = matched['Error_2D'].median()
    max_err  = matched['Error_2D'].max()
    min_err  = matched['Error_2D'].min()

    #  RTK 
    center_lat = matched['RTK_Latitude'].mean()
    center_lon = matched['RTK_Longitude'].mean()

    m = folium.Map(location=[center_lat, center_lon], zoom_start=17, tiles='OpenStreetMap')
    folium.TileLayer('CartoDB positron', name='Light').add_to(m)
    folium.TileLayer(
        tiles='https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}',
        attr='Esri', name='Satellite', overlay=False, control=True
    ).add_to(m)

    #  วงกลม 
    step = 5
    max_circle = max(int(np.ceil(max_err / step)) * step, 15)
    circle_radii = list(range(step, max_circle + step, step))
    circle_colors = ['#e74c3c','#e67e22','#f1c40f','#2ecc71','#3498db',
                     '#9b59b6','#1abc9c','#e74c3c','#e67e22','#f1c40f']

    for i, r in enumerate(circle_radii):
        color = circle_colors[i % len(circle_colors)]
        folium.Circle(
            location=[center_lat, center_lon],
            radius=r,
            color=color,
            fill=False,
            weight=1.5,
            opacity=0.55,
            tooltip=f'{r} m'
        ).add_to(m)
        folium.Marker(
            location=[center_lat + r / 111320, center_lon],
            icon=folium.DivIcon(
                html=f'<div style="font-size:9px;color:{color};font-weight:bold;white-space:nowrap;">{r} m</div>',
                icon_size=(40, 12), icon_anchor=(0, 6)
            )
        ).add_to(m)

    # RTK path 
    rtk_coords = matched[['RTK_Latitude', 'RTK_Longitude']].values.tolist()
    folium.PolyLine(rtk_coords, color='pink', weight=2, opacity=0.6,
                    tooltip='RTK Ground Truth Path').add_to(m)

    # GNSS path 
    gnss_coords = matched[['Latitude', 'Longitude']].values.tolist()
    folium.PolyLine(gnss_coords, color='blue', weight=1.5, opacity=0.4,
                    tooltip='GNSS WLS Path').add_to(m)

    # Plot แต่ละ epoch: GNSS + RTK คู่กัน
    for i, row in matched.iterrows():
        dist = row['Error_2D']

        # สีตามระยะ 
        if dist < 10:
            color = 'green';  fillcolor = 'lightgreen'
        elif dist < 20:
            color = 'blue';   fillcolor = 'lightblue'
        elif dist < 50:
            color = 'orange'; fillcolor = 'yellow'
        else:
            color = 'red';    fillcolor = 'pink'

        # จุด 
        folium.CircleMarker(
            location=[row['Latitude'], row['Longitude']],
            radius=5,
            popup=(f'<b>Epoch #{row["Epoch"]} — GNSS</b><br>'
                   f'Lat: {row["Latitude"]:.6f}<br>'
                   f'Lon: {row["Longitude"]:.6f}<br>'
                   f'Alt: {row["Altitude"]:.1f} m<br>'
                   f'Error 2D: {dist:.1f} m<br>'
                   f'HDOP: {row["HDOP"]:.2f}  PDOP: {row["PDOP"]:.2f}<br>'
                   f'Sats: {row["NumSatellites"]:.0f}  C/N0: {row["MeanCNo"]:.1f} dB-Hz'),
            tooltip=f'GNSS #{row["Epoch"]} | err={dist:.1f} m',
            color=color, fill=True, fillColor=fillcolor, fillOpacity=0.75, weight=2
        ).add_to(m)

        # จุด RTK 
        folium.CircleMarker(
            location=[row['RTK_Latitude'], row['RTK_Longitude']],
            radius=3,
            popup=(f'<b>Epoch #{row["Epoch"]} — RTK</b><br>'
                   f'Lat: {row["RTK_Latitude"]:.6f}<br>'
                   f'Lon: {row["RTK_Longitude"]:.6f}<br>'
                   f'Time diff: {row["Time_Diff_sec"]:.2f} s'),
            tooltip=f'RTK #{row["Epoch"]}',
            color='pink', fill=True, fillColor='pink', fillOpacity=0.9, weight=1
        ).add_to(m)

        # เส้นเชื่อม 
        folium.PolyLine(
            [[row['Latitude'], row['Longitude']],
             [row['RTK_Latitude'], row['RTK_Longitude']]],
            color='gray', weight=1, opacity=0.5,
            tooltip=f'Error: {dist:.1f} m'
        ).add_to(m)

    # Unmatched epochs 
    for i, row in unmatched.iterrows():
        folium.CircleMarker(
            location=[row['Latitude'], row['Longitude']],
            radius=4,
            popup=f'<b>Epoch #{row["Epoch"]} — No RTK match</b><br>Lat: {row["Latitude"]:.6f}<br>Lon: {row["Longitude"]:.6f}',
            tooltip=f'No RTK #{row["Epoch"]}',
            color='gray', fill=True, fillColor='lightgray', fillOpacity=0.5, weight=1
        ).add_to(m)

    #  Legend
    legend_html = f'''
    <div style="position:fixed; top:10px; right:10px; width:240px;
                background:white; border:2px solid grey; z-index:9999;
                padding:12px; border-radius:6px; box-shadow:2px 2px 6px rgba(0,0,0,0.3);">
      <h4 style="margin:0 0 8px 0; border-bottom:2px solid #333;">GNSS WLS vs RTK</h4>
      <p style="margin:4px 0;"><span style="color:green;  font-size:14px;">&#9679;</span> GNSS Error &lt; 10 m</p>
      <p style="margin:4px 0;"><span style="color:blue;   font-size:14px;">&#9679;</span> GNSS Error 10&#8211;20 m</p>
      <p style="margin:4px 0;"><span style="color:orange; font-size:14px;">&#9679;</span> GNSS Error 20&#8211;50 m</p>
      <p style="margin:4px 0;"><span style="color:red;    font-size:14px;">&#9679;</span> GNSS Error &gt; 50 m</p>
      <p style="margin:4px 0;"><span style="color:pink; font-size:12px;">&#9679;</span> RTK Ground Truth</p>
      <p style="margin:4px 0;"><span style="color:gray;   font-size:14px;">&#9679;</span> No RTK match</p>
      <p style="margin:4px 0;"><span style="color:green;  font-size:14px;">&#9135;</span> RTK path &nbsp;
                                <span style="color:blue;">&#9135;</span> GNSS path</p>
      <p style="margin:4px 0;"><span style="color:gray;">&#9135;</span> Error line (GNSS&#8596;RTK)</p>
      <hr style="margin:8px 0;">
      <p style="margin:3px 0; font-size:11px;"><b>Statistics (matched only):</b></p>
      <p style="margin:2px 0 2px 8px; font-size:10px;">Matched : {len(matched)} / {len(results_df)} epochs</p>
      <p style="margin:2px 0 2px 8px; font-size:10px;">Mean    : {mean_err:.1f} m</p>
      <p style="margin:2px 0 2px 8px; font-size:10px;">Median  : {med_err:.1f} m</p>
      <p style="margin:2px 0 2px 8px; font-size:10px;">Max     : {max_err:.1f} m</p>
      <p style="margin:2px 0 2px 8px; font-size:10px;">Min     : {min_err:.1f} m</p>
    </div>
    '''
    m.get_root().html.add_child(folium.Element(legend_html))
    folium.LayerControl().add_to(m)

    m.save(output_file)
    print(f"\nMap saved: {output_file}")
    print(f"File size: {os.path.getsize(output_file)/1024:.1f} KB")
    webbrowser.open('file://' + os.path.abspath(output_file))


# MAIN


if __name__ == "__main__":
    gnss_file   = r"C:\project\gnss_pseudorange_sppแบบมือถือเคลื่อนที่.xlsx"
    rtk_file    = r"C:\Project\epoch_RTKแบบเคลื่อนที่.xlsx"
    output_file = r"C:\project\gnss_outputwith_RTK.xlsx"
    output_map  = r"C:\project\gnss_kinematic_map.html"

    print("="*80)
    print("KINEMATIC GNSS POSITIONING WITH RTK GROUND TRUTH")
    print("="*80)
    
    try:
        print(f"\nReading GNSS: {gnss_file}")
        df_gnss_raw = pd.read_excel(gnss_file)
        print(f"  Loaded: {len(df_gnss_raw)} observations")
        
        print(f"\nReading RTK: {rtk_file}")
        df_rtk = pd.read_excel(rtk_file)
        print(f"  Loaded: {len(df_rtk)} RTK points")
        print(f"  Columns: {list(df_rtk.columns)}")
        
        df_gnss_clean = prepare_kinematic_data(df_gnss_raw)
        
        if len(df_gnss_clean) == 0:
            print("\nNo GNSS data after filtering")
        else:
            # STAGE 1
            results = process_kinematic_positions(df_gnss_clean, verbose=True)
            
            if len(results) == 0:
                print("\nNo positions computed")
            else:
                print(f"\nStage 1 complete: {len(results)} positions computed")
                
                # STAGE 2
                results = match_with_rtk_and_compute_errors(results, df_rtk, verbose=True)
            
            results.to_excel(output_file, index=False)
            
            print("\n" + "="*80)
            print("FINAL RESULTS")
            print("="*80)
            
            print(f"\nTotal epochs: {len(results)}")
            
            matched = results[results['Error_2D'].notna()]
            
            if len(matched) > 0:
                print(f"\nMatched with RTK: {len(matched)}/{len(results)} ({100*len(matched)/len(results):.1f}%)")
                
                print(f"\n2D HORIZONTAL ERROR (vs RTK):")
                print(f"  Mean   : {matched['Error_2D'].mean():.2f} m")
                print(f"  Median : {matched['Error_2D'].median():.2f} m")
                print(f"  Std    : {matched['Error_2D'].std():.2f} m")
                print(f"  Min    : {matched['Error_2D'].min():.2f} m")
                print(f"  Max    : {matched['Error_2D'].max():.2f} m")
                print(f"  68%    : {np.percentile(matched['Error_2D'], 68):.2f} m")
                print(f"  95%    : {np.percentile(matched['Error_2D'], 95):.2f} m")
                
                print(f"\n3D ERROR:")
                print(f"  Mean : {matched['Error_3D'].mean():.2f} m")
                print(f"  Std  : {matched['Error_3D'].std():.2f} m")
                
                print(f"\nALTITUDE ERROR:")
                print(f"  Mean : {matched['Error_Alt'].mean():.2f} m")
                print(f"  Std  : {matched['Error_Alt'].std():.2f} m")
                
                print(f"\nTIME SYNC:")
                print(f"  Mean diff : {matched['Time_Diff_sec'].mean():.3f} s")
                print(f"  Max diff  : {matched['Time_Diff_sec'].max():.3f} s")
                
                print(f"\nQUALITY:")
                print(f"  HDOP : {matched['HDOP'].mean():.2f}")
                print(f"  PDOP : {matched['PDOP'].mean():.2f}")
                print(f"  RMS  : {matched['RMS_Residual'].mean():.2f} m")
                print(f"  Sats : {matched['NumSatellites'].mean():.1f}")
                print(f"  C/N0 : {matched['MeanCNo'].mean():.1f} dB-Hz")
                
                mean_2d = matched['Error_2D'].mean()
                print(f"\n{'='*80}")
                if mean_2d < 15:
                    print(f"OUTSTANDING! {mean_2d:.2f}m")
                elif mean_2d < 20:
                    print(f"EXCELLENT! {mean_2d:.2f}m")
                elif mean_2d < 25:
                    print(f"VERY GOOD! {mean_2d:.2f}m")
                else:
                    print(f"{mean_2d:.2f}m - Standard smartphone")
            else:
                print("\nNo epochs matched with RTK ground truth")
                print(f"\nGNSS Positions (no RTK comparison):")
                print(f"  Mean: {results['Latitude'].mean():.6f}N, {results['Longitude'].mean():.6f}E")
                print(f"  Alt : {results['Altitude'].mean():.1f} +/- {results['Altitude'].std():.1f} m")
                print(f"  HDOP: {results['HDOP'].mean():.2f}")
            
            print(f"\nSaved: {output_file}")

            # พล็อตแผนที่ 
            plot_gnss_map_kinematic(results, output_file=output_map)
            print("="*80)
        
    except Exception as e:
        print(f"\nERROR: {e}")
        import traceback
        traceback.print_exc()