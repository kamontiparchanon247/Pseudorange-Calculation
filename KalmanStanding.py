
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from collections import deque


# GROUND TRUTH - HARDCODED

GROUND_TRUTH_LAT = 13.72728500
GROUND_TRUTH_LON = 100.77642400
GROUND_TRUTH_ALT = -15.90



# COORDINATE CONVERSION

def ecef_to_enu(x_ecef, y_ecef, z_ecef, lat0, lon0, alt0):
    lat0_rad = np.radians(lat0)
    lon0_rad = np.radians(lon0)
    a  = 6378137.0
    e2 = 0.00669437999014
    N  = a / np.sqrt(1 - e2 * np.sin(lat0_rad)**2)
    x0 = (N + alt0) * np.cos(lat0_rad) * np.cos(lon0_rad)
    y0 = (N + alt0) * np.cos(lat0_rad) * np.sin(lon0_rad)
    z0 = (N * (1 - e2) + alt0) * np.sin(lat0_rad)
    dx = x_ecef - x0
    dy = y_ecef - y0
    dz = z_ecef - z0
    sin_lat = np.sin(lat0_rad); cos_lat = np.cos(lat0_rad)
    sin_lon = np.sin(lon0_rad); cos_lon = np.cos(lon0_rad)
    e =  -sin_lon*dx + cos_lon*dy
    n =  -sin_lat*cos_lon*dx - sin_lat*sin_lon*dy + cos_lat*dz
    u =   cos_lat*cos_lon*dx + cos_lat*sin_lon*dy + sin_lat*dz
    return e, n, u

def latlon_to_enu(lat_t, lon_t, alt_t, lat0, lon0, alt0):
    lat_r = np.radians(lat_t); lon_r = np.radians(lon_t)
    a  = 6378137.0; e2 = 0.00669437999014
    N  = a / np.sqrt(1 - e2 * np.sin(lat_r)**2)
    xe = (N + alt_t) * np.cos(lat_r) * np.cos(lon_r)
    ye = (N + alt_t) * np.cos(lat_r) * np.sin(lon_r)
    ze = (N*(1-e2) + alt_t) * np.sin(lat_r)
    return ecef_to_enu(xe, ye, ze, lat0, lon0, alt0)




class ImprovedKalmanFilter:
 

    def __init__(self, dt=1.0,
                 window_x=50,
                 window_y=5,
                 window_z=50,              
                 adaptive_R=True,
                 outlier_threshold=50.0):

        self.dt                = dt
        self.n_states          = 3
        self.adaptive_R        = adaptive_R
        self.outlier_threshold = outlier_threshold
        self.window_x          = window_x
        self.window_y          = window_y
        self.window_z          = window_z

        # Matrices 
        self.F = np.eye(3)
        self.H = np.eye(3)

        # Q: small = stationary target 
        self.Q = np.diag([1.0, 1.0, 1.0])

        self.R_base = np.diag([
            346.0,    # x
            53.0,     # y
            1998.0    # z 
        ])
        self.R = self.R_base.copy()

        # Initial state & covariance 
        self.x = np.zeros((3, 1))
        self.P = np.diag([500.0, 500.0, 500.0])

        #  PER-AXIS moving average windows 
        self.x_win = deque(maxlen=window_x)
        self.y_win = deque(maxlen=window_y)
        self.z_win = deque(maxlen=window_z)   

        # History 
        self.history = {
            'x': [], 'y': [], 'z': [],
            'innovation': [], 'R_scale': [], 'outlier': []
        }
        self.outlier_count = 0
        self.total_count   = 0

    
    def initialize(self, x0, y0, z0):
        self.x = np.array([[x0], [y0], [z0]])
        self.P = np.diag([500.0, 500.0, 500.0])
        for _ in range(self.x_win.maxlen): self.x_win.append(x0)
        for _ in range(self.y_win.maxlen): self.y_win.append(y0)
        for _ in range(self.z_win.maxlen): self.z_win.append(z0)
        self._store_state(x0, y0, z0, innovation=0, R_scale=1.0, outlier=False)

    
    def step(self, z):
        self.total_count += 1
        if z.ndim == 1:
            z = z.reshape(-1, 1)

        #  PREDICT 
        x_pred = self.F @ self.x
        P_pred = self.F @ self.P @ self.F.T + self.Q

        # INNOVATION 
        y_tilde        = z - self.H @ x_pred
        innovation_mag = float(np.linalg.norm(y_tilde))

        # OUTLIER REJECTION 
        is_outlier = innovation_mag > self.outlier_threshold
        R_scale    = 1.0

        if is_outlier:
            self.outlier_count += 1
            self.x = x_pred
            self.P = P_pred
        else:
            # ADAPTIVE R 
            if self.adaptive_R:
                ai = innovation_mag
                if   ai < 5:   R_scale = 0.5
                elif ai < 15:  R_scale = 1.0
                elif ai < 30:  R_scale = 1.8
                else:          R_scale = 3.0
            self.R = self.R_base * R_scale

            # KALMAN UPDATE 
            S = self.H @ P_pred @ self.H.T + self.R
            K = P_pred @ self.H.T @ np.linalg.inv(S)
            self.x = x_pred + K @ y_tilde
            self.P = (np.eye(self.n_states) - K @ self.H) @ P_pred

        x_kf = float(self.x[0])
        y_kf = float(self.x[1])
        z_kf = float(self.x[2])

        # PER-AXIS MOVING AVERAGE 
        self.x_win.append(x_kf)
        self.y_win.append(y_kf)
        self.z_win.append(z_kf)

        x_out = float(np.mean(self.x_win))
        y_out = float(np.mean(self.y_win))
        z_out = float(np.mean(self.z_win))  

        self._store_state(x_out, y_out, z_out,
                          innovation=innovation_mag,
                          R_scale=R_scale if not is_outlier else 99,
                          outlier=is_outlier)
        return x_out, y_out, z_out

   
    def _store_state(self, x, y, z, innovation, R_scale, outlier):
        self.history['x'].append(x)
        self.history['y'].append(y)
        self.history['z'].append(z)
        self.history['innovation'].append(innovation)
        self.history['R_scale'].append(R_scale)
        self.history['outlier'].append(outlier)

    def get_history_dataframe(self):
        return pd.DataFrame(self.history)



# DATA LOADING


def load_wls_data(filepath):
    df = pd.read_excel(filepath)
    print(f"Loaded {len(df)} rows | Columns: {df.columns.tolist()}")

    if 'X_ECEF' in df.columns:
        lat0 = df['Latitude'].iloc[0]
        lon0 = df['Longitude'].iloc[0]
        alt0 = df['Altitude'].iloc[0]
        e, n, u = ecef_to_enu(df['X_ECEF'].values, df['Y_ECEF'].values,
                               df['Z_ECEF'].values, lat0, lon0, alt0)
        df['x'] = e; df['y'] = n; df['z'] = u
        print(f"    ECEF → ENU")
    elif 'Latitude' in df.columns:
        lat0 = df['Latitude'].iloc[0]
        lon0 = df['Longitude'].iloc[0]
        df['x'] = (df['Longitude'] - lon0) * 111000 * np.cos(np.radians(lat0))
        df['y'] = (df['Latitude']  - lat0) * 111000
        df['z'] = df['Altitude'] - df['Altitude'].iloc[0]
        print("    Lat/Lon → local ENU")
    elif 'x' not in df.columns:
        raise ValueError("No position columns found!")

    if 'Epoch' in df.columns and 'timestamp' not in df.columns:
        try:
            if pd.api.types.is_numeric_dtype(df['Epoch']):
                df['timestamp'] = df['Epoch'] - df['Epoch'].iloc[0]
            else:
                df['Epoch'] = pd.to_datetime(df['Epoch'])
                df['timestamp'] = (df['Epoch'] - df['Epoch'].iloc[0]).dt.total_seconds()
        except:
            df['timestamp'] = np.arange(len(df), dtype=float)
    elif 'timestamp' not in df.columns:
        df['timestamp'] = np.arange(len(df), dtype=float)

    return df


def add_ground_truth(df):
    print(f"\n Ground Truth: ({GROUND_TRUTH_LAT}, {GROUND_TRUTH_LON}, alt={GROUND_TRUTH_ALT}m)")
    lat0 = df['Latitude'].iloc[0]
    lon0 = df['Longitude'].iloc[0]
    alt0 = df['Altitude'].iloc[0] if 'Altitude' in df.columns else 0.0
    e_true, n_true, u_true = latlon_to_enu(
        GROUND_TRUTH_LAT, GROUND_TRUTH_LON, GROUND_TRUTH_ALT,
        lat0, lon0, alt0)
    df['x_true'] = e_true
    df['y_true'] = n_true
    df['z_true'] = u_true
    dist = np.sqrt((df['x'].mean()-e_true)**2 + (df['y'].mean()-n_true)**2)
    print(f"   GT  (ENU): E={e_true:.2f}m  N={n_true:.2f}m  U={u_true:.2f}m")
    print(f"   Mean meas: E={df['x'].mean():.2f}m  N={df['y'].mean():.2f}m  U={df['z'].mean():.2f}m")
    print(f"   2D offset from GT: {dist:.2f} m")

    # Bias vs noise diagnosis
    print(f"\n    RAW ERROR DIAGNOSIS:")
    for label, col, tc in [('X','x','x_true'),('Y','y','y_true'),('Z','z','z_true')]:
        err   = df[col] - df[tc]
        bias  = err.mean()
        noise = err.std()
        rms   = np.sqrt((err**2).mean())
        print(f"   {label}: RMS={rms:.2f}m  Bias={bias:.2f}m  Noise(std)={noise:.2f}m")
        if abs(bias) > noise:
            print(f"       BIAS-dominated → smoothing has limited effect on {label}")
        else:
            print(f"       NOISE-dominated → smoothing CAN reduce {label} error")
    return df



# AUTO-FIND BEST WINDOW FOR ALL THREE AXES INDEPENDENTLY


def _rms(series):
    return float(np.sqrt((np.array(series)**2).mean()))


def find_best_windows(df_raw, windows=(2, 5, 10, 20, 30, 50, 80, 100)):
 
    print("\n🔍 SEARCHING BEST WINDOW PER AXIS (X, Y, Z independently)...")
    print(f"\n   {'Win':>6} {'X RMS':>10} {'Y RMS':>10} {'Z RMS':>10}")
    print(f"   {'-'*40}")

    best_x_rms = 9999; best_wx = 5
    best_y_rms = 9999; best_wy = 5
    best_z_rms = 9999; best_wz = 5

    for w in windows:
        kf = ImprovedKalmanFilter(window_x=w, window_y=w, window_z=w,
                                   adaptive_R=True, outlier_threshold=50.0)
        kf.initialize(df_raw['x'].iloc[0], df_raw['y'].iloc[0], df_raw['z'].iloc[0])
        for i in range(1, len(df_raw)):
            kf.step(np.array([df_raw['x'].iloc[i],
                               df_raw['y'].iloc[i],
                               df_raw['z'].iloc[i]]))

        df_t = kf.get_history_dataframe()
        n    = len(df_t)
        rx   = _rms(df_t['x'] - df_raw['x_true'].values[:n])
        ry   = _rms(df_t['y'] - df_raw['y_true'].values[:n])
        rz   = _rms(df_t['z'] - df_raw['z_true'].values[:n])

        tx = " ←X" if rx < best_x_rms else ""
        ty = " ←Y" if ry < best_y_rms else ""
        tz = " ←Z" if rz < best_z_rms else ""
        print(f"   {w:>6} {rx:>9.3f}m{tx}  {ry:>9.3f}m{ty}  {rz:>9.3f}m{tz}")

        if rx < best_x_rms: best_x_rms = rx; best_wx = w
        if ry < best_y_rms: best_y_rms = ry; best_wy = w
        if rz < best_z_rms: best_z_rms = rz; best_wz = w

    print(f"\n    Best window_x = {best_wx}  (X RMS = {best_x_rms:.3f}m)")
    print(f"    Best window_y = {best_wy}  (Y RMS = {best_y_rms:.3f}m)")
    print(f"    Best window_z = {best_wz}  (Z RMS = {best_z_rms:.3f}m)\n")

    return best_wx, best_wy, best_wz



# RUN FILTER

def run_kalman_filter(input_file, output_file,
                      window_x=50, window_y=5, window_z=50,
                      adaptive_R=True, outlier_threshold=50.0):

    print("\n" + "="*70)
    print("  IMPROVED KALMAN FILTER — Per-Axis Window (X + Y + Z)")
    print("="*70)

    df = load_wls_data(input_file)
    df = add_ground_truth(df)

    dt = df['timestamp'].diff().median() if len(df) > 1 else 1.0
    print(f"\n   dt={dt:.3f}s | wx={window_x} | wy={window_y} | wz={window_z}")

    kf = ImprovedKalmanFilter(
        dt=dt,
        window_x=window_x, window_y=window_y, window_z=window_z,
        adaptive_R=adaptive_R, outlier_threshold=outlier_threshold
    )
    kf.initialize(df['x'].iloc[0], df['y'].iloc[0], df['z'].iloc[0])

    print(f"\n   Running on {len(df)} measurements...")
    for i in range(1, len(df)):
        kf.step(np.array([df['x'].iloc[i], df['y'].iloc[i], df['z'].iloc[i]]))
        if (i+1) % 200 == 0:
            print(f"   ... {i+1}/{len(df)}")

    pct = kf.outlier_count / max(kf.total_count, 1) * 100
    print(f"   ✓ Done | Outliers: {kf.outlier_count} ({pct:.1f}%)")

    # Build output
    df_out = kf.get_history_dataframe()
    df_out['x_measured'] = df['x'].values[:len(df_out)]
    df_out['y_measured'] = df['y'].values[:len(df_out)]
    df_out['z_measured'] = df['z'].values[:len(df_out)]
    df_out['timestamp']  = df['timestamp'].values[:len(df_out)]
    df_out['x_true']     = df['x_true'].values[:len(df_out)]
    df_out['y_true']     = df['y_true'].values[:len(df_out)]
    df_out['z_true']     = df['z_true'].values[:len(df_out)]

    for col in ['Latitude','Longitude','Altitude','NumSatellites','GDOP','MeanCNo']:
        if col in df.columns:
            df_out[col] = df[col].values[:len(df_out)]

    for prefix, xc, yc, zc in [('raw','x_measured','y_measured','z_measured'),
                                  ('kf', 'x',         'y',         'z')]:
        df_out[f'err_{prefix}_x']  = df_out[xc] - df_out['x_true']
        df_out[f'err_{prefix}_y']  = df_out[yc] - df_out['y_true']
        df_out[f'err_{prefix}_z']  = df_out[zc] - df_out['z_true']
        df_out[f'err_{prefix}_2d'] = np.sqrt(df_out[f'err_{prefix}_x']**2 +
                                              df_out[f'err_{prefix}_y']**2)
        df_out[f'err_{prefix}_3d'] = np.sqrt(df_out[f'err_{prefix}_x']**2 +
                                              df_out[f'err_{prefix}_y']**2 +
                                              df_out[f'err_{prefix}_z']**2)

    if not output_file.lower().endswith('.xlsx'):
        output_file += '.xlsx'
    df_out.to_excel(output_file, index=False)
    print(f"\n   Saved → {output_file}")

    _print_stats(df_out, window_x, window_y, window_z)
    return df_out, kf



# PRINT STATS


def _print_stats(df, wx=None, wy=None, wz=None):
    print(f"\n{'='*70}")
    print(f"  RESULTS SUMMARY")
    if wx and wy and wz:
        print(f"  window_x={wx}  window_y={wy}  window_z={wz}  (per-axis)")
    print(f"{'='*70}")

    axes = [('X (East)', 'x'), ('Y (North)', 'y'), ('Z (Up)', 'z')]
    print(f"\n  {'Axis':<12} {'Raw RMS':>10} {'KF RMS':>10} {'Improve':>10}  Status")
    print(f"  {'-'*65}")
    for label, ax in axes:
        r   = _rms(df[f'err_raw_{ax}'])
        k   = _rms(df[f'err_kf_{ax}'])
        imp = (r - k) / r * 100
        status = " BELOW 10m!" if k < 10 else f"⚠  need {k-10:.2f}m more"
        print(f"  {label:<12} {r:>9.3f}m {k:>9.3f}m {imp:>9.1f}%  {status}")

    r2d = _rms(df['err_raw_2d']); k2d = _rms(df['err_kf_2d'])
    r3d = _rms(df['err_raw_3d']); k3d = _rms(df['err_kf_3d'])
    print(f"\n  {'2D (Horiz)':<12} {r2d:>9.3f}m {k2d:>9.3f}m {(r2d-k2d)/r2d*100:>9.1f}%")
    print(f"  {'3D (Full)':<12} {r3d:>9.3f}m {k3d:>9.3f}m {(r3d-k3d)/r3d*100:>9.1f}%")
    outliers = int(df['outlier'].sum())
    print(f"\n  Outliers rejected: {outliers} ({outliers/len(df)*100:.1f}%)")
    print(f"{'='*70}\n")



# PLOT


def plot_results(df, wx, wy, wz, save_path="kalman_xyz_plot.png"):
    fig, axes = plt.subplots(3, 2, figsize=(16, 12))
    fig.suptitle(f"Per-Axis Kalman  (wx={wx}, wy={wy}, wz={wz})",
                 fontsize=14, fontweight='bold')

    t = df['timestamp'].values
    axis_info = [
        ('X (East)',  'x', axes[0, 0]),
        ('Y (North)', 'y', axes[1, 0]),
        ('Z (Up)',    'z', axes[2, 0]),
    ]

    for label, ax, a in axis_info:
        raw_err = df[f'err_raw_{ax}'].values
        kf_err  = df[f'err_kf_{ax}'].values
        rms_r   = _rms(df[f'err_raw_{ax}'])
        rms_k   = _rms(df[f'err_kf_{ax}'])
        a.plot(t, raw_err, 'r-', alpha=0.4, linewidth=1,
               label=f'Raw  (RMS={rms_r:.2f}m)')
        a.plot(t, kf_err,  'b-', linewidth=2,
               label=f'Kalman (RMS={rms_k:.2f}m)')
        a.axhline(0,   color='k',     linestyle='--', linewidth=0.6, alpha=0.4)
        a.axhline(10,  color='green', linestyle='--', linewidth=1.5,
                  alpha=0.8, label='±10m target')
        a.axhline(-10, color='green', linestyle='--', linewidth=1.5, alpha=0.8)
        out_mask = df['outlier'].values
        if out_mask.any():
            a.scatter(t[out_mask], raw_err[out_mask], color='orange',
                      s=20, zorder=5, label='Outlier rejected')
        color = 'green' if rms_k < 10 else 'red'
        a.set_title(f'{label}  →  KF={rms_k:.2f}m{"" if rms_k < 10 else ""}',
                    fontsize=11, fontweight='bold', color=color)
        a.set_ylabel(f'{label} Error [m]', fontsize=10)
        a.legend(fontsize=9); a.grid(True, alpha=0.3)

    # 2D Trajectory
    ax_t = axes[0, 1]
    ax_t.plot(df['x_measured'], df['y_measured'], 'r.', alpha=0.3,
              markersize=2, label='Raw')
    ax_t.plot(df['x'], df['y'], 'b-', linewidth=2, label='Kalman')
    ax_t.plot(df['x_true'].iloc[0], df['y_true'].iloc[0],
              'g*', markersize=15, zorder=10, label='Ground Truth')
    ax_t.set_xlabel('East [m]'); ax_t.set_ylabel('North [m]')
    ax_t.set_title('2D Trajectory', fontsize=11, fontweight='bold')
    ax_t.legend(fontsize=9); ax_t.grid(True, alpha=0.3); ax_t.axis('equal')

    # Bar chart - all axes
    ax_b = axes[1, 1]
    lbls    = ['X', 'Y', 'Z', '2D', '3D']
    raw_rms = [_rms(df[f'err_raw_{a}']) for a in ['x','y','z']] + \
              [_rms(df['err_raw_2d']), _rms(df['err_raw_3d'])]
    kf_rms  = [_rms(df[f'err_kf_{a}'])  for a in ['x','y','z']] + \
              [_rms(df['err_kf_2d']),  _rms(df['err_kf_3d'])]
    xp = np.arange(len(lbls)); w = 0.35
    ax_b.bar(xp - w/2, raw_rms, w, label='Raw',    color='red',  alpha=0.7)
    bars = ax_b.bar(xp + w/2, kf_rms, w, label='Kalman', color='blue', alpha=0.7)
    # Color bars green if below 10m
    for bar, val in zip(bars, kf_rms):
        if val < 10:
            bar.set_color('green'); bar.set_alpha(0.8)
    ax_b.axhline(10, color='green', linestyle='--', linewidth=2, label='10m target')
    for i, (r, k) in enumerate(zip(raw_rms, kf_rms)):
        imp = (r - k) / r * 100
        ax_b.text(i, max(r, k) + 0.5, f'{imp:.0f}%',
                  ha='center', fontsize=9, fontweight='bold', color='navy')
    ax_b.set_xticks(xp); ax_b.set_xticklabels(lbls)
    ax_b.set_ylabel('RMS Error [m]')
    ax_b.set_title('RMS by Axis  (green bar = below 10m )', fontsize=11, fontweight='bold')
    ax_b.legend(fontsize=9); ax_b.grid(True, alpha=0.3, axis='y')

    # Innovation
    ax_i = axes[2, 1]
    ax_i.plot(t, df['innovation'], 'purple', linewidth=1, alpha=0.7,
              label='Innovation magnitude')
    ax_i.axhline(50, color='orange', linestyle='--', linewidth=1,
                 label='Outlier threshold (50m)')
    ax_i.set_xlabel('Time [s]')
    ax_i.set_ylabel('Innovation [m]')
    ax_i.set_title('Innovation over time', fontsize=11, fontweight='bold')
    ax_i.legend(fontsize=9); ax_i.grid(True, alpha=0.3)

    plt.tight_layout()
    if not save_path.lower().endswith(('.png','.jpg','.pdf')):
        save_path += '.png'
    plt.savefig(save_path, dpi=200, bbox_inches='tight')
    print(f"   Plot saved → {save_path}")
    plt.show()



# MAIN


if __name__ == "__main__":

    INPUT_FILE  = r"C:\Users\user\Downloads\gnss_results_FINAL.xlsx"
    OUTPUT_FILE = "kalman_xyz_results.xlsx"
    PLOT_FILE   = "kalman_xyz_plot.png"

    ADAPTIVE_R        = True
    OUTLIER_THRESHOLD = 50.0

    try:
        # Load once
        df_raw = load_wls_data(INPUT_FILE)
        df_raw = add_ground_truth(df_raw)

        # Auto-find best window for ALL THREE axes independently
        best_wx, best_wy, best_wz = find_best_windows(
            df_raw,
            windows=[2, 5, 10, 20, 30, 50, 80, 100]
        )

        # Run with best per-axis windows
        df_results, kf = run_kalman_filter(
            input_file=INPUT_FILE,
            output_file=OUTPUT_FILE,
            window_x=best_wx,
            window_y=best_wy,
            window_z=best_wz,    
            adaptive_R=ADAPTIVE_R,
            outlier_threshold=OUTLIER_THRESHOLD
        )

        print("\nGenerating plots...")
        plot_results(df_results, wx=best_wx, wy=best_wy, wz=best_wz,
                     save_path=PLOT_FILE)

        print(f"\n{'='*70}")
        print(f"   Complete!")
        print(f"  Output : {OUTPUT_FILE}")
        print(f"  Plot   : {PLOT_FILE}")
        print(f"{'='*70}\n")

    except FileNotFoundError:
        print(f"\n File not found: {INPUT_FILE}")
        print("   Update INPUT_FILE path and run again.\n")
    except Exception as e:
        import traceback
        print(f"\n Error: {e}")
        traceback.print_exc()
