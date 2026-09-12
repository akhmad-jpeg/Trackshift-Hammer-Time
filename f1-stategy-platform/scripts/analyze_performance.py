import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime
import os

from config import get_db_connection
from stint_analysis import detrend_laps


# ANALYSIS FUNCTIONS
def get_session_summary(conn):
    query = """
    SELECT 
        s.session_id,
        s.track_name,
        s.date,
        COUNT(l.lap_id) as total_laps,
        MIN(CASE WHEN l.lap_time_ms > 0 THEN l.lap_time_ms END) / 1000 as fastest_lap,
        AVG(CASE WHEN l.lap_time_ms > 0 THEN l.lap_time_ms END) / 1000 as avg_lap_time,
        SUM(CASE WHEN l.is_valid = 1 THEN 1 ELSE 0 END) as valid_laps,
        SUM(CASE WHEN l.is_valid = 0 THEN 1 ELSE 0 END) as invalid_laps
    FROM sessions s
    LEFT JOIN laps l ON s.session_id = l.session_id
    GROUP BY s.session_id
    ORDER BY s.date DESC
    """
    return pd.read_sql(query, conn)

def get_tyre_degradation(conn, session_id):
    """Per-stint fuel-adjusted degradation deltas, aggregated by compound+age.

    Each lap is expressed relative to its own stint's pace line (see
    stint_analysis.detrend_laps), so the fuel-burn trend is removed and
    any remaining positive delta at high tyre age is genuine wear.
    """
    query = """
    SELECT 
        l.lap_number,
        l.lap_time_ms / 1000.0 AS lap_time,
        l.tyre_compound,
        l.tyre_age,
        l.is_valid,
        MAX(CASE WHEN se.event_type = 'PitStop' THEN 1 ELSE 0 END) AS has_pit_stop
    FROM laps l
    LEFT JOIN strategy_events se ON l.lap_id = se.lap_id
        AND se.event_type = 'PitStop'
        AND (se.duration_sec IS NULL OR se.duration_sec >= 15)
    WHERE l.session_id = %s AND l.lap_time_ms > 0
    GROUP BY l.lap_id, l.lap_number, l.lap_time_ms, l.tyre_compound, l.tyre_age, l.is_valid
    ORDER BY l.lap_number
    """
    df = pd.read_sql(query, conn, params=(session_id,))
    if len(df) == 0:
        return df

    laps = detrend_laps(df.to_dict('records'))
    out = pd.DataFrame(laps)
    agg = (out.groupby(['tyre_compound', 'tyre_age'], as_index=False)
              .agg(avg_delta=('stint_delta', 'mean'),
                   num_laps=('stint_delta', 'count')))
    agg = agg[agg['num_laps'] > 0]
    return agg[['tyre_compound', 'tyre_age', 'avg_delta', 'num_laps']]

def get_lap_consistency(conn, session_id):
    query = """
    SELECT 
        lap_number,
        lap_time_ms / 1000 as lap_time,
        tyre_compound,
        tyre_age,
        is_valid
    FROM laps
    WHERE session_id = %s AND lap_time_ms > 0
    ORDER BY lap_number
    """
    return pd.read_sql(query, conn, params=(session_id,))

def get_speed_distribution(conn, session_id):
    query = """
    SELECT 
        t.speed,
        l.lap_number
    FROM telemetry t
    JOIN laps l ON t.lap_id = l.lap_id
    WHERE l.session_id = %s AND t.speed > 0
    """
    return pd.read_sql(query, conn, params=(session_id,))

# VISUALIZATION
def plot_tyre_degradation(df, track_name, output_dir):
    plt.figure(figsize=(12, 6))
    
    for compound in df['tyre_compound'].unique():
        compound_data = df[df['tyre_compound'] == compound]
        plt.plot(compound_data['tyre_age'], compound_data['avg_delta'], 
                marker='o', label=compound, linewidth=2)
    
    plt.xlabel('Tyre Age (laps)', fontsize=12)
    plt.ylabel('Delta vs Stint Trend (seconds)', fontsize=12)
    plt.title(f'Tyre Degradation (Fuel-Adjusted) - {track_name}', fontsize=14, fontweight='bold')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    filename = os.path.join(output_dir, 'tyre_degradation.png')
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    print(f"[SAVED] {filename}")
    plt.close()


def plot_lap_times(df, track_name, output_dir):
    plt.figure(figsize=(14, 6))
    
    valid_laps = df[df['is_valid'] == 1]
    invalid_laps = df[df['is_valid'] == 0]
    
    plt.plot(valid_laps['lap_number'], valid_laps['lap_time'], 
            marker='o', color='green', label='Valid Laps', linewidth=2)
    
    if len(invalid_laps) > 0:
        plt.scatter(invalid_laps['lap_number'], invalid_laps['lap_time'], 
                   color='red', s=100, marker='x', label='Invalid Laps', zorder=5)
    
    # Add fastest lap marker
    if len(valid_laps) > 0:
        fastest_lap = valid_laps.loc[valid_laps['lap_time'].idxmin()]
        plt.scatter(fastest_lap['lap_number'], fastest_lap['lap_time'], 
                   color='gold', s=200, marker='*', label='Fastest Lap', zorder=10)
    
    plt.xlabel('Lap Number', fontsize=12)
    plt.ylabel('Lap Time (seconds)', fontsize=12)
    plt.title(f'Lap Time Progression - {track_name}', fontsize=14, fontweight='bold')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    filename = os.path.join(output_dir, 'lap_times.png')
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    print(f"[SAVED] {filename}")
    plt.close()

def plot_speed_distribution(df, track_name, output_dir):
    plt.figure(figsize=(12, 6))
    
    plt.hist(df['speed'], bins=30, color='blue', alpha=0.7, edgecolor='black')
    
    plt.xlabel('Speed (km/h)', fontsize=12)
    plt.ylabel('Frequency', fontsize=12)
    plt.title(f'Speed Distribution - {track_name}', fontsize=14, fontweight='bold')
    plt.grid(True, alpha=0.3, axis='y')
    plt.tight_layout()
    
    filename = os.path.join(output_dir, 'speed_distribution.png')
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    print(f"[SAVED] {filename}")
    plt.close()

def save_summary_report(sessions, lap_stats, output_dir):
    """Save text summary report"""
    filename = os.path.join(output_dir, 'summary_report.txt')
    
    with open(filename, 'w') as f:
        f.write("=" * 60 + "\n")
        f.write("F1 PERFORMANCE ANALYSIS REPORT\n")
        f.write("=" * 60 + "\n\n")
        
        f.write("SESSION SUMMARY\n")
        f.write("-" * 60 + "\n")
        f.write(sessions.to_string(index=False) + "\n\n")
        
        if lap_stats:
            f.write("LAP TIME STATISTICS\n")
            f.write("-" * 60 + "\n")
            f.write(f"Fastest Lap:     {lap_stats['fastest']:.3f}s\n")
            f.write(f"Average Lap:     {lap_stats['mean']:.3f}s\n")
            f.write(f"Std Deviation:   {lap_stats['std']:.3f}s\n")
            f.write(f"Consistency:     {lap_stats['consistency']:.1f}%\n\n")
        
        f.write("Generated on: " + datetime.now().strftime("%Y-%m-%d %H:%M:%S") + "\n")
    
    print(f"[SAVED] {filename}")


# MAIN ANALYSIS
def main():
    print("=" * 60)
    print("F1 PERFORMANCE ANALYSIS")
    print("=" * 60)
    
    conn = get_db_connection()
    
    # Get session summary
    print("\n[1/5] Fetching session data...")
    sessions = get_session_summary(conn)
    
    if len(sessions) == 0:
        print("\n[ERROR] No sessions found in database!")
        print("Run some races first using start_capture.bat")
        conn.close()
        return
    
    print("\n" + "=" * 60)
    print("AVAILABLE SESSIONS")
    print("=" * 60)
    print(sessions.to_string(index=False))
    
    # Let user choose session
    print("\n" + "=" * 60)
    if len(sessions) == 1:
        print("Only one session available. Analyzing it automatically.")
        choice = 0
    else:
        print("Which session would you like to analyze?")
        print("Enter session ID, or press ENTER for most recent:")
        
        user_input = input("> ").strip()
        
        if user_input == "":
            choice = 0  # Most recent
            print(f"Analyzing most recent session (ID: {sessions.iloc[0]['session_id']})")
        else:
            try:
                session_id_input = int(user_input)
                # Find the session in the dataframe
                session_match = sessions[sessions['session_id'] == session_id_input]
                if len(session_match) == 0:
                    print(f"\n[ERROR] Session ID {session_id_input} not found!")
                    conn.close()
                    return
                choice = sessions[sessions['session_id'] == session_id_input].index[0]
            except ValueError:
                print("\n[ERROR] Invalid input! Using most recent session.")
                choice = 0
    
    # Analyze selected session
    selected_session = sessions.iloc[choice]
    # pd.read_sql returns numpy types; the mysql-connector C cursor cannot
    # bind a numpy.int64 as a query parameter, so convert to a plain int
    # before passing it to any parameterised query below.
    session_id = int(selected_session['session_id'])
    track_name = selected_session['track_name']
    
    # Create output directory
    output_dir = f"analysis/session_{session_id}_{track_name}"
    os.makedirs(output_dir, exist_ok=True)
    print(f"\n[OUTPUT] Saving to: {output_dir}/")
    
    print(f"\n[2/5] Analyzing session {session_id} ({track_name})...")
    
    lap_stats = None
    
    # Tyre degradation analysis
    tyre_deg = get_tyre_degradation(conn, session_id)
    if len(tyre_deg) > 0:
        print("\n" + "=" * 60)
        print("TYRE DEGRADATION")
        print("=" * 60)
        print(tyre_deg.to_string(index=False))
        plot_tyre_degradation(tyre_deg, track_name, output_dir)
    
    # Lap consistency analysis
    print(f"\n[3/5] Analyzing lap times...")
    lap_times = get_lap_consistency(conn, session_id)
    if len(lap_times) > 0:
        # lap_time_ms > 0 is filtered in SQL; the extra guard keeps a 0 ms
        # in-progress lap from ever becoming the "fastest" even if a caller
        # feeds this function unfiltered data.
        valid_laps = lap_times[(lap_times['is_valid'] == 1)
                               & (lap_times['lap_time'] > 0)]
        
        if len(valid_laps) > 1:
            std_dev = valid_laps['lap_time'].std()
            mean_time = valid_laps['lap_time'].mean()
            fastest = valid_laps['lap_time'].min()
            
            lap_stats = {
                'fastest': fastest,
                'mean': mean_time,
                'std': std_dev,
                'consistency': (1 - std_dev/mean_time) * 100
            }
            
            print("\n" + "=" * 60)
            print("LAP TIME STATISTICS")
            print("=" * 60)
            print(f"Fastest Lap:     {fastest:.3f}s")
            print(f"Average Lap:     {mean_time:.3f}s")
            print(f"Std Deviation:   {std_dev:.3f}s")
            print(f"Consistency:     {lap_stats['consistency']:.1f}%")
        
        plot_lap_times(lap_times, track_name, output_dir)
    
    # Speed analysis
    print(f"\n[4/5] Analyzing speed data...")
    speed_data = get_speed_distribution(conn, session_id)
    if len(speed_data) > 0:
        print("\n" + "=" * 60)
        print("SPEED STATISTICS")
        print("=" * 60)
        print(f"Max Speed:       {speed_data['speed'].max()} km/h")
        print(f"Average Speed:   {speed_data['speed'].mean():.1f} km/h")
        print(f"Min Speed:       {speed_data['speed'].min()} km/h")
        
        plot_speed_distribution(speed_data, track_name, output_dir)
    
    # Save summary report
    print(f"\n[5/5] Saving summary report...")
    save_summary_report(sessions.iloc[[choice]], lap_stats, output_dir)
    
    conn.close()
    
    print("\n" + "=" * 60)
    print("ANALYSIS COMPLETE")
    print("=" * 60)
    print(f"All files saved to: {output_dir}/")

if __name__ == "__main__":
    main()