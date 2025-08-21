import os
from dotenv import load_dotenv
import mariadb
import pandas as pd
import matplotlib.pyplot as plt

load_dotenv("config.env")

connection = mariadb.connect(
    host=os.getenv('DB_HOST'),
    database=os.getenv('DB_NAME'),
    user=os.getenv('DB_USER'),
    password=os.getenv('DB_PASSWORD')
)


# Create cursor to execute queries
cursor = connection.cursor()

# Query to get all columns from the poi table
cursor.execute("""
SELECT
  p.xfactor_angle_z_mhss	 AS poi_xfactor_angle_z_mhss,
  p.session_trial				AS poi_session_trial,
  p.exit_velo_mph				AS poi_exit_velo_mph,
  p.blast_bat_speed_mph		AS poi_blast_bat_speed_mph,
  s.height_meters				AS session_height,
  s.mass_kilograms			AS session_mass_kilograms
  
  
FROM poi p
JOIN sessions s
  ON s.session = CAST(SUBSTRING_INDEX(p.session_trial, '_', 1) AS UNSIGNED)   
WHERE s.date BETWEEN '2024-01-01' AND '2024-12-31'
""")
results = cursor.fetchall()

# Get column names
column_names = [desc[0] for desc in cursor.description]

# Convert to DataFrame
df = pd.DataFrame(results, columns=column_names)

# Filter out rows where poi_exit_velo_mph equals 0
df_filtered = df[df['poi_blast_bat_speed_mph'] != 0]

print(f"Original data shape: {df.shape}")
print(f"Filtered data shape: {df_filtered.shape}")
print(f"Removed {df.shape[0] - df_filtered.shape[0]} rows where poi_blast_bat_speed_mph = 0")

# Now you can use df_filtered for analysis
print(df_filtered.head())
print(df_filtered.shape)

# Create correlation graph using filtered data
plt.figure(figsize=(10, 6))
plt.scatter(df_filtered['session_mass_kilograms'], df_filtered['poi_blast_bat_speed_mph'], alpha=0.6)
plt.xlabel('Session Mass (kg)')
plt.ylabel('Bat Speed (MPH)')
plt.title('Correlation: Mass vs Bat Speed')

# Calculate and display correlation coefficient using filtered data
correlation = df_filtered['session_mass_kilograms'].corr(df_filtered['poi_blast_bat_speed_mph'])
plt.text(0.05, 0.95, f'Correlation: {correlation:.3f}', transform=plt.gca().transAxes, 
         bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8))

plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

cursor.close()
connection.close()
print("Connection closed")

