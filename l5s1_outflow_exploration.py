import os
import pandas as pd
import mariadb
from dotenv import load_dotenv

def get_poi_data():
    """
    Pull bat speed and L5S1 outflow data from poi table
    Returns: pandas DataFrame with the two columns
    """
    # Connect to database
    load_dotenv("hitting_config.env")
    connection = mariadb.connect(
        host=os.getenv("DB_HOST"),
        port=int(os.getenv("DB_PORT", "3306")),
        database=os.getenv("DB_NAME"),
        user=os.getenv("DB_USER"),
        password=os.getenv("DB_PASSWORD"),
    )
    
    # SQL query
    query = """
    SELECT 
        blast_bat_speed_mph,
        l5s1_outflow_swing
    FROM poi
    WHERE blast_bat_speed_mph IS NOT NULL 
      AND l5s1_outflow_swing IS NOT NULL
    LIMIT 1000
    """
    
    # Execute and convert to DataFrame
    df = pd.read_sql(query, connection)
    connection.close()
    
    return df

# Run it
df = get_poi_data()

correlation = df['blast_bat_speed_mph'].corr(df['l5s1_outflow_swing'])
print(f"\nCorrelation between bat speed and L5S1 outflow: {correlation:.4f}")

import matplotlib.pyplot as plt

# Add this to your existing script
def create_scatter_plot(df):
    plt.figure(figsize=(10, 6))
    plt.scatter(df['blast_bat_speed_mph'], df['l5s1_outflow_swing'], alpha=0.6)
    plt.xlabel('Bat Speed (mph)')
    plt.ylabel('L5S1 Outflow')
    plt.title('L5S1 Outflow vs Bat Speed')
    plt.show()

# Call it after your correlation calculation
create_scatter_plot(df)