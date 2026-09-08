import os
import pandas as pd
import sqlite3

def run_database_demo():
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    train_path = os.path.join(base_dir, "data", "train_FD001.txt")
    
    # Define columns
    col_names = ['unit_id', 'time_cycles', 'op_setting_1', 'op_setting_2', 'op_setting_3'] + [f'sensor_{i}' for i in range(1, 22)]
    
    print("="*60)
    print("DATABASE INTEGRATION DEMO")
    print("="*60)
    
    # 1. Load raw file
    print(f"Loading raw data from: {train_path}")
    df = pd.read_csv(train_path, sep=r'\s+', header=None, names=col_names)
    
    # Attempt PostgreSQL Connection (for Docker)
    conn = None
    db_type = ""
    
    try:
        # Check if psycopg2 is installed and try connecting to docker postgres
        import psycopg2
        from sqlalchemy import create_engine
        
        # Connection string matching docker-compose.yml
        db_url = "postgresql://admin:password123@localhost:5432/cmapss_db"
        engine = create_engine(db_url)
        conn = engine.connect()
        db_type = "PostgreSQL (Docker Container)"
        print("--> Successfully connected to local Docker PostgreSQL database!")
    except Exception as e:
        print("\n[INFO] Local Docker PostgreSQL not detected or database libraries not installed.")
        print("--> Falling back to Python's built-in SQLite database (local file)!")
        
        # SQLite fallback: Creates a local database file in the project directory
        sqlite_path = os.path.join(base_dir, "data", "cmapss.db")
        conn = sqlite3.connect(sqlite_path)
        db_type = f"SQLite (Local file: {sqlite_path})"
    
    # 2. Save data to database
    print(f"\nWriting C-MAPSS data to {db_type} table: 'train_fd001'...")
    # write to SQL
    df.to_sql(name='train_fd001', con=conn, if_exists='replace', index=False)
    print("--> Data successfully written to the database!")
    
    # 3. Query data back using SQL
    print("\nExecuting SQL query: 'SELECT * FROM train_fd001 WHERE unit_id = 1 LIMIT 5'...")
    queried_df = pd.read_sql_query("SELECT * FROM train_fd001 WHERE unit_id = 1 LIMIT 5", conn)
    
    print("\nQuery results (First 5 cycles of Engine 1 from Database):")
    print(queried_df[['unit_id', 'time_cycles', 'sensor_2', 'sensor_3', 'sensor_4']])
    
    # Close connection
    conn.close()
    print("\n" + "="*60)

if __name__ == "__main__":
    run_database_demo()
