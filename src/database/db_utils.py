import os
from sqlalchemy import create_engine, text
from sqlalchemy.exc import OperationalError
from dotenv import load_dotenv

# Find the absolute path to the .env file at the project root
ENV_PATH = os.path.join(os.path.dirname(__file__), "..", "..", ".env")
# Load the environment variables from the .env file into the os environment
load_dotenv(ENV_PATH)

def get_db_engine():
    """
    Creates and returns a SQLAlchemy engine for the PostGIS database.
    Uses environment variables if set, otherwise defaults to local docker-compose values.
    """
    # Default credentials match the docker-compose.yml
    db_user = os.getenv("POSTGRES_USER", "geo_user").strip()
    db_password = os.getenv("POSTGRES_PASSWORD", "geo_password").strip()
    db_host = os.getenv("POSTGRES_HOST", "localhost").strip()
    db_port = os.getenv("POSTGRES_PORT", "5432").strip()
    db_name = os.getenv("POSTGRES_DB", "geo_db").strip()

    connection_string = f"postgresql+psycopg2://{db_user}:{db_password}@{db_host}:{db_port}/{db_name}"
    
    # Create the SQLAlchemy engine
    engine = create_engine(connection_string)
    return engine

def test_connection(engine):
    """
    Tests the database connection and verifies PostGIS is installed.
    """
    try:
        with engine.connect() as conn:
            # Check if PostGIS extension is available
            result = conn.execute(text("SELECT PostGIS_Version();")).fetchone()
            print("Database Connection Successful!")
            print(f"PostGIS Version: {result[0]}")
            return True
    except OperationalError as e:
        print("Failed to connect to the database. Ensure the Docker container is running.")
        print(f"Error: {e}")
        return False
    except Exception as e:
        print("Connected to PostgreSQL, but PostGIS extension might be missing.")
        print(f"Error: {e}")
        return False

if __name__ == "__main__":
    engine = get_db_engine()
    test_connection(engine)
