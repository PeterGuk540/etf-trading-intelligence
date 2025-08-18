#!/usr/bin/env python
"""Initialize database tables and schema"""

import os
import sys
import psycopg2
from psycopg2.extensions import ISOLATION_LEVEL_AUTOCOMMIT
from sqlalchemy import create_engine, MetaData, Table, Column, Integer, String, Float, DateTime, Index
from sqlalchemy.sql import text
from datetime import datetime
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def create_database():
    """Create database if it doesn't exist"""
    db_url = os.getenv('DATABASE_URL', 'postgresql://trading_user:changeme@localhost:5432/etf_trading')
    
    # Parse connection string
    parts = db_url.replace('postgresql://', '').split('@')
    user_pass = parts[0].split(':')
    host_db = parts[1].split('/')
    host_port = host_db[0].split(':')
    
    user = user_pass[0]
    password = user_pass[1]
    host = host_port[0]
    port = host_port[1] if len(host_port) > 1 else '5432'
    dbname = host_db[1]
    
    # Connect to default postgres database
    conn = psycopg2.connect(
        host=host,
        port=port,
        user=user,
        password=password,
        database='postgres'
    )
    conn.set_isolation_level(ISOLATION_LEVEL_AUTOCOMMIT)
    cursor = conn.cursor()
    
    # Check if database exists
    cursor.execute(f"SELECT 1 FROM pg_database WHERE datname = '{dbname}'")
    exists = cursor.fetchone()
    
    if not exists:
        cursor.execute(f"CREATE DATABASE {dbname}")
        logger.info(f"Database {dbname} created successfully")
    else:
        logger.info(f"Database {dbname} already exists")
    
    cursor.close()
    conn.close()
    
    return db_url

def create_tables(db_url):
    """Create required tables"""
    engine = create_engine(db_url)
    metadata = MetaData()
    
    # Market data table
    market_data = Table(
        'market_data',
        metadata,
        Column('id', Integer, primary_key=True),
        Column('symbol', String(10), nullable=False),
        Column('date', DateTime, nullable=False),
        Column('open', Float),
        Column('high', Float),
        Column('low', Float),
        Column('close', Float),
        Column('volume', Float),
        Column('returns', Float),
        Index('idx_symbol_date', 'symbol', 'date', unique=True)
    )
    
    # Predictions table
    predictions = Table(
        'predictions',
        metadata,
        Column('id', Integer, primary_key=True),
        Column('model_version', String(50)),
        Column('symbol', String(10)),
        Column('prediction_date', DateTime),
        Column('target_date', DateTime),
        Column('predicted_return', Float),
        Column('confidence', Float),
        Column('created_at', DateTime, default=datetime.utcnow)
    )
    
    # Backtest results table
    backtest_results = Table(
        'backtest_results',
        metadata,
        Column('id', Integer, primary_key=True),
        Column('model_version', String(50)),
        Column('start_date', DateTime),
        Column('end_date', DateTime),
        Column('sharpe_ratio', Float),
        Column('max_drawdown', Float),
        Column('total_return', Float),
        Column('win_rate', Float),
        Column('created_at', DateTime, default=datetime.utcnow)
    )
    
    # Model registry table
    model_registry = Table(
        'model_registry',
        metadata,
        Column('id', Integer, primary_key=True),
        Column('model_name', String(100)),
        Column('model_version', String(50)),
        Column('model_path', String(500)),
        Column('metrics', String(2000)),  # JSON string
        Column('created_at', DateTime, default=datetime.utcnow),
        Column('is_active', Integer, default=0)
    )
    
    # Create all tables
    metadata.create_all(engine)
    logger.info("All tables created successfully")
    
    # Enable TimescaleDB extension if available
    try:
        with engine.connect() as conn:
            conn.execute(text("CREATE EXTENSION IF NOT EXISTS timescaledb CASCADE;"))
            conn.commit()
            
            # Convert market_data to hypertable
            conn.execute(text(
                "SELECT create_hypertable('market_data', 'date', "
                "if_not_exists => TRUE, migrate_data => TRUE);"
            ))
            conn.commit()
            logger.info("TimescaleDB hypertable created")
    except Exception as e:
        logger.warning(f"Could not enable TimescaleDB: {e}")

def main():
    """Main initialization function"""
    logger.info("Starting database initialization...")
    
    # Load environment variables
    from dotenv import load_dotenv
    load_dotenv()
    
    # Create database
    db_url = create_database()
    
    # Create tables
    create_tables(db_url)
    
    logger.info("Database initialization completed successfully!")

if __name__ == "__main__":
    main()