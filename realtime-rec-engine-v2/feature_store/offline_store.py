"""
Offline feature store using PostgreSQL and Parquet for batch operations.
Supports point-in-time queries, historical analysis, and large-scale feature engineering.
"""

import logging
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Any, Union, Tuple
from datetime import datetime, timezone, timedelta
from dataclasses import dataclass
from pathlib import Path
import pyarrow as pa
import pyarrow.parquet as pq
from sqlalchemy import create_engine, text, MetaData, Table, Column, String, Float, Integer, DateTime, Boolean, JSON
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, Session
from sqlalchemy.pool import QueuePool
import uuid
import json

from app.config import DatabaseConfig

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

Base = declarative_base()


@dataclass
class FeatureDefinition:
    """Feature definition metadata."""
    name: str
    entity_type: str
    data_type: str  # 'numerical', 'categorical', 'text', 'array'
    description: str
    default_value: Any = None
    tags: List[str] = None
    created_at: datetime = None
    updated_at: datetime = None
    
    def __post_init__(self):
        if self.tags is None:
            self.tags = []
        if self.created_at is None:
            self.created_at = datetime.now(timezone.utc)
        if self.updated_at is None:
            self.updated_at = datetime.now(timezone.utc)


class FeatureTable(Base):
    """SQLAlchemy model for feature definitions."""
    __tablename__ = 'feature_definitions'
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    name = Column(String(255), nullable=False, unique=True)
    entity_type = Column(String(100), nullable=False)
    data_type = Column(String(50), nullable=False)
    description = Column(String(1000))
    default_value = Column(JSON)
    tags = Column(JSON)
    created_at = Column(DateTime(timezone=True), default=datetime.now(timezone.utc))
    updated_at = Column(DateTime(timezone=True), default=datetime.now(timezone.utc), onupdate=datetime.now(timezone.utc))


class FeatureValueTable(Base):
    """SQLAlchemy model for feature values."""
    __tablename__ = 'feature_values'
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    entity_type = Column(String(100), nullable=False)
    entity_id = Column(String(255), nullable=False)
    feature_name = Column(String(255), nullable=False)
    value = Column(JSON)
    timestamp = Column(DateTime(timezone=True), nullable=False, default=datetime.now(timezone.utc))
    version = Column(Integer, default=1)
    is_active = Column(Boolean, default=True)
    
    # Composite index for efficient queries
    __table_args__ = (
        {'schema': 'feature_store'}
    )


class OfflineFeatureStore:
    """PostgreSQL-based offline feature store."""
    
    def __init__(self, config: DatabaseConfig):
        self.config = config
        self.engine = None
        self.SessionLocal = None
        
        # Initialize database connection
        self._initialize_database()
        
        # Feature definitions cache
        self.feature_definitions = {}
        
    def _initialize_database(self):
        """Initialize database connection and create tables."""
        # Create database URL
        if self.config.ssl_mode == "require":
            database_url = (
                f"postgresql://{self.config.username}:{self.config.password}@"
                f"{self.config.host}:{self.config.port}/{self.config.database}"
                f"?sslmode={self.config.ssl_mode}"
            )
        else:
            database_url = (
                f"postgresql://{self.config.username}:{self.config.password}@"
                f"{self.config.host}:{self.config.port}/{self.config.database}"
            )
        
        # Create engine with connection pooling
        self.engine = create_engine(
            database_url,
            poolclass=QueuePool,
            pool_size=self.config.pool_size,
            max_overflow=self.config.max_overflow,
            pool_timeout=self.config.pool_timeout,
            pool_recycle=self.config.pool_recycle,
            connect_args={
                'connect_timeout': self.config.connect_timeout,
                'command_timeout': self.config.command_timeout
            }
        )
        
        # Create session factory
        self.SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=self.engine)
        
        # Create tables
        self._create_tables()
        
        logger.info("Database connection initialized")
    
    def _create_tables(self):
        """Create database tables and schema."""
        try:
            # Create schema if it doesn't exist
            with self.engine.connect() as conn:
                conn.execute(text("CREATE SCHEMA IF NOT EXISTS feature_store"))
                conn.commit()
            
            # Create tables
            Base.metadata.create_all(bind=self.engine)
            
            # Create indexes for better performance
            self._create_indexes()
            
            logger.info("Database tables created successfully")
            
        except Exception as e:
            logger.error(f"Failed to create database tables: {e}")
            raise
    
    def _create_indexes(self):
        """Create database indexes for performance."""
        indexes = [
            "CREATE INDEX IF NOT EXISTS idx_feature_values_entity ON feature_store.feature_values(entity_type, entity_id)",
            "CREATE INDEX IF NOT EXISTS idx_feature_values_feature ON feature_store.feature_values(feature_name)",
            "CREATE INDEX IF NOT EXISTS idx_feature_values_timestamp ON feature_store.feature_values(timestamp)",
            "CREATE INDEX IF NOT EXISTS idx_feature_values_entity_feature ON feature_store.feature_values(entity_type, entity_id, feature_name)",
            "CREATE INDEX IF NOT EXISTS idx_feature_values_active ON feature_store.feature_values(is_active) WHERE is_active = true"
        ]
        
        with self.engine.connect() as conn:
            for index_sql in indexes:
                try:
                    conn.execute(text(index_sql))
                except Exception as e:
                    logger.warning(f"Failed to create index: {e}")
            conn.commit()
    
    def register_feature(self, feature_def: FeatureDefinition) -> bool:
        """Register a new feature definition."""
        try:
            with self.SessionLocal() as session:
                # Check if feature already exists
                existing = session.query(FeatureTable).filter(
                    FeatureTable.name == feature_def.name
                ).first()
                
                if existing:
                    logger.warning(f"Feature {feature_def.name} already exists")
                    return False
                
                # Create new feature
                db_feature = FeatureTable(
                    name=feature_def.name,
                    entity_type=feature_def.entity_type,
                    data_type=feature_def.data_type,
                    description=feature_def.description,
                    default_value=feature_def.default_value,
                    tags=feature_def.tags
                )
                
                session.add(db_feature)
                session.commit()
                
                # Cache locally
                self.feature_definitions[feature_def.name] = feature_def
                
                logger.info(f"Registered feature: {feature_def.name}")
                return True
                
        except Exception as e:
            logger.error(f"Failed to register feature {feature_def.name}: {e}")
            return False
    
    def get_feature_definition(self, feature_name: str) -> Optional[FeatureDefinition]:
        """Get feature definition."""
        if feature_name in self.feature_definitions:
            return self.feature_definitions[feature_name]
        
        try:
            with self.SessionLocal() as session:
                db_feature = session.query(FeatureTable).filter(
                    FeatureTable.name == feature_name
                ).first()
                
                if db_feature:
                    feature_def = FeatureDefinition(
                        name=db_feature.name,
                        entity_type=db_feature.entity_type,
                        data_type=db_feature.data_type,
                        description=db_feature.description,
                        default_value=db_feature.default_value,
                        tags=db_feature.tags or [],
                        created_at=db_feature.created_at,
                        updated_at=db_feature.updated_at
                    )
                    
                    self.feature_definitions[feature_name] = feature_def
                    return feature_def
                
        except Exception as e:
            logger.error(f"Failed to get feature definition {feature_name}: {e}")
        
        return None
    
    def write_features(self, entity_type: str, entity_id: str, 
                      features: Dict[str, Any], timestamp: Optional[datetime] = None,
                      version: Optional[int] = None) -> bool:
        """Write feature values to the offline store."""
        try:
            with self.SessionLocal() as session:
                current_time = timestamp or datetime.now(timezone.utc)
                current_version = version or int(current_time.timestamp())
                
                for feature_name, value in features.items():
                    # Check if feature is registered
                    feature_def = self.get_feature_definition(feature_name)
                    if not feature_def:
                        logger.warning(f"Feature {feature_name} not registered, skipping")
                        continue
                    
                    # Deactivate previous versions
                    session.query(FeatureValueTable).filter(
                        FeatureValueTable.entity_type == entity_type,
                        FeatureValueTable.entity_id == entity_id,
                        FeatureValueTable.feature_name == feature_name,
                        FeatureValueTable.is_active == True
                    ).update({'is_active': False})
                    
                    # Insert new feature value
                    feature_value = FeatureValueTable(
                        entity_type=entity_type,
                        entity_id=entity_id,
                        feature_name=feature_name,
                        value=value,
                        timestamp=current_time,
                        version=current_version,
                        is_active=True
                    )
                    
                    session.add(feature_value)
                
                session.commit()
                return True
                
        except Exception as e:
            logger.error(f"Failed to write features for {entity_type}:{entity_id}: {e}")
            return False
    
    def write_features_batch(self, feature_data: List[Dict[str, Any]]) -> int:
        """Write multiple feature records in batch."""
        try:
            with self.SessionLocal() as session:
                inserted_count = 0
                
                for data in feature_data:
                    entity_type = data['entity_type']
                    entity_id = data['entity_id']
                    features = data['features']
                    timestamp = data.get('timestamp', datetime.now(timezone.utc))
                    version = data.get('version', int(timestamp.timestamp()))
                    
                    for feature_name, value in features.items():
                        # Check if feature is registered
                        feature_def = self.get_feature_definition(feature_name)
                        if not feature_def:
                            continue
                        
                        # Deactivate previous versions
                        session.query(FeatureValueTable).filter(
                            FeatureValueTable.entity_type == entity_type,
                            FeatureValueTable.entity_id == entity_id,
                            FeatureValueTable.feature_name == feature_name,
                            FeatureValueTable.is_active == True
                        ).update({'is_active': False})
                        
                        # Insert new feature value
                        feature_value = FeatureValueTable(
                            entity_type=entity_type,
                            entity_id=entity_id,
                            feature_name=feature_name,
                            value=value,
                            timestamp=timestamp,
                            version=version,
                            is_active=True
                        )
                        
                        session.add(feature_value)
                        inserted_count += 1
                
                session.commit()
                return inserted_count
                
        except Exception as e:
            logger.error(f"Failed to write batch features: {e}")
            return 0
    
    def get_features(self, entity_type: str, entity_id: str,
                    feature_names: List[str], 
                    timestamp: Optional[datetime] = None) -> Dict[str, Any]:
        """Get feature values for an entity at a specific point in time."""
        try:
            with self.SessionLocal() as session:
                query = session.query(FeatureValueTable).filter(
                    FeatureValueTable.entity_type == entity_type,
                    FeatureValueTable.entity_id == entity_id,
                    FeatureValueTable.feature_name.in_(feature_names),
                    FeatureValueTable.is_active == True
                )
                
                # Point-in-time query
                if timestamp:
                    query = query.filter(FeatureValueTable.timestamp <= timestamp)
                
                results = query.all()
                
                # Convert to dictionary
                features = {}
                for result in results:
                    features[result.feature_name] = result.value
                
                return features
                
        except Exception as e:
            logger.error(f"Failed to get features for {entity_type}:{entity_id}: {e}")
            return {}
    
    def get_historical_features(self, entity_type: str, entity_id: str,
                               feature_names: List[str],
                               start_time: datetime, end_time: datetime) -> pd.DataFrame:
        """Get historical feature values within a time range."""
        try:
            with self.SessionLocal() as session:
                query = session.query(FeatureValueTable).filter(
                    FeatureValueTable.entity_type == entity_type,
                    FeatureValueTable.entity_id == entity_id,
                    FeatureValueTable.feature_name.in_(feature_names),
                    FeatureValueTable.timestamp >= start_time,
                    FeatureValueTable.timestamp <= end_time
                ).order_by(FeatureValueTable.timestamp)
                
                results = query.all()
                
                # Convert to DataFrame
                data = []
                for result in results:
                    data.append({
                        'timestamp': result.timestamp,
                        'feature_name': result.feature_name,
                        'value': result.value,
                        'version': result.version
                    })
                
                df = pd.DataFrame(data)
                
                # Pivot to have features as columns
                if not df.empty:
                    df = df.pivot_table(
                        index='timestamp',
                        columns='feature_name',
                        values='value',
                        aggfunc='first'
                    ).reset_index()
                
                return df
                
        except Exception as e:
            logger.error(f"Failed to get historical features for {entity_type}:{entity_id}: {e}")
            return pd.DataFrame()
    
    def get_feature_statistics(self, feature_name: str, 
                              entity_type: Optional[str] = None,
                              time_range: Optional[Tuple[datetime, datetime]] = None) -> Dict[str, Any]:
        """Get statistics for a feature."""
        try:
            with self.SessionLocal() as session:
                query = session.query(FeatureValueTable).filter(
                    FeatureValueTable.feature_name == feature_name,
                    FeatureValueTable.is_active == True
                )
                
                if entity_type:
                    query = query.filter(FeatureValueTable.entity_type == entity_type)
                
                if time_range:
                    start_time, end_time = time_range
                    query = query.filter(
                        FeatureValueTable.timestamp >= start_time,
                        FeatureValueTable.timestamp <= end_time
                    )
                
                results = query.all()
                
                if not results:
                    return {}
                
                # Extract numeric values for statistics
                numeric_values = []
                for result in results:
                    try:
                        value = float(result.value)
                        numeric_values.append(value)
                    except (ValueError, TypeError):
                        continue
                
                if numeric_values:
                    return {
                        'count': len(numeric_values),
                        'mean': np.mean(numeric_values),
                        'std': np.std(numeric_values),
                        'min': np.min(numeric_values),
                        'max': np.max(numeric_values),
                        'median': np.median(numeric_values),
                        'q25': np.percentile(numeric_values, 25),
                        'q75': np.percentile(numeric_values, 75)
                    }
                else:
                    return {'count': len(results)}
                
        except Exception as e:
            logger.error(f"Failed to get feature statistics for {feature_name}: {e}")
            return {}
    
    def export_to_parquet(self, entity_type: str, feature_names: List[str],
                         output_path: str, start_time: Optional[datetime] = None,
                         end_time: Optional[datetime] = None) -> bool:
        """Export features to Parquet file."""
        try:
            with self.SessionLocal() as session:
                query = session.query(FeatureValueTable).filter(
                    FeatureValueTable.entity_type == entity_type,
                    FeatureValueTable.feature_name.in_(feature_names),
                    FeatureValueTable.is_active == True
                )
                
                if start_time:
                    query = query.filter(FeatureValueTable.timestamp >= start_time)
                
                if end_time:
                    query = query.filter(FeatureValueTable.timestamp <= end_time)
                
                results = query.all()
                
                # Convert to DataFrame
                data = []
                for result in results:
                    data.append({
                        'entity_id': result.entity_id,
                        'feature_name': result.feature_name,
                        'value': result.value,
                        'timestamp': result.timestamp,
                        'version': result.version
                    })
                
                df = pd.DataFrame(data)
                
                if df.empty:
                    logger.warning("No data to export")
                    return False
                
                # Pivot to have features as columns
                df_pivot = df.pivot_table(
                    index=['entity_id', 'timestamp'],
                    columns='feature_name',
                    values='value',
                    aggfunc='first'
                ).reset_index()
                
                # Write to Parquet
                table = pa.Table.from_pandas(df_pivot)
                pq.write_table(table, output_path)
                
                logger.info(f"Exported {len(df_pivot)} rows to {output_path}")
                return True
                
        except Exception as e:
            logger.error(f"Failed to export to Parquet: {e}")
            return False
    
    def import_from_parquet(self, parquet_path: str, entity_type: str,
                           timestamp_column: str = 'timestamp',
                           entity_id_column: str = 'entity_id') -> int:
        """Import features from Parquet file."""
        try:
            # Read Parquet file
            df = pd.read_parquet(parquet_path)
            
            # Convert to feature data format
            feature_data = []
            for _, row in df.iterrows():
                entity_id = row[entity_id_column]
                timestamp = row[timestamp_column]
                
                # Extract feature columns (exclude metadata columns)
                feature_columns = [col for col in df.columns 
                                 if col not in [timestamp_column, entity_id_column]]
                
                features = {}
                for col in feature_columns:
                    if pd.notna(row[col]):
                        features[col] = row[col]
                
                if features:
                    feature_data.append({
                        'entity_type': entity_type,
                        'entity_id': str(entity_id),
                        'features': features,
                        'timestamp': timestamp
                    })
            
            # Write to database
            inserted_count = self.write_features_batch(feature_data)
            
            logger.info(f"Imported {inserted_count} feature values from {parquet_path}")
            return inserted_count
            
        except Exception as e:
            logger.error(f"Failed to import from Parquet: {e}")
            return 0
    
    def cleanup_old_versions(self, retention_days: int = 30) -> int:
        """Clean up old feature versions."""
        try:
            cutoff_date = datetime.now(timezone.utc) - timedelta(days=retention_days)
            
            with self.SessionLocal() as session:
                # Delete old inactive versions
                deleted_count = session.query(FeatureValueTable).filter(
                    FeatureValueTable.is_active == False,
                    FeatureValueTable.timestamp < cutoff_date
                ).delete()
                
                session.commit()
                
                logger.info(f"Cleaned up {deleted_count} old feature versions")
                return deleted_count
                
        except Exception as e:
            logger.error(f"Failed to cleanup old versions: {e}")
            return 0
    
    def get_store_stats(self) -> Dict[str, Any]:
        """Get offline store statistics."""
        try:
            with self.SessionLocal() as session:
                # Feature count
                feature_count = session.query(FeatureTable).count()
                
                # Feature value count
                value_count = session.query(FeatureValueTable).count()
                
                # Active value count
                active_count = session.query(FeatureValueTable).filter(
                    FeatureValueTable.is_active == True
                ).count()
                
                # Entity types
                entity_types = session.query(FeatureValueTable.entity_type).distinct().all()
                entity_type_count = len(entity_types)
                
                # Date range
                min_date = session.query(func.min(FeatureValueTable.timestamp)).scalar()
                max_date = session.query(func.max(FeatureValueTable.timestamp)).scalar()
                
                return {
                    'feature_definitions': feature_count,
                    'total_feature_values': value_count,
                    'active_feature_values': active_count,
                    'entity_types': entity_type_count,
                    'date_range': {
                        'min': min_date,
                        'max': max_date
                    }
                }
                
        except Exception as e:
            logger.error(f"Failed to get store stats: {e}")
            return {}
    
    def health_check(self) -> bool:
        """Check database health."""
        try:
            with self.SessionLocal() as session:
                session.execute(text("SELECT 1"))
                return True
        except Exception as e:
            logger.error(f"Database health check failed: {e}")
            return False
    
    def close(self):
        """Close database connections."""
        if self.engine:
            self.engine.dispose()
        
        logger.info("Database connections closed")


# Example usage
def main():
    """Example usage of offline feature store."""
    from app.config import Config
    
    # Load configuration
    config = Config()
    
    # Initialize offline store
    offline_store = OfflineFeatureStore(config.database)
    
    try:
        # Register some features
        user_age_feature = FeatureDefinition(
            name='user_age',
            entity_type='user',
            data_type='numerical',
            description='User age in years'
        )
        
        user_preferences_feature = FeatureDefinition(
            name='user_preferences',
            entity_type='user',
            data_type='array',
            description='User content preferences'
        )
        
        offline_store.register_feature(user_age_feature)
        offline_store.register_feature(user_preferences_feature)
        
        # Write some features
        features = {
            'user_age': 30,
            'user_preferences': ['action', 'comedy', 'drama'],
            'last_login': datetime.now(timezone.utc).isoformat()
        }
        
        success = offline_store.write_features('user', 'user_123', features)
        logger.info(f"Write features: {success}")
        
        # Get features
        retrieved = offline_store.get_features(
            'user', 'user_123', ['user_age', 'user_preferences']
        )
        logger.info(f"Retrieved features: {retrieved}")
        
        # Get statistics
        stats = offline_store.get_feature_statistics('user_age')
        logger.info(f"Feature statistics: {stats}")
        
        # Get store stats
        store_stats = offline_store.get_store_stats()
        logger.info(f"Store stats: {store_stats}")
        
        # Health check
        is_healthy = offline_store.health_check()
        logger.info(f"Database healthy: {is_healthy}")
        
    finally:
        offline_store.close()


if __name__ == "__main__":
    main()
