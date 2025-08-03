"""
Database connection management.
"""

import os
from typing import Optional, Generator
from contextlib import contextmanager
import pymongo
from pymongo import MongoClient
from pymongo.database import Database
import psycopg2
from psycopg2.extras import RealDictCursor
import redis
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, Session
from sqlalchemy.ext.declarative import declarative_base


# SQLAlchemy base for ORM models
Base = declarative_base()


class DatabaseConnection:
    """
    Manages connections to various database backends.
    
    Supports PostgreSQL, MongoDB, and Redis.
    """
    
    def __init__(
        self,
        db_type: str = "postgresql",
        connection_string: Optional[str] = None,
        pool_size: int = 10,
        max_overflow: int = 20
    ):
        """
        Initialize database connection.
        
        Args:
            db_type: Type of database ('postgresql', 'mongodb', 'redis')
            connection_string: Database connection string
            pool_size: Connection pool size
            max_overflow: Maximum overflow connections
        """
        self.db_type = db_type
        self.connection_string = connection_string or self._get_default_connection()
        
        # Initialize connection based on type
        if db_type == "postgresql":
            self._init_postgresql(pool_size, max_overflow)
        elif db_type == "mongodb":
            self._init_mongodb()
        elif db_type == "redis":
            self._init_redis()
        else:
            raise ValueError(f"Unsupported database type: {db_type}")
    
    def _get_default_connection(self) -> str:
        """Get default connection string from environment."""
        if self.db_type == "postgresql":
            return os.getenv(
                "DATABASE_URL",
                "postgresql://user:password@localhost:5432/robo_rlhf"
            )
        elif self.db_type == "mongodb":
            return os.getenv(
                "MONGODB_URL",
                "mongodb://localhost:27017/robo_rlhf"
            )
        elif self.db_type == "redis":
            return os.getenv(
                "REDIS_URL",
                "redis://localhost:6379/0"
            )
        return ""
    
    def _init_postgresql(self, pool_size: int, max_overflow: int) -> None:
        """Initialize PostgreSQL connection."""
        # Create SQLAlchemy engine
        self.engine = create_engine(
            self.connection_string,
            pool_size=pool_size,
            max_overflow=max_overflow,
            pool_pre_ping=True,  # Verify connections before using
            echo=False  # Set to True for SQL debugging
        )
        
        # Create session factory
        self.SessionLocal = sessionmaker(
            autocommit=False,
            autoflush=False,
            bind=self.engine
        )
        
        # Create tables if they don't exist
        Base.metadata.create_all(bind=self.engine)
    
    def _init_mongodb(self) -> None:
        """Initialize MongoDB connection."""
        self.client = MongoClient(self.connection_string)
        
        # Extract database name from connection string
        db_name = self.connection_string.split("/")[-1].split("?")[0]
        self.database = self.client[db_name or "robo_rlhf"]
        
        # Create indexes
        self._create_mongodb_indexes()
    
    def _init_redis(self) -> None:
        """Initialize Redis connection."""
        self.redis_client = redis.from_url(
            self.connection_string,
            decode_responses=True
        )
        
        # Test connection
        self.redis_client.ping()
    
    def _create_mongodb_indexes(self) -> None:
        """Create MongoDB indexes for better performance."""
        # Demonstrations collection
        demos = self.database.demonstrations
        demos.create_index([("episode_id", pymongo.ASCENDING)], unique=True)
        demos.create_index([("timestamp", pymongo.DESCENDING)])
        demos.create_index([("success", pymongo.ASCENDING)])
        
        # Preferences collection
        prefs = self.database.preferences
        prefs.create_index([("pair_id", pymongo.ASCENDING)], unique=True)
        prefs.create_index([("annotator_id", pymongo.ASCENDING)])
        prefs.create_index([("timestamp", pymongo.DESCENDING)])
        
        # Training runs collection
        runs = self.database.training_runs
        runs.create_index([("run_id", pymongo.ASCENDING)], unique=True)
        runs.create_index([("start_time", pymongo.DESCENDING)])
        runs.create_index([("status", pymongo.ASCENDING)])
    
    @contextmanager
    def get_session(self) -> Generator[Session, None, None]:
        """
        Get database session (PostgreSQL).
        
        Yields:
            Database session
        """
        if self.db_type != "postgresql":
            raise ValueError("Sessions only available for PostgreSQL")
        
        session = self.SessionLocal()
        try:
            yield session
            session.commit()
        except Exception:
            session.rollback()
            raise
        finally:
            session.close()
    
    def get_mongodb_database(self) -> Database:
        """Get MongoDB database instance."""
        if self.db_type != "mongodb":
            raise ValueError("MongoDB database only available for MongoDB connection")
        return self.database
    
    def get_redis_client(self) -> redis.Redis:
        """Get Redis client instance."""
        if self.db_type != "redis":
            raise ValueError("Redis client only available for Redis connection")
        return self.redis_client
    
    def execute_query(self, query: str, params: Optional[dict] = None) -> list:
        """
        Execute raw SQL query (PostgreSQL only).
        
        Args:
            query: SQL query string
            params: Query parameters
            
        Returns:
            Query results
        """
        if self.db_type != "postgresql":
            raise ValueError("Raw queries only supported for PostgreSQL")
        
        with self.engine.connect() as conn:
            result = conn.execute(query, params or {})
            return result.fetchall()
    
    def health_check(self) -> bool:
        """
        Check if database connection is healthy.
        
        Returns:
            True if connection is healthy
        """
        try:
            if self.db_type == "postgresql":
                with self.engine.connect() as conn:
                    conn.execute("SELECT 1")
            elif self.db_type == "mongodb":
                self.client.admin.command('ping')
            elif self.db_type == "redis":
                self.redis_client.ping()
            return True
        except Exception:
            return False
    
    def close(self) -> None:
        """Close database connections."""
        if self.db_type == "postgresql":
            self.engine.dispose()
        elif self.db_type == "mongodb":
            self.client.close()
        elif self.db_type == "redis":
            self.redis_client.close()


# Global database connection instance
_db_connection: Optional[DatabaseConnection] = None


def init_db(
    db_type: str = "postgresql",
    connection_string: Optional[str] = None
) -> DatabaseConnection:
    """
    Initialize global database connection.
    
    Args:
        db_type: Type of database
        connection_string: Connection string
        
    Returns:
        Database connection instance
    """
    global _db_connection
    _db_connection = DatabaseConnection(db_type, connection_string)
    return _db_connection


def get_db() -> DatabaseConnection:
    """
    Get global database connection.
    
    Returns:
        Database connection instance
    """
    if _db_connection is None:
        return init_db()
    return _db_connection


class CacheManager:
    """
    Manages caching with Redis backend.
    """
    
    def __init__(self, redis_client: Optional[redis.Redis] = None):
        """
        Initialize cache manager.
        
        Args:
            redis_client: Redis client instance
        """
        self.redis = redis_client or get_db().get_redis_client()
        self.default_ttl = 3600  # 1 hour
    
    def get(self, key: str) -> Optional[str]:
        """Get value from cache."""
        return self.redis.get(key)
    
    def set(
        self,
        key: str,
        value: str,
        ttl: Optional[int] = None
    ) -> None:
        """Set value in cache with TTL."""
        self.redis.setex(
            key,
            ttl or self.default_ttl,
            value
        )
    
    def delete(self, key: str) -> None:
        """Delete key from cache."""
        self.redis.delete(key)
    
    def exists(self, key: str) -> bool:
        """Check if key exists in cache."""
        return self.redis.exists(key) > 0
    
    def invalidate_pattern(self, pattern: str) -> int:
        """
        Invalidate all keys matching pattern.
        
        Args:
            pattern: Key pattern (e.g., "demo:*")
            
        Returns:
            Number of keys deleted
        """
        keys = self.redis.keys(pattern)
        if keys:
            return self.redis.delete(*keys)
        return 0