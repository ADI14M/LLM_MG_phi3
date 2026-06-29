import psycopg2
from psycopg2 import pool
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import DB_HOST, DB_NAME, DB_USER, DB_PASSWORD, DB_PORT

class DatabasePool:
    _pool = None

    @classmethod
    def initialize(cls):
        if cls._pool is None:
            try:
                cls._pool = psycopg2.pool.SimpleConnectionPool(
                    1, 20,
                    host=DB_HOST,
                    database=DB_NAME,
                    user=DB_USER,
                    password=DB_PASSWORD,
                    port=DB_PORT,
                    connect_timeout=10
                )
            except Exception as e:
                print(f"[DatabasePool] Failed to initialize connection pool: {e}")

    @classmethod
    def get_connection(cls):
        if cls._pool is None:
            cls.initialize()
        return cls._pool.getconn()

    @classmethod
    def release_connection(cls, conn):
        if cls._pool and conn:
            cls._pool.putconn(conn)
