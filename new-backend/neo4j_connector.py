import os
import logging
from typing import Dict, List, Any, Optional

import pandas as pd
from neo4j import GraphDatabase, basic_auth
from neo4j.exceptions import ServiceUnavailable, AuthError

logger = logging.getLogger(__name__)


class Neo4jConnector:
    def __init__(self, uri: str = None, user: str = None, password: str = None):
        self.uri = uri or os.getenv('NEO4J_URI', 'bolt://localhost:7687')
        self.user = user or os.getenv('NEO4J_USER', 'neo4j')
        self.password = password or os.getenv('NEO4J_PASSWORD', 'clustering123')

        self.driver = None
        self._connect()

    def _connect(self):
        try:
            self.driver = GraphDatabase.driver(
                self.uri,
                auth=basic_auth(self.user, self.password),
                max_connection_lifetime=3600,
                max_connection_pool_size=50,
                connection_acquisition_timeout=60
            )

            with self.driver.session() as session:
                session.run("RETURN 1")

            logger.info(f"Successfully connected to Neo4j at {self.uri}")
            self._initialize_database()

        except ServiceUnavailable as e:
            logger.error(f"Failed to connect to Neo4j: {e}")
            raise
        except AuthError as e:
            logger.error(f"Authentication failed for Neo4j: {e}")
            raise
        except Exception as e:
            logger.error(f"Unexpected error connecting to Neo4j: {e}")
            raise

    def _initialize_database(self):
        try:
            with self.driver.session() as session:
                constraints = [
                    "CREATE CONSTRAINT app_name_unique IF NOT EXISTS FOR (a:App) REQUIRE a.name IS UNIQUE",
                    "CREATE CONSTRAINT review_id_unique IF NOT EXISTS FOR (r:Review) REQUIRE r.id IS UNIQUE",
                    "CREATE CONSTRAINT feature_text_unique IF NOT EXISTS FOR (f:Feature) REQUIRE f.text IS UNIQUE",
                    "CREATE CONSTRAINT cluster_id_app_unique IF NOT EXISTS FOR (c:Cluster) REQUIRE (c.id, c.app_name) IS UNIQUE"
                ]

                for constraint in constraints:
                    try:
                        session.run(constraint)
                    except Exception as e:
                        logger.debug(f"Constraint might already exist: {e}")

                # Create indexes for better performance
                indexes = [
                    "CREATE INDEX app_name_index IF NOT EXISTS FOR (a:App) ON (a.name)",
                    "CREATE INDEX review_score_index IF NOT EXISTS FOR (r:Review) ON (r.score)",
                    "CREATE INDEX review_date_index IF NOT EXISTS FOR (r:Review) ON (r.date)",
                    "CREATE INDEX feature_text_index IF NOT EXISTS FOR (f:Feature) ON (f.text)",
                    "CREATE INDEX cluster_name_index IF NOT EXISTS FOR (c:Cluster) ON (c.name)",
                    "CREATE INDEX cluster_app_index IF NOT EXISTS FOR (c:Cluster) ON (c.app_name)"
                ]

                for index in indexes:
                    try:
                        session.run(index)
                    except Exception as e:
                        logger.debug(f"Index might already exist: {e}")

                logger.info("Database schema initialized successfully")

        except Exception as e:
            logger.error(f"Failed to initialize database schema: {e}")
            raise

    def test_connection(self) -> bool:
        try:
            with self.driver.session() as session:
                result = session.run("RETURN 1 as test")
                return result.single()["test"] == 1
        except Exception as e:
            logger.error(f"Connection test failed: {e}")
            return False

    def execute_read_query(self, query: str, parameters: Dict = None) -> List[Dict]:
        try:
            with self.driver.session() as session:
                result = session.run(query, parameters or {})
                return [record.data() for record in result]
        except Exception as e:
            logger.error(f"Read query failed: {query}, Error: {e}")
            raise

    def execute_write_query(self, query: str, parameters: Dict = None) -> Any:
        try:
            with self.driver.session() as session:
                result = session.run(query, parameters or {})
                return result.single()
        except Exception as e:
            logger.error(f"Write query failed: {query}, Error: {e}")
            raise

    def execute_write_transaction(self, queries: List[Dict]) -> bool:
        try:
            def _transaction_function(tx):
                for query_info in queries:
                    query = query_info['query']
                    parameters = query_info.get('parameters', {})
                    tx.run(query, parameters)

            with self.driver.session() as session:
                session.execute_write(_transaction_function)

            return True

        except Exception as e:
            logger.error(f"Transaction failed: {e}")
            raise

    def get_database_stats(self) -> Dict[str, Any]:
        try:
            queries = {
                'apps_count': "MATCH (a:App) RETURN count(a) as count",
                'reviews_count': "MATCH (r:Review) RETURN count(r) as count",
                'features_count': "MATCH (f:Feature) RETURN count(f) as count",
                'clusters_count': "MATCH (c:Cluster) RETURN count(c) as count",
                'relationships_count': "MATCH ()-[r]->() RETURN count(r) as count"
            }

            stats = {}
            for stat_name, query in queries.items():
                result = self.execute_read_query(query)
                stats[stat_name] = result[0]['count'] if result else 0

            return stats

        except Exception as e:
            logger.error(f"Failed to get database stats: {e}")
            return {}

    def clear_app_data(self, app_name: str) -> bool:
        try:
            query = """
            MATCH (a:App {name: $app_name})
            OPTIONAL MATCH (a)-[:HAS_REVIEW]->(r:Review)
            OPTIONAL MATCH (a)-[:HAS_FEATURE]->(f:Feature)
            OPTIONAL MATCH (a)-[:HAS_CLUSTER]->(c:Cluster)
            OPTIONAL MATCH (c)-[:CONTAINS_FEATURE]->(cf:Feature)
            DETACH DELETE a, r, f, c
            """

            self.execute_write_query(query, {'app_name': app_name})
            logger.info(f"Cleared all data for app: {app_name}")
            return True

        except Exception as e:
            logger.error(f"Failed to clear app data: {e}")
            return False

    def get_node_count(self, label: str) -> int:
        try:
            query = f"MATCH (n:{label}) RETURN count(n) as count"
            result = self.execute_read_query(query)
            return result[0]['count'] if result else 0
        except Exception as e:
            logger.error(f"Failed to get node count for {label}: {e}")
            return 0

    def get_relationship_count(self, rel_type: str = None) -> int:
        try:
            if rel_type:
                query = f"MATCH ()-[r:{rel_type}]->() RETURN count(r) as count"
            else:
                query = "MATCH ()-[r]->() RETURN count(r) as count"

            result = self.execute_read_query(query)
            return result[0]['count'] if result else 0
        except Exception as e:
            logger.error(f"Failed to get relationship count: {e}")
            return 0

    def health_check(self) -> Dict[str, Any]:
        health_status = {
            'connection': False,
            'read_access': False,
            'write_access': False,
            'stats': {}
        }

        try:
            health_status['connection'] = self.test_connection()

            if health_status['connection']:
                try:
                    self.execute_read_query("MATCH (n) RETURN count(n) LIMIT 1")
                    health_status['read_access'] = True
                except:
                    pass

                try:
                    self.execute_write_query("CREATE (t:HealthTest {timestamp: $ts}) DELETE t",
                                             {'ts': str(pd.Timestamp.now())})
                    health_status['write_access'] = True
                except:
                    pass

                health_status['stats'] = self.get_database_stats()

        except Exception as e:
            logger.error(f"Health check failed: {e}")

        return health_status

    def close(self):
        if self.driver:
            self.driver.close()
            logger.info("Neo4j connection closed")

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
