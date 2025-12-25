-- Enable required PostgreSQL extensions
CREATE EXTENSION IF NOT EXISTS vector;
CREATE EXTENSION IF NOT EXISTS age;

-- Load AGE extension into the search path
LOAD 'age';
SET search_path = ag_catalog, "$user", public;
