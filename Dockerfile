FROM apache/age:release_PG16_1.6.0

RUN apt-get update && apt-get install -y ca-certificates

RUN update-ca-certificates \
&& apt-get install -y --no-install-recommends git build-essential postgresql-server-dev-16 \
&& git clone --depth 1 https://github.com/pgvector/pgvector.git /tmp/pgvector \
&& cd /tmp/pgvector \
&& make PG_CONFIG=/usr/lib/postgresql/16/bin/pg_config \
&& make install \
&& cd / \
&& rm -rf /tmp/pgvector \
&& apt-get purge -y --auto-remove git build-essential postgresql-server-dev-16 \
&& rm -rf /var/lib/apt/lists/*

COPY init/01_enable_extensions.sql /docker-entrypoint-initdb.d/01_enable_extensions.sql
