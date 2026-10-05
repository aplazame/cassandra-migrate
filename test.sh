#!/bin/sh

set -e

coverage erase
coverage run --source cassandra_migrate -m pytest
coverage report --include='cassandra_migrate/**' --omit='cassandra_migrate/test/**'
