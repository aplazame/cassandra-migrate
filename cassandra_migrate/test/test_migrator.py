from types import SimpleNamespace
from unittest.mock import Mock, call, patch

import pytest

from cassandra_migrate import (
    ConcurrentMigration,
    FailedMigration,
    InconsistentState,
    Migration,
    Migrator,
    UnknownMigration,
)
from cassandra_migrate.migrator import (
    CREATE_DB_VERSION,
    DELETE_DB_VERSION,
    FINALIZE_DB_VERSION,
)


def make_migration(name):
    return Migration(
        path=f"/migrations/{name}",
        name=name,
        is_python=False,
        content=f"CREATE TABLE {name};",
        checksum=name.encode(),
    )


def make_version(version, migration, state=Migration.State.SUCCEEDED):
    return SimpleNamespace(
        id=f"id-{version}",
        version=version,
        name=migration.name,
        content=migration.content,
        checksum=migration.checksum,
        state=state,
    )


def make_migrator(migrations, versions=()):
    migrator = Migrator.__new__(Migrator)
    migrator.config = SimpleNamespace(
        migrations=migrations,
        keyspace="test_keyspace",
        migrations_table="migrations",
        migrations_path="/migrations",
    )
    migrator.cluster = Mock()
    migrator._session = Mock()
    migrator._execute = Mock(return_value=list(versions))
    return migrator


def test_get_target_version_resolves_latest_numeric_and_name():
    migrations = [make_migration("v001_first.cql"),
                  make_migration("v002_second.cql")]
    migrator = make_migrator(migrations)

    assert migrator._get_target_version(None) == 2
    assert migrator._get_target_version(1) == 1
    assert migrator._get_target_version("2") == 2
    assert migrator._get_target_version("v002_second.cql") == 2


@pytest.mark.parametrize("target", [0, -1, "missing.cql"])
def test_get_target_version_rejects_invalid_target(target):
    migrator = make_migrator([make_migration("v001_first.cql")])

    with pytest.raises(ValueError, match="Invalid database version"):
        migrator._get_target_version(target)


def test_verify_migrations_returns_pending_versions_in_order():
    first = make_migration("v001_first.cql")
    second = make_migration("v002_second.cql")
    migrator = make_migrator([first, second], [make_version(1, first)])

    last_version, current_versions, pending = migrator._verify_migrations(
        [first, second])

    assert last_version == 1
    assert current_versions == [make_version(1, first)]
    assert pending == [(2, second)]


def test_verify_migrations_returns_all_as_pending_for_empty_database():
    first = make_migration("v001_first.cql")
    second = make_migration("v002_second.cql")
    migrator = make_migrator([first, second])

    assert migrator._verify_migrations([first, second]) == (
        None, [], [(1, first), (2, second)])


@pytest.mark.parametrize(
    ("state", "error"),
    [
        (Migration.State.FAILED, FailedMigration),
        (Migration.State.IN_PROGRESS, ConcurrentMigration),
    ],
)
def test_verify_migrations_rejects_failed_or_in_progress_versions(state, error):
    migration = make_migration("v001_first.cql")
    migrator = make_migrator(
        [migration], [make_version(1, migration, state=state)])

    with pytest.raises(error):
        migrator._verify_migrations([migration])


@pytest.mark.parametrize(
    ("state", "ignore_option", "expected_last_version", "expected_pending"),
    [
        (Migration.State.FAILED, "ignore_failed", None, 1),
        (Migration.State.IN_PROGRESS, "ignore_concurrent", 1, 2),
    ],
)
def test_verify_migrations_can_ignore_failed_or_concurrent_version(
        state, ignore_option, expected_last_version, expected_pending):
    migrations = [make_migration("v001_first.cql"),
                  make_migration("v002_second.cql")]
    versions = [make_version(1, migrations[0], state=state)]
    migrator = make_migrator(migrations, versions)

    result = migrator._verify_migrations(
        migrations, **{ignore_option: True})

    assert result[0] == expected_last_version
    assert result[2] == list(enumerate(
        migrations[expected_pending - 1:], expected_pending))


def test_verify_migrations_rejects_database_versions_without_files():
    migration = make_migration("v001_first.cql")
    migrator = make_migrator(
        [migration],
        [make_version(1, migration), make_version(2, make_migration("v002"))],
    )

    with pytest.raises(UnknownMigration):
        migrator._verify_migrations([migration])


def test_verify_migrations_rejects_changed_migration_content():
    configured = make_migration("v001_first.cql")
    stored = make_version(1, configured)
    changed = configured._replace(content="DROP TABLE first;")
    migrator = make_migrator([changed], [stored])

    with pytest.raises(InconsistentState):
        migrator._verify_migrations([changed])


def test_advance_applies_only_through_target_version():
    migrations = [make_migration(f"v00{version}.cql")
                  for version in range(1, 4)]
    migrator = make_migrator(migrations)
    migrator._apply_migration = Mock()

    migrator._advance(list(enumerate(migrations, 1)), 2, [])

    assert migrator._session.execute.call_args_list == [call("USE test_keyspace;")]
    assert migrator._apply_migration.call_args_list == [
        call(1, migrations[0], skip=False),
        call(2, migrations[1], skip=False),
    ]
    migrator.cluster.refresh_schema_metadata.assert_called_once_with()


@pytest.mark.parametrize(
    ("skip", "expected_state"),
    [
        (False, Migration.State.SUCCEEDED),
        (True, Migration.State.SKIPPED),
    ],
)
def test_apply_migration_executes_or_skips_and_finalizes(skip, expected_state):
    migration = make_migration("v001_first.cql")
    migrator = make_migrator([migration])
    migrator._create_version = Mock(return_value="version-id")
    migrator._apply_cql_migration = Mock()
    migrator._execute.return_value = [SimpleNamespace(applied=True)]

    with patch("cassandra_migrate.migrator.sys.path", []):
        migrator._apply_migration(1, migration, skip=skip)

    if skip:
        migrator._apply_cql_migration.assert_not_called()
    else:
        migrator._apply_cql_migration.assert_called_once_with(1, migration)
    migrator._execute.assert_called_once_with(
        migrator._q(FINALIZE_DB_VERSION),
        (expected_state, "version-id", Migration.State.IN_PROGRESS),
    )


def test_apply_migration_marks_failed_when_execution_raises():
    migration = make_migration("v001_first.cql")
    migrator = make_migrator([migration])
    migrator._create_version = Mock(return_value="version-id")
    migrator._apply_cql_migration = Mock(side_effect=RuntimeError("failed"))
    migrator._execute.return_value = [SimpleNamespace(applied=True)]

    with patch("cassandra_migrate.migrator.sys.path", []):
        with pytest.raises(FailedMigration):
            migrator._apply_migration(1, migration)

    migrator._execute.assert_called_once_with(
        migrator._q(FINALIZE_DB_VERSION),
        (Migration.State.FAILED, "version-id", Migration.State.IN_PROGRESS),
    )


def test_create_version_returns_id_when_compare_and_set_is_applied():
    migration = make_migration("v001_first.cql")
    migrator = make_migrator([migration])
    version_id = object()
    migrator._execute.return_value = [SimpleNamespace(applied=True)]

    with patch("cassandra_migrate.migrator.uuid.uuid4",
               return_value=version_id):
        result = migrator._create_version(1, migration)

    assert result is version_id
    migrator._execute.assert_called_once_with(
        migrator._q(CREATE_DB_VERSION),
        (version_id, 1, migration.name, migration.content,
         bytearray(migration.checksum), Migration.State.IN_PROGRESS),
    )


@pytest.mark.parametrize("result", [[], [SimpleNamespace(applied=False)]])
def test_create_version_rejects_unapplied_compare_and_set(result):
    migration = make_migration("v001_first.cql")
    migrator = make_migrator([migration])
    migrator._execute.return_value = result

    with pytest.raises(ConcurrentMigration):
        migrator._create_version(1, migration)


def test_cleanup_previous_versions_deletes_failed_version():
    migration = make_migration("v001_first.cql")
    failed_version = make_version(
        1, migration, state=Migration.State.FAILED)
    migrator = make_migrator([migration])
    migrator._execute.return_value = [SimpleNamespace(applied=True)]

    migrator._cleanup_previous_versions([failed_version])

    migrator._execute.assert_called_once_with(
        migrator._q(DELETE_DB_VERSION),
        (failed_version.id, Migration.State.FAILED),
    )


def test_cleanup_previous_versions_rejects_concurrent_delete():
    migration = make_migration("v001_first.cql")
    failed_version = make_version(
        1, migration, state=Migration.State.FAILED)
    migrator = make_migrator([migration])
    migrator._execute.return_value = [SimpleNamespace(applied=False)]

    with pytest.raises(ConcurrentMigration):
        migrator._cleanup_previous_versions([failed_version])


def test_cleanup_previous_versions_does_not_delete_succeeded_version():
    migration = make_migration("v001_first.cql")
    migrator = make_migrator([migration])

    migrator._cleanup_previous_versions([make_version(1, migration)])

    migrator._execute.assert_not_called()


def test_apply_cql_migration_wraps_session_errors():
    migration = make_migration("v001_first.cql")._replace(
        content="CREATE TABLE first; CREATE TABLE second;")
    migrator = make_migrator([migration])
    migrator._session.execute.side_effect = [
        None, RuntimeError("database unavailable")]

    with pytest.raises(FailedMigration) as error:
        migrator._apply_cql_migration(1, migration)

    assert error.value.version == 1
    assert error.value.migration_name == migration.name
    assert migrator._session.execute.call_args_list == [
        call("CREATE TABLE first"),
        call("CREATE TABLE second"),
    ]
