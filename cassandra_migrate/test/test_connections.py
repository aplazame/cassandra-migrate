from __future__ import annotations

import ssl
from types import SimpleNamespace
from unittest.mock import patch

import pytest
from cassandra import ConsistencyLevel
from cassandra.cluster import EXEC_PROFILE_DEFAULT
from cassandra.io.asyncorereactor import AsyncoreConnection

from cassandra_migrate import MigratorBundle, MigratorKeyspace
from cassandra_migrate.cli import main


@pytest.mark.parametrize("verify_mode", [ssl.CERT_REQUIRED, ssl.CERT_NONE])
def test_keyspaces_configures_tls_authentication_and_driver(verify_mode: int) -> None:
    """Configure Keyspaces without certificates, credentials or a network."""
    config = SimpleNamespace(profiles={"prod": {"replication": {}}})
    with (
        patch("cassandra_migrate.migrator.Cluster") as cluster,
        patch("cassandra_migrate.migrator.ssl.SSLContext") as context,
        patch("cassandra_sigv4.auth.SigV4AuthProvider") as auth,
    ):
        migrator = MigratorKeyspace(
            config, profile="prod", bundle_path="/cert.pem",
            region_name="eu-west-1",
            keyspaces_host="cassandra.eu-west-1.amazonaws.com",
            verify_mode=verify_mode,
        )

    context.assert_called_once_with(ssl.PROTOCOL_TLS_CLIENT)
    context.return_value.load_verify_locations.assert_called_once_with("/cert.pem")
    assert context.return_value.minimum_version == ssl.TLSVersion.TLSv1_2
    assert context.return_value.verify_mode == verify_mode
    if verify_mode == ssl.CERT_NONE:
        assert context.return_value.check_hostname is False
    auth.assert_called_once_with(region_name="eu-west-1")
    kwargs = cluster.call_args.kwargs
    assert cluster.call_args.args == (["cassandra.eu-west-1.amazonaws.com"],)
    assert kwargs["port"] == 9142
    assert kwargs["protocol_version"] == 4
    assert kwargs["connection_class"] is AsyncoreConnection
    assert kwargs["ssl_context"] is context.return_value
    assert kwargs["auth_provider"] is auth.return_value
    profile = kwargs["execution_profiles"][EXEC_PROFILE_DEFAULT]
    assert profile.consistency_level == ConsistencyLevel.LOCAL_QUORUM
    assert migrator.current_profile is config.profiles["prod"]
    cluster.return_value.connect.assert_not_called()


@pytest.mark.parametrize(
    ("arguments", "connection", "expected"),
    [
        (["-K", "/cert.pem", "--aws-region", "eu-west-1"], "MigratorKeyspace",
         {"bundle_path": "/cert.pem", "region_name": "eu-west-1",
          "keyspaces_host": "cassandra.eu-west-1.amazonaws.com",
          "verify_mode": ssl.CERT_REQUIRED}),
        (["-K", "/cert.pem", "--aws-verify-mode", "None"], "MigratorKeyspace",
         {"verify_mode": ssl.CERT_NONE}),
        (["-K", "/cert.pem", "--aws-verify-mode", "required"], "MigratorKeyspace",
         {"verify_mode": ssl.CERT_REQUIRED}),
        (["-K", "/cert.pem", "--aws-host", "custom.example"], "MigratorKeyspace",
         {"bundle_path": "/cert.pem", "region_name": "eu-central-1",
          "keyspaces_host": "custom.example"}),
        (["-b", "/bundle.zip", "-u", "client", "-P", "example"], "MigratorBundle",
         {"bundle_path": "/bundle.zip", "user": "client", "password": "example"}),
        (["-H", "first,second", "-p", "9142"], "Migrator",
         {"hosts": ["first", "second"], "port": 9142}),
    ],
)
def test_cli_routes_connection(
    arguments: list[str], connection: str, expected: dict[str, object],
) -> None:
    """Pass the YAML profile and connection arguments to the selected driver."""
    with (
        patch("sys.argv", ["cassandra-migrate", "-m", "prod", *arguments, "status"]),
        patch("cassandra_migrate.cli.MigrationConfig.load") as load,
        patch("cassandra_migrate.cli.MigratorKeyspace") as keyspaces,
        patch("cassandra_migrate.cli.MigratorBundle") as bundle,
        patch("cassandra_migrate.cli.Migrator") as normal,
    ):
        main()
    connections = {"MigratorKeyspace": keyspaces, "MigratorBundle": bundle,
                   "Migrator": normal}
    selected = connections.pop(connection)
    assert selected.call_args.kwargs["config"] is load.return_value
    assert selected.call_args.kwargs["profile"] == "prod"
    for key, value in expected.items():
        assert selected.call_args.kwargs[key] == value
    selected.return_value.__enter__.return_value.status.assert_called_once()
    for unused in connections.values():
        unused.assert_not_called()


def test_generate_does_not_create_connection() -> None:
    """Generate a migration locally without initializing any driver."""
    with (
        patch("sys.argv", ["cassandra-migrate", "generate", "new table"]),
        patch("cassandra_migrate.cli.MigrationConfig.load"),
        patch("cassandra_migrate.cli.Migration.generate", return_value="/v001.cql"),
        patch("cassandra_migrate.cli.sys.stdin.isatty", return_value=False),
        patch("cassandra_migrate.cli.Migrator") as normal,
        patch("cassandra_migrate.cli.MigratorBundle") as bundle,
        patch("cassandra_migrate.cli.MigratorKeyspace") as keyspaces,
    ):
        main()
    for connection in (normal, bundle, keyspaces):
        connection.assert_not_called()


def test_cli_rejects_conflicting_connection_modes() -> None:
    """Reject simultaneous DataStax and AWS connection options."""
    with patch("sys.argv", ["cassandra-migrate", "-b", "bundle.zip",
                            "-K", "cert.pem", "status"]):
        with pytest.raises(SystemExit) as error:
            main()
    assert error.value.code == 2


@pytest.mark.parametrize(
    "arguments",
    [
        ["--aws-verify-mode", "none"],
        ["-b", "/bundle.zip", "--aws-verify-mode", "none"],
        ["-K", "/cert.pem", "--aws-verify-mode", "invalid"],
    ],
)
def test_cli_rejects_invalid_tls_verification_options(arguments: list[str]) -> None:
    """Reject invalid modes or disabling verification outside AWS Keyspaces."""
    with patch("sys.argv", ["cassandra-migrate", *arguments, "status"]):
        with pytest.raises(SystemExit) as error:
            main()
    assert error.value.code == 2


def test_cli_warns_when_keyspaces_verification_is_disabled(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Warn explicitly when opting out of certificate and hostname verification."""
    with (
        patch("sys.argv", ["cassandra-migrate", "-K", "/cert.pem",
                           "--aws-verify-mode", "none", "status"]),
        patch("cassandra_migrate.cli.MigrationConfig.load"),
        patch("cassandra_migrate.cli.MigratorKeyspace"),
    ):
        main()
    assert "certificate and hostname verification are disabled" in caplog.text


def test_keyspaces_disables_verification_on_real_ssl_context() -> None:
    """Disable hostname checking before setting CERT_NONE on a real SSLContext."""
    config = SimpleNamespace(profiles={"prod": {}})
    with (
        patch("cassandra_migrate.migrator.Cluster") as cluster,
        patch("cassandra_migrate.migrator.ssl.SSLContext.load_verify_locations"),
        patch("cassandra_sigv4.auth.SigV4AuthProvider"),
    ):
        MigratorKeyspace(config, profile="prod", verify_mode=ssl.CERT_NONE)
    context = cluster.call_args.kwargs["ssl_context"]
    assert context.verify_mode == ssl.CERT_NONE
    assert context.check_hostname is False


def test_bundle_configures_cloud_driver_without_connecting() -> None:
    """Configure a DataStax bundle without reading it or opening a session."""
    config = SimpleNamespace(profiles={"prod": {}})
    with (
        patch("cassandra_migrate.migrator.Cluster") as cluster,
        patch("cassandra_migrate.migrator.PlainTextAuthProvider") as auth,
    ):
        migrator = MigratorBundle(
            config, profile="prod", bundle_path="/bundle.zip",
            user="client", password="example",
        )
    auth.assert_called_once_with("client", "example")
    cluster.assert_called_once_with(
        cloud={"secure_connect_bundle": "/bundle.zip"},
        auth_provider=auth.return_value,
        connection_class=AsyncoreConnection,
        protocol_version=4,
    )
    assert migrator.current_profile is config.profiles["prod"]
    cluster.return_value.connect.assert_not_called()
