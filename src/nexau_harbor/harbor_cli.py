from __future__ import annotations

from nexau_harbor.harbor_compat import install_harbor_task_directory_permissions_patch


def main() -> None:
    """Run Harbor with the repository's narrow 0.21 compatibility fixes."""

    install_harbor_task_directory_permissions_patch()

    from harbor.cli.main import app

    app()


if __name__ == "__main__":
    main()
