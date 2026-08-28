PREPARE_DEBIAN_CA_CERTIFICATES_JAVA = "\n".join(
    (
        "set -e",
        "if command -v dpkg-query >/dev/null 2>&1; then",
        "  package_status=\"$(dpkg-query -W -f='${db:Status-Abbrev}' "
        "ca-certificates-java 2>/dev/null || true)\"",
        '  case "$package_status" in',
        "    i*)",
        "      install -d -m 0755 /etc/ssl/certs/java",
        '      case "$package_status" in',
        "        ii*) ;;",
        "        *) DEBIAN_FRONTEND=noninteractive dpkg --configure "
        "ca-certificates-java ;;",
        "      esac",
        "      ;;",
        "  esac",
        "fi",
        "",
    )
)
