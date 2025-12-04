# Security Policy

## Supported Versions

| Version | Supported          |
| ------- | ------------------ |
| 1.x.x   | :white_check_mark: |
| < 1.0   | :x:                |

## Reporting a Vulnerability

We take security seriously. If you discover a security vulnerability, please report it responsibly.

### Reporting Channel

- **Preferred**: Open a security advisory on the repository
- **Alternative**: Email security concerns to the maintainers listed in MAINTAINERS.md
- **Do NOT**: Open public issues for security vulnerabilities

### Response SLA

- **Acknowledgement**: Within 24 hours
- **Initial Assessment**: Within 72 hours
- **Fix Timeline**: Depends on severity
  - Critical: 7 days
  - High: 14 days
  - Medium: 30 days
  - Low: 90 days

### What to Include

1. Description of the vulnerability
2. Steps to reproduce
3. Potential impact assessment
4. Suggested fix (if available)
5. Your contact information

### Security Measures

This project implements the following security practices:

#### Type Safety
- Written in Rust with compile-time type checking
- No unsafe code blocks without explicit justification
- Memory safety guaranteed by ownership model

#### Cryptographic Integrity
- SHAKE256 (d=256) for file checksums
- FIPS 202 compliant implementation
- No custom cryptography

#### Supply Chain Security
- SPDX headers on all source files
- Pinned dependencies (no floating versions)
- Regular dependency audits via `cargo audit`
- SBOM generation available

#### Container Security
- Chainguard Wolfi base images (minimal attack surface)
- Non-root container execution
- No privileged operations required
- Podman (rootless) preferred over Docker

### Security Headers (if web-facing)

Not applicable - this is a CLI tool.

### Disclosure Policy

We follow coordinated disclosure:

1. Reporter submits vulnerability
2. We acknowledge within 24 hours
3. We assess and develop fix
4. Reporter is credited (unless anonymity requested)
5. Public disclosure after fix is released

### Security Audit History

| Date       | Auditor    | Scope              | Findings |
| ---------- | ---------- | ------------------ | -------- |
| 2024-01-01 | Self-audit | Full codebase      | N/A      |

### Hall of Fame

Security researchers who have responsibly disclosed vulnerabilities:

- (None yet - be the first!)
