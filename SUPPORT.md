<!-- SPDX-FileCopyrightText: 2024 Joshua Jewell -->
<!-- SPDX-License-Identifier: MIT -->

# Support

This document describes how to get help with the VAE Dataset Normalizer.

## Getting Help

### Documentation

Start with the documentation:

- [README.adoc](README.adoc) - Overview and quick start
- [QUICKSTART.md](QUICKSTART.md) - Step-by-step guide
- [CONTRIBUTING.adoc](CONTRIBUTING.adoc) - Contribution guidelines

### Issue Tracker

For bugs and feature requests, use the issue tracker:

- **GitLab**: https://gitlab.com/hyperpolymath/zerostep/-/issues

Before opening an issue:

1. Search existing issues to avoid duplicates
2. Include reproduction steps for bugs
3. Provide system information (OS, Rust version, etc.)

### Security Issues

For security vulnerabilities, **do not** open a public issue.

See [SECURITY.md](SECURITY.md) for responsible disclosure procedures.

## Support Channels

| Channel | Response Time | Use For |
|---------|---------------|---------|
| Issue tracker | 1-7 days | Bugs, features, documentation |
| Email | 7-14 days | Private inquiries |
| Security email | 48 hours | Vulnerabilities |

## What We Support

### Supported Versions

| Version | Status | Support |
|---------|--------|---------|
| 1.x | Current | Full |
| 0.x | Legacy | Security only |

### Supported Platforms

- **Rust**: 1.70+
- **Julia**: 1.9+
- **Isabelle**: 2023+
- **OS**: Linux, macOS, Windows (via WSL2)
- **Containers**: Podman with Chainguard Wolfi

## Self-Help Resources

### Common Issues

**Build fails with missing dependencies**:
```bash
# Check all dependencies
just check-deps

# Install with Nix (recommended)
nix develop
```

**Checksum verification fails**:
```bash
# Verify dataset integrity
vae-normalizer verify -o /path/to/output --checksums -d /path/to/dataset
```

**Julia package errors**:
```bash
# Reinstall Julia dependencies
just julia-setup
```

### Debug Mode

Enable verbose output for troubleshooting:

```bash
vae-normalizer -v normalize -d /path/to/dataset -o /path/to/output
```

## Community

This is a research project maintained by volunteers. Please be patient
and respectful in all interactions.

See [CODE_OF_CONDUCT.md](CODE_OF_CONDUCT.md) for community guidelines.
