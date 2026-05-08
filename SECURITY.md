# Security Policy

## Supported Versions

| Version | Supported          |
| ------- | ------------------ |
| 3.0.x   | ✅ Active support  |
| < 3.0   | ❌ No longer supported |

## Reporting a Vulnerability

If you discover a security vulnerability within Nexus AI Studio, please report it responsibly:

1. **DO NOT** open a public GitHub issue for security vulnerabilities
2. Email the maintainer directly or use GitHub's private vulnerability reporting feature
3. Include a detailed description of the vulnerability and steps to reproduce

### What to expect:

- **Acknowledgment** within 48 hours
- **Assessment** within 1 week
- **Fix/Patch** within 2 weeks for critical vulnerabilities

## Security Best Practices

When using Nexus AI Studio:

- **Never commit** your `.streamlit/secrets.toml` file (it's in `.gitignore` by default)
- **Use environment variables** or Streamlit secrets for API keys
- **Review datasets** before uploading — the app processes data locally
- **Keep dependencies** updated with `pip install --upgrade -r requirements.txt`

## Scope

This security policy applies to the Nexus AI Studio codebase and its dependencies.

---

Thank you for helping keep Nexus AI Studio secure! ⬡
