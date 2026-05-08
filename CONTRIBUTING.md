# Contributing to Nexus AI Studio

Thank you for considering contributing to **Nexus AI Studio**! 🎉

We welcome contributions of all kinds — bug fixes, new features, documentation improvements, and more.

## 📋 Table of Contents

- [Code of Conduct](#code-of-conduct)
- [Getting Started](#getting-started)
- [Development Setup](#development-setup)
- [Making Changes](#making-changes)
- [Pull Request Process](#pull-request-process)
- [Style Guidelines](#style-guidelines)
- [Reporting Bugs](#reporting-bugs)
- [Suggesting Features](#suggesting-features)

## 📜 Code of Conduct

This project adheres to the [Code of Conduct](CODE_OF_CONDUCT.md). By participating, you are expected to uphold this code.

## 🚀 Getting Started

1. **Fork** the repository on GitHub
2. **Clone** your fork locally:
   ```bash
   git clone https://github.com/YOUR-USERNAME/nexus-ai-studio.git
   cd nexus-ai-studio
   ```
3. **Create a branch** for your changes:
   ```bash
   git checkout -b feature/your-feature-name
   ```

## 🛠️ Development Setup

```bash
# Create virtual environment
python -m venv venv

# Activate it
# Windows:
venv\Scripts\activate
# macOS/Linux:
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Run the app locally
streamlit run app.py
```

## ✏️ Making Changes

1. Make your changes in a dedicated branch
2. Test your changes locally by running the Streamlit app
3. Ensure your code follows the project's style guidelines
4. Write clear, concise commit messages

### Commit Message Format

```
<type>: <short description>

<optional longer description>
```

**Types:**
- `feat` — new feature
- `fix` — bug fix
- `docs` — documentation changes
- `style` — formatting, missing semi-colons, etc.
- `refactor` — code restructuring
- `test` — adding tests
- `chore` — maintenance tasks

## 🔄 Pull Request Process

1. Update the `README.md` if your changes affect the documentation
2. Update the `CHANGELOG.md` with notes under the `[Unreleased]` section
3. Ensure the app runs without errors
4. Submit your pull request using the [PR template](.github/pull_request_template.md)
5. Request a review from a maintainer

## 🎨 Style Guidelines

### Python
- Follow **PEP 8** conventions
- Use meaningful variable names
- Add comments for complex logic
- Keep functions focused and concise

### CSS (Streamlit Custom Styles)
- Use CSS custom properties (variables) defined in the `:root` selector
- Follow the existing naming convention (`nx-*` prefix for custom components)
- Maintain the dark theme color palette

### UI Components
- Use the `sec()` helper for section headers
- Follow the existing layout patterns (columns, tabs, expanders)
- Maintain consistency with the custom badge and score components

## 🐛 Reporting Bugs

Use the [Bug Report template](.github/ISSUE_TEMPLATE/bug_report.md) and include:

1. A clear description of the bug
2. Steps to reproduce
3. Expected vs actual behavior
4. Screenshots if applicable
5. Your environment details (OS, Python version, browser)

## 💡 Suggesting Features

Use the [Feature Request template](.github/ISSUE_TEMPLATE/feature_request.md) and include:

1. A clear description of the feature
2. The problem it solves
3. Proposed implementation approach (if any)
4. Any alternatives you've considered

---

**Thank you for making Nexus AI Studio better!** ⬡
