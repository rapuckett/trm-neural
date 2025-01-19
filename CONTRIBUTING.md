# Contributing to TRM

Thank you for your interest in contributing to the Thematic Resonance Memory (TRM) project! We've designed TRM to be a collaborative effort, and we're excited to welcome new contributors who share our vision of advancing theme-based text understanding.

## Getting Started

The first step in contributing is understanding how TRM works. We recommend reading through our README.md and trying out the basic examples. Our approach to text understanding is based on the metaphor of a radio telescope array, where different "antennas" detect different aspects of meaning. Familiarizing yourself with this concept will help you understand the codebase better.

### Development Environment Setup

First, fork the repository and clone your fork:

```bash
git clone https://github.com/rapuckett/trm.git
cd trm
```

Create and activate a virtual environment:

```bash
python -m venv venv
# On Windows:
venv\Scripts\activate
# On Unix or MacOS:
source venv/bin/activate
```

Install development dependencies:

```bash
pip install -e ".[dev]"
```

## Development Workflow

We follow a standard GitHub flow:

1. Create a new branch for your feature:
   ```bash
   git checkout -b feature/your-feature-name
   ```

2. Make your changes, following our code style guidelines.

3. Run the test suite:
   ```bash
   pytest
   ```

4. Format your code:
   ```bash
   black trm tests
   isort trm tests
   ```

5. Commit your changes using conventional commit messages:
   ```
   feat: add new thematic projection capability
   fix: correct theme strength calculation
   docs: improve encoder documentation
   test: add tests for theme synthesis
   ```

6. Push to your fork and submit a pull request.

## Code Style Guidelines

We maintain high standards for code quality to ensure TRM remains maintainable and reliable:

### Python Style

- Follow PEP 8 conventions
- Use type hints for all function arguments and return values
- Keep functions focused and under 50 lines where possible
- Use descriptive variable names that reflect our radio telescope metaphor

Example of good style:

```python
def analyze_theme_strength(
    self,
    text: str,
    theme_idx: int
) -> float:
    """
    Analyze how strongly a specific theme resonates in the input text.

    Args:
        text: The input text to analyze
        theme_idx: Index of the theme "antenna" to use

    Returns:
        Normalized strength of the theme's resonance (0.0 to 1.0)
    """
    theme_projection = self._project_to_theme(text, theme_idx)
    return self._calculate_resonance_strength(theme_projection)
```

### Documentation

- All public functions must have docstrings following Google style
- Include examples in docstrings when behavior isn't obvious
- Keep the radio telescope metaphor consistent in documentation
- Update relevant documentation when changing functionality

## Testing

We take testing seriously:

- Write tests for all new functionality
- Maintain or improve code coverage (currently at X%)
- Include both unit tests and integration tests
- Test edge cases and potential failure modes
- Use pytest fixtures for common test setups

Example test:

```python
def test_theme_orthogonality():
    """Ensure different themes capture distinct semantic patterns."""
    encoder = ThematicEncoder(num_themes=2)
    text = "Sample text for theme analysis"
    
    theme1_strength = encoder.analyze_theme_strength(text, 0)
    theme2_strength = encoder.analyze_theme_strength(text, 1)
    
    # Themes should respond differently to the same input
    assert abs(theme1_strength - theme2_strength) > 0.1
```

## Pull Request Process

1. Ensure your PR includes:
   - A clear description of the changes
   - Any updates to documentation
   - New or updated tests
   - A note about what type of change it is (feature/bugfix/etc.)

2. Wait for CI checks to pass

3. Address any review comments promptly

4. Once approved, maintainers will merge your PR

## Community Guidelines

We strive to maintain a welcoming and inclusive community:

- Be respectful and professional in all interactions
- Welcome newcomers and help them get started
- Focus on constructive feedback in code reviews
- Credit others' ideas and contributions
- Raise issues early to discuss major changes

## Getting Help

- Check existing issues and discussions before creating new ones
- Join our community chat for quick questions
- Tag issues appropriately (#help-wanted, #good-first-issue, etc.)
- Reach out to maintainers if you're stuck

## Recognition

We value all contributions and maintain an AUTHORS file listing contributors. Significant contributions may also be highlighted in release notes.

Remember, whether you're fixing a typo in documentation or implementing a major feature, all contributions are valuable to the project!

## Questions?

If anything in this guide isn't clear, please open an issue with the label #contributing-guide. We're happy to help and will use your feedback to improve this document.

Thank you for contributing to TRM! 🚀