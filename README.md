# TRM: Thematic Resonance Memory

TRM is a neural framework for understanding text through multiple semantic "frequencies" or themes. Just as a radio telescope array captures different wavelengths of electromagnetic radiation to build a complete picture of celestial objects, TRM uses multiple specialized neural projections to capture different aspects of meaning in text.

## 🌟 Key Features

TRM introduces several innovative approaches to text understanding:

- **Multiple Thematic Projections**: Rather than forcing all semantic information through a single embedding space, TRM projects text into multiple specialized semantic spaces, each tuned to different aspects of meaning.

- **Thematic Synthesis**: TRM combines information from different thematic projections using attention mechanisms, allowing it to create rich, multi-dimensional understanding of text.

- **Semantic Frequency Analysis**: The system can analyze how strongly different themes resonate within a text, providing insights into its semantic composition.

## 🚀 Quick Start

### Installation

Install TRM using pip:

```bash
pip install trm-neural
```

For development installation with additional tools:

```bash
git clone https://github.com/yourusername/trm.git
cd trm
pip install -e ".[dev]"
```

### Basic Usage

Here's a simple example of analyzing thematic patterns in text:

```python
from trm import ThemeAnalyzer

# Initialize the analyzer
analyzer = ThemeAnalyzer()

# Analyze a text sample
text = """
The quantum computer manipulates information using delicate quantum states,
while simultaneously protecting these states from environmental interference
through sophisticated error correction protocols.
"""

# Get thematic strengths
theme_patterns = analyzer.analyze_text(text)

# Print the strengths of different themes
for theme_id, strength in theme_patterns.items():
    print(f"Theme {theme_id}: {strength:.3f}")
```

## 🔬 Core Components

TRM consists of three main components:

### 1. ThematicEncoder

The encoder module implements the core neural architecture, creating multiple "thematic antennas" that detect different semantic patterns:

```python
from trm.encoder import ThematicEncoder

encoder = ThematicEncoder(
    base_model="microsoft/deberta-v3-large",
    num_themes=8,
    projection_dims=384
)
```

### 2. ThematicGenerator

The generator processes text through thematic projections and synthesizes the results:

```python
from trm.generator import ThematicGenerator

generator = ThematicGenerator(
    model=encoder,
    tokenizer=tokenizer
)
```

### 3. ThemeAnalyzer

The analyzer provides high-level tools for understanding thematic patterns:

```python
from trm.analyzer import ThemeAnalyzer

analyzer = ThemeAnalyzer()
results = analyzer.compare_texts(text1, text2)
```

## 📚 Technical Details

### The Radio Telescope Metaphor

TRM's architecture is inspired by radio telescope arrays, where different receivers capture different wavelengths of electromagnetic radiation. In TRM:

- Each thematic projection is like a specialized antenna
- Different themes capture different "frequencies" of meaning
- Thematic synthesis combines these signals into a coherent understanding

### Theme Detection

Themes are learned through a combination of:
- Contrastive learning across different semantic domains
- Attention-based synthesis of thematic signals
- Orthogonality constraints to ensure theme diversity

## 🛠 Development

### Running Tests

```bash
# Run all tests
pytest

# Run tests with coverage report
pytest --cov=trm tests/
```

### Code Style

We use black for code formatting and isort for import sorting:

```bash
# Format code
black trm tests
isort trm tests
```

## 📝 Citation

If you use TRM in your research, please cite:

```bibtex
@software{trm2024,
  title={TRM: Thematic Resonance Memory},
  author={Richard Puckett},
  year={2024},
  url={https://github.com/rapuckett/trm}
}
```

## 🤝 Contributing

We welcome contributions! Please see our [Contributing Guide](CONTRIBUTING.md) for details on how to get started.

## 📄 License

TRM is released under the MIT License. See the LICENSE file for more details.

## 🙏 Acknowledgments

TRM builds upon ideas from several fields:
- Sparse Distributed Memory (Kanerva, 1988)
- Modern transformer architectures
- Radio interferometry techniques
- Cognitive science theories of semantic memory

We thank the open source community for their valuable tools and libraries that make this work possible.