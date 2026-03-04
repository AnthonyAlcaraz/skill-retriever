---
name: science-plots
description: Create publication-quality scientific figures using Python's SciencePlots library with Matplotlib. Use when generating plots for academic papers, IEEE/Nature journals, presentations, or theses. Covers style combinations, color cycles, LaTeX rendering, and journal-specific formatting.
license: MIT
metadata:
  author: AnthonyAlcaraz
  version: "1.0"
  source_post: "https://www.linkedin.com/posts/giannis-tolios_datascience-python-deeplearning-ugcPost-7426679771193028608"
compatibility: Requires Python 3.7+, matplotlib, and optionally LaTeX for full rendering.
---

# SciencePlots - Publication-Quality Scientific Figures

Generate matplotlib figures that comply with academic journal formatting standards (IEEE, Nature, Springer) using the SciencePlots library.

## Setup

```bash
pip install SciencePlots
```

For full LaTeX rendering (recommended for publications):
- Windows: Install MiKTeX or TeX Live
- macOS: `brew install --cask mactex`
- Linux: `sudo apt install texlive-full`

For no-LaTeX fallback, use the `no-latex` style.

## Core Usage Pattern

```python
import matplotlib.pyplot as plt
import scienceplots  # REQUIRED since v2.0.0

# Single style
plt.style.use('science')

# Combine styles (order matters - later styles override earlier ones)
plt.style.use(['science', 'ieee'])
```

## Available Styles

### Base Styles
| Style | Use Case |
|-------|----------|
| `science` | General scientific plotting (serif fonts, clean grid) |
| `ieee` | IEEE journal papers (single-column width, B&W compatible) |
| `nature` | Nature journal (sans-serif, specific dimensions) |
| `notebook` | Jupyter notebooks (larger fonts, wider figures) |
| `no-latex` | When LaTeX is not installed |

### Modifier Styles (combine with base)
| Style | Effect |
|-------|--------|
| `grid` | Add background grid |
| `high-vis` | Thicker lines, larger markers for presentations |
| `scatter` | Optimized for scatter plots |
| `vibrant` | Vibrant color cycle |
| `muted` | Muted color cycle |
| `bright` | Colorblind-safe bright palette |
| `retro` | Retro color scheme |

### Language Styles
| Style | Font Support |
|-------|-------------|
| `cjk-tc` | Traditional Chinese |
| `cjk-sc` | Simplified Chinese |
| `cjk-jp` | Japanese |
| `cjk-kr` | Korean |
| `russian` | Russian Cyrillic |
| `turkish` | Turkish characters |

## Common Recipes

### IEEE Paper Figure
```python
import matplotlib.pyplot as plt
import numpy as np
import scienceplots

plt.style.use(['science', 'ieee'])

fig, ax = plt.subplots()
x = np.linspace(0, 10, 100)
ax.plot(x, np.sin(x), label='Model A')
ax.plot(x, np.cos(x), label='Model B')
ax.set_xlabel('Time [s]')
ax.set_ylabel('Amplitude')
ax.legend()
fig.savefig('ieee_figure.pdf', dpi=300, bbox_inches='tight')
```

### Nature Journal Figure
```python
plt.style.use(['science', 'nature'])

fig, ax = plt.subplots()
# Nature uses sans-serif fonts automatically
ax.plot(x, y)
fig.savefig('nature_figure.pdf', dpi=300, bbox_inches='tight')
```

### Presentation (High Visibility)
```python
plt.style.use(['science', 'high-vis', 'grid'])

fig, ax = plt.subplots(figsize=(8, 6))
ax.plot(x, y, linewidth=2)
fig.savefig('presentation.png', dpi=150, bbox_inches='tight')
```

### No LaTeX Fallback
```python
plt.style.use(['science', 'no-latex'])
# Uses mathtext instead of LaTeX - no external dependency needed
```

### Colorblind-Safe Figures
```python
plt.style.use(['science', 'bright'])
# Uses Paul Tol's bright qualitative color scheme
```

### Context Manager (Temporary Style)
```python
with plt.style.context(['science', 'ieee']):
    fig, ax = plt.subplots()
    ax.plot(x, y)
    fig.savefig('temp_style.pdf')
# Style reverts after context manager exits
```

## Multi-Panel Figures

```python
plt.style.use(['science', 'ieee'])

fig, axes = plt.subplots(1, 3, figsize=(7, 2.5))
for ax, style_name in zip(axes, ['Default', 'Scatter', 'High-vis']):
    ax.plot(x, np.sin(x))
    ax.set_title(style_name)

fig.savefig('multi_panel.pdf', dpi=300, bbox_inches='tight')
```

## Output Best Practices

| Format | When to Use |
|--------|------------|
| PDF | Journal submissions (vector, lossless) |
| PNG (300 dpi) | Web/screen display |
| SVG | Editable vector graphics |
| EPS | Legacy journal requirements |

Always use `bbox_inches='tight'` to avoid whitespace cropping issues.

## Custom Style Overrides

```python
plt.style.use(['science', 'ieee'])

# Override specific parameters after loading styles
plt.rcParams.update({
    'font.size': 12,
    'axes.linewidth': 1.5,
    'figure.figsize': (6, 4),
})
```

## Troubleshooting

- **"Style not found"**: Ensure `import scienceplots` is called before `plt.style.use()`
- **LaTeX errors**: Use `['science', 'no-latex']` as fallback
- **CJK font missing**: Install appropriate CJK fonts for your OS
- **Figures too small**: Combine with `notebook` style or set `figure.figsize` manually

## Citation

If used in academic work, cite: DOI 10.5281/zenodo.4106649

## References

- Repository: https://github.com/garrettj403/SciencePlots
- PyPI: https://pypi.org/project/SciencePlots/
- Inspired by: Giannis Tolios LinkedIn post on scientific visualization best practices
