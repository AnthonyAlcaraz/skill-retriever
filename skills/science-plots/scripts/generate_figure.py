"""
Generate a publication-quality scientific figure using SciencePlots.

Usage:
    python generate_figure.py --style ieee --output figure.pdf
    python generate_figure.py --style nature --output figure.png --dpi 300
    python generate_figure.py --style "science,high-vis,grid" --output presentation.png

Arguments:
    --style: Comma-separated list of SciencePlots styles (default: science,ieee)
    --output: Output file path (default: figure.pdf)
    --dpi: Resolution in DPI (default: 300)
    --no-latex: Use no-latex fallback rendering
"""

import argparse
import sys

def check_dependencies():
    """Verify required packages are installed."""
    missing = []
    try:
        import matplotlib
    except ImportError:
        missing.append('matplotlib')
    try:
        import scienceplots
    except ImportError:
        missing.append('SciencePlots')

    if missing:
        print(f"Missing dependencies: {', '.join(missing)}")
        print(f"Install with: pip install {' '.join(missing)}")
        sys.exit(1)


def generate_demo_figure(styles, output_path, dpi=300):
    """Generate a demo figure with the specified styles."""
    import matplotlib.pyplot as plt
    import numpy as np
    import scienceplots  # noqa: F401

    plt.style.use(styles)

    x = np.linspace(0, 2 * np.pi, 200)
    fig, ax = plt.subplots()

    functions = [
        (np.sin(x), r'$\sin(x)$'),
        (np.cos(x), r'$\cos(x)$'),
        (np.sin(2 * x), r'$\sin(2x)$'),
    ]

    for y, label in functions:
        ax.plot(x, y, label=label)

    ax.set_xlabel(r'$x$')
    ax.set_ylabel(r'$f(x)$')
    ax.legend()
    ax.set_title('SciencePlots Demo')

    fig.savefig(output_path, dpi=dpi, bbox_inches='tight')
    print(f"Figure saved to: {output_path}")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description='Generate publication-quality scientific figures')
    parser.add_argument('--style', default='science,ieee', help='Comma-separated SciencePlots styles')
    parser.add_argument('--output', default='figure.pdf', help='Output file path')
    parser.add_argument('--dpi', type=int, default=300, help='Resolution in DPI')
    parser.add_argument('--no-latex', action='store_true', help='Use no-latex fallback')
    args = parser.parse_args()

    check_dependencies()

    styles = [s.strip() for s in args.style.split(',')]
    if args.no_latex and 'no-latex' not in styles:
        styles.append('no-latex')

    generate_demo_figure(styles, args.output, args.dpi)


if __name__ == '__main__':
    main()
