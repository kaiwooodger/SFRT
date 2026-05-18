# Methods Overleaf Bundle

This folder is a direct upload bundle for Overleaf.

## Files

- `main.tex`
  Standalone LaTeX document that compiles the Methods section by itself.
- `sections/methods_section.tex`
  Section-only Methods text for inserting into a larger manuscript.
- `sections/methods_figures.tex`
  Figure environments and captions used by the Methods section.
- `tables/`
  Table files already referenced from the Methods section.
- `figures/`
  Local copies of the Methods figures used in the LaTeX files.

## Suggested Overleaf use

1. Upload the entire folder.
2. Set `main.tex` as the main document if you want a standalone Methods compile.
3. If you already have a manuscript project, copy:
   - `sections/methods_section.tex`
   - `sections/methods_figures.tex`
   - `tables/`
   - `figures/`
4. Then `\input{sections/methods_section}` and `\input{sections/methods_figures}` from your manuscript.

## Notes

- The text reflects the final project framing as a **computational biological risk-analysis** workflow, not a clinically validated optimizer.
- The figures and tables are already wired to the labels referenced in the text.
