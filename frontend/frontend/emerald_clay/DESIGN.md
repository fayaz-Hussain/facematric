---
name: FaceMetric Neo-Clinical
colors:
  surface: '#0e1511'
  surface-dim: '#0e1511'
  surface-bright: '#343b37'
  surface-container-lowest: '#09100c'
  surface-container-low: '#161d19'
  surface-container: '#1a211d'
  surface-container-high: '#252b28'
  surface-container-highest: '#303632'
  on-surface: '#dde4de'
  on-surface-variant: '#bfc9c4'
  inverse-surface: '#dde4de'
  inverse-on-surface: '#2b322e'
  outline: '#89938f'
  outline-variant: '#404945'
  surface-tint: '#95d3c1'
  primary: '#ffffff'
  on-primary: '#00382d'
  primary-container: '#b0efdc'
  on-primary-container: '#326f60'
  inverse-primary: '#2b695a'
  secondary: '#c5c6cd'
  on-secondary: '#2e3035'
  secondary-container: '#47494e'
  on-secondary-container: '#b7b8be'
  tertiary: '#ffffff'
  on-tertiary: '#003829'
  tertiary-container: '#9df4d2'
  on-tertiary-container: '#0f7257'
  error: '#ffb4ab'
  on-error: '#690005'
  error-container: '#93000a'
  on-error-container: '#ffdad6'
  primary-fixed: '#b0efdc'
  primary-fixed-dim: '#95d3c1'
  on-primary-fixed: '#002019'
  on-primary-fixed-variant: '#0b5043'
  secondary-fixed: '#e2e2e9'
  secondary-fixed-dim: '#c5c6cd'
  on-secondary-fixed: '#191c20'
  on-secondary-fixed-variant: '#45474c'
  tertiary-fixed: '#9df4d2'
  tertiary-fixed-dim: '#82d7b7'
  on-tertiary-fixed: '#002117'
  on-tertiary-fixed-variant: '#00513d'
  background: '#0e1511'
  on-background: '#dde4de'
  surface-variant: '#303632'
  sky-mint: '#B8F7E4'
  dark-graphite: '#25272C'
  graphite-muted: rgba(37, 39, 44, 0.7)
  mint-low-opacity: rgba(184, 247, 228, 0.1)
typography:
  display-xl:
    fontFamily: Manrope
    fontSize: 48px
    fontWeight: '800'
    lineHeight: '1'
    letterSpacing: -0.05em
  headline-lg:
    fontFamily: Manrope
    fontSize: 36px
    fontWeight: '700'
    lineHeight: '1.2'
  headline-md:
    fontFamily: Manrope
    fontSize: 24px
    fontWeight: '600'
    lineHeight: '1.3'
  body-md:
    fontFamily: Inter
    fontSize: 16px
    fontWeight: '400'
    lineHeight: '1.5'
  label-xs-caps:
    fontFamily: Inter
    fontSize: 12px
    fontWeight: '600'
    lineHeight: '1'
    letterSpacing: 0.15em
rounded:
  sm: 0.125rem
  DEFAULT: 0.25rem
  md: 0.375rem
  lg: 0.5rem
  xl: 0.75rem
  full: 9999px
spacing:
  container-max: 1280px
  gutter: 1rem
  section-gap: 2.5rem
  card-padding: 1.5rem
---

## Brand & Style
The FaceMetric brand identity is a fusion of high-precision scientific analysis and modern digital aesthetics. It targets a tech-literate audience seeking objective, data-driven insights. The brand personality is "Clinical yet Electric"—it feels rigorous and authoritative like a laboratory, but maintains the high-energy pulse of contemporary technology.

The visual style is a **High-Contrast Digital Noir**. It utilizes a deep graphite foundation to minimize eye strain and emphasize the "Sky Mint" data visualizations. This approach borrows from **Minimalism** (focused data, plenty of negative space) and **Modern Tech** (glow effects, precise geometric overlays, and bento-box layouts) to create an interface that feels like a professional diagnostic tool from the near future.

## Colors
The palette is intentionally restricted to create a focused, high-contrast environment. 

- **Primary (Sky Mint):** Used for all data-rich elements, active icons, and primary call-to-actions. In this dark mode context, it acts as the "light source."
- **Secondary (Dark Graphite):** The foundational canvas. It provides a sophisticated, non-black dark mode that allows the mint accents to pop without being jarring.
- **Surface Strategy:** Cards and interactive containers invert the main theme. They use a "Sky Mint" surface with "Dark Graphite" text to create immediate visual hierarchy and distinguish analysis results from the application shell.
- **Accents:** Use varying opacities of Sky Mint (10-50%) for geometric grids, scan lines, and secondary typography to maintain a monochromatic technical feel.

## Typography
The typographic system emphasizes clarity and a "technical manual" aesthetic. 

- **Headlines (Manrope):** Chosen for its modern, geometric construction which complements the facial symmetry theme. Bold weights are used to anchor sections.
- **Body & Labels (Inter):** Utilized for its high legibility and neutral, utilitarian character. 
- **The "Data" Look:** Critical metrics use a condensed, tight-tracked Manrope for a "dashboard" feel. Labels always appear in uppercase with generous letter spacing (15%+) to mimic architectural or scientific labeling.
- **Mobile Scaling:** Headline-lg (36px) should scale down to 28px on mobile devices to maintain readability without overwhelming the viewport.

## Layout & Spacing
The system uses a **Bento-style Fixed Grid** for analysis results. 

- **Grid Logic:** A 12-column system is used on desktop. Components span 3, 4, 6, or 12 columns depending on information density.
- **Rhythm:** A consistent 4px/8px base unit is used. 16px (1rem) is the standard gutter between grid items.
- **Margins:** Large outer margins (20px on mobile, auto-centered on desktop with a 1280px max-width) ensure the focus remains on the central data.
- **Responsive Behavior:** On mobile, the grid collapses into a single column. On tablet, a 2-column masonry or equal-height grid is preferred.

## Elevation & Depth
Depth is created through **Tonal Contrast** rather than traditional shadows.

- **The Shell:** The background is at the lowest elevation (Dark Graphite).
- **Interactive Surfaces:** Cards sit "above" the background by using the inverted Sky Mint color. These cards are flat and use a thin, low-opacity border (15% graphite) instead of shadows to maintain a "printed report" feel.
- **Overlays:** Technical overlays (scan lines, geometric nodes) utilize a "soft glow" (box-shadow: 0 0 8px #B8F7E4) to simulate a light-emitting screen within the interface.
- **Transparency:** Use backdrop filters (blur) and semi-transparent fills for headers to suggest a high-tech glass aesthetic without full glassmorphism.

## Shapes
The shape language is precise and controlled. 

- **Containers:** Cards and primary buttons use "Soft" corners (0.5rem / 8px). This strikes a balance between friendly modernism and clinical sharpness.
- **Geometric Elements:** Facial mapping nodes and status indicators are perfectly circular (9999px) to contrast against the rectangular grid.
- **Special Shapes:** Organic face shapes (ovals, hearts) are represented using high-percentage border-radii (e.g., 50%/60%) to mimic anatomical curves within the geometric system.

## Components
- **Buttons:** Primary buttons are "Accent-BG" (Mint surface/Graphite text). They are uppercase, tracked wide, and have no shadow. Hover states should simply adjust opacity to 90%.
- **Cards (Bento):** The core unit. High-contrast Mint background with Graphite text. Each card must include a `label-xs-caps` header to identify the metric.
- **Data Visualizations:** 
    - **Progress Circles:** Use a heavy stroke for the primary metric and a low-opacity "track" stroke.
    - **Proportion Bars:** 8px height, rounded-full, with a vertical "Ideal" marker to indicate target ratios.
- **Analysis Overlays:** Thin (1px) lines in Sky Mint. Use "dot" nodes at intersection points to signify data extraction nodes.
- **Status Chips:** Small, pill-shaped, using the inverted color of their container (Graphite on Mint) to ensure they are the most legible element on the card.