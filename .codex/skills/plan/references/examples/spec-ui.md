# Example: UI spec tree

### Top-level (PM level)

```md
Summary: Sidebar navigation for switching between top-level sections of the app.

## Guarantees
- User can switch between all top-level sections by clicking sidebar items.
- User can collapse the sidebar to icon-only mode.
- Sidebar is visible on screens wider than 768px; collapsed to icons below.
- Navigation state persists across page refreshes (URL-driven).
- Exactly one section is active at any time.

## Constraints
- Sidebar items cannot be reordered or customized by users (v1).
```

### Mid-level (designer level)

```md
Summary: Sidebar items show hover feedback and selected state.

## Guarantees
- User can hover items to see visual feedback before clicking.
- Selected item always corresponds to the current route.
- Hover: accent-colored background at muted opacity.
- Selected: accent-colored underline below the item label.
- Items display icon (20px) + label, vertically centered, 40px row height.
- Collapsed mode: icon only, tooltip on hover shows label.

## Constraints
- Disabled items show no hover feedback and ignore clicks.
```

### Leaf (detailed design)

```md
Summary: Sidebar selected item underline treatment.

## Guarantees
- 2px solid underline using theme accent color token.
- Positioned 4px below the label baseline.
- Transition: 100ms ease-in on route change.
- Underline width matches label text width, not full item width.

## Rationale
- Text-width underline feels lighter than full-width; avoids visual heaviness in dense navigation.
- 100ms keeps transitions perceptible but not sluggish.
```
