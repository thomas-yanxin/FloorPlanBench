`pascal_floorplan.txt` is copied verbatim (apart from its terminal newline) from
Pascal Editor's `SYSTEM_PROMPT` in `packages/mcp/src/tools/vision/analyze-floorplan-image.ts`.

Source commit: `559c0d16133b3aa82fa73b5e7ba42d49aaf160d4`

Source: https://github.com/pascalorg/editor/blob/559c0d16133b3aa82fa73b5e7ba42d49aaf160d4/packages/mcp/src/tools/vision/analyze-floorplan-image.ts

`reconstruction_floorplan.txt` is FloorPlanBench's extended default prompt. Its contract
retains Pascal's walls, rooms, dimensions and confidence fields and adds hosted openings.
The added tracing and opening instructions are local additions, not the upstream prompt.

Finite-number, size and renderability checks are local safeguards; they are not accuracy
measurements. Pascal's original schema does not include doors, windows, or furniture.
Model-reported confidence is not used as a leaderboard score. See PASCAL-LICENSE for
the upstream MIT license.
