The default system prompt is copied verbatim (apart from its terminal newline) from
Pascal Editor's `SYSTEM_PROMPT` in `packages/mcp/src/tools/vision/analyze-floorplan-image.ts`.

Source commit: `559c0d16133b3aa82fa73b5e7ba42d49aaf160d4`

Source: https://github.com/pascalorg/editor/blob/559c0d16133b3aa82fa73b5e7ba42d49aaf160d4/packages/mcp/src/tools/vision/analyze-floorplan-image.ts

Its output schema is mirrored with Pydantic. Finite-number, size and renderability
checks are local safeguards; they are not accuracy measurements. Pascal's schema
does not include doors, windows, or furniture. Model-reported confidence is not used
as a leaderboard score. See PASCAL-LICENSE for the upstream MIT license.
