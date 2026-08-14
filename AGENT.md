# Codex Working Rules

## Julia example script style

- For manually run Julia example files, prefer the direct experiment-script style when the file is intended to be edited and executed by hand: define parameters near the bottom of the file and call the driver function directly.
- If a script loads the package source by
  `include("../../src/GrassmannTensorNetworks.jl")`, import the local module with
  `using .GrassmannTensorNetworks`, not `using GrassmannTensorNetworks`.
- Do not use triple-quoted strings (`""" ... """`) as comments in Julia. They are evaluated string literals and may interpolate expressions such as `$(Dbond)`. Use Julia block comments `#= ... =#` or ordinary `#` comments instead.
- `flush(stdout)` is not required for numerical correctness. Keep it after progress-printing statements in long-running server examples when immediate log visibility is useful; it may be removed from short scripts or final-only output.
