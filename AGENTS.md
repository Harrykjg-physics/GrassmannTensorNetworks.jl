# Codex Working Rules

## Julia example script style

- For manually run Julia example files, prefer the direct experiment-script style when the file is intended to be edited and executed by hand: define parameters near the bottom of the file and call the driver function directly.
- If a script loads the package source by
  `include("../../src/GrassmannTensorNetworks.jl")`, import the local module with
  `using Main.GrassmannTensorNetworks`, not `using GrassmannTensorNetworks`.
