# TODOs for the project

## Water + Calcite System

### Coding

- [ ] Add a `DensityProfile` class to `src/colvar` module to compute the density profile of a given atom selection along a given axis.
MDAnalysis has a `DensityAnalysis` class in the `analysis.density` module [here](https://docs.mdanalysis.org/stable/documentation_pages/analysis/density.html), but it is not parallelized.
- [ ] Add a `WaterDynamics` class to `src/colvar` module to compute the water orientational and translational relaxation times.
MDAnalysis has a `WaterOrientationalRelaxation`, `AngularDistribution`, and `SurvivalProbability` classes in the `analysis.waterdynamics` module [here](https://docs.mdanalysis.org/stable/documentation_pages/analysis/waterdynamics.html), but they are not parallelized.

### Analysis

- [ ] Compute the density profile of water along the z-axis.
Determine the number, position and width of the layers.
This will be used to define the interfacial water structure in the absence of ions and polymer.
- [ ] Compute the PMF of water along the z-axis.
This will be used to determine the free energy of water adsorption to the calcite surface and the free energy of water transport through the interfacial region.
- [ ] Compute the water orientational and translational relaxation times.
This will be used to quantify the kinetic frustration of interfacial water molecules.

## Water + Calcite + Ions System

### Coding

- [ ] Add a `WaterBridge` class to `src/colvar` module to compute the number of water bridges between two atom selections.
MDAnalysis has a `WaterBridgeAnalysis` class in the `analysis.hydrogenbonds` module [here](https://docs.mdanalysis.org/stable/documentation_pages/analysis/wbridge_analysis.html), but it is not parallelized.

### Analysis

- [ ] Compute the density profile of water along the z-axis.
This will be used to determine how aqueous calcium and carbonate ions perturb the interfacial water structure.
- [ ] Compute g(r) of the ions about the calcite surface.
This will quantify the size and distribution of the electrical double layer.
- [ ] Compute the PMF of ions along the z-axis.
This will be used to determine the free energy of ion adsorption to the calcite surface and the free energy of ion transport through the interfacial region.
- [ ] Compute the number of solvating water molecules about the aqueous ions as a function of distance from the surface.
This will aid in quantifying the role of solvation in the adsorption of ions to the calcite surface.
- [ ] Compute the distribution of water bridges between the aqueous ions and the calcite surface split into calcium and carbonate ions.
This will aid in quantifying the role of water bridges in the adsorption of ions to the calcite surface.

## Water + Calcite + Monomers System

### Analysis

- [ ] Compute the density profile of water along the z-axis.
This will be used to determine how monomers perturb the interfacial water structure.
- [ ] Compute the PMF of monomers along the z-axis.
This will be used to determine the free energy of monomer adsorption to the calcite surface and the free energy of monomer transport through the interfacial region.
- [ ] Compute the number of solvating water molecules about the monomers as a function of distance from the surface.
This will aid in quantifying the role of solvation in the adsorption of monomers to the calcite surface.
- [ ] Compute the distribution of water bridges between the monomers and the calcite surface split into calcium and carbonate ions.
This will aid in quantifying the role of water bridges in the adsorption of monomers to the calcite surface.
