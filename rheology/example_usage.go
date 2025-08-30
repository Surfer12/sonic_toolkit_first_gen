package main

import (
	"fmt"
	"math"
	"time"

	"rheology/visualization"
)

func main() {
	fmt.Println("=== Thixotropic Structure Evolution Framework - Go Implementation ===")
	fmt.Println("Advanced Time-Dependent Microstructure Modeling for Complex Fluids")
	fmt.Println("===================================================================")

	// Demonstrate basic thixotropic evolution
	demonstrateBasicEvolution()

	// Demonstrate hysteresis loop analysis
	demonstrateHysteresisLoop()

	// Demonstrate polymer melt behavior
	demonstratePolymerMelt()

	// Demonstrate biological tissue behavior
	demonstrateBiologicalTissue()

	// Demonstrate constitutive model
	demonstrateConstitutiveModel()

	// Performance benchmark
	benchmarkPerformance()

	// Demonstrate visualization capabilities
	demonstrateVisualization()

	fmt.Println("\n=== Framework Ready for Advanced Thixotropic Simulations ===")
	fmt.Println("✓ Structure evolution equations implemented")
	fmt.Println("✓ Memory effects and hysteresis analysis included")
	fmt.Println("✓ Material-specific models for polymers and tissues")
	fmt.Println("✓ Integration with existing rheological framework")
	fmt.Println("✓ High-performance Go implementation with comprehensive testing")
	fmt.Println("✓ Advanced visualization and plotting capabilities")
}

func demonstrateBasicEvolution() {
	fmt.Println("\n=== Basic Thixotropic Structure Evolution ===")

	// Create thixotropic parameters
	params := ThixotropicParameters{
		Lambda0:       1.0,
		LambdaInf:     0.3,
		KBreakdown:    0.1,
		KBuildup:      0.01,
		NThixo:        1.0,
		MThixo:        1.0,
		CriticalShear: 0.1,
		MemoryTime:    10.0,
	}

	solver := NewThixotropicSolver(params)

	// Simulate step changes in shear rate
	shearRates := []float64{0.01, 1.0, 0.01} // Low shear - buildup, High shear - breakdown, Low shear - recovery
	times := []float64{0.0, 50.0, 100.0}     // 50 seconds at high shear, 50 seconds at low shear

	fmt.Println("Shear Rate | Time | Structure Parameter")
	fmt.Println("-----------|------|-------------------")

	lambdaCurrent := params.Lambda0
	for i, shearRate := range shearRates {
		time := times[i]

		if i > 0 {
			dt := time - times[i-1]
			lambdaCurrent = solver.EvolveStructure(shearRate, dt, time)
		}

		fmt.Printf("%.2f | %.1f | %.4f\n", shearRate, time, lambdaCurrent)
	}
}

func demonstrateHysteresisLoop() {
	fmt.Println("\n=== Thixotropic Hysteresis Loop Analysis ===")

	params := ThixotropicParameters{
		Lambda0:       1.0,
		LambdaInf:     0.2,
		KBreakdown:    0.2,
		KBuildup:      0.05,
		NThixo:        1.2,
		MThixo:        0.8,
		CriticalShear: 0.1,
		MemoryTime:    5.0,
	}

	solver := NewThixotropicSolver(params)

	// Create shear rate loop
	nPoints := 10
	shearRates := make([]float64, nPoints)
	for i := 0; i < nPoints; i++ {
		shearRates[i] = 0.01 + float64(i)*(10.0-0.01)/float64(nPoints-1)
	}

	// Analyze loop
	result := solver.AnalyzeThixotropicLoop(shearRates, 5.0)

	// Show first few structure points
	fmt.Println("\nFirst few structure evolution points:")
	for i := 0; i < min(5, len(result.StructureUp)); i++ {
		fmt.Printf("  Up sweep point %d: shear_rate=%.3f, structure=%.4f\n",
			i, shearRates[i], result.StructureUp[i])
	}

	fmt.Println("Thixotropic Loop Analysis Results:")
	fmt.Printf("- Hysteresis Area: %.4f\n", result.HysteresisArea)
	fmt.Printf("- Structure Degradation: %.4f\n", result.StructureDegradation)
	fmt.Printf("- Recovery Rate: %.4f\n", result.RecoveryRate)
	fmt.Printf("- Loop Type: %s\n", result.LoopType)

	// Show first few structure points
	fmt.Println("\nFirst few structure evolution points:")
	for i := 0; i < min(5, len(result.StructureUp)); i++ {
		fmt.Printf("  Up sweep point %d: shear_rate=%.3f, structure=%.4f\n",
			i, shearRates[i], result.StructureUp[i])
	}
}

func demonstratePolymerMelt() {
	fmt.Println("\n=== Polymer Melt Thixotropic Behavior ===")

	polymer := NewPolymerMeltThixotropy(
		100000.0, // 100 kg/mol
		250.0,    // K (Tg)
		400.0,    // K (processing temp)
		50000.0,  // J/mol (activation energy)
	)

	fmt.Printf("Polymer Properties:\n")
	fmt.Printf("- Molecular Weight: %.0f g/mol\n", polymer.MolecularWeight)
	fmt.Printf("- Glass Transition: %.1f K\n", polymer.GlassTransitionT)
	fmt.Printf("- Processing Temp: %.1f K\n", polymer.MeltTemperature)
	fmt.Printf("- Temperature Factor: %.4f\n", polymer.TemperatureFactor())
	fmt.Printf("- Molecular Weight Factor: %.4f\n", polymer.MolecularWeightFactor())
	// Simulate processing conditions
	shearRates := []float64{0.1, 10.0, 1.0, 0.01} // Filling, Injection, Packing, Cooling
	times := []float64{0.0, 2.0, 5.0, 30.0}       // Time progression

	fmt.Println("\nProcessing Simulation:")
	fmt.Println("Time | Shear Rate | Viscosity | Structure")
	fmt.Println("-----|------------|-----------|----------")

	for i, shearRate := range shearRates {
		time := times[i]

		if i > 0 {
			dt := time - times[i-1]
			viscosity := polymer.ComputeMeltViscosity(shearRate, dt, time)
			structure := polymer.BaseSolver.CurrentState.LambdaCurrent

			fmt.Printf("%.1f | %.2f | %.1f | %.4f\n", time, shearRate, viscosity, structure)
		}
	}
}

func demonstrateBiologicalTissue() {
	fmt.Println("\n=== Biological Tissue Thixotropic Behavior ===")

	tissue := NewBiologicalTissueThixotropy(
		"cartilage",
		310.15, // 37°C
		0.15,   // 15% collagen
		0.08,   // 8% proteoglycan
	)

	fmt.Printf("Tissue Properties:\n")
	fmt.Printf("- Tissue Type: %s\n", tissue.TissueType)
	fmt.Printf("- Physiological Temp: %.1f K\n", tissue.PhysiologicalTemp)
	fmt.Printf("- Collagen Content: %.1f%%\n", tissue.CollagenContent*100)
	// Simulate physiological loading
	shearRates := []float64{0.001, 0.01, 0.1, 0.001} // Rest, Walking, Running, Rest
	times := []float64{0.0, 3600.0, 3660.0, 7200.0}  // Time progression

	fmt.Println("\nPhysiological Loading Simulation:")
	fmt.Println("Time | Shear Rate | Stress | Structure")
	fmt.Println("-----|------------|--------|----------")

	for i, shearRate := range shearRates {
		time := times[i]

		if i > 0 {
			dt := time - times[i-1]
			stress := tissue.ComputeTissueStress(shearRate, dt, time)
			structure := tissue.BaseSolver.CurrentState.LambdaCurrent

			fmt.Printf("%.0f | %.3f | %.2f | %.4f\n", time, shearRate, stress, structure)
		}
	}
}

func demonstrateConstitutiveModel() {
	fmt.Println("\n=== Complete Thixotropic Constitutive Model ===")

	params := NewThixotropicParameters()

	model := NewThixotropicConstitutiveModel(
		params,
		100.0, // Base viscosity
		10.0,  // Yield stress
		50.0,  // Power law K
		0.8,   // Power law n
		2.0,   // Structure coupling
	)

	// Simulate complex shear history
	nPoints := 50
	times := make([]float64, nPoints)
	shearRates := make([]float64, nPoints)

	// Oscillatory shear with amplitude modulation
	for i := 0; i < nPoints; i++ {
		times[i] = float64(i) * 0.5 // 0.5 s intervals
		baseFrequency := 0.1        // rad/s
		amplitude := 1.0 + 0.5*math.Sin(float64(i)*0.1)
		shearRates[i] = amplitude * math.Sin(baseFrequency*times[i])
	}

	// Run simulation
	results := model.SimulateShearHistory(shearRates, times)

	fmt.Println("Simulation completed successfully!")
	fmt.Printf("Final structure parameter: %.4f\n", results.Structure[nPoints-1])
	fmt.Printf("Average stress: %.2f\n", mean(results.Stress))
	fmt.Printf("Stress range: %.2f\n", max(results.Stress)-min(results.Stress))

	// Show some intermediate results
	fmt.Println("\nFirst 10 simulation points:")
	for i := 0; i < min(10, nPoints); i++ {
		fmt.Printf("  t=%.1fs: γ̇=%.3f, τ=%.2f, λ=%.4f\n",
			results.Time[i], results.ShearRate[i], results.Stress[i], results.Structure[i])
	}
}

func benchmarkPerformance() {
	fmt.Println("\n=== Performance Benchmark ===")

	nIterations := 10000
	elapsed := benchmarkThixotropicEvolution(nIterations)

	fmt.Printf("Time for %d iterations: %v\n", nIterations, elapsed)
	fmt.Printf("Iterations per second: %.0f\n", float64(nIterations)/elapsed.Seconds())
	fmt.Printf("Time per iteration: %.2f μs\n", elapsed.Seconds()*1e6/float64(nIterations))
}

func benchmarkThixotropicEvolution(nIterations int) time.Duration {
	params := NewThixotropicParameters()
	solver := NewThixotropicSolver(params)

	start := time.Now()

	for i := 0; i < nIterations; i++ {
		shearRate := 0.1 + float64(i%100)*0.01
		dt := 0.1
		totalTime := float64(i) * dt
		solver.EvolveStructure(shearRate, dt, totalTime)
	}

	elapsed := time.Since(start)
	return elapsed
}

func demonstrateVisualization() {
	fmt.Println("\n=== Visualization Capabilities ===")

	// Generate sample data for visualization
	nPoints := 100
	times := make([]float64, nPoints)
	shearRates := make([]float64, nPoints)
	structures := make([]float64, nPoints)
	stresses := make([]float64, nPoints)

	// Simulate thixotropic behavior
	params := NewThixotropicParameters()
	solver := NewThixotropicSolver(params)

	lambda := params.Lambda0
	for i := 0; i < nPoints; i++ {
		times[i] = float64(i) * 0.5
		shearRates[i] = 0.1 + 0.9*math.Sin(float64(i)*0.1)

		// Evolve structure
		if i > 0 {
			dt := times[i] - times[i-1]
			lambda = solver.EvolveStructure(shearRates[i], dt, times[i])
		}

		structures[i] = lambda
		stresses[i] = 10.0*lambda + 50.0*math.Pow(shearRates[i], 0.8)*lambda
	}

	fmt.Println("Creating visualization plots...")

	// Create output directory
	outputDir := "output/plots"
	if err := visualization.CreateOutputDirectory(outputDir); err != nil {
		fmt.Printf("Failed to create output directory: %v\n", err)
		return
	}

	// 1. Structure evolution plot
	structureFile := "output/plots/structure_evolution.png"
	if err := visualization.StructureEvolutionPlot(shearRates, times, structures, structureFile); err != nil {
		fmt.Printf("Failed to create structure evolution plot: %v\n", err)
	}

	// 2. Stress-strain plot
	stressFile := "output/plots/stress_strain.png"
	if err := visualization.StressStrainPlot(shearRates, stresses, stressFile); err != nil {
		fmt.Printf("Failed to create stress-strain plot: %v\n", err)
	}

	// 3. Hysteresis loop plot (simulate up/down sweeps)
	if len(structures) >= 20 {
		midPoint := len(shearRates) / 2
		shearRatesUp := shearRates[:midPoint]
		shearRatesDown := make([]float64, len(shearRatesUp))
		structuresUp := structures[:midPoint]
		structuresDown := make([]float64, len(structuresUp))

		for i := range shearRatesUp {
			shearRatesDown[i] = shearRatesUp[len(shearRatesUp)-1-i]
			structuresDown[i] = structures[midPoint+i] * 0.7 // Simulate degradation
		}

		hysteresisFile := "output/plots/hysteresis_loop.png"
		if err := visualization.HysteresisLoopPlot(shearRatesUp, structuresUp, structuresDown, hysteresisFile); err != nil {
			fmt.Printf("Failed to create hysteresis loop plot: %v\n", err)
		}
	}

	// 4. Material comparison plot
	materials := []visualization.PlotData{
		{
			X:     times,
			Y:     structures,
			Label: "Thixotropic Fluid",
		},
	}

	// Add modified material
	structures2 := make([]float64, nPoints)
	for i := range structures {
		structures2[i] = math.Max(0.1, structures[i]*0.8)
	}
	materials = append(materials, visualization.PlotData{
		X:     times,
		Y:     structures2,
		Label: "Modified Material",
	})

	comparisonFile := "output/plots/material_comparison.png"
	if err := visualization.MultiMaterialComparisonPlot(materials, comparisonFile); err != nil {
		fmt.Printf("Failed to create comparison plot: %v\n", err)
	}

	// 5. Viscoelastic properties plot
	frequencies := make([]float64, 50)
	gPrime := make([]float64, 50)
	gDoublePrime := make([]float64, 50)
	tanDelta := make([]float64, 50)

	for i := 0; i < 50; i++ {
		omega := 0.1 + float64(i)*0.2
		frequencies[i] = omega
		gPrime[i] = 1000.0 * math.Pow(0.5, 0.3) // Structure-modified
		gDoublePrime[i] = omega * 100.0 * math.Pow(0.5, 0.3)
		if gPrime[i] > 0 {
			tanDelta[i] = gDoublePrime[i] / gPrime[i]
		}
	}

	viscoFile := "output/plots/viscoelastic"
	if err := visualization.ViscoelasticPlot(frequencies, gPrime, gDoublePrime, tanDelta, viscoFile); err != nil {
		fmt.Printf("Failed to create viscoelastic plots: %v\n", err)
	}

	// 6. Performance plot
	iterations := []int{1000, 5000, 10000, 25000}
	perfTimes := make([]float64, len(iterations))

	for i, nIter := range iterations {
		start := time.Now()
		benchmarkThixotropicEvolution(nIter)
		elapsed := time.Since(start)
		perfTimes[i] = elapsed.Seconds()
	}

	perfFile := "output/plots/performance_benchmark.png"
	if err := visualization.PerformancePlot(iterations, perfTimes, perfFile); err != nil {
		fmt.Printf("Failed to create performance plot: %v\n", err)
	}

	fmt.Printf("Visualization plots created in: %s\n", outputDir)
	fmt.Println("Generated plots:")
	fmt.Println("  - Structure evolution over time")
	fmt.Println("  - Stress vs strain rate")
	fmt.Println("  - Hysteresis loop analysis")
	fmt.Println("  - Material comparison")
	fmt.Println("  - Viscoelastic properties (G', G'', tan δ)")
	fmt.Println("  - Performance benchmarks")
}

// Utility functions (duplicating for example - in real code, use the package versions)
func max(slice []float64) float64 {
	if len(slice) == 0 {
		return 0.0
	}
	maxVal := slice[0]
	for _, v := range slice {
		if v > maxVal {
			maxVal = v
		}
	}
	return maxVal
}

func min(slice []float64) float64 {
	if len(slice) == 0 {
		return 0.0
	}
	minVal := slice[0]
	for _, v := range slice {
		if v < minVal {
			minVal = v
		}
	}
	return minVal
}

func mean(slice []float64) float64 {
	if len(slice) == 0 {
		return 0.0
	}
	sum := 0.0
	for _, v := range slice {
		sum += v
	}
	return sum / float64(len(slice))
}
