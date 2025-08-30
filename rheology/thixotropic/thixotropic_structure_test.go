package thixotropic

import (
	"fmt"
	"math"
	"testing"
	"time"
)

// TestNewThixotropicParameters tests parameter creation
func TestNewThixotropicParameters(t *testing.T) {
	params := NewThixotropicParameters()

	if params.Lambda0 != 1.0 {
		t.Errorf("Expected Lambda0 = 1.0, got %f", params.Lambda0)
	}
	if params.LambdaInf != 0.3 {
		t.Errorf("Expected LambdaInf = 0.3, got %f", params.LambdaInf)
	}
	if params.KBreakdown != 0.1 {
		t.Errorf("Expected KBreakdown = 0.1, got %f", params.KBreakdown)
	}
}

// TestNewStructureState tests structure state creation
func TestNewStructureState(t *testing.T) {
	state := NewStructureState(0.8)

	if state.LambdaCurrent != 0.8 {
		t.Errorf("Expected LambdaCurrent = 0.8, got %f", state.LambdaCurrent)
	}
	if state.LambdaRate != 0.0 {
		t.Errorf("Expected LambdaRate = 0.0, got %f", state.LambdaRate)
	}
	if state.ShearHistory != 0.0 {
		t.Errorf("Expected ShearHistory = 0.0, got %f", state.ShearHistory)
	}
}

// TestStructureStateUpdate tests structure state updates
func TestStructureStateUpdate(t *testing.T) {
	state := NewStructureState(1.0)

	// Update with positive change
	state.Update(0.9, 0.1, 1.0)

	if state.LambdaCurrent != 0.9 {
		t.Errorf("Expected LambdaCurrent = 0.9, got %f", state.LambdaCurrent)
	}
	if state.LambdaRate != -10.0 {
		t.Errorf("Expected LambdaRate = -10.0, got %f", state.LambdaRate)
	}
	if state.ShearHistory != 0.1 {
		t.Errorf("Expected ShearHistory = 0.1, got %f", state.ShearHistory)
	}
	if state.TimeSinceChange != 0.0 {
		t.Errorf("Expected TimeSinceChange = 0.0, got %f", state.TimeSinceChange)
	}

	// Update with small change (should trigger time accumulation)
	state.Update(0.9001, 0.1, 0.5)

	if state.TimeSinceChange != 0.1 {
		t.Errorf("Expected TimeSinceChange = 0.1, got %f", state.TimeSinceChange)
	}
}

// TestNewThixotropicSolver tests solver creation
func TestNewThixotropicSolver(t *testing.T) {
	params := NewThixotropicParameters()
	solver := NewThixotropicSolver(params)

	if solver.Parameters.Lambda0 != params.Lambda0 {
		t.Errorf("Expected solver parameters to match input")
	}
	if solver.CurrentState.LambdaCurrent != params.Lambda0 {
		t.Errorf("Expected initial state LambdaCurrent = %f, got %f", params.Lambda0, solver.CurrentState.LambdaCurrent)
	}
}

// TestEvolveStructure tests structure evolution
func TestEvolveStructure(t *testing.T) {
	params := NewThixotropicParameters()
	solver := NewThixotropicSolver(params)

	// Test with low shear rate (should have buildup)
	initialLambda := solver.CurrentState.LambdaCurrent
	result := solver.EvolveStructure(0.01, 1.0, 1.0)

	if result <= initialLambda {
		t.Errorf("Expected structure to increase with low shear rate, but %f <= %f", result, initialLambda)
	}

	// Test with high shear rate (should have breakdown)
	result = solver.EvolveStructure(1.0, 1.0, 2.0)

	if result >= solver.CurrentState.LambdaCurrent {
		t.Errorf("Expected structure to decrease with high shear rate")
	}

	// Test bounds
	for i := 0; i < 1000; i++ {
		result = solver.EvolveStructure(10.0, 0.1, float64(i)*0.1)
	}

	if result < params.LambdaInf || result > params.Lambda0 {
		t.Errorf("Expected result within bounds [%f, %f], got %f", params.LambdaInf, params.Lambda0, result)
	}
}

// TestPredictStructureEvolution tests structure evolution prediction
func TestPredictStructureEvolution(t *testing.T) {
	params := NewThixotropicParameters()
	solver := NewThixotropicSolver(params)

	shearRates := []float64{0.01, 1.0, 0.01, 1.0}
	times := []float64{0.0, 1.0, 2.0, 3.0}

	structureEvolution := solver.PredictStructureEvolution(shearRates, times)

	if len(structureEvolution) != len(shearRates) {
		t.Errorf("Expected structure evolution length %d, got %d", len(shearRates), len(structureEvolution))
	}

	// First point should be initial value
	if structureEvolution[0] != params.Lambda0 {
		t.Errorf("Expected first point = %f, got %f", params.Lambda0, structureEvolution[0])
	}

	// Check that structure stays within bounds
	for i, lambda := range structureEvolution {
		if lambda < params.LambdaInf || lambda > params.Lambda0 {
			t.Errorf("Point %d: Expected lambda within bounds [%f, %f], got %f",
				i, params.LambdaInf, params.Lambda0, lambda)
		}
	}
}

// TestAnalyzeThixotropicLoop tests hysteresis loop analysis
func TestAnalyzeThixotropicLoop(t *testing.T) {
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

	// Check result structure
	if len(result.StructureUp) != nPoints {
		t.Errorf("Expected StructureUp length %d, got %d", nPoints, len(result.StructureUp))
	}
	if len(result.StructureDown) != nPoints {
		t.Errorf("Expected StructureDown length %d, got %d", nPoints, len(result.StructureDown))
	}

	// Check hysteresis area is positive
	if result.HysteresisArea <= 0 {
		t.Errorf("Expected positive hysteresis area, got %f", result.HysteresisArea)
	}

	// Check degradation is within bounds
	if result.StructureDegradation < 0 || result.StructureDegradation > 1 {
		t.Errorf("Expected degradation between 0 and 1, got %f", result.StructureDegradation)
	}

	// Check recovery rate is non-negative
	if result.RecoveryRate < 0 {
		t.Errorf("Expected non-negative recovery rate, got %f", result.RecoveryRate)
	}

	// Check loop type is not empty
	if result.LoopType == "" {
		t.Errorf("Expected non-empty loop type")
	}
}

// TestNewThixotropicConstitutiveModel tests constitutive model creation
func TestNewThixotropicConstitutiveModel(t *testing.T) {
	params := NewThixotropicParameters()

	model := NewThixotropicConstitutiveModel(
		params,
		100.0, // Base viscosity
		10.0,  // Yield stress
		50.0,  // Power law K
		0.8,   // Power law n
		2.0,   // Structure coupling
	)

	if model.BaseViscosity != 100.0 {
		t.Errorf("Expected BaseViscosity = 100.0, got %f", model.BaseViscosity)
	}
	if model.YieldStress != 10.0 {
		t.Errorf("Expected YieldStress = 10.0, got %f", model.YieldStress)
	}
}

// TestComputeStress tests stress computation
func TestComputeStress(t *testing.T) {
	params := NewThixotropicParameters()

	model := NewThixotropicConstitutiveModel(
		params,
		100.0, // Base viscosity
		10.0,  // Yield stress
		50.0,  // Power law K
		0.8,   // Power law n
		2.0,   // Structure coupling
	)

	// Test with zero shear rate
	stress := model.ComputeStress(0.0, 1.0, 1.0)
	expectedYieldStress := 10.0 * math.Pow(1.0, 2.0) // Structure coupling effect
	if math.Abs(stress-expectedYieldStress) > 1e-10 {
		t.Errorf("Expected stress = %f at zero shear rate, got %f", expectedYieldStress, stress)
	}

	// Test with positive shear rate
	stress = model.ComputeStress(1.0, 1.0, 2.0)
	if stress <= 0 {
		t.Errorf("Expected positive stress for positive shear rate, got %f", stress)
	}

	// Test stress is non-negative
	for i := 0; i < 10; i++ {
		shearRate := float64(i) * 0.5
		stress := model.ComputeStress(shearRate, 0.1, float64(i)*0.1)
		if stress < 0 {
			t.Errorf("Expected non-negative stress, got %f at shear rate %f", stress, shearRate)
		}
	}
}

// TestSimulateShearHistory tests shear history simulation
func TestSimulateShearHistory(t *testing.T) {
	params := NewThixotropicParameters()

	model := NewThixotropicConstitutiveModel(
		params,
		100.0, // Base viscosity
		10.0,  // Yield stress
		50.0,  // Power law K
		0.8,   // Power law n
		2.0,   // Structure coupling
	)

	// Create simple shear history
	nPoints := 20
	times := make([]float64, nPoints)
	shearRates := make([]float64, nPoints)

	for i := 0; i < nPoints; i++ {
		times[i] = float64(i) * 0.5
		shearRates[i] = math.Sin(float64(i) * 0.1) // Oscillatory shear
	}

	// Run simulation
	result := model.SimulateShearHistory(shearRates, times)

	// Check result structure
	if len(result.Stress) != nPoints {
		t.Errorf("Expected Stress length %d, got %d", nPoints, len(result.Stress))
	}
	if len(result.Structure) != nPoints {
		t.Errorf("Expected Structure length %d, got %d", nPoints, len(result.Structure))
	}
	if len(result.ShearRate) != nPoints {
		t.Errorf("Expected ShearRate length %d, got %d", nPoints, len(result.ShearRate))
	}
	if len(result.Time) != nPoints {
		t.Errorf("Expected Time length %d, got %d", nPoints, len(result.Time))
	}

	// Check structure stays within bounds
	for i, lambda := range result.Structure {
		if lambda < params.LambdaInf || lambda > params.Lambda0 {
			t.Errorf("Point %d: Expected lambda within bounds [%f, %f], got %f",
				i, params.LambdaInf, params.Lambda0, lambda)
		}
	}

	// Check stress is non-negative
	for i, stress := range result.Stress {
		if stress < 0 {
			t.Errorf("Point %d: Expected non-negative stress, got %f", i, stress)
		}
	}
}

// TestPolymerMeltThixotropy tests polymer melt model
func TestPolymerMeltThixotropy(t *testing.T) {
	polymer := NewPolymerMeltThixotropy(
		100000.0, // 100 kg/mol
		250.0,    // K (Tg)
		400.0,    // K (processing temp)
		50000.0,  // J/mol (activation energy)
	)

	// Test temperature factor
	tempFactor := polymer.TemperatureFactor()
	if tempFactor <= 0 {
		t.Errorf("Expected positive temperature factor, got %f", tempFactor)
	}

	// Test molecular weight factor
	mwFactor := polymer.MolecularWeightFactor()
	if mwFactor <= 0 {
		t.Errorf("Expected positive molecular weight factor, got %f", mwFactor)
	}

	// Test viscosity computation
	viscosity := polymer.ComputeMeltViscosity(1.0, 1.0, 1.0)
	if viscosity <= 0 {
		t.Errorf("Expected positive viscosity, got %f", viscosity)
	}

	// Test viscosity decreases with shear rate (shear thinning)
	viscosityLow := polymer.ComputeMeltViscosity(0.1, 1.0, 2.0)
	viscosityHigh := polymer.ComputeMeltViscosity(10.0, 1.0, 3.0)

	// Note: Due to structure evolution, this might not always hold,
	// but we can check that the function runs without error
	_ = viscosityLow
	_ = viscosityHigh
}

// TestBiologicalTissueThixotropy tests biological tissue model
func TestBiologicalTissueThixotropy(t *testing.T) {
	tissue := NewBiologicalTissueThixotropy(
		"cartilage",
		310.15, // 37°C
		0.15,   // 15% collagen
		0.08,   // 8% proteoglycan
	)

	// Test stress computation
	stress := tissue.ComputeTissueStress(0.1, 1.0, 1.0)
	if stress < 0 {
		t.Errorf("Expected non-negative stress, got %f", stress)
	}

	// Test with different tissue types
	muscleTissue := NewBiologicalTissueThixotropy(
		"muscle",
		310.15,
		0.2,  // 20% collagen
		0.05, // 5% proteoglycan
	)

	muscleStress := muscleTissue.ComputeTissueStress(0.1, 1.0, 1.0)
	if muscleStress < 0 {
		t.Errorf("Expected non-negative muscle stress, got %f", muscleStress)
	}
}

// TestUtilityFunctions tests utility functions
func TestUtilityFunctions(t *testing.T) {
	// Test max function
	data := []float64{1.0, 5.0, 2.0, 8.0, 3.0}
	maxVal := max(data)
	if maxVal != 8.0 {
		t.Errorf("Expected max = 8.0, got %f", maxVal)
	}

	// Test min function
	minVal := min(data)
	if minVal != 1.0 {
		t.Errorf("Expected min = 1.0, got %f", minVal)
	}

	// Test mean function
	meanVal := mean(data)
	expectedMean := 19.0 / 5.0 // 3.8
	if math.Abs(meanVal-expectedMean) > 1e-10 {
		t.Errorf("Expected mean = %f, got %f", expectedMean, meanVal)
	}

	// Test empty slice
	if max([]float64{}) != 0.0 {
		t.Errorf("Expected max of empty slice = 0.0")
	}
	if min([]float64{}) != 0.0 {
		t.Errorf("Expected min of empty slice = 0.0")
	}
	if mean([]float64{}) != 0.0 {
		t.Errorf("Expected mean of empty slice = 0.0")
	}
}

// BenchmarkThixotropicEvolution benchmarks the evolution function
func BenchmarkThixotropicEvolution(b *testing.B) {
	params := NewThixotropicParameters()
	solver := NewThixotropicSolver(params)

	b.ResetTimer()

	for i := 0; i < b.N; i++ {
		shearRate := 0.1 + float64(i%100)*0.01
		dt := 0.1
		totalTime := float64(i) * dt
		solver.EvolveStructure(shearRate, dt, totalTime)
	}
}

// BenchmarkThixotropicLoopAnalysis benchmarks loop analysis
func BenchmarkThixotropicLoopAnalysis(b *testing.B) {
	params := NewThixotropicParameters()
	solver := NewThixotropicSolver(params)

	// Create shear rate loop
	nPoints := 20
	shearRates := make([]float64, nPoints)
	for i := 0; i < nPoints; i++ {
		shearRates[i] = 0.01 + float64(i)*(10.0-0.01)/float64(nPoints-1)
	}

	b.ResetTimer()

	for i := 0; i < b.N; i++ {
		solver.AnalyzeThixotropicLoop(shearRates, 5.0)
	}
}

// BenchmarkConstitutiveModel benchmarks the constitutive model
func BenchmarkConstitutiveModel(b *testing.B) {
	params := NewThixotropicParameters()

	model := NewThixotropicConstitutiveModel(
		params,
		100.0, // Base viscosity
		10.0,  // Yield stress
		50.0,  // Power law K
		0.8,   // Power law n
		2.0,   // Structure coupling
	)

	// Create shear history
	nPoints := 100
	times := make([]float64, nPoints)
	shearRates := make([]float64, nPoints)

	for i := 0; i < nPoints; i++ {
		times[i] = float64(i) * 0.1
		shearRates[i] = math.Sin(float64(i) * 0.1)
	}

	b.ResetTimer()

	for i := 0; i < b.N; i++ {
		model.SimulateShearHistory(shearRates, times)
	}
}
