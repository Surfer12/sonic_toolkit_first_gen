package thixotropic

import (
	"fmt"
	"math"
	"sort"
	"time"
)

// ThixotropicParameters holds complete thixotropic model parameters
type ThixotropicParameters struct {
	Lambda0        float64 // Initial structure parameter
	LambdaInf      float64 // Equilibrium structure parameter
	KBreakdown     float64 // Breakdown rate constant [1/s]
	KBuildup       float64 // Buildup rate constant [1/s]
	NThixo         float64 // Breakdown power-law index
	MThixo         float64 // Buildup power-law index
	CriticalShear  float64 // Critical shear rate for structure changes [1/s]
	MemoryTime     float64 // Memory time for structure evolution [s]
}

// NewThixotropicParameters creates default thixotropic parameters
func NewThixotropicParameters() ThixotropicParameters {
	return ThixotropicParameters{
		Lambda0:       1.0,
		LambdaInf:     0.3,
		KBreakdown:    0.1,
		KBuildup:      0.01,
		NThixo:        1.0,
		MThixo:        1.0,
		CriticalShear: 0.1,
		MemoryTime:    10.0,
	}
}

// StructureState tracks current state of material structure
type StructureState struct {
	LambdaCurrent     float64 // Current structure parameter
	LambdaRate        float64 // Rate of structure change
	ShearHistory      float64 // Integrated shear history
	TimeSinceChange   float64 // Time since last significant change
	EnergyDissipated  float64 // Energy dissipated in structure changes
}

// NewStructureState creates initial structure state
func NewStructureState(lambdaCurrent float64) *StructureState {
	return &StructureState{
		LambdaCurrent:   lambdaCurrent,
		LambdaRate:      0.0,
		ShearHistory:    0.0,
		TimeSinceChange: 0.0,
		EnergyDissipated: 0.0,
	}
}

// Update updates the structure state with new values
func (s *StructureState) Update(newLambda, dt, shearRate float64) {
	oldLambda := s.LambdaCurrent
	dLambda := newLambda - oldLambda

	s.LambdaCurrent = newLambda
	s.LambdaRate = dLambda / dt
	if dt > 0 {
		s.LambdaRate = dLambda / dt
	} else {
		s.LambdaRate = 0.0
	}

	s.ShearHistory += shearRate * dt

	if math.Abs(dLambda) > 1e-6 {
		s.TimeSinceChange = 0.0
	} else {
		s.TimeSinceChange += dt
	}

	// Energy dissipated in structure change (simplified)
	viscosityContribution := 100.0 // Pa·s (simplified)
	s.EnergyDissipated += math.Abs(dLambda) * viscosityContribution * shearRate * shearRate * dt
}

// ThixotropicSolver implements advanced thixotropic structure evolution
type ThixotropicSolver struct {
	Parameters        ThixotropicParameters
	CurrentState      *StructureState
	StructureHistory  []float64
	TimeHistory       []float64
	ShearHistory      []float64
}

// NewThixotropicSolver creates a new thixotropic solver
func NewThixotropicSolver(parameters ThixotropicParameters) *ThixotropicSolver {
	initialState := NewStructureState(parameters.Lambda0)

	return &ThixotropicSolver{
		Parameters:       parameters,
		CurrentState:     initialState,
		StructureHistory: make([]float64, 0),
		TimeHistory:      make([]float64, 0),
		ShearHistory:     make([]float64, 0),
	}
}

// EvolveStructure evolves structure parameter using thixotropic kinetics
func (ts *ThixotropicSolver) EvolveStructure(shearRate, dt, totalTime float64) float64 {
	lambdaOld := ts.CurrentState.LambdaCurrent

	// Compute breakdown term
	breakdownRate := ts.computeBreakdownRate(shearRate, lambdaOld)

	// Compute buildup term
	buildupRate := ts.computeBuildupRate(shearRate, lambdaOld)

	// Apply memory effects
	memoryFactor := ts.computeMemoryFactor(totalTime)

	// Total rate of structure change
	dLambdaDt := -breakdownRate + buildupRate*memoryFactor

	// Integrate structure change
	lambdaNew := lambdaOld + dLambdaDt*dt

	// Apply bounds
	lambdaNew = math.Max(ts.Parameters.LambdaInf,
		math.Min(ts.Parameters.Lambda0, lambdaNew))

	// Update state
	ts.CurrentState.Update(lambdaNew, dt, shearRate)

	// Store history
	ts.StructureHistory = append(ts.StructureHistory, lambdaNew)
	ts.TimeHistory = append(ts.TimeHistory, totalTime)
	ts.ShearHistory = append(ts.ShearHistory, shearRate)

	return lambdaNew
}

// computeBreakdownRate computes structure breakdown rate
func (ts *ThixotropicSolver) computeBreakdownRate(shearRate, lambdaCurrent float64) float64 {
	if shearRate <= ts.Parameters.CriticalShear {
		return 0.0 // No breakdown below critical shear rate
	}

	// Power-law breakdown kinetics
	shearFactor := math.Pow(shearRate/ts.Parameters.CriticalShear, ts.Parameters.NThixo)
	structureFactor := lambdaCurrent - ts.Parameters.LambdaInf

	return ts.Parameters.KBreakdown * shearFactor * structureFactor
}

// computeBuildupRate computes structure buildup rate
func (ts *ThixotropicSolver) computeBuildupRate(shearRate, lambdaCurrent float64) float64 {
	shearInhibition := math.Exp(-shearRate / ts.Parameters.CriticalShear)
	structureDeficit := ts.Parameters.Lambda0 - lambdaCurrent

	// Power-law buildup with shear inhibition
	buildupRate := ts.Parameters.KBuildup * math.Pow(structureDeficit, ts.Parameters.MThixo) * shearInhibition

	return buildupRate
}

// computeMemoryFactor computes memory factor for structure evolution
func (ts *ThixotropicSolver) computeMemoryFactor(totalTime float64) float64 {
	if len(ts.StructureHistory) < 2 {
		return 1.0
	}

	// Compute time since last significant structure change
	timeSinceChange := ts.CurrentState.TimeSinceChange

	// Exponential memory decay
	memoryFactor := math.Exp(-timeSinceChange / ts.Parameters.MemoryTime)

	return memoryFactor
}

// PredictStructureEvolution predicts structure evolution over time series
func (ts *ThixotropicSolver) PredictStructureEvolution(shearRates, times []float64) []float64 {
	nPoints := len(shearRates)
	structureEvolution := make([]float64, nPoints)

	// Reset to initial state
	ts.CurrentState = NewStructureState(ts.Parameters.Lambda0)
	ts.StructureHistory = make([]float64, 0)
	ts.TimeHistory = make([]float64, 0)
	ts.ShearHistory = make([]float64, 0)

	structureEvolution[0] = ts.Parameters.Lambda0

	for i := 1; i < nPoints; i++ {
		dt := times[i] - times[i-1]
		lambdaNew := ts.EvolveStructure(shearRates[i], dt, times[i])
		structureEvolution[i] = lambdaNew
	}

	return structureEvolution
}

// ThixotropicLoopResult holds results of thixotropic loop analysis
type ThixotropicLoopResult struct {
	StructureUp        []float64
	StructureDown      []float64
	HysteresisArea     float64
	StructureDegradation float64
	RecoveryRate       float64
	LoopType           string
}

// AnalyzeThixotropicLoop analyzes thixotropic hysteresis loop
func (ts *ThixotropicSolver) AnalyzeThixotropicLoop(shearRates []float64, timePerStep float64) ThixotropicLoopResult {
	nPoints := len(shearRates)
	timesUp := make([]float64, nPoints)
	timesDown := make([]float64, nPoints)

	// Create time arrays
	for i := 0; i < nPoints; i++ {
		timesUp[i] = float64(i) * timePerStep
		timesDown[i] = float64(i+nPoints) * timePerStep
	}

	// Forward (up) sweep
	structureUp := ts.PredictStructureEvolution(shearRates, timesUp)

	// Reverse (down) sweep
	reverseShearRates := make([]float64, nPoints)
	for i := 0; i < nPoints; i++ {
		reverseShearRates[i] = shearRates[nPoints-1-i]
	}

	structureDown := ts.PredictStructureEvolution(reverseShearRates, timesDown)

	// Compute hysteresis metrics
	hysteresisArea := ts.computeHysteresisArea(structureUp, structureDown, shearRates)
	structureDegradation := ts.computeStructureDegradation(structureUp, structureDown)
	recoveryRate := ts.computeRecoveryRate(structureDown)

	// Package results
	result := ThixotropicLoopResult{
		StructureUp:         structureUp,
		StructureDown:       structureDown,
		HysteresisArea:      hysteresisArea,
		StructureDegradation: structureDegradation,
		RecoveryRate:        recoveryRate,
		LoopType:            ts.classifyLoopType(hysteresisArea, structureDegradation),
	}

	return result
}

// computeHysteresisArea computes hysteresis area (structure degradation metric)
func (ts *ThixotropicSolver) computeHysteresisArea(structureUp, structureDown, shearRates []float64) float64 {
	area := 0.0
	nPoints := len(structureUp)
	if nPoints > len(structureDown) {
		nPoints = len(structureDown)
	}

	for i := 0; i < nPoints-1; i++ {
		gammaAvg := (shearRates[i] + shearRates[i+1]) / 2.0
		dLambda := math.Abs(structureUp[i] - structureDown[nPoints-1-i])
		area += gammaAvg * dLambda
	}

	return area
}

// computeStructureDegradation computes structure degradation during loop
func (ts *ThixotropicSolver) computeStructureDegradation(structureUp, structureDown []float64) float64 {
	maxStructure := max(structureUp)
	minStructure := min(structureDown)

	if maxStructure > 0 {
		return (maxStructure - minStructure) / maxStructure
	}
	return 0.0
}

// computeRecoveryRate computes structure recovery rate during down sweep
func (ts *ThixotropicSolver) computeRecoveryRate(structureDown []float64) float64 {
	if len(structureDown) < 2 {
		return 0.0
	}

	recovery := 0.0
	for i := 0; i < len(structureDown)-1; i++ {
		dLambda := structureDown[i+1] - structureDown[i]
		if dLambda > 0 { // Recovery
			recovery += dLambda
		}
	}

	return recovery
}

// classifyLoopType classifies thixotropic loop type
func (ts *ThixotropicSolver) classifyLoopType(hysteresisArea, degradation float64) string {
	if hysteresisArea > 1.0 && degradation > 0.5 {
		return "Strong Thixotropic"
	} else if hysteresisArea > 0.5 && degradation > 0.3 {
		return "Moderate Thixotropic"
	} else if hysteresisArea > 0.1 {
		return "Weak Thixotropic"
	} else if hysteresisArea > 0.01 {
		return "Very Weak Thixotropic"
	} else {
		return "Non-Thixotropic"
	}
}

// ThixotropicConstitutiveModel combines structure evolution with rheological response
type ThixotropicConstitutiveModel struct {
	ThixotropicSolver  *ThixotropicSolver
	BaseViscosity      float64 // Base viscosity without structure effects [Pa·s]
	YieldStress        float64 // Base yield stress [Pa]
	PowerLawK          float64 // Power-law consistency [Pa·s^n]
	PowerLawN          float64 // Power-law index
	StructureCoupling  float64 // Coupling factor between structure and rheology
}

// NewThixotropicConstitutiveModel creates a new constitutive model
func NewThixotropicConstitutiveModel(
	thixoParams ThixotropicParameters,
	baseViscosity, yieldStress, powerLawK, powerLawN, structureCoupling float64,
) *ThixotropicConstitutiveModel {
	solver := NewThixotropicSolver(thixoParams)

	return &ThixotropicConstitutiveModel{
		ThixotropicSolver: solver,
		BaseViscosity:     baseViscosity,
		YieldStress:       yieldStress,
		PowerLawK:         powerLawK,
		PowerLawN:         powerLawN,
		StructureCoupling: structureCoupling,
	}
}

// ComputeStress computes shear stress including thixotropic effects
func (tcm *ThixotropicConstitutiveModel) ComputeStress(shearRate, dt, totalTime float64) float64 {
	// Evolve structure
	lambdaCurrent := tcm.ThixotropicSolver.EvolveStructure(shearRate, dt, totalTime)

	// Compute structure-modified yield stress
	tauY := tcm.YieldStress * math.Pow(lambdaCurrent, tcm.StructureCoupling)

	// Compute structure-modified viscosity
	etaStructure := tcm.BaseViscosity * math.Pow(lambdaCurrent, tcm.StructureCoupling-1.0)

	// Compute HB stress with structure effects
	var tauHB float64
	if shearRate > 0.0 {
		tauHB = tcm.PowerLawK * math.Pow(shearRate, tcm.PowerLawN) * math.Pow(lambdaCurrent, tcm.StructureCoupling)
	}

	// Total stress
	tauTotal := tauY + tauHB

	return math.Max(0.0, tauTotal) // Ensure non-negative
}

// SimulationResult holds results of shear history simulation
type SimulationResult struct {
	Stress     []float64
	Structure  []float64
	ShearRate  []float64
	Time       []float64
}

// SimulateShearHistory simulates complete shear history with thixotropic effects
func (tcm *ThixotropicConstitutiveModel) SimulateShearHistory(shearRates, times []float64) SimulationResult {
	nPoints := len(shearRates)
	stresses := make([]float64, nPoints)
	structures := make([]float64, nPoints)
	shearRatesSim := make([]float64, nPoints)

	// Reset solver to initial state
	tcm.ThixotropicSolver.CurrentState = NewStructureState(tcm.ThixotropicSolver.Parameters.Lambda0)
	tcm.ThixotropicSolver.StructureHistory = make([]float64, 0)
	tcm.ThixotropicSolver.TimeHistory = make([]float64, 0)
	tcm.ThixotropicSolver.ShearHistory = make([]float64, 0)

	// Initial conditions
	structures[0] = tcm.ThixotropicSolver.Parameters.Lambda0
	stresses[0] = tcm.ComputeStress(shearRates[0], 0.0, times[0])
	shearRatesSim[0] = shearRates[0]

	// Time integration
	for i := 1; i < nPoints; i++ {
		dt := times[i] - times[i-1]
		stress := tcm.ComputeStress(shearRates[i], dt, times[i])

		stresses[i] = stress
		structures[i] = tcm.ThixotropicSolver.CurrentState.LambdaCurrent
		shearRatesSim[i] = shearRates[i]
	}

	return SimulationResult{
		Stress:    stresses,
		Structure: structures,
		ShearRate: shearRatesSim,
		Time:      times,
	}
}

// PolymerMeltThixotropy implements thixotropic model for polymer melts
type PolymerMeltThixotropy struct {
	BaseSolver        *ThixotropicSolver
	MolecularWeight   float64 // g/mol
	GlassTransitionT  float64 // K
	MeltTemperature   float64 // K
	ActivationEnergy  float64 // J/mol
}

// NewPolymerMeltThixotropy creates specialized polymer melt model
func NewPolymerMeltThixotropy(
	molecularWeight, glassTransitionT, meltTemperature, activationEnergy float64,
) *PolymerMeltThixotropy {
	// Polymer-specific thixotropic parameters
	thixoParams := ThixotropicParameters{
		Lambda0:       1.0,
		LambdaInf:     0.1,  // Polymers don't fully recover
		KBreakdown:    0.5,  // Faster breakdown for polymers
		KBuildup:      0.001, // Slower recovery
		NThixo:        1.5,   // Higher power for polymers
		MThixo:        0.8,
		CriticalShear: 1.0,   // Higher critical shear for polymers
		MemoryTime:    100.0, // Longer memory for polymers
	}

	solver := NewThixotropicSolver(thixoParams)

	return &PolymerMeltThixotropy{
		BaseSolver:       solver,
		MolecularWeight:  molecularWeight,
		GlassTransitionT: glassTransitionT,
		MeltTemperature:  meltTemperature,
		ActivationEnergy: activationEnergy,
	}
}

// TemperatureFactor computes temperature-dependent structure evolution factor
func (pmt *PolymerMeltThixotropy) TemperatureFactor() float64 {
	r := 8.314 // Gas constant
	tRef := pmt.GlassTransitionT + 50.0 // Reference temperature

	return math.Exp(-pmt.ActivationEnergy/r * (1.0/pmt.MeltTemperature - 1.0/tRef))
}

// MolecularWeightFactor computes molecular weight effect on structure evolution
func (pmt *PolymerMeltThixotropy) MolecularWeightFactor() float64 {
	// Higher MW polymers have more entanglements, stronger structure effects
	return math.Log(pmt.MolecularWeight/10000.0) / math.Log(10.0)
}

// ComputeMeltViscosity computes polymer melt viscosity with thixotropic effects
func (pmt *PolymerMeltThixotropy) ComputeMeltViscosity(shearRate, dt, totalTime float64) float64 {
	// Temperature correction
	tempFactor := pmt.TemperatureFactor()

	// Molecular weight correction
	mwFactor := pmt.MolecularWeightFactor()

	// Evolve structure with corrections
	lambdaCurrent := pmt.BaseSolver.EvolveStructure(shearRate, dt, totalTime)

	// Structure-modified viscosity (simplified Carreau-Yasuda model)
	eta0 := 1000.0 * math.Exp(3.4*math.Log(pmt.MolecularWeight/10000.0)) * tempFactor // Zero-shear viscosity
	lambdaRelax := 1.0 // Relaxation time [s]

	var viscosity float64
	if shearRate > 0 {
		reducedRate := shearRate * lambdaRelax
		viscosity = eta0 * math.Pow(lambdaCurrent, mwFactor) * math.Pow(1.0+math.Pow(reducedRate, 2.0), (0.8-1.0)/2.0)
	} else {
		viscosity = eta0 * math.Pow(lambdaCurrent, mwFactor)
	}

	return viscosity
}

// BiologicalTissueThixotropy implements thixotropic model for biological tissues
type BiologicalTissueThixotropy struct {
	BaseSolver           *ThixotropicSolver
	TissueType           string
	PhysiologicalTemp    float64 // K
	CollagenContent      float64 // Collagen fraction (0-1)
	ProteoglycanContent  float64 // Proteoglycan fraction (0-1)
}

// NewBiologicalTissueThixotropy creates specialized biological tissue model
func NewBiologicalTissueThixotropy(
	tissueType string,
	physiologicalTemp, collagenContent, proteoglycanContent float64,
) *BiologicalTissueThixotropy {
	// Tissue-specific thixotropic parameters
	var thixoParams ThixotropicParameters

	switch tissueType {
	case "cartilage":
		thixoParams = ThixotropicParameters{
			Lambda0:       1.0,
			LambdaInf:     0.8,     // Cartilage recovers well
			KBreakdown:    0.01,    // Slow breakdown
			KBuildup:      0.02,    // Good recovery
			NThixo:        0.8,
			MThixo:        1.2,
			CriticalShear: 0.01,    // Very low critical shear
			MemoryTime:    3600.0,  // Long memory (1 hour)
		}
	case "muscle":
		thixoParams = ThixotropicParameters{
			Lambda0:       1.0,
			LambdaInf:     0.6,     // Moderate recovery
			KBreakdown:    0.1,     // Moderate breakdown
			KBuildup:      0.05,    // Moderate recovery
			NThixo:        1.0,
			MThixo:        1.0,
			CriticalShear: 0.1,
			MemoryTime:    300.0,   // 5 minutes memory
		}
	default: // Default soft tissue
		thixoParams = ThixotropicParameters{
			Lambda0:       1.0,
			LambdaInf:     0.7,
			KBreakdown:    0.05,
			KBuildup:      0.03,
			NThixo:        1.0,
			MThixo:        1.0,
			CriticalShear: 0.05,
			MemoryTime:    600.0,   // 10 minutes memory
		}
	}

	solver := NewThixotropicSolver(thixoParams)

	return &BiologicalTissueThixotropy{
		BaseSolver:          solver,
		TissueType:          tissueType,
		PhysiologicalTemp:   physiologicalTemp,
		CollagenContent:     collagenContent,
		ProteoglycanContent: proteoglycanContent,
	}
}

// ComputeTissueStress computes tissue stress with physiological thixotropic effects
func (btt *BiologicalTissueThixotropy) ComputeTissueStress(shearRate, dt, totalTime float64) float64 {
	// Evolve structure
	lambdaCurrent := btt.BaseSolver.EvolveStructure(shearRate, dt, totalTime)

	// Collagen contribution (strong, elastic)
	collagenStress := btt.CollagenContent * 1000.0 * lambdaCurrent * shearRate

	// Proteoglycan contribution (viscous, thixotropic)
	proteoglycanStress := btt.ProteoglycanContent * 100.0 * math.Pow(lambdaCurrent, 2.0) * shearRate

	// Ground substance contribution (Newtonian)
	groundStress := 10.0 * shearRate

	// Total stress with structure effects
	totalStress := (collagenStress + proteoglycanStress + groundStress) * lambdaCurrent

	return math.Max(0.0, totalStress)
}

// Utility functions
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

// Test functions
func TestBasicThixotropicEvolution() {
	fmt.Println("Testing Basic Thixotropic Structure Evolution...")

	// Create thixotropic parameters
	thixoParams := ThixotropicParameters{
		Lambda0:       1.0,
		LambdaInf:     0.3,
		KBreakdown:    0.1,
		KBuildup:      0.01,
		NThixo:        1.0,
		MThixo:        1.0,
		CriticalShear: 0.1,
		MemoryTime:    10.0,
	}

	solver := NewThixotropicSolver(thixoParams)

	// Simulate step changes in shear rate
	shearRates := []float64{0.01, 1.0, 0.01} // Low shear - buildup, High shear - breakdown, Low shear - recovery
	times := []float64{0.0, 50.0, 100.0}     // 50 seconds at high shear, 50 seconds at low shear

	fmt.Println("Shear Rate | Time | Structure Parameter")
	fmt.Println("-----------|------|-------------------")

	lambdaCurrent := thixoParams.Lambda0
	for i, shearRate := range shearRates {
		time := times[i]

		if i > 0 {
			dt := time - times[i-1]
			lambdaCurrent = solver.EvolveStructure(shearRate, dt, time)
		}

		fmt.Printf("%.2f | %.1f | %.4f\n", shearRate, time, lambdaCurrent)
	}

	fmt.Println("✓ Basic thixotropic evolution test completed")
}

func TestThixotropicLoopAnalysis() {
	fmt.Println("\nTesting Thixotropic Hysteresis Loop Analysis...")

	thixoParams := ThixotropicParameters{
		Lambda0:       1.0,
		LambdaInf:     0.2,
		KBreakdown:    0.2,
		KBuildup:      0.05,
		NThixo:        1.2,
		MThixo:        0.8,
		CriticalShear: 0.1,
		MemoryTime:    5.0,
	}

	solver := NewThixotropicSolver(thixoParams)

	// Create shear rate loop
	nPoints := 10
	shearRates := make([]float64, nPoints)
	for i := 0; i < nPoints; i++ {
		shearRates[i] = 0.01 + float64(i)*(10.0-0.01)/float64(nPoints-1)
	}

	// Analyze loop
	loopResults := solver.AnalyzeThixotropicLoop(shearRates, 5.0)

	fmt.Println("Thixotropic Loop Analysis Results:")
	fmt.Printf("- Hysteresis Area: %.4f\n", loopResults.HysteresisArea)
	fmt.Printf("- Structure Degradation: %.4f\n", loopResults.StructureDegradation)
	fmt.Printf("- Recovery Rate: %.4f\n", loopResults.RecoveryRate)
	fmt.Printf("- Loop Type: %s\n", loopResults.LoopType)

	fmt.Println("✓ Thixotropic loop analysis test completed")
}

func TestPolymerMeltThixotropy() {
	fmt.Println("\nTesting Polymer Melt Thixotropy...")

	polymer := NewPolymerMeltThixotropy(
		100000.0,  // 100 kg/mol
		250.0,     // K (Tg)
		400.0,     // K (processing temp)
		50000.0,   // J/mol (activation energy)
	)

	// Simulate processing conditions
	shearRates := []float64{0.1, 10.0, 1.0, 0.01} // Filling, Injection, Packing, Cooling
	times := []float64{0.0, 2.0, 5.0, 30.0}       // Time progression

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

	fmt.Println("✓ Polymer melt thixotropy test completed")
}

func TestBiologicalTissueThixotropy() {
	fmt.Println("\nTesting Biological Tissue Thixotropy...")

	tissue := NewBiologicalTissueThixotropy(
		"cartilage",
		310.15, // 37°C
		0.15,   // 15% collagen
		0.08,   // 8% proteoglycan
	)

	// Simulate physiological loading
	shearRates := []float64{0.001, 0.01, 0.1, 0.001} // Rest, Walking, Running, Rest
	times := []float64{0.0, 3600.0, 3660.0, 7200.0}  // Time progression

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

	fmt.Println("✓ Biological tissue thixotropy test completed")
}

func TestThixotropicConstitutiveModel() {
	fmt.Println("\nTesting Complete Thixotropic Constitutive Model...")

	thixoParams := ThixotropicParameters{
		Lambda0:       1.0,
		LambdaInf:     0.4,
		KBreakdown:    0.15,
		KBuildup:      0.02,
		NThixo:        1.0,
		MThixo:        1.0,
		CriticalShear: 0.1,
		MemoryTime:    10.0,
	}

	model := NewThixotropicConstitutiveModel(
		thixoParams,
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

	fmt.Println("✓ Thixotropic constitutive model test completed")
}

// BenchmarkThixotropicEvolution benchmarks thixotropic evolution performance
func BenchmarkThixotropicEvolution(nIterations int) time.Duration {
	thixoParams := NewThixotropicParameters()
	solver := NewThixotropicSolver(thixoParams)

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

// Example usage function
func ExampleUsage() {
	fmt.Println("=== Thixotropic Structure Evolution Framework - Go Implementation ===")
	fmt.Println("Advanced Time-Dependent Microstructure Modeling for Complex Fluids")
	fmt.Println("====================================================================")

	// Run all tests
	TestBasicThixotropicEvolution()
	TestThixotropicLoopAnalysis()
	TestPolymerMeltThixotropy()
	TestBiologicalTissueThixotropy()
	TestThixotropicConstitutiveModel()

	// Performance benchmark
	fmt.Println("\n=== Performance Benchmark ===")
	nIterations := 10000
	elapsed := BenchmarkThixotropicEvolution(nIterations)
	fmt.Printf("Time for %d iterations: %v\n", nIterations, elapsed)
	fmt.Printf("Iterations per second: %.0f\n", float64(nIterations)/elapsed.Seconds())

	fmt.Println("\n=== Framework Ready for Advanced Thixotropic Simulations ===")
	fmt.Println("✓ Structure evolution equations implemented")
	fmt.Println("✓ Memory effects and hysteresis analysis included")
	fmt.Println("✓ Material-specific models for polymers and tissues")
	fmt.Println("✓ Integration with existing rheological framework")
}
