package visualization

import (
	"fmt"
	"image/color"
	"log"
	"math"
	"os"
	"path/filepath"

	"gonum.org/v1/plot"
	"gonum.org/v1/plot/plotter"
	"gonum.org/v1/plot/vg"
	"gonum.org/v1/plot/vg/draw"
	"gonum.org/v1/gonum/mat"
)

// PlotStyle defines the visual style for plots
type PlotStyle struct {
	Title          string
	XLabel         string
	YLabel         string
	Width          vg.Length
	Height         vg.Length
	LineWidth      vg.Length
	PointSize      vg.Length
	Colors         []color.Color
	Grid           bool
	Legend         bool
	LegendPosition draw.Corner
}

// DefaultPlotStyle returns a default plot style
func DefaultPlotStyle() PlotStyle {
	return PlotStyle{
		Width:          8 * vg.Inch,
		Height:         6 * vg.Inch,
		LineWidth:      vg.Points(2),
		PointSize:      vg.Points(3),
		Colors:         []color.Color{color.RGBA{R: 46, G: 125, B: 50, A: 255}, color.RGBA{R: 244, G: 67, B: 54, A: 255}, color.RGBA{R: 33, G: 150, B: 243, A: 255}, color.RGBA{R: 255, G: 193, B: 7, A: 255}},
		Grid:           true,
		Legend:         true,
		LegendPosition: draw.NE,
	}
}

// PlotData represents data for plotting
type PlotData struct {
	X     []float64
	Y     []float64
	Label string
}

// StructureEvolutionPlot creates a plot of structure parameter evolution over time
func StructureEvolutionPlot(shearRates, times, structures []float64, filename string) error {
	if len(shearRates) != len(times) || len(times) != len(structures) {
		return fmt.Errorf("data arrays must have the same length")
	}

	style := DefaultPlotStyle()
	style.Title = "Thixotropic Structure Evolution"
	style.XLabel = "Time (s)"
	style.YLabel = "Structure Parameter λ"

	p := plot.New()
	p.Title.Text = style.Title
	p.X.Label.Text = style.XLabel
	p.Y.Label.Text = style.YLabel

	if style.Grid {
		p.Add(plotter.NewGrid())
	}

	// Create points for structure evolution
	pts := make(plotter.XYs, len(times))
	for i := range times {
		pts[i].X = times[i]
		pts[i].Y = structures[i]
	}

	line, err := plotter.NewLine(pts)
	if err != nil {
		return err
	}
	line.LineStyle.Width = style.LineWidth
	line.LineStyle.Color = style.Colors[0]

	p.Add(line)

	// Create second axis for shear rate
	p2 := plot.New()
	p2.X.Label.Text = "Time (s)"
	p2.Y.Label.Text = "Shear Rate (1/s)"
	p2.Y.Position = draw.PosRight
	p2.Y.Tick.Label = func(float64) string { return "" } // Hide tick labels

	pts2 := make(plotter.XYs, len(times))
	for i := range times {
		pts2[i].X = times[i]
		pts2[i].Y = shearRates[i]
	}

	line2, err := plotter.NewLine(pts2)
	if err != nil {
		return err
	}
	line2.LineStyle.Width = style.LineWidth
	line2.LineStyle.Color = style.Colors[1]

	p2.Add(line2)

	// Save the plot
	if err := p.Save(style.Width, style.Height, filename); err != nil {
		return err
	}

	fmt.Printf("Structure evolution plot saved to: %s\n", filename)
	return nil
}

// HysteresisLoopPlot creates a hysteresis loop plot
func HysteresisLoopPlot(shearRates, structureUp, structureDown []float64, filename string) error {
	if len(shearRates) != len(structureUp) || len(structureUp) != len(structureDown) {
		return fmt.Errorf("data arrays must have the same length")
	}

	style := DefaultPlotStyle()
	style.Title = "Thixotropic Hysteresis Loop"
	style.XLabel = "Shear Rate (1/s)"
	style.YLabel = "Structure Parameter λ"

	p := plot.New()
	p.Title.Text = style.Title
	p.X.Label.Text = style.XLabel
	p.Y.Label.Text = style.YLabel

	if style.Grid {
		p.Add(plotter.NewGrid())
	}

	// Create points for up sweep
	ptsUp := make(plotter.XYs, len(shearRates))
	for i := range shearRates {
		ptsUp[i].X = shearRates[i]
		ptsUp[i].Y = structureUp[i]
	}

	lineUp, err := plotter.NewLine(ptsUp)
	if err != nil {
		return err
	}
	lineUp.LineStyle.Width = style.LineWidth
	lineUp.LineStyle.Color = style.Colors[0]

	// Create points for down sweep
	ptsDown := make(plotter.XYs, len(shearRates))
	for i := range shearRates {
		ptsDown[i].X = shearRates[i]
		ptsDown[i].Y = structureDown[i]
	}

	lineDown, err := plotter.NewLine(ptsDown)
	if err != nil {
		return err
	}
	lineDown.LineStyle.Width = style.LineWidth
	lineDown.LineStyle.Color = style.Colors[1]

	// Add lines to plot
	p.Add(lineUp, lineDown)

	// Add legend
	if style.Legend {
		legend := plot.NewLegend()
		legend.Add("Up Sweep", lineUp)
		legend.Add("Down Sweep", lineDown)
		legend.Top = true
		legend.Left = true
		p.Legend = legend
	}

	// Save the plot
	if err := p.Save(style.Width, style.Height, filename); err != nil {
		return err
	}

	fmt.Printf("Hysteresis loop plot saved to: %s\n", filename)
	return nil
}

// StressStrainPlot creates a stress vs strain/strain-rate plot
func StressStrainPlot(strainRates, stresses []float64, filename string) error {
	if len(strainRates) != len(stresses) {
		return fmt.Errorf("data arrays must have the same length")
	}

	style := DefaultPlotStyle()
	style.Title = "Stress vs Strain Rate"
	style.XLabel = "Strain Rate (1/s)"
	style.YLabel = "Stress (Pa)"

	p := plot.New()
	p.Title.Text = style.Title
	p.X.Label.Text = style.XLabel
	p.Y.Label.Text = style.YLabel

	if style.Grid {
		p.Add(plotter.NewGrid())
	}

	// Create points
	pts := make(plotter.XYs, len(strainRates))
	for i := range strainRates {
		pts[i].X = strainRates[i]
		pts[i].Y = stresses[i]
	}

	scatter, err := plotter.NewScatter(pts)
	if err != nil {
		return err
	}
	scatter.GlyphStyle.Color = style.Colors[0]
	scatter.GlyphStyle.Shape = draw.CrossGlyph{}
	scatter.GlyphStyle.Radius = style.PointSize

	line, err := plotter.NewLine(pts)
	if err != nil {
		return err
	}
	line.LineStyle.Width = style.LineWidth
	line.LineStyle.Color = style.Colors[0]

	p.Add(scatter, line)

	// Save the plot
	if err := p.Save(style.Width, style.Height, filename); err != nil {
		return err
	}

	fmt.Printf("Stress-strain plot saved to: %s\n", filename)
	return nil
}

// MultiMaterialComparisonPlot compares different materials
func MultiMaterialComparisonPlot(datasets []PlotData, filename string) error {
	style := DefaultPlotStyle()
	style.Title = "Material Comparison"
	style.XLabel = "Time (s)"
	style.YLabel = "Structure Parameter λ"

	p := plot.New()
	p.Title.Text = style.Title
	p.X.Label.Text = style.XLabel
	p.Y.Label.Text = style.YLabel

	if style.Grid {
		p.Add(plotter.NewGrid())
	}

	// Add each dataset
	for i, data := range datasets {
		if len(data.X) != len(data.Y) {
			return fmt.Errorf("dataset %d: X and Y arrays must have the same length", i)
		}

		pts := make(plotter.XYs, len(data.X))
		for j := range data.X {
			pts[j].X = data.X[j]
			pts[j].Y = data.Y[j]
		}

		line, err := plotter.NewLine(pts)
		if err != nil {
			return err
		}
		line.LineStyle.Width = style.LineWidth
		line.LineStyle.Color = style.Colors[i%len(style.Colors)]

		p.Add(line)

		// Add to legend
		if style.Legend {
			if p.Legend == nil {
				p.Legend = plot.NewLegend()
				p.Legend.Top = true
				p.Legend.Left = true
			}
			p.Legend.Add(data.Label, line)
		}
	}

	// Save the plot
	if err := p.Save(style.Width, style.Height, filename); err != nil {
		return err
	}

	fmt.Printf("Multi-material comparison plot saved to: %s\n", filename)
	return nil
}

// ViscoelasticPlot creates plots for viscoelastic properties
func ViscoelasticPlot(frequencies, gPrime, gDoublePrime, tanDelta []float64, filename string) error {
	if len(frequencies) != len(gPrime) || len(gPrime) != len(gDoublePrime) || len(gDoublePrime) != len(tanDelta) {
		return fmt.Errorf("all data arrays must have the same length")
	}

	style := DefaultPlotStyle()
	style.Title = "Viscoelastic Properties"
	style.Width = 12 * vg.Inch
	style.Height = 8 * vg.Inch

	// Create subplots
	p1 := plot.New()
	p1.Title.Text = "Storage and Loss Moduli"
	p1.X.Label.Text = "Frequency (rad/s)"
	p1.Y.Label.Text = "Modulus (Pa)"

	if style.Grid {
		p1.Add(plotter.NewGrid())
	}

	// Storage modulus (G')
	ptsGPrime := make(plotter.XYs, len(frequencies))
	for i := range frequencies {
		ptsGPrime[i].X = frequencies[i]
		ptsGPrime[i].Y = gPrime[i]
	}

	lineGPrime, err := plotter.NewLine(ptsGPrime)
	if err != nil {
		return err
	}
	lineGPrime.LineStyle.Width = style.LineWidth
	lineGPrime.LineStyle.Color = style.Colors[0]

	// Loss modulus (G'')
	ptsGDoublePrime := make(plotter.XYs, len(frequencies))
	for i := range frequencies {
		ptsGDoublePrime[i].X = frequencies[i]
		ptsGDoublePrime[i].Y = gDoublePrime[i]
	}

	lineGDoublePrime, err := plotter.NewLine(ptsGDoublePrime)
	if err != nil {
		return err
	}
	lineGDoublePrime.LineStyle.Width = style.LineWidth
	lineGDoublePrime.LineStyle.Color = style.Colors[1]

	p1.Add(lineGPrime, lineGDoublePrime)

	// Add legend
	if style.Legend {
		legend := plot.NewLegend()
		legend.Add("G' (Storage)", lineGPrime)
		legend.Add("G'' (Loss)", lineGDoublePrime)
		legend.Top = true
		legend.Left = true
		p1.Legend = legend
	}

	// Create second plot for tan delta
	p2 := plot.New()
	p2.Title.Text = "Loss Tangent"
	p2.X.Label.Text = "Frequency (rad/s)"
	p2.Y.Label.Text = "tan δ"

	if style.Grid {
		p2.Add(plotter.NewGrid())
	}

	ptsTanDelta := make(plotter.XYs, len(frequencies))
	for i := range frequencies {
		ptsTanDelta[i].X = frequencies[i]
		ptsTanDelta[i].Y = tanDelta[i]
	}

	lineTanDelta, err := plotter.NewLine(ptsTanDelta)
	if err != nil {
		return err
	}
	lineTanDelta.LineStyle.Width = style.LineWidth
	lineTanDelta.LineStyle.Color = style.Colors[2]

	p2.Add(lineTanDelta)

	// Save both plots
	if err := p1.Save(style.Width, style.Height, filename+"_moduli.png"); err != nil {
		return err
	}

	if err := p2.Save(style.Width, style.Height, filename+"_tandelta.png"); err != nil {
		return err
	}

	fmt.Printf("Viscoelastic plots saved to: %s_moduli.png and %s_tandelta.png\n", filename, filename)
	return nil
}

// ContourPlot creates a contour plot for 2D data (simplified)
func ContourPlot(x, y, z [][]float64, filename string) error {
	if len(x) == 0 || len(x[0]) != len(y[0]) || len(x) != len(z) {
		return fmt.Errorf("invalid data dimensions for contour plot")
	}

	style := DefaultPlotStyle()
	style.Title = "Contour Plot"
	style.XLabel = "X Variable"
	style.YLabel = "Y Variable"

	p := plot.New()
	p.Title.Text = style.Title
	p.X.Label.Text = style.XLabel
	p.Y.Label.Text = style.YLabel

	if style.Grid {
		p.Add(plotter.NewGrid())
	}

	// For simplicity, create a scatter plot with color coding
	// In a full implementation, you'd use a proper contour plotting library
	var allPoints plotter.XYs

	for i := range x {
		for j := range x[i] {
			allPoints = append(allPoints, plotter.XY{X: x[i][j], Y: y[i][j]})
		}
	}

	if len(allPoints) > 0 {
		scatter, err := plotter.NewScatter(allPoints)
		if err != nil {
			return err
		}
		scatter.GlyphStyle.Color = style.Colors[0]
		scatter.GlyphStyle.Shape = draw.CircleGlyph{}
		scatter.GlyphStyle.Radius = style.PointSize

		p.Add(scatter)
	}

	// Save the plot
	if err := p.Save(style.Width, style.Height, filename); err != nil {
		return err
	}

	fmt.Printf("Contour plot saved to: %s\n", filename)
	return nil
}

// CreateOutputDirectory creates the output directory if it doesn't exist
func CreateOutputDirectory(dir string) error {
	if _, err := os.Stat(dir); os.IsNotExist(err) {
		err := os.MkdirAll(dir, 0755)
		if err != nil {
			return fmt.Errorf("failed to create output directory: %v", err)
		}
	}
	return nil
}

// BatchPlot creates multiple plots for a complete analysis
func BatchPlot(shearRates, times, structures, stresses []float64, outputDir string) error {
	// Create output directory
	if err := CreateOutputDirectory(outputDir); err != nil {
		return err
	}

	// Structure evolution plot
	structureFile := filepath.Join(outputDir, "structure_evolution.png")
	if err := StructureEvolutionPlot(shearRates, times, structures, structureFile); err != nil {
		log.Printf("Failed to create structure evolution plot: %v", err)
	}

	// Stress-strain plot
	stressFile := filepath.Join(outputDir, "stress_strain.png")
	if err := StressStrainPlot(shearRates, stresses, stressFile); err != nil {
		log.Printf("Failed to create stress-strain plot: %v", err)
	}

	// Hysteresis loop plot (if we have up/down data)
	if len(structures) >= 20 {
		// Simulate up and down sweeps
		midPoint := len(shearRates) / 2
		shearRatesUp := shearRates[:midPoint]
		shearRatesDown := make([]float64, len(shearRatesUp))
		structuresUp := structures[:midPoint]
		structuresDown := make([]float64, len(structuresUp))

		for i := range shearRatesUp {
			shearRatesDown[i] = shearRatesUp[len(shearRatesUp)-1-i]
			structuresDown[i] = structures[midPoint+i] * 0.7 // Simulate degradation
		}

		hysteresisFile := filepath.Join(outputDir, "hysteresis_loop.png")
		if err := HysteresisLoopPlot(shearRatesUp, structuresUp, structuresDown, hysteresisFile); err != nil {
			log.Printf("Failed to create hysteresis loop plot: %v", err)
		}
	}

	fmt.Printf("Batch plots completed. Files saved to: %s\n", outputDir)
	return nil
}

// PerformancePlot creates performance benchmark plots
func PerformancePlot(iterations []int, times []float64, filename string) error {
	if len(iterations) != len(times) {
		return fmt.Errorf("iterations and times arrays must have the same length")
	}

	style := DefaultPlotStyle()
	style.Title = "Performance Benchmark"
	style.XLabel = "Iterations"
	style.YLabel = "Time (seconds)"

	p := plot.New()
	p.Title.Text = style.Title
	p.X.Label.Text = style.XLabel
	p.Y.Label.Text = style.YLabel

	if style.Grid {
		p.Add(plotter.NewGrid())
	}

	// Create points
	pts := make(plotter.XYs, len(iterations))
	for i := range iterations {
		pts[i].X = float64(iterations[i])
		pts[i].Y = times[i]
	}

	scatter, err := plotter.NewScatter(pts)
	if err != nil {
		return err
	}
	scatter.GlyphStyle.Color = style.Colors[0]
	scatter.GlyphStyle.Shape = draw.CrossGlyph{}
	scatter.GlyphStyle.Radius = style.PointSize

	line, err := plotter.NewLine(pts)
	if err != nil {
		return err
	}
	line.LineStyle.Width = style.LineWidth
	line.LineStyle.Color = style.Colors[0]

	p.Add(scatter, line)

	// Save the plot
	if err := p.Save(style.Width, style.Height, filename); err != nil {
		return err
	}

	fmt.Printf("Performance plot saved to: %s\n", filename)
	return nil
}

// CreateDemoPlots creates demonstration plots for the thixotropic framework
func CreateDemoPlots() error {
	outputDir := "output/plots"

	// Create output directory
	if err := CreateOutputDirectory(outputDir); err != nil {
		return err
	}

	// Generate sample data
	nPoints := 100
	times := make([]float64, nPoints)
	shearRates := make([]float64, nPoints)
	structures := make([]float64, nPoints)
	stresses := make([]float64, nPoints)

	// Simulate structure evolution
	lambda := 1.0
	for i := 0; i < nPoints; i++ {
		times[i] = float64(i) * 0.5
		shearRates[i] = 0.1 + 0.9*math.Sin(float64(i)*0.1)

		// Simple structure evolution model
		if shearRates[i] > 0.5 {
			lambda -= 0.01 // Breakdown
		} else {
			lambda += 0.005 // Recovery
		}
		lambda = math.Max(0.2, math.Min(1.0, lambda))

		structures[i] = lambda
		stresses[i] = 10.0*lambda + 50.0*math.Pow(shearRates[i], 0.8)*lambda
	}

	// Create batch plots
	if err := BatchPlot(shearRates, times, structures, stresses, outputDir); err != nil {
		return fmt.Errorf("failed to create batch plots: %v", err)
	}

	// Create multi-material comparison
	materials := []PlotData{
		{
			X:     times,
			Y:     structures,
			Label: "Generic Fluid",
		},
	}

	// Add modified versions for comparison
	structures2 := make([]float64, nPoints)
	for i := range structures {
		structures2[i] = math.Max(0.1, structures[i]*0.8) // Different material
	}
	materials = append(materials, PlotData{
		X:     times,
		Y:     structures2,
		Label: "Modified Material",
	})

	comparisonFile := filepath.Join(outputDir, "material_comparison.png")
	if err := MultiMaterialComparisonPlot(materials, comparisonFile); err != nil {
		return fmt.Errorf("failed to create comparison plot: %v", err)
	}

	// Create viscoelastic plots
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

	viscoFile := filepath.Join(outputDir, "viscoelastic")
	if err := ViscoelasticPlot(frequencies, gPrime, gDoublePrime, tanDelta, viscoFile); err != nil {
		return fmt.Errorf("failed to create viscoelastic plots: %v", err)
	}

	fmt.Printf("Demo plots created successfully in: %s\n", outputDir)
	return nil
}
