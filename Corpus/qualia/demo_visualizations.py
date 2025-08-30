#!/usr/bin/env python3
"""
Reverse Koopman Penetration Testing Framework - Visualization Demo
Python-based visualization using matplotlib and plotly for security data analysis
"""

import numpy as np
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import pandas as pd
import seaborn as sns
from datetime import datetime, timedelta
import json
import os
from pathlib import Path

# Set style for matplotlib
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

class SecurityVisualizationDemo:
    """Demonstration of security data visualization capabilities"""

    def __init__(self):
        self.output_dir = Path("visualizations")
        self.output_dir.mkdir(exist_ok=True)

        # Sample security findings data
        self.findings_data = self.generate_sample_findings()
        self.time_series_data = self.generate_time_series_data()

    def generate_sample_findings(self):
        """Generate sample security findings data"""
        findings = []

        # Java application vulnerabilities
        java_vulns = [
            {"type": "SQL Injection", "severity": "High", "framework": "Java", "location": "Database Layer"},
            {"type": "Buffer Overflow", "severity": "Critical", "framework": "Java", "location": "Memory Management"},
            {"type": "Weak Encryption", "severity": "Medium", "framework": "Java", "location": "Cryptography"},
            {"type": "Input Validation", "severity": "Medium", "framework": "Java", "location": "Input Processing"},
            {"type": "Authentication", "severity": "High", "framework": "Java", "location": "Security Layer"},
        ]

        # GPTOSS AI model vulnerabilities
        gptoss_vulns = [
            {"type": "Prompt Injection", "severity": "Critical", "framework": "GPTOSS", "location": "AI Model"},
            {"type": "Model Inversion", "severity": "High", "framework": "GPTOSS", "location": "AI Model"},
            {"type": "Data Leakage", "severity": "High", "framework": "GPTOSS", "location": "AI Model"},
            {"type": "Jailbreak Attack", "severity": "Critical", "framework": "GPTOSS", "location": "AI Model"},
            {"type": "Membership Inference", "severity": "Medium", "framework": "GPTOSS", "location": "AI Model"},
        ]

        findings.extend(java_vulns)
        findings.extend(gptoss_vulns)

        return pd.DataFrame(findings)

    def generate_time_series_data(self):
        """Generate time series data for findings over time"""
        base_time = datetime.now()
        times = [base_time + timedelta(minutes=i*30) for i in range(24)]

        data = []
        for i, time in enumerate(times):
            # Simulate increasing findings over time
            java_count = min(5, i // 2)
            gptoss_count = min(3, i // 4)

            data.extend([
                {"timestamp": time, "framework": "Java", "findings": java_count, "type": "Security Findings"},
                {"timestamp": time, "framework": "GPTOSS", "findings": gptoss_count, "type": "Security Findings"}
            ])

        return pd.DataFrame(data)

    def create_severity_pie_chart(self):
        """Create pie chart showing severity distribution"""
        print("Creating severity pie chart...")

        severity_counts = self.findings_data['severity'].value_counts()

        plt.figure(figsize=(10, 8))
        colors = ['#ff4444', '#ff8800', '#ffaa00', '#4488ff', '#666666']

        plt.pie(severity_counts.values, labels=severity_counts.index,
                autopct='%1.1f%%', colors=colors[:len(severity_counts)],
                startangle=90, shadow=True, explode=[0.1 if x == 'Critical' else 0 for x in severity_counts.index])

        plt.title('Security Findings by Severity', fontsize=16, fontweight='bold')
        plt.axis('equal')

        output_path = self.output_dir / "severity_pie_chart.png"
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()

        print(f"✅ Saved severity pie chart to {output_path}")

    def create_vulnerability_bar_chart(self):
        """Create bar chart showing vulnerability types by framework"""
        print("Creating vulnerability bar chart...")

        plt.figure(figsize=(12, 8))

        # Group by framework and type
        grouped = self.findings_data.groupby(['framework', 'type']).size().unstack(fill_value=0)

        # Create stacked bar chart
        ax = grouped.plot(kind='bar', stacked=True, figsize=(12, 8))

        plt.title('Vulnerability Types by Framework', fontsize=16, fontweight='bold')
        plt.xlabel('Framework', fontsize=12)
        plt.ylabel('Number of Findings', fontsize=12)
        plt.legend(title='Vulnerability Type', bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.xticks(rotation=45)
        plt.tight_layout()

        output_path = self.output_dir / "vulnerability_bar_chart.png"
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()

        print(f"✅ Saved vulnerability bar chart to {output_path}")

    def create_time_series_plot(self):
        """Create time series plot of findings over time"""
        print("Creating time series plot...")

        plt.figure(figsize=(14, 8))

        # Plot time series for each framework
        for framework in self.time_series_data['framework'].unique():
            data = self.time_series_data[self.time_series_data['framework'] == framework]
            plt.plot(data['timestamp'], data['findings'],
                    label=framework, marker='o', linewidth=2, markersize=4)

        plt.title('Security Findings Over Time', fontsize=16, fontweight='bold')
        plt.xlabel('Time', fontsize=12)
        plt.ylabel('Number of Findings', fontsize=12)
        plt.legend(title='Framework')
        plt.grid(True, alpha=0.3)
        plt.xticks(rotation=45)
        plt.tight_layout()

        output_path = self.output_dir / "time_series_plot.png"
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()

        print(f"✅ Saved time series plot to {output_path}")

    def create_heatmap_data(self):
        """Create heatmap showing vulnerability distribution"""
        print("Creating security heatmap...")

        # Create pivot table for heatmap
        heatmap_data = pd.pivot_table(
            self.findings_data,
            values='type',  # We'll count occurrences
            index='severity',
            columns='framework',
            aggfunc='count',
            fill_value=0
        )

        plt.figure(figsize=(10, 8))

        # Create heatmap
        sns.heatmap(heatmap_data,
                   annot=True,
                   fmt='d',
                   cmap='YlOrRd',
                   cbar_kws={'label': 'Number of Findings'})

        plt.title('Security Findings Heatmap', fontsize=16, fontweight='bold')
        plt.xlabel('Framework', fontsize=12)
        plt.ylabel('Severity', fontsize=12)
        plt.tight_layout()

        output_path = self.output_dir / "security_heatmap.png"
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()

        print(f"✅ Saved security heatmap to {output_path}")

    def create_interactive_dashboard(self):
        """Create interactive dashboard using Plotly"""
        print("Creating interactive dashboard...")

        # Create subplots
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=('Severity Distribution', 'Framework Comparison', 'Timeline', 'Heatmap'),
            specs=[[{'type': 'pie'}, {'type': 'bar'}],
                   [{'type': 'scatter'}, {'type': 'heatmap'}]]
        )

        # Pie chart for severity
        severity_counts = self.findings_data['severity'].value_counts()
        colors = ['#ff4444', '#ff8800', '#ffaa00', '#4488ff', '#666666']

        fig.add_trace(
            go.Pie(
                labels=severity_counts.index,
                values=severity_counts.values,
                name="Severity",
                marker_colors=colors[:len(severity_counts)]
            ),
            row=1, col=1
        )

        # Bar chart for frameworks
        framework_counts = self.findings_data['framework'].value_counts()

        fig.add_trace(
            go.Bar(
                x=framework_counts.index,
                y=framework_counts.values,
                name="Framework",
                marker_color=['#1f77b4', '#ff7f0e']
            ),
            row=1, col=2
        )

        # Timeline scatter plot
        for framework in self.time_series_data['framework'].unique():
            data = self.time_series_data[self.time_series_data['framework'] == framework]
            fig.add_trace(
                go.Scatter(
                    x=data['timestamp'],
                    y=data['findings'],
                    mode='lines+markers',
                    name=framework,
                    line=dict(width=2)
                ),
                row=2, col=1
            )

        # Heatmap
        heatmap_data = pd.pivot_table(
            self.findings_data,
            values='type',
            index='severity',
            columns='framework',
            aggfunc='count',
            fill_value=0
        )

        fig.add_trace(
            go.Heatmap(
                z=heatmap_data.values,
                x=heatmap_data.columns,
                y=heatmap_data.index,
                colorscale='YlOrRd',
                name="Heatmap"
            ),
            row=2, col=2
        )

        # Update layout
        fig.update_layout(
            height=800,
            width=1200,
            title_text="Interactive Security Dashboard",
            showlegend=True
        )

        # Save as HTML
        output_path = self.output_dir / "interactive_dashboard.html"
        fig.write_html(str(output_path))

        print(f"✅ Saved interactive dashboard to {output_path}")

    def create_koopman_visualization(self):
        """Create visualization of koopman operator concepts"""
        print("Creating koopman operator visualization...")

        # Generate sample koopman data
        t = np.linspace(0, 10, 100)
        x = np.sin(t) + 0.1 * np.cos(5 * t)  # Nonlinear system

        # Observable functions
        obs1 = x  # Identity
        obs2 = x**2  # Quadratic
        obs3 = np.sin(x)  # Sinusoidal

        fig, axes = plt.subplots(2, 2, figsize=(15, 12))

        # Original system
        axes[0, 0].plot(t, x, 'b-', linewidth=2)
        axes[0, 0].set_title('Original Nonlinear System')
        axes[0, 0].set_xlabel('Time')
        axes[0, 0].set_ylabel('x(t)')
        axes[0, 0].grid(True, alpha=0.3)

        # Observable functions
        axes[0, 1].plot(obs1, obs2, 'r.', alpha=0.6)
        axes[0, 1].set_title('Observable Space: x vs x²')
        axes[0, 1].set_xlabel('x(t)')
        axes[0, 1].set_ylabel('x(t)²')
        axes[0, 1].grid(True, alpha=0.3)

        axes[1, 0].plot(obs1, obs3, 'g.', alpha=0.6)
        axes[1, 0].set_title('Observable Space: x vs sin(x)')
        axes[1, 0].set_xlabel('x(t)')
        axes[1, 0].set_ylabel('sin(x(t))')
        axes[1, 0].grid(True, alpha=0.3)

        # Phase portrait with koopman modes
        axes[1, 1].plot(x[:-1], x[1:], 'b-', alpha=0.7)
        axes[1, 1].set_title('Phase Portrait (Koopman Linearization)')
        axes[1, 1].set_xlabel('x(t)')
        axes[1, 1].set_ylabel('x(t+1)')
        axes[1, 1].grid(True, alpha=0.3)

        # Add linear approximation
        x_range = np.linspace(min(x), max(x), 10)
        axes[1, 1].plot(x_range, x_range, 'r--', label='Linear Approximation')
        axes[1, 1].legend()

        plt.suptitle('Koopman Operator Visualization for Security Analysis', fontsize=16, fontweight='bold')
        plt.tight_layout()

        output_path = self.output_dir / "koopman_visualization.png"
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()

        print(f"✅ Saved koopman visualization to {output_path}")

    def create_comprehensive_report(self):
        """Create comprehensive HTML report with all visualizations"""
        print("Creating comprehensive HTML report...")

        html_content = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>Reverse Koopman Penetration Testing Report</title>
            <style>
                body {{
                    font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
                    margin: 40px;
                    background-color: #f8f9fa;
                }}
                .container {{
                    max-width: 1200px;
                    margin: 0 auto;
                    background: white;
                    padding: 30px;
                    border-radius: 10px;
                    box-shadow: 0 0 20px rgba(0,0,0,0.1);
                }}
                h1 {{
                    color: #2c3e50;
                    text-align: center;
                    margin-bottom: 30px;
                }}
                .section {{
                    margin-bottom: 40px;
                    padding: 20px;
                    border: 1px solid #dee2e6;
                    border-radius: 8px;
                }}
                .chart-container {{
                    text-align: center;
                    margin: 20px 0;
                }}
                .stats-grid {{
                    display: grid;
                    grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
                    gap: 20px;
                    margin: 20px 0;
                }}
                .stat-card {{
                    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                    color: white;
                    padding: 20px;
                    border-radius: 8px;
                    text-align: center;
                }}
                .metric {{
                    font-size: 2em;
                    font-weight: bold;
                    margin-bottom: 5px;
                }}
                .label {{
                    font-size: 0.9em;
                    opacity: 0.9;
                }}
                table {{
                    width: 100%;
                    border-collapse: collapse;
                    margin: 20px 0;
                }}
                th, td {{
                    padding: 12px;
                    text-align: left;
                    border-bottom: 1px solid #ddd;
                }}
                th {{
                    background-color: #f8f9fa;
                    font-weight: bold;
                }}
                .severity-critical {{ color: #dc3545; font-weight: bold; }}
                .severity-high {{ color: #fd7e14; font-weight: bold; }}
                .severity-medium {{ color: #ffc107; font-weight: bold; }}
                .severity-low {{ color: #28a745; font-weight: bold; }}
            </style>
        </head>
        <body>
            <div class="container">
                <h1>🔒 Reverse Koopman Penetration Testing Report</h1>

                <div class="stats-grid">
                    <div class="stat-card">
                        <div class="metric">{len(self.findings_data)}</div>
                        <div class="label">Total Findings</div>
                    </div>
                    <div class="stat-card">
                        <div class="metric">{self.findings_data[self.findings_data['severity'] == 'Critical'].shape[0]}</div>
                        <div class="label">Critical Issues</div>
                    </div>
                    <div class="stat-card">
                        <div class="metric">{self.findings_data['framework'].nunique()}</div>
                        <div class="label">Frameworks Tested</div>
                    </div>
                    <div class="stat-card">
                        <div class="metric">{self.findings_data['type'].nunique()}</div>
                        <div class="label">Vulnerability Types</div>
                    </div>
                </div>

                <div class="section">
                    <h2>📊 Executive Summary</h2>
                    <p>This report presents the findings from a comprehensive security assessment using the Reverse Koopman Penetration Testing Framework. The assessment covered both traditional Java application security testing and advanced GPTOSS 2.0 AI model security evaluation.</p>

                    <h3>Key Findings:</h3>
                    <ul>
                        <li><strong>{self.findings_data[self.findings_data['severity'] == 'Critical'].shape[0]} Critical vulnerabilities</strong> requiring immediate attention</li>
                        <li><strong>{self.findings_data['framework'].value_counts().iloc[0]} {self.findings_data['framework'].mode().iloc[0]}</strong> framework vulnerabilities identified</li>
                        <li><strong>{self.findings_data['type'].nunique()} different vulnerability types</strong> detected across all frameworks</li>
                        <li><strong>AI Model Security:</strong> {self.findings_data[self.findings_data['framework'] == 'GPTOSS'].shape[0]} GPTOSS-specific findings</li>
                    </ul>
                </div>

                <div class="section">
                    <h2>📈 Security Findings Overview</h2>
                    <div class="chart-container">
                        <img src="severity_pie_chart.png" alt="Severity Distribution" style="max-width: 100%; height: auto;">
                    </div>
                </div>

                <div class="section">
                    <h2>🔍 Vulnerability Analysis by Framework</h2>
                    <div class="chart-container">
                        <img src="vulnerability_bar_chart.png" alt="Vulnerability Types" style="max-width: 100%; height: auto;">
                    </div>
                </div>

                <div class="section">
                    <h2>⏰ Findings Timeline</h2>
                    <div class="chart-container">
                        <img src="time_series_plot.png" alt="Findings Over Time" style="max-width: 100%; height: auto;">
                    </div>
                </div>

                <div class="section">
                    <h2>🌡️ Security Heatmap</h2>
                    <div class="chart-container">
                        <img src="security_heatmap.png" alt="Security Heatmap" style="max-width: 100%; height: auto;">
                    </div>
                </div>

                <div class="section">
                    <h2>🧠 Koopman Operator Analysis</h2>
                    <div class="chart-container">
                        <img src="koopman_visualization.png" alt="Koopman Visualization" style="max-width: 100%; height: auto;">
                    </div>
                    <p><em>The koopman operator provides mathematical linearization of nonlinear systems, enabling advanced security analysis and anomaly detection.</em></p>
                </div>

                <div class="section">
                    <h2>📋 Detailed Findings</h2>
                    <table>
                        <thead>
                            <tr>
                                <th>Framework</th>
                                <th>Vulnerability Type</th>
                                <th>Severity</th>
                                <th>Location</th>
                            </tr>
                        </thead>
                        <tbody>
        """

        # Add table rows
        for _, finding in self.findings_data.iterrows():
            severity_class = f"severity-{finding['severity'].lower()}"
            html_content += f"""
                            <tr>
                                <td>{finding['framework']}</td>
                                <td>{finding['type']}</td>
                                <td class="{severity_class}">{finding['severity']}</td>
                                <td>{finding['location']}</td>
                            </tr>
            """

        html_content += """
                        </tbody>
                    </table>
                </div>

                <div class="section">
                    <h2>🎯 Recommendations</h2>
                    <h3>Immediate Actions (Critical):</h3>
                    <ul>
                        <li>Address all critical vulnerabilities before deployment</li>
                        <li>Implement comprehensive input validation and sanitization</li>
                        <li>Strengthen AI model security mechanisms</li>
                        <li>Review and update cryptographic implementations</li>
                    </ul>

                    <h3>Short-term (High Priority):</h3>
                    <ul>
                        <li>Implement automated security testing in CI/CD pipeline</li>
                        <li>Regular dependency vulnerability scanning</li>
                        <li>Enhanced logging and monitoring</li>
                        <li>Security training for development team</li>
                    </ul>

                    <h3>Long-term (Enhancement):</h3>
                    <ul>
                        <li>Implement advanced threat detection using koopman operators</li>
                        <li>Develop comprehensive security metrics dashboard</li>
                        <li>Establish security incident response procedures</li>
                        <li>Regular security audits and penetration testing</li>
                    </ul>
                </div>

                <div class="section">
                    <h2>🔗 Interactive Dashboard</h2>
                    <p>For an interactive experience, open the <a href="interactive_dashboard.html">interactive dashboard</a> in your web browser.</p>
                </div>

                <div class="section">
                    <h2>📞 Contact & Support</h2>
                    <p>
                        Generated by: Reverse Koopman Penetration Testing Framework<br>
                        Report Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}<br>
                        Framework Version: 2.0<br>
                        Contact: security@reversekoopman.dev
                    </p>
                </div>
            </div>
        </body>
        </html>
        """

        output_path = self.output_dir / "comprehensive_report.html"
        with open(output_path, 'w') as f:
            f.write(html_content)

        print(f"✅ Saved comprehensive report to {output_path}")

    def run_all_visualizations(self):
        """Run all visualization demos"""
        print("🎨 Starting Security Visualization Demo")
        print("=" * 50)

        try:
            # Create all visualizations
            self.create_severity_pie_chart()
            self.create_vulnerability_bar_chart()
            self.create_time_series_plot()
            self.create_heatmap_data()
            self.create_koopman_visualization()
            self.create_interactive_dashboard()
            self.create_comprehensive_report()

            print("\n" + "=" * 50)
            print("✅ All visualizations created successfully!")
            print(f"📁 Output directory: {self.output_dir}")
            print("\nGenerated files:")
            for file in sorted(self.output_dir.glob("*")):
                print(f"  • {file.name}")

            print("\n🔗 Open comprehensive_report.html to view the full report")
            print("🔗 Open interactive_dashboard.html for interactive analysis")

        except Exception as e:
            print(f"❌ Error creating visualizations: {e}")
            import traceback
            traceback.print_exc()


def main():
    """Main function"""
    demo = SecurityVisualizationDemo()
    demo.run_all_visualizations()


if __name__ == "__main__":
    main()
