"""
Report Generator for ACES

Generates comprehensive reports including executive summaries, detailed analysis,
and visualizations for military logistics scenarios and decision support.
"""

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Any, Union
from datetime import datetime
import json
from reportlab.lib import colors
from reportlab.lib.pagesizes import letter, A4
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image, Table, TableStyle
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
import io

from eye.analytics.decision_analyzer import DecisionAnalyzer
from eye.analytics.audit_logger import AuditLogger
from eye.analytics.strategy_comparator import StrategyComparator, ComparisonResult, BenchmarkResult
from eye.domain.environment import LogisticsEnvironment


class ReportGenerator:
    """
    Generates comprehensive reports for ACES scenarios and analytics.

    Supports multiple output formats including PDF, HTML, and JSON for
    executive summaries, detailed analysis, and decision support documentation.
    """

    def __init__(self):
        """Initialize the report generator with default styling."""
        # Set up matplotlib/seaborn styling
        plt.style.use('seaborn-v0_8')
        sns.set_palette("husl")

        # ReportLab styles
        self.styles = getSampleStyleSheet()
        self.title_style = ParagraphStyle(
            'CustomTitle',
            parent=self.styles['Heading1'],
            fontSize=24,
            spaceAfter=30,
            alignment=1  # Center
        )
        self.heading_style = self.styles['Heading2']
        self.normal_style = self.styles['Normal']

    def generate_executive_summary(
        self,
        scenario_name: str,
        benchmark_result: BenchmarkResult,
        comparison_results: List[ComparisonResult],
        key_insights: List[str],
        recommendations: List[str],
        output_path: Optional[Path] = None,
        format: str = 'pdf'
    ) -> Union[str, bytes]:
        """
        Generate an executive summary report.

        Args:
            scenario_name: Name of the scenario
            benchmark_result: Benchmark results from strategy comparison
            comparison_results: Detailed comparison results
            key_insights: Key findings and insights
            recommendations: Strategic recommendations
            output_path: Optional path to save the report
            format: Output format ('pdf', 'html', 'markdown')

        Returns:
            Report content as string or bytes
        """
        if format == 'pdf':
            return self._generate_pdf_executive_summary(
                scenario_name, benchmark_result, comparison_results,
                key_insights, recommendations, output_path
            )
        elif format == 'html':
            return self._generate_html_executive_summary(
                scenario_name, benchmark_result, comparison_results,
                key_insights, recommendations, output_path
            )
        elif format == 'markdown':
            return self._generate_markdown_executive_summary(
                scenario_name, benchmark_result, comparison_results,
                key_insights, recommendations, output_path
            )
        else:
            raise ValueError(f"Unsupported format: {format}")

    def _generate_pdf_executive_summary(
        self,
        scenario_name: str,
        benchmark_result: BenchmarkResult,
        comparison_results: List[ComparisonResult],
        key_insights: List[str],
        recommendations: List[str],
        output_path: Optional[Path] = None
    ) -> bytes:
        """Generate PDF executive summary."""
        buffer = io.BytesIO()
        doc = SimpleDocTemplate(buffer, pagesize=A4)
        story = []

        # Title
        story.append(Paragraph(f"ACES Executive Summary: {scenario_name}", self.title_style))
        story.append(Paragraph(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
                              self.styles['Italic']))
        story.append(Spacer(1, 12))

        # Executive Summary Section
        story.append(Paragraph("Executive Summary", self.heading_style))
        summary_text = f"""
        This report presents the results of strategic analysis for the {scenario_name} scenario.
        The analysis compared {len(benchmark_result.strategies)} different strategies,
        with {benchmark_result.best_strategy} emerging as the top performer.
        """
        story.append(Paragraph(summary_text, self.normal_style))
        story.append(Spacer(1, 12))

        # Key Results
        story.append(Paragraph("Key Results", self.heading_style))

        # Rankings table
        rankings_data = [['Rank', 'Strategy', 'Confidence']]
        for strategy, rank in sorted(benchmark_result.rankings.items(), key=lambda x: x[1]):
            confidence = benchmark_result.confidence_levels[strategy]
            rankings_data.append([str(rank), strategy, f"{confidence:.2f}"])

        rankings_table = Table(rankings_data)
        rankings_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
            ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, 0), 14),
            ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
            ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
            ('GRID', (0, 0), (-1, -1), 1, colors.black)
        ]))
        story.append(rankings_table)
        story.append(Spacer(1, 12))

        # Key Insights
        story.append(Paragraph("Key Insights", self.heading_style))
        for insight in key_insights:
            story.append(Paragraph(f"• {insight}", self.normal_style))
        story.append(Spacer(1, 12))

        # Recommendations
        story.append(Paragraph("Recommendations", self.heading_style))
        for rec in recommendations:
            story.append(Paragraph(f"• {rec}", self.normal_style))
        story.append(Spacer(1, 12))

        # Statistical Summary
        story.append(Paragraph("Statistical Summary", self.heading_style))
        significant_comparisons = [r for r in comparison_results if r.significant]
        story.append(Paragraph(f"Significant Comparisons: {len(significant_comparisons)}/{len(comparison_results)}",
                              self.normal_style))

        doc.build(story)

        pdf_bytes = buffer.getvalue()
        buffer.close()

        if output_path:
            output_path.parent.mkdir(parents=True, exist_ok=True)
            with open(output_path, 'wb') as f:
                f.write(pdf_bytes)

        return pdf_bytes

    def _generate_html_executive_summary(
        self,
        scenario_name: str,
        benchmark_result: BenchmarkResult,
        comparison_results: List[ComparisonResult],
        key_insights: List[str],
        recommendations: List[str],
        output_path: Optional[Path] = None
    ) -> str:
        """Generate HTML executive summary."""
        html_content = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>ACES Executive Summary: {scenario_name}</title>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 40px; }}
                .header {{ text-align: center; border-bottom: 2px solid #333; padding-bottom: 20px; }}
                .section {{ margin: 30px 0; }}
                .ranking-table {{ border-collapse: collapse; width: 100%; }}
                .ranking-table th, .ranking-table td {{ border: 1px solid #ddd; padding: 8px; text-align: center; }}
                .ranking-table th {{ background-color: #f2f2f2; }}
                ul {{ margin: 10px 0; }}
            </style>
        </head>
        <body>
            <div class="header">
                <h1>ACES Executive Summary: {scenario_name}</h1>
                <p><em>Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</em></p>
            </div>

            <div class="section">
                <h2>Executive Summary</h2>
                <p>This report presents the results of strategic analysis for the {scenario_name} scenario.
                The analysis compared {len(benchmark_result.strategies)} different strategies,
                with {benchmark_result.best_strategy} emerging as the top performer.</p>
            </div>

            <div class="section">
                <h2>Key Results</h2>
                <table class="ranking-table">
                    <tr><th>Rank</th><th>Strategy</th><th>Confidence</th></tr>
        """

        for strategy, rank in sorted(benchmark_result.rankings.items(), key=lambda x: x[1]):
            confidence = benchmark_result.confidence_levels[strategy]
            html_content += f"<tr><td>{rank}</td><td>{strategy}</td><td>{confidence:.2f}</td></tr>"

        html_content += """
                </table>
            </div>

            <div class="section">
                <h2>Key Insights</h2>
                <ul>
        """

        for insight in key_insights:
            html_content += f"<li>{insight}</li>"

        html_content += """
                </ul>
            </div>

            <div class="section">
                <h2>Recommendations</h2>
                <ul>
        """

        for rec in recommendations:
            html_content += f"<li>{rec}</li>"

        html_content += """
                </ul>
            </div>

            <div class="section">
                <h2>Statistical Summary</h2>
        """

        significant_comparisons = [r for r in comparison_results if r.significant]
        html_content += f"<p>Significant Comparisons: {len(significant_comparisons)}/{len(comparison_results)}</p>"

        html_content += """
        </body>
        </html>
        """

        if output_path:
            output_path.parent.mkdir(parents=True, exist_ok=True)
            with open(output_path, 'w', encoding='utf-8') as f:
                f.write(html_content)

        return html_content

    def _generate_markdown_executive_summary(
        self,
        scenario_name: str,
        benchmark_result: BenchmarkResult,
        comparison_results: List[ComparisonResult],
        key_insights: List[str],
        recommendations: List[str],
        output_path: Optional[Path] = None
    ) -> str:
        """Generate Markdown executive summary."""
        md_content = f"""# ACES Executive Summary: {scenario_name}

*Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}*

## Executive Summary

This report presents the results of strategic analysis for the {scenario_name} scenario.
The analysis compared {len(benchmark_result.strategies)} different strategies,
with {benchmark_result.best_strategy} emerging as the top performer.

## Key Results

| Rank | Strategy | Confidence |
|------|----------|------------|
"""

        for strategy, rank in sorted(benchmark_result.rankings.items(), key=lambda x: x[1]):
            confidence = benchmark_result.confidence_levels[strategy]
            md_content += f"| {rank} | {strategy} | {confidence:.2f} |\n"

        md_content += "\n## Key Insights\n\n"
        for insight in key_insights:
            md_content += f"- {insight}\n"

        md_content += "\n## Recommendations\n\n"
        for rec in recommendations:
            md_content += f"- {rec}\n"

        md_content += "\n## Statistical Summary\n\n"
        significant_comparisons = [r for r in comparison_results if r.significant]
        md_content += f"Significant Comparisons: {len(significant_comparisons)}/{len(comparison_results)}\n"

        if output_path:
            output_path.parent.mkdir(parents=True, exist_ok=True)
            with open(output_path, 'w', encoding='utf-8') as f:
                f.write(md_content)

        return md_content

    def generate_detailed_analysis_report(
        self,
        scenario_name: str,
        environment: LogisticsEnvironment,
        decision_analyzer: DecisionAnalyzer,
        audit_logger: AuditLogger,
        strategy_comparator: StrategyComparator,
        output_path: Optional[Path] = None,
        include_visualizations: bool = True
    ) -> str:
        """
        Generate a detailed analysis report with comprehensive metrics and visualizations.

        Args:
            scenario_name: Name of the scenario
            environment: The logistics environment
            decision_analyzer: Decision analyzer instance
            audit_logger: Audit logger instance
            strategy_comparator: Strategy comparator instance
            output_path: Optional path to save the report
            include_visualizations: Whether to include charts and plots

        Returns:
            Detailed report as markdown string
        """
        report_lines = []
        report_lines.append(f"# Detailed Analysis Report: {scenario_name}")
        report_lines.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report_lines.append("")

        # Environment Summary
        report_lines.append("## Environment Summary")
        env_info = environment.get_scenario_info()
        report_lines.append(f"- **Scenario Type:** {env_info.get('scenario_type', 'Unknown')}")
        report_lines.append(f"- **Assets:** {env_info.get('num_assets', 0)}")
        report_lines.append(f"- **Threats:** {env_info.get('num_threats', 0)}")
        report_lines.append(f"- **Time Steps:** {env_info.get('max_steps', 0)}")
        report_lines.append("")

        # Decision Analysis Summary
        report_lines.append("## Decision Analysis Summary")
        decision_summary = decision_analyzer.get_summary_stats()
        report_lines.append(f"- **Total Decisions Analyzed:** {decision_summary.get('total_decisions', 0)}")
        report_lines.append(f"- **Average Confidence:** {decision_summary.get('avg_confidence', 0):.3f}")
        report_lines.append(f"- **High-Risk Decisions:** {decision_summary.get('high_risk_count', 0)}")
        report_lines.append("")

        # Audit Summary
        report_lines.append("## Audit Summary")
        audit_summary = audit_logger.get_summary_stats()
        report_lines.append(f"- **Total Logged Decisions:** {audit_summary.get('total_decisions', 0)}")
        report_lines.append(f"- **Flagged Incidents:** {audit_summary.get('flagged_incidents', 0)}")
        report_lines.append(f"- **After-Action Reviews:** {audit_summary.get('reviews_generated', 0)}")
        report_lines.append("")

        # Strategy Comparison Summary
        report_lines.append("## Strategy Comparison Summary")
        if strategy_comparator.comparison_history:
            total_comparisons = len(strategy_comparator.comparison_history)
            significant_comparisons = sum(1 for c in strategy_comparator.comparison_history if c.significant)
            report_lines.append(f"- **Total Comparisons:** {total_comparisons}")
            report_lines.append(f"- **Significant Differences:** {significant_comparisons}")
            report_lines.append(f"- **Significance Rate:** {significant_comparisons/total_comparisons:.1%}")
        else:
            report_lines.append("- No strategy comparisons available")
        report_lines.append("")

        # Detailed Comparison Results
        if strategy_comparator.comparison_history:
            report_lines.append("## Detailed Comparison Results")
            for i, comparison in enumerate(strategy_comparator.comparison_history, 1):
                report_lines.append(f"### Comparison {i}: {comparison.strategy_a} vs {comparison.strategy_b}")
                report_lines.append(f"- **Metric:** {comparison.metric}")
                report_lines.append(f"- **Sample Sizes:** {comparison.sample_size_a} vs {comparison.sample_size_b}")
                report_lines.append(f"- **Means:** {comparison.mean_a:.3f} vs {comparison.mean_b:.3f}")
                report_lines.append(f"- **P-value:** {comparison.p_value:.4f}")
                report_lines.append(f"- **Significant:** {'Yes' if comparison.significant else 'No'}")
                report_lines.append(f"- **Effect Size:** {comparison.effect_size:.3f}")
                report_lines.append("")

        # Recommendations
        report_lines.append("## Recommendations")
        recommendations = self._generate_recommendations(
            decision_analyzer, audit_logger, strategy_comparator
        )
        for rec in recommendations:
            report_lines.append(f"- {rec}")
        report_lines.append("")

        report_content = "\n".join(report_lines)

        if output_path:
            output_path.parent.mkdir(parents=True, exist_ok=True)
            with open(output_path, 'w', encoding='utf-8') as f:
                f.write(report_content)

        return report_content

    def _generate_recommendations(
        self,
        decision_analyzer: DecisionAnalyzer,
        audit_logger: AuditLogger,
        strategy_comparator: StrategyComparator
    ) -> List[str]:
        """Generate recommendations based on analysis results."""
        recommendations = []

        # Decision analysis recommendations
        decision_stats = decision_analyzer.get_summary_stats()
        if decision_stats.get('avg_confidence', 0) < 0.7:
            recommendations.append("Consider additional training or scenario refinement to improve decision confidence")

        if decision_stats.get('high_risk_count', 0) > 10:
            recommendations.append("Review high-risk decision patterns and implement additional safeguards")

        # Audit recommendations
        audit_stats = audit_logger.get_summary_stats()
        if audit_stats.get('flagged_incidents', 0) > 5:
            recommendations.append("Conduct detailed review of flagged incidents and update decision protocols")

        # Strategy recommendations
        if strategy_comparator.comparison_history:
            significant_comparisons = [c for c in strategy_comparator.comparison_history if c.significant]
            if len(significant_comparisons) > 0:
                recommendations.append(f"Found {len(significant_comparisons)} statistically significant strategy differences - prioritize winning strategies")
            else:
                recommendations.append("No significant strategy differences found - consider alternative evaluation metrics")

        if not recommendations:
            recommendations.append("Analysis complete - no specific recommendations at this time")

        return recommendations

    def generate_scenario_comparison_report(
        self,
        scenario_results: Dict[str, Dict[str, Any]],
        output_path: Optional[Path] = None
    ) -> str:
        """
        Generate a report comparing multiple scenarios.

        Args:
            scenario_results: Dictionary mapping scenario names to their results
            output_path: Optional path to save the report

        Returns:
            Scenario comparison report as string
        """
        report_lines = []
        report_lines.append("# Scenario Comparison Report")
        report_lines.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report_lines.append("")

        # Summary table
        report_lines.append("## Scenario Summary")
        report_lines.append("| Scenario | Best Strategy | Mean Reward | Confidence | Key Metrics |")
        report_lines.append("|----------|---------------|-------------|------------|-------------|")

        for scenario_name, results in scenario_results.items():
            best_strategy = results.get('best_strategy', 'N/A')
            mean_reward = results.get('mean_reward', 0)
            confidence = results.get('confidence', 0)
            key_metrics = results.get('key_metrics', 'N/A')
            report_lines.append(f"| {scenario_name} | {best_strategy} | {mean_reward:.3f} | {confidence:.2f} | {key_metrics} |")

        report_lines.append("")

        # Detailed analysis
        report_lines.append("## Detailed Analysis")
        for scenario_name, results in scenario_results.items():
            report_lines.append(f"### {scenario_name}")
            report_lines.append(f"- **Description:** {results.get('description', 'No description available')}")
            report_lines.append(f"- **Challenges:** {results.get('challenges', 'N/A')}")
            report_lines.append(f"- **Performance Notes:** {results.get('performance_notes', 'N/A')}")
            report_lines.append("")

        report_content = "\n".join(report_lines)

        if output_path:
            output_path.parent.mkdir(parents=True, exist_ok=True)
            with open(output_path, 'w', encoding='utf-8') as f:
                f.write(report_content)

        return report_content
