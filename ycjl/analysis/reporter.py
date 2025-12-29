"""
报告生成器
==========

生成分析报告
"""

import numpy as np
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, field
from datetime import datetime
import json
import os

from .logger import DataLogger
from .analyzer import PerformanceAnalyzer, TrendAnalyzer, AnomalyAnalyzer


@dataclass
class ReportSection:
    """报告章节"""
    title: str
    content: str
    data: Dict = field(default_factory=dict)
    subsections: List['ReportSection'] = field(default_factory=list)


@dataclass
class Report:
    """分析报告"""
    title: str
    generated_at: str
    summary: str
    sections: List[ReportSection] = field(default_factory=list)
    metadata: Dict = field(default_factory=dict)


class ReportGenerator:
    """
    报告生成器

    功能:
    - 自动生成分析报告
    - 多种输出格式
    - 可定制模板
    """

    def __init__(self, logger: DataLogger = None):
        self.logger = logger
        self.perf_analyzer = PerformanceAnalyzer()
        self.trend_analyzer = TrendAnalyzer()
        self.anomaly_analyzer = AnomalyAnalyzer()

    def generate_report(self, title: str = "系统运行分析报告",
                        include_sections: List[str] = None) -> Report:
        """
        生成完整报告

        Parameters:
            title: 报告标题
            include_sections: 包含的章节

        Returns:
            Report
        """
        if include_sections is None:
            include_sections = ['overview', 'performance', 'trends', 'anomalies', 'recommendations']

        report = Report(
            title=title,
            generated_at=datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            summary="",
            sections=[],
            metadata={}
        )

        # 获取数据
        if self.logger:
            stats = self.logger.get_statistics()
            report.metadata['data_stats'] = stats

        # 生成各章节
        if 'overview' in include_sections:
            report.sections.append(self._generate_overview())

        if 'performance' in include_sections:
            report.sections.append(self._generate_performance_section())

        if 'trends' in include_sections:
            report.sections.append(self._generate_trend_section())

        if 'anomalies' in include_sections:
            report.sections.append(self._generate_anomaly_section())

        if 'recommendations' in include_sections:
            report.sections.append(self._generate_recommendations())

        # 生成摘要
        report.summary = self._generate_summary(report)

        return report

    def _generate_overview(self) -> ReportSection:
        """生成概述章节"""
        if not self.logger:
            return ReportSection(
                title="系统概述",
                content="无可用数据"
            )

        stats = self.logger.get_statistics()

        content = f"""
## 数据概述

- 记录总数: {stats.get('total_records', 0)}
- 记录时长: {stats.get('duration', 0):.2f} 秒
- 通道数量: {len(stats.get('channels', []))}
- 通道列表: {', '.join(stats.get('channels', [])[:10])}
"""

        # 各通道统计
        channel_stats = stats.get('channel_stats', {})
        if channel_stats:
            content += "\n## 通道统计\n\n"
            content += "| 通道 | 均值 | 标准差 | 最小值 | 最大值 |\n"
            content += "|------|------|--------|--------|--------|\n"

            for ch, cs in list(channel_stats.items())[:10]:
                content += f"| {ch} | {cs['mean']:.3f} | {cs['std']:.3f} | {cs['min']:.3f} | {cs['max']:.3f} |\n"

        return ReportSection(
            title="系统概述",
            content=content,
            data=stats
        )

    def _generate_performance_section(self) -> ReportSection:
        """生成性能分析章节"""
        if not self.logger:
            return ReportSection(title="性能分析", content="无可用数据")

        # 获取关键通道数据
        data = self.logger.get_timeseries()
        time_array = data.get('time', np.array([]))

        content = "## 控制性能分析\n\n"
        perf_data = {}

        # 分析各控制回路
        control_pairs = [
            ('pool_level', 'pool_level_sp'),
            ('pipe_flow', 'flow_sp'),
            ('pipe_pressure', 'pressure_sp')
        ]

        for pv_name, sp_name in control_pairs:
            pv = data.get(pv_name)
            sp = data.get(sp_name)

            if pv is not None and len(pv) > 10:
                if sp is None:
                    sp = np.ones_like(pv) * np.mean(pv)

                metrics = self.perf_analyzer.analyze(time_array, sp, pv)

                content += f"### {pv_name}\n\n"
                content += f"- MAE: {metrics.mae:.4f}\n"
                content += f"- RMSE: {metrics.rmse:.4f}\n"
                content += f"- IAE: {metrics.iae:.4f}\n"

                if metrics.settling_time > 0:
                    content += f"- 调节时间: {metrics.settling_time:.2f}s\n"
                if metrics.overshoot > 0:
                    content += f"- 超调量: {metrics.overshoot:.2f}%\n"

                content += "\n"
                perf_data[pv_name] = {
                    'mae': metrics.mae,
                    'rmse': metrics.rmse,
                    'iae': metrics.iae
                }

        return ReportSection(
            title="性能分析",
            content=content,
            data=perf_data
        )

    def _generate_trend_section(self) -> ReportSection:
        """生成趋势分析章节"""
        if not self.logger:
            return ReportSection(title="趋势分析", content="无可用数据")

        data = self.logger.get_timeseries()
        time_array = data.get('time', np.array([]))

        content = "## 趋势分析\n\n"
        trend_data = {}

        for channel in self.logger.channels[:10]:
            values = data.get(channel)
            if values is not None and len(values) > 10:
                # 线性趋势
                trend = self.trend_analyzer.analyze_linear_trend(time_array, values)

                content += f"### {channel}\n\n"
                content += f"- 趋势类型: {trend.trend_type}\n"
                content += f"- 斜率: {trend.slope:.6f}\n"
                content += f"- R²: {trend.r_squared:.4f}\n"
                content += f"- 置信度: {trend.confidence:.2%}\n"

                # 周期性
                seasonality = self.trend_analyzer.detect_seasonality(values)
                if seasonality.get('has_seasonality'):
                    content += f"- 检测到周期性, 主周期: {seasonality['main_period']}\n"

                # 变化点
                change_points = self.trend_analyzer.detect_change_points(values)
                if change_points:
                    content += f"- 检测到 {len(change_points)} 个变化点\n"

                content += "\n"

                trend_data[channel] = {
                    'trend_type': trend.trend_type,
                    'slope': trend.slope,
                    'r_squared': trend.r_squared
                }

        return ReportSection(
            title="趋势分析",
            content=content,
            data=trend_data
        )

    def _generate_anomaly_section(self) -> ReportSection:
        """生成异常分析章节"""
        if not self.logger:
            return ReportSection(title="异常检测", content="无可用数据")

        data = self.logger.get_timeseries()
        time_array = data.get('time', np.array([]))

        content = "## 异常检测\n\n"
        anomaly_data = {}
        total_anomalies = 0

        for channel in self.logger.channels[:10]:
            values = data.get(channel)
            if values is not None and len(values) > 10:
                anomalies = self.anomaly_analyzer.batch_detect(
                    time_array, values, method='zscore'
                )

                if anomalies:
                    total_anomalies += len(anomalies)

                    content += f"### {channel}\n\n"
                    content += f"检测到 {len(anomalies)} 个异常点:\n\n"

                    for ano in anomalies[:5]:  # 只显示前5个
                        content += f"- 时间: {ano.timestamp:.2f}, "
                        content += f"值: {ano.value:.4f}, "
                        content += f"偏差: {ano.deviation:.4f}, "
                        content += f"严重程度: {ano.severity:.2f}\n"

                    if len(anomalies) > 5:
                        content += f"- ... 还有 {len(anomalies) - 5} 个异常\n"

                    content += "\n"

                    anomaly_data[channel] = {
                        'count': len(anomalies),
                        'max_severity': max(a.severity for a in anomalies)
                    }

        content = f"共检测到 {total_anomalies} 个异常点\n\n" + content

        return ReportSection(
            title="异常检测",
            content=content,
            data={'total_anomalies': total_anomalies, 'by_channel': anomaly_data}
        )

    def _generate_recommendations(self) -> ReportSection:
        """生成建议章节"""
        recommendations = []

        if self.logger:
            stats = self.logger.get_statistics()
            data = self.logger.get_timeseries()

            # 基于分析结果生成建议
            for channel, cs in stats.get('channel_stats', {}).items():
                # 高变异性
                if cs['std'] / (abs(cs['mean']) + 1e-6) > 0.3:
                    recommendations.append(
                        f"- {channel} 变异系数较高，建议检查控制回路稳定性"
                    )

                # 接近限值
                if 'level' in channel.lower():
                    if cs['min'] < 2.0:
                        recommendations.append(
                            f"- {channel} 曾接近下限，建议增加安全裕度"
                        )

            # 通用建议
            if stats.get('total_records', 0) < 100:
                recommendations.append("- 数据量较少，建议延长监测时间以获得更可靠的分析")

        if not recommendations:
            recommendations.append("- 系统运行正常，无特殊建议")

        content = "## 运行建议\n\n" + "\n".join(recommendations)

        return ReportSection(
            title="运行建议",
            content=content,
            data={'recommendations': recommendations}
        )

    def _generate_summary(self, report: Report) -> str:
        """生成报告摘要"""
        summary = []

        # 数据统计
        if 'data_stats' in report.metadata:
            stats = report.metadata['data_stats']
            summary.append(f"本报告基于 {stats.get('total_records', 0)} 条记录")
            summary.append(f"覆盖时长 {stats.get('duration', 0):.1f} 秒")

        # 各章节摘要
        for section in report.sections:
            if section.title == "异常检测" and section.data:
                total = section.data.get('total_anomalies', 0)
                summary.append(f"共检测到 {total} 个异常点")

            if section.title == "性能分析" and section.data:
                avg_mae = np.mean([d['mae'] for d in section.data.values()])
                summary.append(f"平均控制误差: {avg_mae:.4f}")

        return ". ".join(summary) + "." if summary else "报告生成完成。"

    def export_markdown(self, report: Report, filename: str):
        """导出为Markdown"""
        content = f"# {report.title}\n\n"
        content += f"*生成时间: {report.generated_at}*\n\n"
        content += f"**摘要**: {report.summary}\n\n"
        content += "---\n\n"

        for section in report.sections:
            content += f"# {section.title}\n\n"
            content += section.content + "\n\n"

        with open(filename, 'w', encoding='utf-8') as f:
            f.write(content)

    def export_json(self, report: Report, filename: str):
        """导出为JSON"""
        data = {
            'title': report.title,
            'generated_at': report.generated_at,
            'summary': report.summary,
            'metadata': report.metadata,
            'sections': [
                {
                    'title': s.title,
                    'content': s.content,
                    'data': s.data
                }
                for s in report.sections
            ]
        }

        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)

    def export_html(self, report: Report, filename: str):
        """导出为HTML"""
        html = f"""<!DOCTYPE html>
<html>
<head>
    <meta charset="utf-8">
    <title>{report.title}</title>
    <style>
        body {{ font-family: Arial, sans-serif; margin: 40px; }}
        h1 {{ color: #333; }}
        h2 {{ color: #666; border-bottom: 1px solid #ddd; }}
        .summary {{ background: #f5f5f5; padding: 15px; border-radius: 5px; }}
        table {{ border-collapse: collapse; width: 100%; }}
        th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
        th {{ background: #4CAF50; color: white; }}
    </style>
</head>
<body>
    <h1>{report.title}</h1>
    <p><em>生成时间: {report.generated_at}</em></p>
    <div class="summary">
        <strong>摘要:</strong> {report.summary}
    </div>
    <hr>
"""

        for section in report.sections:
            html += f"<h2>{section.title}</h2>\n"
            # 简单的Markdown到HTML转换
            content = section.content.replace('\n\n', '</p><p>')
            content = content.replace('\n', '<br>')
            html += f"<p>{content}</p>\n"

        html += "</body></html>"

        with open(filename, 'w', encoding='utf-8') as f:
            f.write(html)
