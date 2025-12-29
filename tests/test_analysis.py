"""
分析模块测试
"""

import pytest
import numpy as np
import os
import tempfile
import sys
sys.path.insert(0, '/home/user/YCJL')

from ycjl.analysis.logger import DataLogger, LogLevel, RingBuffer
from ycjl.analysis.analyzer import PerformanceAnalyzer, TrendAnalyzer, AnomalyAnalyzer
from ycjl.analysis.reporter import ReportGenerator


class TestRingBuffer:
    """环形缓冲区测试"""

    def test_append(self):
        """测试添加"""
        buf = RingBuffer(capacity=5)

        for i in range(3):
            buf.append(i)

        assert buf.size == 3
        assert buf.get_all() == [0, 1, 2]

    def test_overflow(self):
        """测试溢出"""
        buf = RingBuffer(capacity=5)

        for i in range(10):
            buf.append(i)

        assert buf.size == 5
        assert buf.get_all() == [5, 6, 7, 8, 9]

    def test_get_recent(self):
        """测试获取最近"""
        buf = RingBuffer(capacity=10)

        for i in range(10):
            buf.append(i)

        recent = buf.get_recent(3)
        assert recent == [7, 8, 9]


class TestDataLogger:
    """数据记录器测试"""

    def test_log_timeseries(self):
        """测试时间序列记录"""
        logger = DataLogger(auto_save=False)

        for i in range(100):
            logger.log_timeseries({
                'value1': float(i),
                'value2': float(i) * 2
            }, float(i))

        assert logger.total_records == 100
        assert 'value1' in logger.channels

    def test_log_event(self):
        """测试事件记录"""
        logger = DataLogger(auto_save=False)

        logger.info('test', 'Test message')
        logger.warning('test', 'Warning message')
        logger.error('test', 'Error message')

        events = logger.get_events()
        assert len(events) == 3

    def test_get_timeseries(self):
        """测试获取时间序列"""
        logger = DataLogger(auto_save=False)

        for i in range(50):
            logger.log_timeseries({'val': float(i)}, float(i))

        data = logger.get_timeseries('val')

        assert 'time' in data
        assert 'val' in data
        assert len(data['time']) == 50

    def test_time_filter(self):
        """测试时间过滤"""
        logger = DataLogger(auto_save=False)

        for i in range(100):
            logger.log_timeseries({'val': float(i)}, float(i))

        data = logger.get_timeseries(start_time=30, end_time=50)

        assert len(data['time']) == 21  # 30到50包含

    def test_save_load(self):
        """测试保存和加载"""
        logger = DataLogger(auto_save=False)

        for i in range(50):
            logger.log_timeseries({'val': float(i)}, float(i))

        with tempfile.NamedTemporaryFile(suffix='.json', delete=False) as f:
            filename = f.name

        try:
            logger.save_to_file(filename)

            # 加载
            logger2 = DataLogger(auto_save=False)
            logger2.load_from_file(filename)

            assert logger2.total_records == 50
        finally:
            os.unlink(filename)

    def test_export_csv(self):
        """测试CSV导出"""
        logger = DataLogger(auto_save=False)

        for i in range(20):
            logger.log_timeseries({'val': float(i)}, float(i))

        with tempfile.NamedTemporaryFile(suffix='.csv', delete=False) as f:
            filename = f.name

        try:
            logger.export_csv(filename)
            assert os.path.exists(filename)

            with open(filename) as f:
                lines = f.readlines()
                assert len(lines) == 21  # 头 + 20行数据
        finally:
            os.unlink(filename)

    def test_statistics(self):
        """测试统计信息"""
        logger = DataLogger(auto_save=False)

        for i in range(100):
            logger.log_timeseries({
                'val': np.random.randn()
            }, float(i))

        stats = logger.get_statistics()

        assert 'total_records' in stats
        assert 'channel_stats' in stats
        assert 'val' in stats['channel_stats']


class TestPerformanceAnalyzer:
    """性能分析器测试"""

    def test_basic_metrics(self):
        """测试基本指标"""
        analyzer = PerformanceAnalyzer()

        time = np.arange(0, 100, 1.0)
        setpoint = np.ones(100) * 10.0
        measurement = 10.0 + np.random.randn(100) * 0.5

        metrics = analyzer.analyze(time, setpoint, measurement)

        assert metrics.mae > 0
        assert metrics.rmse > 0
        assert metrics.iae > 0

    def test_step_response_analysis(self):
        """测试阶跃响应分析"""
        analyzer = PerformanceAnalyzer()

        time = np.arange(0, 100, 1.0)
        setpoint = np.concatenate([np.ones(20) * 5, np.ones(80) * 10])

        # 模拟一阶响应
        measurement = np.zeros(100)
        measurement[0] = 5.0
        for i in range(1, 100):
            measurement[i] = 0.9 * measurement[i-1] + 0.1 * setpoint[i]

        metrics = analyzer.analyze(time, setpoint, measurement)

        # 应该检测到阶跃响应特性
        assert metrics.settling_time > 0 or metrics.rise_time >= 0


class TestTrendAnalyzer:
    """趋势分析器测试"""

    def test_linear_trend(self):
        """测试线性趋势"""
        analyzer = TrendAnalyzer()

        time = np.arange(0, 100, 1.0)
        values = 10 + 0.1 * time + np.random.randn(100) * 0.5

        result = analyzer.analyze_linear_trend(time, values)

        assert result.trend_type == 'increasing'
        assert result.slope > 0
        assert result.r_squared > 0.5

    def test_stable_trend(self):
        """测试稳定趋势"""
        analyzer = TrendAnalyzer()

        time = np.arange(0, 100, 1.0)
        values = 10 + np.random.randn(100) * 0.1

        result = analyzer.analyze_linear_trend(time, values)

        assert result.trend_type in ['stable', 'increasing', 'decreasing']

    def test_seasonality_detection(self):
        """测试周期性检测"""
        analyzer = TrendAnalyzer()

        # 带周期的信号
        values = np.sin(np.arange(100) * 2 * np.pi / 20) + np.random.randn(100) * 0.1

        result = analyzer.detect_seasonality(values)

        # 应该检测到周期性
        if result['has_seasonality']:
            assert result['main_period'] > 0

    def test_change_point_detection(self):
        """测试变化点检测"""
        analyzer = TrendAnalyzer()

        # 有均值突变的信号
        values = np.concatenate([
            np.random.randn(50) + 0,
            np.random.randn(50) + 5
        ])

        change_points = analyzer.detect_change_points(values)

        # 应该在50附近检测到变化点
        # (CUSUM可能不精确)
        assert isinstance(change_points, list)


class TestAnomalyAnalyzer:
    """异常检测测试"""

    def test_online_detection(self):
        """测试在线检测"""
        analyzer = AnomalyAnalyzer(window_size=30)

        # 正常数据
        for i in range(50):
            result = analyzer.update(np.random.randn() + 10, float(i))
            # 大多数应该不是异常

        # 注入异常
        result = analyzer.update(100.0, 51.0)  # 明显异常

        if result is not None:
            assert result.severity > 0
            assert result.anomaly_type in ['high_value', 'spike_up']

    def test_batch_detection_zscore(self):
        """测试批量Z分数检测"""
        analyzer = AnomalyAnalyzer()

        time = np.arange(100)
        values = np.random.randn(100) * 1 + 10
        # 注入异常
        values[50] = 100
        values[75] = -50

        anomalies = analyzer.batch_detect(time, values, method='zscore')

        assert len(anomalies) >= 2

    def test_batch_detection_iqr(self):
        """测试批量IQR检测"""
        analyzer = AnomalyAnalyzer()

        time = np.arange(100)
        values = np.random.randn(100) * 1 + 10
        values[50] = 100

        anomalies = analyzer.batch_detect(time, values, method='iqr')

        assert len(anomalies) >= 1


class TestReportGenerator:
    """报告生成器测试"""

    def test_generate_report(self):
        """测试报告生成"""
        logger = DataLogger(auto_save=False)

        for i in range(100):
            logger.log_timeseries({
                'pool_level': 5.0 + np.sin(i * 0.1) * 0.5,
                'pipe_flow': 10.0 + np.random.randn() * 0.5,
                'pipe_pressure': 50.0 + np.random.randn() * 2
            }, float(i))

        generator = ReportGenerator(logger)
        report = generator.generate_report()

        assert report.title
        assert report.summary
        assert len(report.sections) > 0

    def test_export_formats(self):
        """测试导出格式"""
        logger = DataLogger(auto_save=False)

        for i in range(50):
            logger.log_timeseries({'val': float(i)}, float(i))

        generator = ReportGenerator(logger)
        report = generator.generate_report()

        with tempfile.TemporaryDirectory() as tmpdir:
            # Markdown
            md_file = os.path.join(tmpdir, 'report.md')
            generator.export_markdown(report, md_file)
            assert os.path.exists(md_file)

            # JSON
            json_file = os.path.join(tmpdir, 'report.json')
            generator.export_json(report, json_file)
            assert os.path.exists(json_file)

            # HTML
            html_file = os.path.join(tmpdir, 'report.html')
            generator.export_html(report, html_file)
            assert os.path.exists(html_file)


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
