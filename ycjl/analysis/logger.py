"""
数据记录器
==========

高效的时间序列数据记录:
- 内存缓冲
- 文件持久化
- 压缩存储
- 查询接口
"""

import numpy as np
from typing import Dict, List, Optional, Any, Union
from dataclasses import dataclass, field
from enum import Enum, auto
from collections import deque
import time
import json
import os
from datetime import datetime


class LogLevel(Enum):
    """日志级别"""
    DEBUG = auto()
    INFO = auto()
    WARNING = auto()
    ERROR = auto()
    CRITICAL = auto()


@dataclass
class LogEntry:
    """日志条目"""
    timestamp: float
    level: LogLevel
    source: str
    message: str
    data: Dict = field(default_factory=dict)


@dataclass
class TimeSeriesPoint:
    """时间序列点"""
    timestamp: float
    values: Dict[str, float]


class RingBuffer:
    """环形缓冲区"""

    def __init__(self, capacity: int = 10000):
        self.capacity = capacity
        self.buffer: List = [None] * capacity
        self.head = 0
        self.tail = 0
        self.size = 0

    def append(self, item):
        """添加元素"""
        self.buffer[self.tail] = item
        self.tail = (self.tail + 1) % self.capacity

        if self.size < self.capacity:
            self.size += 1
        else:
            self.head = (self.head + 1) % self.capacity

    def get_all(self) -> List:
        """获取所有元素"""
        if self.size == 0:
            return []

        if self.head < self.tail:
            return self.buffer[self.head:self.tail]
        else:
            return self.buffer[self.head:] + self.buffer[:self.tail]

    def get_recent(self, n: int) -> List:
        """获取最近n个元素"""
        all_items = self.get_all()
        return all_items[-n:] if n < len(all_items) else all_items

    def clear(self):
        """清空"""
        self.buffer = [None] * self.capacity
        self.head = 0
        self.tail = 0
        self.size = 0


class DataLogger:
    """
    数据记录器

    功能:
    - 多通道时间序列记录
    - 事件日志
    - 内存缓冲+文件持久化
    - 高效查询
    """

    def __init__(self,
                 buffer_size: int = 10000,
                 log_dir: str = None,
                 auto_save: bool = True,
                 save_interval: int = 1000):
        """
        Parameters:
            buffer_size: 缓冲区大小
            log_dir: 日志目录
            auto_save: 自动保存
            save_interval: 保存间隔(条数)
        """
        self.buffer_size = buffer_size
        self.log_dir = log_dir or "/tmp/ycjl_logs"
        self.auto_save = auto_save
        self.save_interval = save_interval

        # 时间序列缓冲
        self.timeseries_buffer = RingBuffer(buffer_size)

        # 事件日志缓冲
        self.event_buffer = RingBuffer(buffer_size)

        # 通道名称
        self.channels: List[str] = []

        # 统计
        self.total_records = 0
        self.start_time = time.time()

        # 确保日志目录存在
        os.makedirs(self.log_dir, exist_ok=True)

    def log_timeseries(self, values: Dict[str, float], timestamp: float = None):
        """
        记录时间序列数据

        Parameters:
            values: 通道值 {channel_name: value}
            timestamp: 时间戳 (None=当前时间)
        """
        if timestamp is None:
            timestamp = time.time()

        point = TimeSeriesPoint(timestamp=timestamp, values=values)
        self.timeseries_buffer.append(point)

        # 更新通道列表
        for channel in values.keys():
            if channel not in self.channels:
                self.channels.append(channel)

        self.total_records += 1

        # 自动保存
        if self.auto_save and self.total_records % self.save_interval == 0:
            self._auto_save()

    def log_event(self, level: LogLevel, source: str, message: str,
                  data: Dict = None, timestamp: float = None):
        """
        记录事件

        Parameters:
            level: 日志级别
            source: 来源
            message: 消息
            data: 附加数据
            timestamp: 时间戳
        """
        if timestamp is None:
            timestamp = time.time()

        entry = LogEntry(
            timestamp=timestamp,
            level=level,
            source=source,
            message=message,
            data=data or {}
        )
        self.event_buffer.append(entry)

    def info(self, source: str, message: str, data: Dict = None):
        """记录INFO级别"""
        self.log_event(LogLevel.INFO, source, message, data)

    def warning(self, source: str, message: str, data: Dict = None):
        """记录WARNING级别"""
        self.log_event(LogLevel.WARNING, source, message, data)

    def error(self, source: str, message: str, data: Dict = None):
        """记录ERROR级别"""
        self.log_event(LogLevel.ERROR, source, message, data)

    def get_timeseries(self, channel: str = None,
                       start_time: float = None,
                       end_time: float = None,
                       max_points: int = None) -> Dict[str, np.ndarray]:
        """
        获取时间序列数据

        Parameters:
            channel: 通道名 (None=所有)
            start_time: 起始时间
            end_time: 结束时间
            max_points: 最大点数

        Returns:
            {'time': [...], 'channel1': [...], ...}
        """
        all_points = self.timeseries_buffer.get_all()

        # 时间过滤
        if start_time is not None or end_time is not None:
            filtered = []
            for point in all_points:
                if start_time and point.timestamp < start_time:
                    continue
                if end_time and point.timestamp > end_time:
                    continue
                filtered.append(point)
            all_points = filtered

        # 限制点数
        if max_points and len(all_points) > max_points:
            step = len(all_points) // max_points
            all_points = all_points[::step]

        if not all_points:
            return {'time': np.array([])}

        # 构建结果
        result = {
            'time': np.array([p.timestamp for p in all_points])
        }

        if channel:
            result[channel] = np.array([
                p.values.get(channel, np.nan) for p in all_points
            ])
        else:
            for ch in self.channels:
                result[ch] = np.array([
                    p.values.get(ch, np.nan) for p in all_points
                ])

        return result

    def get_events(self, level: LogLevel = None,
                   source: str = None,
                   start_time: float = None,
                   end_time: float = None) -> List[LogEntry]:
        """
        获取事件日志

        Parameters:
            level: 过滤级别
            source: 过滤来源
            start_time: 起始时间
            end_time: 结束时间

        Returns:
            日志条目列表
        """
        all_events = self.event_buffer.get_all()

        filtered = []
        for event in all_events:
            if level and event.level != level:
                continue
            if source and event.source != source:
                continue
            if start_time and event.timestamp < start_time:
                continue
            if end_time and event.timestamp > end_time:
                continue
            filtered.append(event)

        return filtered

    def get_statistics(self) -> Dict:
        """获取记录统计"""
        ts_data = self.timeseries_buffer.get_all()

        if not ts_data:
            return {
                'total_records': 0,
                'channels': [],
                'duration': 0
            }

        start_ts = ts_data[0].timestamp
        end_ts = ts_data[-1].timestamp

        # 每通道统计
        channel_stats = {}
        for ch in self.channels:
            values = [p.values.get(ch) for p in ts_data if ch in p.values]
            if values:
                channel_stats[ch] = {
                    'count': len(values),
                    'mean': np.mean(values),
                    'std': np.std(values),
                    'min': np.min(values),
                    'max': np.max(values)
                }

        return {
            'total_records': self.total_records,
            'buffer_size': self.timeseries_buffer.size,
            'channels': self.channels,
            'duration': end_ts - start_ts,
            'start_time': start_ts,
            'end_time': end_ts,
            'channel_stats': channel_stats
        }

    def _auto_save(self):
        """自动保存到文件"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = os.path.join(self.log_dir, f"data_{timestamp}.json")

        self.save_to_file(filename)

    def save_to_file(self, filename: str):
        """
        保存到文件

        Parameters:
            filename: 文件路径
        """
        ts_data = self.timeseries_buffer.get_all()
        events = self.event_buffer.get_all()

        data = {
            'metadata': {
                'start_time': self.start_time,
                'total_records': self.total_records,
                'channels': self.channels
            },
            'timeseries': [
                {'timestamp': p.timestamp, 'values': p.values}
                for p in ts_data
            ],
            'events': [
                {
                    'timestamp': e.timestamp,
                    'level': e.level.name,
                    'source': e.source,
                    'message': e.message,
                    'data': e.data
                }
                for e in events
            ]
        }

        with open(filename, 'w') as f:
            json.dump(data, f, indent=2)

    def load_from_file(self, filename: str):
        """
        从文件加载

        Parameters:
            filename: 文件路径
        """
        with open(filename, 'r') as f:
            data = json.load(f)

        # 清空当前数据
        self.clear()

        # 加载时间序列
        for item in data.get('timeseries', []):
            self.log_timeseries(item['values'], item['timestamp'])

        # 加载事件
        for item in data.get('events', []):
            level = LogLevel[item['level']]
            self.log_event(
                level=level,
                source=item['source'],
                message=item['message'],
                data=item.get('data', {}),
                timestamp=item['timestamp']
            )

    def export_csv(self, filename: str, channels: List[str] = None):
        """
        导出为CSV

        Parameters:
            filename: 文件路径
            channels: 导出的通道
        """
        if channels is None:
            channels = self.channels

        data = self.get_timeseries()

        with open(filename, 'w') as f:
            # 写入头
            header = ['timestamp'] + channels
            f.write(','.join(header) + '\n')

            # 写入数据
            times = data.get('time', [])
            for i, t in enumerate(times):
                row = [str(t)]
                for ch in channels:
                    if ch in data:
                        row.append(str(data[ch][i]))
                    else:
                        row.append('')
                f.write(','.join(row) + '\n')

    def clear(self):
        """清空所有数据"""
        self.timeseries_buffer.clear()
        self.event_buffer.clear()
        self.channels = []
        self.total_records = 0


class RotatingLogger(DataLogger):
    """
    滚动日志记录器

    自动按时间或大小滚动日志文件
    """

    def __init__(self,
                 max_file_size: int = 10 * 1024 * 1024,  # 10MB
                 max_files: int = 10,
                 **kwargs):
        super().__init__(**kwargs)

        self.max_file_size = max_file_size
        self.max_files = max_files
        self.current_file_size = 0
        self.file_count = 0

    def _auto_save(self):
        """滚动保存"""
        # 检查是否需要滚动
        if self.current_file_size >= self.max_file_size:
            self.file_count += 1
            self.current_file_size = 0

            # 删除最旧的文件
            if self.file_count > self.max_files:
                self._cleanup_old_files()

        super()._auto_save()

    def _cleanup_old_files(self):
        """清理旧文件"""
        files = sorted([
            f for f in os.listdir(self.log_dir)
            if f.startswith('data_') and f.endswith('.json')
        ])

        while len(files) > self.max_files:
            oldest = files.pop(0)
            os.remove(os.path.join(self.log_dir, oldest))
