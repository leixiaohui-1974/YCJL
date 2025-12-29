"""
部署接口测试
===========

测试工程部署接口和配置验证功能
"""

import pytest
import json
import tempfile
import os
from datetime import datetime

from ycjl.deployment import (
    DeploymentEnvironment,
    SystemStatus,
    ValidationResult,
    SystemHealth,
    ConfigValidator,
    DeploymentManager,
    SCADAInterface,
    create_production_deployment,
    create_testing_deployment
)
from ycjl.config.config_database import ProjectParams


class TestConfigValidator:
    """配置验证器测试"""

    def test_validate_all(self):
        """测试完整验证"""
        validator = ConfigValidator(ProjectParams)
        result = validator.validate_all()

        assert isinstance(result, ValidationResult)
        assert isinstance(result.is_valid, bool)
        assert isinstance(result.errors, list)
        assert isinstance(result.warnings, list)
        assert isinstance(result.info, list)

    def test_validation_result_structure(self):
        """测试验证结果结构"""
        validator = ConfigValidator(ProjectParams)
        result = validator.validate_all()

        # 应有info记录验证过程
        assert len(result.info) > 0

    def test_water_level_validation(self):
        """测试水位验证"""
        validator = ConfigValidator(ProjectParams)
        result = validator.validate_all()

        # 正常配置应该通过水位验证
        water_level_errors = [e for e in result.errors if '水位' in e]
        assert len(water_level_errors) == 0


class TestDeploymentManager:
    """部署管理器测试"""

    def test_create_development_manager(self):
        """测试创建开发环境管理器"""
        manager = DeploymentManager(DeploymentEnvironment.DEVELOPMENT)
        assert manager.environment == DeploymentEnvironment.DEVELOPMENT
        assert manager.status == SystemStatus.INITIALIZING

    def test_create_testing_manager(self):
        """测试创建测试环境管理器"""
        manager = create_testing_deployment()
        assert manager.environment == DeploymentEnvironment.TESTING

    def test_validate_configuration(self):
        """测试配置验证"""
        manager = DeploymentManager()
        result = manager.validate_configuration()
        assert isinstance(result, ValidationResult)

    def test_initialize_system(self):
        """测试系统初始化"""
        manager = DeploymentManager()
        success = manager.initialize_system()

        # 初始化应该成功
        assert success is True
        assert manager.status == SystemStatus.READY

    def test_start_stop(self):
        """测试启动和停止"""
        manager = DeploymentManager()
        manager.initialize_system()

        # 启动
        success = manager.start()
        assert success is True
        assert manager.status == SystemStatus.RUNNING
        assert manager.start_time is not None

        # 停止
        manager.stop()
        assert manager.status == SystemStatus.SHUTDOWN

    def test_start_without_init(self):
        """测试未初始化时启动"""
        manager = DeploymentManager()
        success = manager.start()
        # 未初始化应该启动失败
        assert success is False

    def test_get_health(self):
        """测试获取健康状态"""
        manager = DeploymentManager()
        health = manager.get_health()

        assert isinstance(health, SystemHealth)
        assert isinstance(health.timestamp, datetime)
        assert health.status == manager.status
        assert isinstance(health.component_status, dict)

    def test_export_configuration(self):
        """测试导出配置"""
        manager = DeploymentManager()

        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            temp_path = f.name

        try:
            manager.export_configuration(temp_path)

            # 检查文件存在
            assert os.path.exists(temp_path)

            # 检查内容
            with open(temp_path, 'r', encoding='utf-8') as f:
                data = json.load(f)

            assert 'version' in data
            assert 'environment' in data
            assert 'summary' in data
            assert 'validation' in data
        finally:
            if os.path.exists(temp_path):
                os.unlink(temp_path)


class TestSCADAInterface:
    """SCADA接口测试"""

    def test_create_interface(self):
        """测试创建接口"""
        scada = SCADAInterface()
        assert scada.connected is False

    def test_point_table(self):
        """测试点表"""
        scada = SCADAInterface()
        points = scada.get_point_table()

        assert 'RSV_LEVEL' in points
        assert 'INTAKE_FLOW' in points
        assert 'POOL_LEVEL' in points

    def test_point_structure(self):
        """测试点表结构"""
        scada = SCADAInterface()
        points = scada.get_point_table()

        for point_id, point in points.items():
            assert 'description' in point
            assert 'unit' in point
            assert 'type' in point
            assert point['type'] in ['AI', 'AO', 'DI', 'DO']
            assert 'range' in point

    def test_connect_disconnect(self):
        """测试连接和断开"""
        scada = SCADAInterface()

        success = scada.connect()
        assert success is True
        assert scada.connected is True

        scada.disconnect()
        assert scada.connected is False

    def test_read_without_connect(self):
        """测试未连接时读取"""
        scada = SCADAInterface()
        value = scada.read_point('RSV_LEVEL')
        assert value is None

    def test_write_without_connect(self):
        """测试未连接时写入"""
        scada = SCADAInterface()
        success = scada.write_point('VALVE_INLINE_1_CMD', 50.0)
        assert success is False

    def test_read_unknown_point(self):
        """测试读取未知点"""
        scada = SCADAInterface()
        scada.connect()
        value = scada.read_point('UNKNOWN_POINT')
        assert value is None

    def test_write_readonly_point(self):
        """测试写入只读点"""
        scada = SCADAInterface()
        scada.connect()
        # AI点不可写
        success = scada.write_point('RSV_LEVEL', 370.0)
        assert success is False

    def test_write_out_of_range(self):
        """测试写入超范围值"""
        scada = SCADAInterface()
        scada.connect()
        # 阀门开度范围是0-100
        success = scada.write_point('VALVE_INLINE_1_CMD', 150.0)
        assert success is False


class TestDeploymentEnvironments:
    """部署环境测试"""

    def test_all_environments(self):
        """测试所有环境"""
        environments = [
            DeploymentEnvironment.DEVELOPMENT,
            DeploymentEnvironment.TESTING,
            DeploymentEnvironment.STAGING,
            DeploymentEnvironment.PRODUCTION
        ]

        for env in environments:
            manager = DeploymentManager(env)
            assert manager.environment == env


class TestSystemStatus:
    """系统状态测试"""

    def test_status_transitions(self):
        """测试状态转换"""
        manager = DeploymentManager()

        # 初始状态
        assert manager.status == SystemStatus.INITIALIZING

        # 初始化后
        manager.initialize_system()
        assert manager.status == SystemStatus.READY

        # 启动后
        manager.start()
        assert manager.status == SystemStatus.RUNNING

        # 停止后
        manager.stop()
        assert manager.status == SystemStatus.SHUTDOWN


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
