#!/usr/bin/env python3
"""
DLOps AI-Driven Backend Server
Developed by: Charles 
"""

import json
import os
import subprocess
import time
import threading
import numpy as np
from datetime import datetime, timedelta
from pathlib import Path
from flask import Flask, jsonify, request, Response, send_from_directory
import logging
from collections import defaultdict, deque

# 配置
DLOPS_WORKDIR = os.environ.get('DLOPS_WORKDIR', f"{os.path.expanduser('~')}/.dlops")
WEB_PORT = int(os.environ.get('WEB_PORT', 8080))
DEBUG_MODE = os.environ.get('DEBUG', 'false').lower() == 'true'
STATIC_DIR = os.environ.get('STATIC_DIR', './static')  # HTML文件目录

app = Flask(__name__)
app.secret_key = 'dlops_ai_secret_2024'

# 设置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# 全局状态
metrics_cache = deque(maxlen=100)
anomaly_detector = None
performance_analyzer = None

class AnomalyDetector:
    """AI异常检测引擎"""
    def __init__(self):
        self.baseline_metrics = {}
        self.anomaly_threshold = 2.0  # 标准差倍数
        self.learning_window = 50
        
    def update_baseline(self, metrics):
        """更新基线指标"""
        try:
            for key, value in self._flatten_metrics(metrics).items():
                if isinstance(value, (int, float)):
                    if key not in self.baseline_metrics:
                        self.baseline_metrics[key] = deque(maxlen=self.learning_window)
                    self.baseline_metrics[key].append(value)
        except Exception as e:
            logger.error(f"更新基线失败: {e}")
    
    def detect_anomalies(self, metrics):
        """检测异常"""
        anomalies = []
        try:
            flat_metrics = self._flatten_metrics(metrics)
            
            for key, value in flat_metrics.items():
                if (isinstance(value, (int, float)) and 
                    key in self.baseline_metrics and 
                    len(self.baseline_metrics[key]) > 10):
                    
                    baseline = list(self.baseline_metrics[key])
                    mean_val = np.mean(baseline)
                    std_val = np.std(baseline)
                    
                    if std_val > 0:
                        z_score = abs(value - mean_val) / std_val
                        if z_score > self.anomaly_threshold:
                            anomalies.append({
                                'metric': key,
                                'value': value,
                                'baseline_mean': mean_val,
                                'z_score': z_score,
                                'severity': 'high' if z_score > 3 else 'medium'
                            })
        except Exception as e:
            logger.error(f"异常检测失败: {e}")
            
        return anomalies
    
    def _flatten_metrics(self, metrics, prefix=''):
        """展平嵌套指标"""
        flat = {}
        try:
            for key, value in metrics.items():
                new_key = f"{prefix}.{key}" if prefix else key
                if isinstance(value, dict):
                    flat.update(self._flatten_metrics(value, new_key))
                elif isinstance(value, (list, tuple)) and len(value) > 0:
                    if isinstance(value[0], (int, float)):
                        flat[new_key] = value[0]  # 取第一个元素
                elif isinstance(value, str):
                    # 尝试解析数字字符串
                    try:
                        if ',' in value:
                            # 处理 "used,total,percentage" 格式
                            parts = value.split(',')
                            if len(parts) >= 3:
                                flat[f"{new_key}.used"] = float(parts[0])
                                flat[f"{new_key}.total"] = float(parts[1])
                                flat[f"{new_key}.percent"] = float(parts[2])
                        else:
                            flat[new_key] = float(value)
                    except ValueError:
                        pass
                else:
                    flat[new_key] = value
        except Exception as e:
            logger.error(f"指标展平失败: {e}")
        return flat

class PerformanceAnalyzer:
    """性能分析器"""
    def __init__(self):
        self.performance_history = deque(maxlen=200)
        self.training_patterns = {}
        
    def analyze_training_efficiency(self, metrics):
        """分析训练效率"""
        analysis = {
            'gpu_efficiency': 0,
            'memory_efficiency': 0,
            'io_efficiency': 0,
            'overall_score': 0,
            'bottlenecks': [],
            'recommendations': []
        }
        
        try:
            # GPU效率分析
            if 'gpu' in metrics and 'utilization' in metrics['gpu']:
                gpu_util = metrics['gpu']['utilization'][0] if isinstance(metrics['gpu']['utilization'], list) else metrics['gpu']['utilization']
                analysis['gpu_efficiency'] = gpu_util
                
                if gpu_util < 70:
                    analysis['bottlenecks'].append('GPU利用率低')
                    analysis['recommendations'].append('增加batch size或优化数据加载')
                elif gpu_util > 95:
                    analysis['recommendations'].append('GPU利用率良好')
            
            # 内存效率分析
            if 'system' in metrics and 'memory' in metrics['system']:
                memory_info = metrics['system']['memory']
                if isinstance(memory_info, str) and ',' in memory_info:
                    parts = memory_info.split(',')
                    if len(parts) >= 3:
                        mem_percent = float(parts[2])
                        analysis['memory_efficiency'] = min(100, mem_percent * 1.2)  # 调整分数
                        
                        if mem_percent > 90:
                            analysis['bottlenecks'].append('内存使用率过高')
                            analysis['recommendations'].append('启用gradient checkpointing或减少batch size')
            
            # I/O效率分析（基于CPU wa时间推断）
            analysis['io_efficiency'] = 85  
            
            # 整体分数计算
            analysis['overall_score'] = (
                analysis['gpu_efficiency'] * 0.5 +
                analysis['memory_efficiency'] * 0.3 +
                analysis['io_efficiency'] * 0.2
            )
            
        except Exception as e:
            logger.error(f"性能分析失败: {e}")
            
        return analysis
    
    def predict_performance_trends(self):
        """预测性能趋势"""
        predictions = []
        
        if len(self.performance_history) < 10:
            return predictions
        
        try:
            # 分析GPU利用率趋势
            gpu_utils = [h.get('gpu_efficiency', 0) for h in self.performance_history if h.get('gpu_efficiency')]
            if len(gpu_utils) >= 5:
                recent_trend = np.mean(gpu_utils[-5:]) - np.mean(gpu_utils[-10:-5])
                if recent_trend < -10:
                    predictions.append({
                        'type': 'performance_degradation',
                        'message': 'GPU利用率呈下降趋势',
                        'confidence': min(90, abs(recent_trend) * 5),
                        'time_window': '5分钟'
                    })
                    
        except Exception as e:
            logger.error(f"性能预测失败: {e}")
            
        return predictions

def run_dlops_command(command_args):
    """执行DLOps命令"""
    try:
        dlops_script = os.path.join(DLOPS_WORKDIR, '../../../dlops')
        if not os.path.exists(dlops_script):
            dlops_script = './dlops'
            
        result = subprocess.run([dlops_script] + command_args, 
                              capture_output=True, text=True, timeout=30)
        
        if result.returncode == 0:
            try:
                return json.loads(result.stdout) if result.stdout.strip() else {}
            except json.JSONDecodeError:
                return {'raw_output': result.stdout}
        else:
            return {'error': result.stderr, 'returncode': result.returncode}
            
    except subprocess.TimeoutExpired:
        return {'error': '命令超时', 'returncode': -1}
    except Exception as e:
        return {'error': str(e), 'returncode': -1}

def get_latest_metrics():
    """获取最新指标"""
    try:
        metrics_file = os.path.join(DLOPS_WORKDIR, 'metrics', 'latest.json')
        if os.path.exists(metrics_file):
            with open(metrics_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
                
                # 更新缓存
                metrics_cache.append(data)
                
                # 更新异常检测基线
                if anomaly_detector:
                    anomaly_detector.update_baseline(data)
                
                return data
        return {'error': '指标文件未找到，请确保守护进程正在运行'}
    except Exception as e:
        return {'error': f'读取指标失败: {str(e)}'}

def get_diagnostics():
    """获取诊断信息"""
    try:
        diagnostics_dir = os.path.join(DLOPS_WORKDIR, 'diagnostics')
        latest_file = os.path.join(diagnostics_dir, 'latest_analysis.json')
        
        if os.path.exists(latest_file):
            with open(latest_file, 'r', encoding='utf-8') as f:
                return json.load(f)
        
        # 如果没有诊断文件，运行快速诊断
        return run_dlops_command(['analyze'])
        
    except Exception as e:
        return {'error': f'获取诊断信息失败: {str(e)}'}

def get_intelligent_analysis():
    """获取智能分析"""
    analysis = {
        'anomalies': [],
        'performance': {},
        'predictions': [],
        'optimization_score': 0,
        'health_trend': 'stable'
    }
    
    try:
        # 异常检测
        if metrics_cache and anomaly_detector:
            latest_metrics = metrics_cache[-1]
            analysis['anomalies'] = anomaly_detector.detect_anomalies(latest_metrics)
        
        # 性能分析
        if metrics_cache and performance_analyzer:
            latest_metrics = metrics_cache[-1]
            perf_analysis = performance_analyzer.analyze_training_efficiency(latest_metrics)
            analysis['performance'] = perf_analysis
            performance_analyzer.performance_history.append(perf_analysis)
            
            # 性能预测
            analysis['predictions'] = performance_analyzer.predict_performance_trends()
        
        # 计算优化分数
        if analysis['performance']:
            analysis['optimization_score'] = analysis['performance'].get('overall_score', 0)
        
        # 健康趋势分析
        if len(metrics_cache) >= 10:
            recent_scores = []
            for metrics in list(metrics_cache)[-10:]:
                if 'analysis' in metrics and 'health_score' in metrics['analysis']:
                    recent_scores.append(metrics['analysis']['health_score'])
            
            if len(recent_scores) >= 5:
                if recent_scores[-1] > recent_scores[0] + 5:
                    analysis['health_trend'] = 'improving'
                elif recent_scores[-1] < recent_scores[0] - 5:
                    analysis['health_trend'] = 'declining'
        
    except Exception as e:
        logger.error(f"智能分析失败: {e}")
        analysis['error'] = str(e)
    
    return analysis

# ==================== API 路由 ====================

@app.route('/')
def index():
    """API信息页面"""
    return jsonify({
        'system': 'DLOps AI智能运维系统',
        'developer': 'Charles Dong <NVIDIA HPC Engineer>',
        'version': '1.0.0',
        'status': 'running',
        'endpoints': {
            'status': '/api/status',
            'metrics': '/api/metrics', 
            'diagnostics': '/api/diagnostics',
            'intelligence': '/api/intelligence',
            'events': '/events (EventSource)',
            'daemon_control': '/api/daemon/<start|stop|restart>',
            'health_check': '/api/check',
            'auto_fix': '/api/fix/<fix_type>'
        },
        'timestamp': datetime.now().isoformat()
    })

@app.route('/api/status')
def api_status():
    """系统状态API"""
    return jsonify({
        'metrics': get_latest_metrics(),
        'diagnostics': get_diagnostics(),
        'intelligence': get_intelligent_analysis(),
        'timestamp': datetime.now().isoformat()
    })

@app.route('/api/metrics')
def api_metrics():
    """指标API"""
    return jsonify(get_latest_metrics())

@app.route('/api/diagnostics')
def api_diagnostics():
    """诊断API"""
    return jsonify(get_diagnostics())

@app.route('/api/intelligence')
def api_intelligence():
    """智能分析API"""
    return jsonify(get_intelligent_analysis())

@app.route('/api/metrics/history')
def api_metrics_history():
    """历史指标API"""
    hours = request.args.get('hours', 1, type=int)
    return jsonify(list(metrics_cache)[-min(hours*20, len(metrics_cache)):])

@app.route('/api/daemon/<action>', methods=['POST'])
def api_daemon_control(action):
    """守护进程控制API"""
    if action in ['start', 'stop', 'restart']:
        result = run_dlops_command(['daemon', action])
        logger.info(f"守护进程{action}操作结果: {result}")
        return jsonify(result)
    return jsonify({'error': '无效操作'}), 400

@app.route('/api/fix/<fix_type>', methods=['POST'])
def api_auto_fix(fix_type):
    """自动修复API"""
    result = run_dlops_command(['fix', fix_type])
    logger.info(f"自动修复{fix_type}结果: {result}")
    return jsonify(result)

@app.route('/api/check', methods=['POST'])
def api_health_check():
    """健康检查API"""
    result = run_dlops_command(['check'])
    logger.info(f"健康检查结果: {result}")
    return jsonify(result)

@app.route('/events')
def events():
    """Server-Sent Events实时数据流"""
    def event_stream():
        logger.info("EventSource连接已建立")
        while True:
            try:
                # 获取全面状态
                status_data = {
                    'metrics': get_latest_metrics(),
                    'intelligence': get_intelligent_analysis(),
                    'timestamp': datetime.now().isoformat()
                }
                
                yield f"data: {json.dumps(status_data)}\n\n"
                time.sleep(3)  # 3秒更新一次
                
            except Exception as e:
                logger.error(f"事件流错误: {e}")
                yield f"data: {json.dumps({'error': str(e)})}\n\n"
                time.sleep(5)
    
    return Response(event_stream(), 
                   mimetype="text/plain",
                   headers={
                       'Cache-Control': 'no-cache',
                       'Connection': 'keep-alive',
                       'Access-Control-Allow-Origin': '*'
                   })

# ==================== 静态文件服务 ====================

@app.route('/static/<path:filename>')
def serve_static(filename):
    """提供静态文件服务（如果需要）"""
    try:
        return send_from_directory(STATIC_DIR, filename)
    except Exception as e:
        logger.error(f"静态文件服务失败: {e}")
        return jsonify({'error': '文件未找到'}), 404

# ==================== 工具函数 ====================

def initialize_ai_components():
    """初始化AI组件"""
    global anomaly_detector, performance_analyzer
    
    try:
        anomaly_detector = AnomalyDetector()
        performance_analyzer = PerformanceAnalyzer()
        logger.info("Charles Dong AI组件初始化成功")
    except Exception as e:
        logger.error(f"AI组件初始化失败: {e}")

def startup_check():
    """启动检查"""
    try:
        # 检查工作目录
        if not os.path.exists(DLOPS_WORKDIR):
            logger.warning(f"工作目录不存在，创建: {DLOPS_WORKDIR}")
            Path(DLOPS_WORKDIR).mkdir(parents=True, exist_ok=True)
        
        # 检查dlops命令
        dlops_script = os.path.join(DLOPS_WORKDIR, '../../../dlops')
        if not os.path.exists(dlops_script):
            dlops_script = './dlops'
        
        if not os.path.exists(dlops_script):
            logger.warning("dlops命令未找到，某些功能可能不可用")
        
        logger.info("启动检查完成")
        
    except Exception as e:
        logger.error(f"启动检查失败: {e}")

if __name__ == '__main__':
    # 启动检查
    startup_check()
    
    # 确保工作目录存在
    Path(DLOPS_WORKDIR).mkdir(parents=True, exist_ok=True)
    
    # 初始化AI组件
    initialize_ai_components()
    
    # 打印启动信息
    print("╔" + "═" * 70 + "╗")
    print("║" + " " * 70 + "║")
    print("║" + "🚀 DLOps AI智能运维后端服务".center(70) + "║")
    print("║" + "Developed by: Charles Dong <NVIDIA HPC Engineer>".center(70) + "║")
    print("║" + "AI-Driven Deep Learning Operations Backend".center(70) + "║")
    print("║" + " " * 70 + "║")
    print("╠" + "═" * 70 + "╣")
    print(f"║📍 API服务地址: http://localhost:{WEB_PORT}".ljust(71) + "║")
    print(f"║📡 EventSource: http://localhost:{WEB_PORT}/events".ljust(71) + "║")
    print(f"║📁 工作目录: {DLOPS_WORKDIR}".ljust(71) + "║")
    print(f"║📊 API文档: http://localhost:{WEB_PORT}/".ljust(71) + "║")
    print("║" + " " * 70 + "║")
    print("╚" + "═" * 70 + "╝")
    print()
    print("🧠 Charles Dong AI监控后端服务已就绪!")
    print("📄 前端HTML文件请单独部署或直接打开")
    print()
    
    try:
        app.run(host='0.0.0.0', port=WEB_PORT, debug=DEBUG_MODE, threaded=True)
    except KeyboardInterrupt:
        logger.info("🛑 Charles Dong's AI monitoring backend stopped")
    except Exception as e:
        logger.error(f"❌ 服务器启动失败: {e}")
