#!/usr/bin/env python3
"""
DLOps AI-Driven Operations Assistant
Developed by Senior Engineer Charles Dong
"""

import os
import sys
import json
import time
import signal
import subprocess
import threading
import argparse
import platform
import psutil
from datetime import datetime, timedelta
from pathlib import Path
from collections import deque, defaultdict
import logging

try:
    import numpy as np
    from flask import Flask, jsonify, Response, render_template_string
except ImportError:
    print("Installing required dependencies...")
    subprocess.run([sys.executable, "-m", "pip", "install", "flask", "numpy", "psutil"], check=True)
    import numpy as np
    from flask import Flask, jsonify, Response, render_template_string
    import psutil

VERSION = "2.0.2"
WORKDIR = Path.home() / ".dlops"
MONITOR_INTERVAL = 3
IS_WINDOWS = platform.system() == "Windows"
IS_LINUX = platform.system() == "Linux"

class Colors:
    RED = '\033[0;31m'
    GREEN = '\033[0;32m'
    YELLOW = '\033[1;33m'
    BLUE = '\033[0;34m'
    NC = '\033[0m'

class DLOpsSystem:
    def __init__(self):
        self.workdir = WORKDIR
        self.pidfile = self.workdir / "daemon.pid"
        self.metricsdir = self.workdir / "metrics"
        self.diagnosticsdir = self.workdir / "diagnostics"
        self.knowledge_base = self.workdir / "knowledge_base.json"
        
        self.metrics_cache = deque(maxlen=100)
        self.anomaly_detector = AnomalyDetector()
        self.performance_analyzer = PerformanceAnalyzer()
        
        self.init_system()
        
    def init_system(self):
        for dir_path in [self.workdir, self.metricsdir, self.diagnosticsdir, 
                         self.workdir / "fixes", self.workdir / "logs"]:
            dir_path.mkdir(parents=True, exist_ok=True)
        
        if not self.knowledge_base.exists():
            with open(self.knowledge_base, 'w', encoding='utf-8') as f:
                json.dump({"issues": {}, "fixes": {}, "patterns": {}, "baselines": {}}, f)
    
    def show_banner(self):
        print(f"{Colors.BLUE}╔══════════════════════════════════════════════════════════════╗{Colors.NC}")
        print(f"{Colors.BLUE}║                    🚀 DLOps AI智能运维系统                    ║{Colors.NC}")
        print(f"{Colors.BLUE}║                Deep Learning Operations Assistant            ║{Colors.NC}")
        print(f"{Colors.BLUE}╠══════════════════════════════════════════════════════════════╣{Colors.NC}")
        print(f"{Colors.BLUE}║                                                              ║{Colors.NC}")
        print(f"{Colors.BLUE}║            🎯 Senior Engineer: Charles Dong                  ║{Colors.NC}")
        print(f"{Colors.BLUE}║            🧠 AI-Driven Diagnostics & Auto-Fix              ║{Colors.NC}")
        print(f"{Colors.BLUE}║            📊 Version: {VERSION} ({platform.system()})        ║{Colors.NC}")
        print(f"{Colors.BLUE}║                                                              ║{Colors.NC}")
        print(f"{Colors.BLUE}╚══════════════════════════════════════════════════════════════╝{Colors.NC}")
        print(f"{Colors.GREEN}✨ Developed by Senior Engineer Charles Dong{Colors.NC}")
        print(f"{Colors.YELLOW}🔬 AI-Powered • Predictive Maintenance • Auto-Healing{Colors.NC}")
        print()

    def log(self, level, msg, component="main"):
        timestamp = datetime.now().isoformat()
        log_msg = f"[{timestamp}][{level}][{component}] {msg}"
        
        with open(self.workdir / "logs" / "system.log", "a", encoding='utf-8') as f:
            f.write(log_msg + "\n")
        
        color_map = {
            "ERROR": f"{Colors.RED}🚨 {msg}{Colors.NC}",
            "WARN": f"{Colors.YELLOW}⚠️  {msg}{Colors.NC}",
            "INFO": f"{Colors.GREEN}ℹ️  {msg}{Colors.NC}",
            "DEBUG": f"{Colors.BLUE}🔧 {msg}{Colors.NC}"
        }
        
        print(color_map.get(level, msg))

    def safe_float(self, value, fallback=0.0):
        try:
            return float(str(value).replace('W', '').replace('%', '').strip())
        except (ValueError, TypeError):
            return fallback

    def safe_nvidia_query(self, query, fallback="0"):
        try:
            nvidia_cmd = "nvidia-smi.exe" if IS_WINDOWS else "nvidia-smi"
            result = subprocess.run(
                [nvidia_cmd, f"--query-gpu={query}", "--format=csv,noheader,nounits"],
                capture_output=True, text=True, timeout=5
            )
            if result.returncode == 0 and result.stdout.strip():
                return result.stdout.strip().split('\n')[0]
        except (subprocess.TimeoutExpired, FileNotFoundError):
            pass
        return fallback

    def collect_advanced_metrics(self):
        timestamp = datetime.now().isoformat()
        
        cpu_util = self.get_cpu_utilization()
        memory_stats = self.get_memory_stats()
        
        gpu_metrics = {}
        if self.check_nvidia_smi():
            gpu_metrics = {
                "utilization": self.safe_float(self.safe_nvidia_query("utilization.gpu", "0")),
                "temperature": self.safe_float(self.safe_nvidia_query("temperature.gpu", "0")),
                "memory_used": self.safe_float(self.safe_nvidia_query("memory.used", "0")),
                "memory_total": self.safe_float(self.safe_nvidia_query("memory.total", "1")),
                "power_draw": self.safe_float(self.safe_nvidia_query("power.draw", "0"))
            }
        
        training_processes = self.get_training_processes()
        
        metrics = {
            "timestamp": timestamp,
            "system": {
                "cpu_utilization": cpu_util,
                "memory": memory_stats,
                "disk_usage": self.get_disk_usage()
            },
            "gpu": gpu_metrics,
            "training": {
                "active_processes": training_processes
            },
            "analysis": self.analyze_system_health()
        }
        
        latest_file = self.metricsdir / "latest.json"
        history_file = self.metricsdir / f"history_{datetime.now().strftime('%Y%m%d')}.jsonl"
        
        with open(latest_file, 'w', encoding='utf-8') as f:
            json.dump(metrics, f, indent=2)
        
        with open(history_file, 'a', encoding='utf-8') as f:
            f.write(json.dumps(metrics) + '\n')
        
        self.metrics_cache.append(metrics)
        self.anomaly_detector.update_baseline(metrics)
        
        return metrics

    def check_nvidia_smi(self):
        """检查nvidia-smi是否可用"""
        try:
            nvidia_cmd = "nvidia-smi.exe" if IS_WINDOWS else "nvidia-smi"
            if IS_WINDOWS:
                result = subprocess.run(["where", "nvidia-smi"], capture_output=True, timeout=5)
            else:
                result = subprocess.run(["which", "nvidia-smi"], capture_output=True, timeout=5)
            return result.returncode == 0
        except:
            return False

    def get_cpu_utilization(self):
        """跨平台CPU使用率获取"""
        try:
            return psutil.cpu_percent(interval=1)
        except Exception as e:
            self.log("DEBUG", f"CPU监控失败，使用系统命令: {e}")
            
        try:
            if IS_WINDOWS:
                result = subprocess.run([
                    "wmic", "cpu", "get", "loadpercentage", "/value"
                ], capture_output=True, text=True, timeout=10)
                
                for line in result.stdout.split('\n'):
                    if 'LoadPercentage=' in line:
                        return float(line.split('=')[1].strip())
            else:
                result = subprocess.run(["top", "-bn1"], capture_output=True, text=True, timeout=10)
                for line in result.stdout.split('\n'):
                    if 'Cpu(s)' in line:
                        return float(line.split()[1].replace('%us,', ''))
        except Exception as e:
            self.log("DEBUG", f"系统命令CPU监控失败: {e}")
        return 0.0

    def get_memory_stats(self):
        """跨平台内存统计"""
        try:
            memory = psutil.virtual_memory()
            used_gb = memory.used // (1024**3)
            total_gb = memory.total // (1024**3)
            percent = memory.percent
            return f"{used_gb},{total_gb},{percent:.1f}"
        except Exception as e:
            self.log("DEBUG", f"内存监控失败，使用系统命令: {e}")
            
        try:
            if IS_WINDOWS:
                result = subprocess.run([
                    "wmic", "OS", "get", "TotalVisibleMemorySize,FreePhysicalMemory", "/value"
                ], capture_output=True, text=True, timeout=10)
                
                total_kb = free_kb = 0
                for line in result.stdout.split('\n'):
                    if 'TotalVisibleMemorySize=' in line:
                        total_kb = int(line.split('=')[1].strip())
                    elif 'FreePhysicalMemory=' in line:
                        free_kb = int(line.split('=')[1].strip())
                
                if total_kb > 0:
                    used_kb = total_kb - free_kb
                    used_gb = used_kb // (1024**2)
                    total_gb = total_kb // (1024**2)
                    percent = (used_kb / total_kb) * 100
                    return f"{used_gb},{total_gb},{percent:.1f}"
            else:
                result = subprocess.run(["free"], capture_output=True, text=True, timeout=10)
                for line in result.stdout.split('\n'):
                    if line.startswith('Mem:'):
                        parts = line.split()
                        total, used = int(parts[1]), int(parts[2])
                        percent = (used / total) * 100
                        return f"{used//1024//1024},{total//1024//1024},{percent:.1f}"
        except Exception as e:
            self.log("DEBUG", f"系统命令内存监控失败: {e}")
        return "0,1,0.0"

    def get_disk_usage(self):
        """跨平台磁盘使用率"""
        try:
            if IS_WINDOWS:
                disk_usage = psutil.disk_usage('C:')
            else:
                disk_usage = psutil.disk_usage('/')
            return int(disk_usage.percent)
        except Exception as e:
            self.log("DEBUG", f"磁盘监控失败，使用系统命令: {e}")
            
        try:
            if IS_WINDOWS:
                result = subprocess.run([
                    "wmic", "logicaldisk", "where", "size>0", "get", "size,freespace,caption"
                ], capture_output=True, text=True, timeout=10)
                
                for line in result.stdout.split('\n'):
                    if 'C:' in line:
                        parts = line.split()
                        if len(parts) >= 3:
                            free_space = int(parts[1])
                            total_space = int(parts[2])
                            used_percent = ((total_space - free_space) / total_space) * 100
                            return int(used_percent)
            else:
                result = subprocess.run(["df", "/"], capture_output=True, text=True, timeout=10)
                for line in result.stdout.split('\n')[1:]:
                    if line.strip():
                        return int(line.split()[4].replace('%', ''))
        except Exception as e:
            self.log("DEBUG", f"系统命令磁盘监控失败: {e}")
        return 0

    def get_training_processes(self):
        """跨平台训练进程计数"""
        try:
            count = 0
            for proc in psutil.process_iter(['name', 'cmdline']):
                try:
                    cmdline = ' '.join(proc.info['cmdline'] or [])
                    if 'python' in proc.info['name'].lower() and 'train' in cmdline.lower():
                        count += 1
                except (psutil.NoSuchProcess, psutil.AccessDenied):
                    continue
            return count
        except Exception as e:
            self.log("DEBUG", f"进程监控失败，使用系统命令: {e}")
            
        try:
            if IS_WINDOWS:
                result = subprocess.run([
                    "tasklist", "/fo", "csv", "/fi", "IMAGENAME eq python.exe"
                ], capture_output=True, text=True, timeout=10)
                return len([line for line in result.stdout.split('\n') if 'python.exe' in line]) - 1
            else:
                result = subprocess.run(["ps", "aux"], capture_output=True, text=True, timeout=10)
                return len([p for p in result.stdout.split('\n') if 'python' in p and 'train' in p])
        except Exception as e:
            self.log("DEBUG", f"系统命令进程监控失败: {e}")
        return 0

    def analyze_system_health(self):
        issues = []
        recommendations = []
        predictions = []
        auto_fixes = []
        
        if self.check_nvidia_smi():
            gpu_temp = self.safe_float(self.safe_nvidia_query("temperature.gpu", "0"))
            gpu_util = self.safe_float(self.safe_nvidia_query("utilization.gpu", "0"))
            gpu_power = self.safe_float(self.safe_nvidia_query("power.draw", "0"))
            
            if gpu_temp > 85:
                issues.append(f"GPU过热 ({gpu_temp:.1f}°C)")
                recommendations.append("检查散热，考虑降低功耗限制")
                if gpu_power > 100:
                    auto_fixes.append(f"nvidia-smi -pl {int(gpu_power * 0.8)}")
            
            if gpu_util < 10:
                issues.append("GPU利用率偏低")
                recommendations.append("增加batch size或优化数据加载")
            
            if gpu_util > 95 and gpu_temp > 80:
                predictions.append("GPU过热风险预警：5分钟内可能触发热限制")
        
        memory_stats = self.get_memory_stats()
        if memory_stats:
            mem_percent = float(memory_stats.split(',')[2])
            if mem_percent > 90:
                issues.append(f"内存使用率过高 ({mem_percent:.1f}%)")
                recommendations.append("考虑gradient checkpointing或减少缓存")
                if IS_LINUX:
                    auto_fixes.append("sync && echo 3 > /proc/sys/vm/drop_caches")
        
        return {
            "timestamp": datetime.now().isoformat(),
            "issues": issues,
            "recommendations": recommendations,
            "predictions": predictions,
            "auto_fixes": auto_fixes,
            "severity": self.get_severity_level(len(issues)),
            "health_score": self.calculate_health_score()
        }

    def calculate_health_score(self):
        score = 100
        
        if self.check_nvidia_smi():
            gpu_temp = self.safe_float(self.safe_nvidia_query("temperature.gpu", "0"))
            gpu_util = self.safe_float(self.safe_nvidia_query("utilization.gpu", "0"))
            
            if gpu_temp > 85: score -= 20
            if gpu_temp > 90: score -= 30
            if gpu_util < 50: score -= 10
        
        memory_stats = self.get_memory_stats()
        if memory_stats:
            mem_percent = float(memory_stats.split(',')[2])
            if mem_percent > 90: score -= 15
        
        disk_usage = self.get_disk_usage()
        if disk_usage > 90: score -= 25
        
        return max(0, score)

    def get_severity_level(self, issue_count):
        if issue_count == 0: return "healthy"
        elif issue_count <= 2: return "warning"
        else: return "critical"

    def start_daemon(self):
        if self.pidfile.exists():
            try:
                with open(self.pidfile, encoding='utf-8') as f:
                    pid = int(f.read().strip())
                if IS_WINDOWS:
                    # Windows进程检查
                    try:
                        result = subprocess.run(["tasklist", "/FI", f"PID eq {pid}"], 
                                              capture_output=True, text=True)
                        if str(pid) not in result.stdout:
                            raise OSError("Process not found")
                    except:
                        raise OSError("Process not found")
                else:
                    os.kill(pid, 0)
                self.log("WARN", f"Daemon already running (PID: {pid})")
                return False
            except (OSError, ValueError):
                self.pidfile.unlink()
        
        print(f"{Colors.GREEN}🚀 Senior Engineer Charles Dong's AI Monitoring System{Colors.NC}")
        self.log("INFO", "🚀 Starting enhanced monitoring daemon...")
        
        daemon_process = threading.Thread(target=self.daemon_loop, daemon=True)
        daemon_process.start()
        
        with open(self.pidfile, 'w', encoding='utf-8') as f:
            f.write(str(os.getpid()))
        
        self.log("INFO", f"✅ Charles Dong's AI system started successfully (PID: {os.getpid()})")
        return True

    def daemon_loop(self):
        self.log("INFO", "🔄 AI monitoring daemon started by Charles Dong", "daemon")
        
        while True:
            try:
                self.collect_advanced_metrics()
                time.sleep(MONITOR_INTERVAL)
            except KeyboardInterrupt:
                break
            except Exception as e:
                self.log("ERROR", f"Daemon error: {e}", "daemon")
                time.sleep(5)

    def stop_daemon(self):
        if not self.pidfile.exists():
            self.log("INFO", "Daemon not running")
            return
        
        try:
            with open(self.pidfile, encoding='utf-8') as f:
                pid = int(f.read().strip())
            
            if IS_WINDOWS:
                subprocess.run(["taskkill", "/F", "/PID", str(pid)], check=True)
            else:
                os.kill(pid, signal.SIGTERM)
            
            self.pidfile.unlink()
            self.log("INFO", "Daemon stopped successfully")
        except (OSError, ValueError, subprocess.CalledProcessError) as e:
            self.log("ERROR", f"Failed to stop daemon: {e}")

    def daemon_status(self):
        if not self.pidfile.exists():
            print(f"{Colors.RED}❌ 守护进程未运行{Colors.NC}")
            print(f"{Colors.YELLOW}💡 启动命令: python dlops.py daemon start{Colors.NC}")
            return False
        
        try:
            with open(self.pidfile, encoding='utf-8') as f:
                pid = int(f.read().strip())
            
            if IS_WINDOWS:
                result = subprocess.run(["tasklist", "/FI", f"PID eq {pid}"], 
                                      capture_output=True, text=True)
                if str(pid) not in result.stdout:
                    raise OSError("Process not found")
            else:
                os.kill(pid, 0)
                
            print(f"{Colors.GREEN}✅ Charles Dong's AI监控守护进程运行中 (PID: {pid}){Colors.NC}")
            print(f"{Colors.YELLOW}📁 工作目录: {self.workdir}{Colors.NC}")
            return True
        except OSError:
            self.pidfile.unlink()
            print(f"{Colors.RED}❌ 守护进程未运行{Colors.NC}")
            return False

    def show_metrics(self):
        latest_file = self.metricsdir / "latest.json"
        if not latest_file.exists():
            print(f"{Colors.YELLOW}⚠️ 无指标数据，请先启动守护进程{Colors.NC}")
            return
        
        with open(latest_file, encoding='utf-8') as f:
            metrics = json.load(f)
        
        print(f"{Colors.GREEN}📊 Charles Dong's AI监控指标：{Colors.NC}")
        print(json.dumps(metrics, indent=2, ensure_ascii=False))

    def run_diagnostic_wizard(self):
        self.log("INFO", "🧙‍♂️ 启动智能诊断向导...")
        
        print(f"{Colors.BLUE}=== Charles Dong's AI Diagnostic Wizard ==={Colors.NC}")
        print()
        
        analysis = self.analyze_system_health()
        
        print(f"{Colors.YELLOW}📋 发现的问题：{Colors.NC}")
        for issue in analysis['issues']:
            print(f"  ❌ {issue}")
        if not analysis['issues']:
            print("  ✅ 未发现问题")
        
        print(f"\n{Colors.BLUE}💡 AI优化建议：{Colors.NC}")
        for rec in analysis['recommendations']:
            print(f"  💡 {rec}")
        if not analysis['recommendations']:
            print("  ✅ 系统运行良好")
        
        print(f"\n{Colors.GREEN}🔮 预测分析：{Colors.NC}")
        for pred in analysis['predictions']:
            print(f"  🔮 {pred}")
        if not analysis['predictions']:
            print("  ✅ 无风险预警")
        
        print(f"\n{Colors.BLUE}🏥 系统健康分数: {analysis['health_score']}/100{Colors.NC}")

    def start_web_server(self, port=8080):
        print(f"{Colors.BLUE}╔══════════════════════════════════════════════════════════════╗{Colors.NC}")
        print(f"{Colors.BLUE}║              🌐 Charles Dong's AI Web Dashboard              ║{Colors.NC}")
        print(f"{Colors.BLUE}║                Senior Engineer's Monitoring System          ║{Colors.NC}")
        print(f"{Colors.BLUE}╚══════════════════════════════════════════════════════════════╝{Colors.NC}")
        print()
        print(f"{Colors.GREEN}👨‍💻 Developed by Senior Engineer Charles Dong{Colors.NC}")
        print(f"{Colors.YELLOW}🚀 Access your AI monitoring dashboard at: http://localhost:{port}{Colors.NC}")
        print()
        
        app = Flask(__name__)
        app.secret_key = 'charles_dong_dlops_2025'
        
        @app.route('/')
        def dashboard():
            try:
                with open('index.html', 'r', encoding='utf-8') as f:
                    return f.read()
            except FileNotFoundError:
                return f"""
                <h1>🚀 Charles Dong's DLOps AI System</h1>
                <p>index.html not found. Please ensure index.html is in the same directory.</p>
                <p>👨‍💻 Developed by Senior Engineer Charles Dong</p>
                <p>📊 API available at: <a href="/api/status">/api/status</a></p>
                """
        
        @app.route('/api/status')
        def api_status():
            metrics = self.get_latest_metrics()
            intelligence = self.get_intelligent_analysis()
            return jsonify({
                'metrics': metrics,
                'intelligence': intelligence,
                'timestamp': datetime.now().isoformat()
            })
        
        @app.route('/api/metrics')
        def api_metrics():
            return jsonify(self.get_latest_metrics())
        
        @app.route('/api/daemon/<action>', methods=['POST'])
        def api_daemon_control(action):
            if action == 'start':
                success = self.start_daemon()
                return jsonify({'success': success, 'message': 'Daemon started' if success else 'Already running'})
            elif action == 'stop':
                self.stop_daemon()
                return jsonify({'success': True, 'message': 'Daemon stopped'})
            return jsonify({'error': 'Invalid action'}), 400
        
        @app.route('/events')
        def events():
            def event_stream():
                while True:
                    try:
                        status_data = {
                            'metrics': self.get_latest_metrics(),
                            'intelligence': self.get_intelligent_analysis(),
                            'timestamp': datetime.now().isoformat()
                        }
                        yield f"data: {json.dumps(status_data)}\n\n"
                        time.sleep(3)
                    except Exception as e:
                        yield f"data: {json.dumps({'error': str(e)})}\n\n"
                        time.sleep(5)
            
            return Response(event_stream(), mimetype="text/plain")
        
        try:
            app.run(host='0.0.0.0', port=port, debug=False, threaded=True)
        except KeyboardInterrupt:
            print(f"\n{Colors.BLUE}🛑 Charles Dong's monitoring system stopped{Colors.NC}")

    def get_latest_metrics(self):
        latest_file = self.metricsdir / "latest.json"
        if latest_file.exists():
            with open(latest_file, encoding='utf-8') as f:
                return json.load(f)
        return {'error': 'No metrics available'}

    def get_intelligent_analysis(self):
        analysis = {
            'anomalies': [],
            'performance': {},
            'predictions': [],
            'optimization_score': 0,
            'health_trend': 'stable'
        }
        
        if self.metrics_cache:
            latest_metrics = self.metrics_cache[-1]
            analysis['anomalies'] = self.anomaly_detector.detect_anomalies(latest_metrics)
            analysis['performance'] = self.performance_analyzer.analyze_training_efficiency(latest_metrics)
            analysis['predictions'] = self.performance_analyzer.predict_performance_trends()
            
            if analysis['performance']:
                analysis['optimization_score'] = analysis['performance'].get('overall_score', 0)
        
        return analysis

    def show_dashboard(self):
        try:
            while True:
                os.system('cls' if IS_WINDOWS else 'clear')
                print(f"{Colors.BLUE}╔══════════════════════════════════════════════════════════════╗{Colors.NC}")
                print(f"{Colors.BLUE}║                🚀 Charles Dong's AI监控仪表板                 ║{Colors.NC}")
                print(f"{Colors.BLUE}╚══════════════════════════════════════════════════════════════╝{Colors.NC}")
                print()
                
                latest_file = self.metricsdir / "latest.json"
                if latest_file.exists():
                    with open(latest_file, encoding='utf-8') as f:
                        metrics = json.load(f)
                    
                    health_score = metrics.get('analysis', {}).get('health_score', 0)
                    print(f"{Colors.GREEN}🏥 系统健康分数: {health_score}/100{Colors.NC}")
                    
                    if 'gpu' in metrics and metrics['gpu']:
                        gpu = metrics['gpu']
                        gpu_util = gpu.get('utilization', 0)
                        gpu_temp = gpu.get('temperature', 0)
                        gpu_mem_used = gpu.get('memory_used', 0)
                        gpu_mem_total = gpu.get('memory_total', 1)
                        
                        print(f"{Colors.YELLOW}🎮 GPU状态{Colors.NC}")
                        print(f"   利用率: {gpu_util}% | 温度: {gpu_temp}°C | 显存: {gpu_mem_used}MB/{gpu_mem_total}MB")
                    
                    if 'system' in metrics:
                        system = metrics['system']
                        cpu_util = system.get('cpu_utilization', 0)
                        memory_info = system.get('memory', '0,1,0').split(',')
                        mem_used = float(memory_info[0]) if len(memory_info) > 0 else 0
                        mem_total = float(memory_info[1]) if len(memory_info) > 1 else 1
                        mem_percent = float(memory_info[2]) if len(memory_info) > 2 else 0
                        
                        print(f"{Colors.GREEN}💻 CPU: {cpu_util}% | 内存: {mem_used:.1f}GB/{mem_total:.1f}GB ({mem_percent:.1f}%){Colors.NC}")
                    
                    training_procs = metrics.get('training', {}).get('active_processes', 0)
                    print(f"{Colors.BLUE}🏋️ 训练进程: {training_procs}{Colors.NC}")
                else:
                    print(f"{Colors.RED}❌ 无监控数据，请先启动守护进程{Colors.NC}")
                
                print()
                print(f"{Colors.BLUE}最后更新: {datetime.now().strftime('%H:%M:%S')} | 按 Ctrl+C 退出{Colors.NC}")
                print(f"{Colors.GREEN}👨‍💻 Senior Engineer Charles Dong's System ({platform.system()}){Colors.NC}")
                
                time.sleep(3)
        
        except KeyboardInterrupt:
            print(f"\n{Colors.GREEN}👋 Dashboard closed{Colors.NC}")

class AnomalyDetector:
    def __init__(self):
        self.baseline_metrics = {}
        self.anomaly_threshold = 2.0
        self.learning_window = 50
        
    def update_baseline(self, metrics):
        try:
            flat_metrics = self._flatten_metrics(metrics)
            for key, value in flat_metrics.items():
                if isinstance(value, (int, float)):
                    if key not in self.baseline_metrics:
                        self.baseline_metrics[key] = deque(maxlen=self.learning_window)
                    self.baseline_metrics[key].append(value)
        except Exception:
            pass
    
    def detect_anomalies(self, metrics):
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
        except Exception:
            pass
        return anomalies
    
    def _flatten_metrics(self, metrics, prefix=''):
        flat = {}
        try:
            for key, value in metrics.items():
                new_key = f"{prefix}.{key}" if prefix else key
                if isinstance(value, dict):
                    flat.update(self._flatten_metrics(value, new_key))
                elif isinstance(value, (int, float)):
                    flat[new_key] = value
                elif isinstance(value, str) and ',' in value:
                    parts = value.split(',')
                    for i, part in enumerate(parts):
                        try:
                            flat[f"{new_key}.{i}"] = float(part)
                        except ValueError:
                            pass
        except Exception:
            pass
        return flat

class PerformanceAnalyzer:
    def __init__(self):
        self.performance_history = deque(maxlen=200)
        
    def analyze_training_efficiency(self, metrics):
        analysis = {
            'gpu_efficiency': 0,
            'memory_efficiency': 0,
            'overall_score': 0,
            'bottlenecks': [],
            'recommendations': []
        }
        
        try:
            if 'gpu' in metrics and 'utilization' in metrics['gpu']:
                gpu_util = metrics['gpu']['utilization']
                analysis['gpu_efficiency'] = gpu_util
                
                if gpu_util < 70:
                    analysis['bottlenecks'].append('GPU利用率低')
                    analysis['recommendations'].append('增加batch size或优化数据加载')
            
            if 'system' in metrics and 'memory' in metrics['system']:
                memory_info = metrics['system']['memory']
                if ',' in memory_info:
                    mem_percent = float(memory_info.split(',')[2])
                    analysis['memory_efficiency'] = min(100, mem_percent * 1.2)
                    
                    if mem_percent > 90:
                        analysis['bottlenecks'].append('内存使用率过高')
                        analysis['recommendations'].append('启用gradient checkpointing')
            
            analysis['overall_score'] = (analysis['gpu_efficiency'] * 0.6 + analysis['memory_efficiency'] * 0.4)
            
        except Exception:
            pass
        
        return analysis
    
    def predict_performance_trends(self):
        predictions = []
        
        if len(self.performance_history) >= 10:
            try:
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
            except Exception:
                pass
                
        return predictions

def main():
    dlops = DLOpsSystem()
    
    parser = argparse.ArgumentParser(description="DLOps AI Intelligent Operations System")
    subparsers = parser.add_subparsers(dest='command', help='Available commands')
    
    daemon_parser = subparsers.add_parser('daemon', help='Daemon management')
    daemon_parser.add_argument('action', choices=['start', 'stop', 'status', 'restart'])
    
    web_parser = subparsers.add_parser('web', help='Web server')
    web_parser.add_argument('action', nargs='?', default='start')
    web_parser.add_argument('--port', type=int, default=8080)
    
    subparsers.add_parser('check', help='Run diagnostic wizard')
    subparsers.add_parser('metrics', help='Show current metrics')
    subparsers.add_parser('analyze', help='Run system analysis')
    subparsers.add_parser('dashboard', help='Show terminal dashboard')
    
    args = parser.parse_args()
    
    if not args.command:
        dlops.show_banner()
        parser.print_help()
        return
    
    if args.command == 'daemon':
        if args.action == 'start':
            dlops.show_banner()
            dlops.start_daemon()
            
            try:
                while dlops.pidfile.exists():
                    time.sleep(1)
            except KeyboardInterrupt:
                dlops.stop_daemon()
                
        elif args.action == 'stop':
            dlops.stop_daemon()
        elif args.action == 'status':
            dlops.daemon_status()
        elif args.action == 'restart':
            dlops.stop_daemon()
            time.sleep(2)
            dlops.show_banner()
            dlops.start_daemon()
    
    elif args.command == 'web':
        dlops.show_banner()
        dlops.start_web_server(args.port)
    
    elif args.command == 'check':
        dlops.show_banner()
        dlops.run_diagnostic_wizard()
    
    elif args.command == 'metrics':
        dlops.show_metrics()
    
    elif args.command == 'analyze':
        print(f"{Colors.BLUE}🧠 Charles Dong's AI系统分析：{Colors.NC}")
        analysis = dlops.analyze_system_health()
        print(json.dumps(analysis, indent=2, ensure_ascii=False))
    
    elif args.command == 'dashboard':
        dlops.show_dashboard()

if __name__ == "__main__":
    main()