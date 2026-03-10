
#!/usr/bin/env python3
"""
Ollama Service Monitoring
Checks service availability, API, model load and sends notifications when problems occur
Web interface for viewing graphs in browser
"""

import requests
import time
import smtplib
import logging
import os
import json
import threading
from datetime import datetime, timedelta
from email.mime.text import MIMEText
from typing import Optional, List, Dict, Any
from dataclasses import dataclass, field, asdict
import psutil
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

matplotlib.use('Agg')  # Headless mode for server

# Configuration
CONFIG = {
    "ollama_host": "http://localhost:11434",
    "check_interval": 60,  # seconds
    "timeout": 10,  # request timeout in seconds
    "history_file": "ollama_history.json",
    "graph_dir": "graphs",
    "web_port": 8080,
    "web_host": "0.0.0.0",
    "graph_width": 12,
    "graph_height": 6,
    "history_hours": 24,  # store history for 24 hours
    "email": {
        "enabled": False,
        "smtp_server": "smtp.example.com",
        "smtp_port": 587,
        "username": "alert@example.com",
        "password": "your_password",
        "from_addr": "alert@example.com",
        "to_addrs": ["admin@example.com"],
    },
    "telegram": {
        "enabled": False,
        "bot_token": "YOUR_BOT_TOKEN",
        "chat_id": "YOUR_CHAT_ID",
    },
    "log_file": "ollama_monitor.log",
}


@dataclass
class ModelInfo:
    """Model information"""
    name: str
    size: int  # in bytes
    modified_at: str
    digest: str
    
    @property
    def size_mb(self) -> float:
        return self.size / (1024 * 1024)
    
    @property
    def size_gb(self) -> float:
        return self.size / (1024 * 1024 * 1024)


@dataclass
class RunningModel:
    """Loaded/active model"""
    model: str
    digest: str
    duration: int  # nanoseconds
    done: bool
    
    @property
    def duration_ms(self) -> float:
        return self.duration / 1_000_000
    
    @property
    def duration_sec(self) -> float:
        return self.duration / 1_000_000_000


@dataclass
class MetricPoint:
    """Metric point for chart"""
    timestamp: str
    response_time: float
    models_loaded: int
    models_available: int
    memory_percent: float
    ollama_memory_mb: float
    service_up: bool


@dataclass
class OllamaStatus:
    """Ollama service status"""
    service_running: bool
    api_reachable: bool
    models_available: int
    models_loaded: List[str] = field(default_factory=list)
    running_tasks: List[RunningModel] = field(default_factory=list)
    total_models_size_gb: float = 0.0
    system_memory_percent: float = 0.0
    ollama_memory_mb: float = 0.0
    response_time: float = 0.0
    error_message: Optional[str] = None


class HistoryManager:
    """Metrics history management"""
    
    def __init__(self, config: dict):
        self.config = config
        self.history_file = config.get("history_file", "ollama_history.json")
        self.history_hours = config.get("history_hours", 24)
        self.metrics: List[MetricPoint] = []
        self._load_history()
    
    def _load_history(self):
        """Load history from file"""
        if os.path.exists(self.history_file):
            try:
                with open(self.history_file, 'r') as f:
                    data = json.load(f)
                    self.metrics = [MetricPoint(**m) for m in data]
                    self._cleanup_old()
            except Exception:
                self.metrics = []
    
    def _cleanup_old(self):
        """Delete old records"""
        cutoff = datetime.now() - timedelta(hours=self.history_hours)
        cutoff_str = cutoff.isoformat()
        self.metrics = [m for m in self.metrics if m.timestamp >= cutoff_str]
    
    def _save_history(self):
        """Save history to file"""
        try:
            with open(self.history_file, 'w') as f:
                json.dump([asdict(m) for m in self.metrics], f)
        except Exception:
            pass
    
    def add_metric(self, status: OllamaStatus):
        """Add new metric"""
        metric = MetricPoint(
            timestamp=datetime.now().isoformat(),
            response_time=status.response_time,
            models_loaded=len(status.models_loaded),
            models_available=status.models_available,
            memory_percent=status.system_memory_percent,
            ollama_memory_mb=status.ollama_memory_mb,
            service_up=status.service_running
        )
        self.metrics.append(metric)
        self._cleanup_old()
        self._save_history()
    
    def get_history_data(self) -> Dict[str, List]:
        """Get data for charts"""
        timestamps = []
        response_times = []
        models_loaded = []
        memory_percent = []
        
        for m in self.metrics:
            timestamps.append(datetime.fromisoformat(m.timestamp))
            response_times.append(m.response_time)
            models_loaded.append(m.models_loaded)
            memory_percent.append(m.memory_percent)
        
        return {
            'timestamps': timestamps,
            'response_time': response_times,
            'models_loaded': models_loaded,
            'memory_percent': memory_percent
        }
    
    def get_latest_status(self) -> Dict[str, Any]:
        """Get latest status for display"""
        if not self.metrics:
            return {
                'service_up': False,
                'response_time': 0,
                'models_loaded': 0,
                'models_available': 0,
                'memory_percent': 0,
                'ollama_memory_mb': 0,
            }
        
        latest = self.metrics[-1]
        return {
            'service_up': latest.service_up,
            'response_time': latest.response_time,
            'models_loaded': latest.models_loaded,
            'models_available': latest.models_available,
            'memory_percent': latest.memory_percent,
            'ollama_memory_mb': latest.ollama_memory_mb,
            'last_update': latest.timestamp,
        }


class GraphGenerator:
    """Graph generation"""
    
    def __init__(self, config: dict):
        self.config = config
        self.graph_dir = config.get("graph_dir", "graphs")
        os.makedirs(self.graph_dir, exist_ok=True)
    
    def generate_all(self, history: HistoryManager):
        """Generate all graphs"""
        data = history.get_history_data()
        
        if not data['timestamps']:
            return
        
        self._generate_response_time_graph(data)
        self._generate_models_graph(data)
        self._generate_memory_graph(data)
        self._generate_combined_graph(data)
    
    def _setup_figure(self):
        """Setup figure"""
        width = self.config.get("graph_width", 12)
        height = self.config.get("graph_height", 6)
        return plt.figure(figsize=(width, height))
    
    def _format_time_axis(self, ax):
        """Time axis formatting"""
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
        ax.xaxis.set_major_locator(mdates.AutoDateLocator())
        plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha='right')
    
    def _generate_response_time_graph(self, data: Dict[str, List]):
        """Response time graph"""
        fig = self._setup_figure()
        
        plt.plot(data['timestamps'], data['response_time'], 
                 color='#2196F3', linewidth=2, label='Response time (sec)')
        
        # Add moving average
        if len(data['response_time']) > 5:
            import numpy as np
            window = min(5, len(data['response_time']))
            rolling = np.convolve(data['response_time'], 
                                   np.ones(window)/window, mode='valid')
            plt.plot(data['timestamps'][:len(rolling)], rolling,
                    color='#FF5722', linewidth=2, linestyle='--', 
                    label=f'Average (window={window})')
        
        plt.xlabel('Time')
        plt.ylabel('Response time (sec)')
        plt.title('📈 Ollama response time')
        plt.grid(True, alpha=0.3)
        plt.legend()
        self._format_time_axis(plt.gca())
        
        plt.tight_layout()
        plt.savefig(f'{self.graph_dir}/response_time.png', dpi=100)
        plt.close()
    
    def _generate_models_graph(self, data: Dict[str, List]):
        """Loaded models graph"""
        fig = self._setup_figure()
        
        plt.plot(data['timestamps'], data['models_loaded'],
                 color='#4CAF50', linewidth=2, marker='o', markersize=3,
                 label='Models loaded')
        
        plt.fill_between(data['timestamps'], data['models_loaded'], 
                         alpha=0.3, color='#4CAF50')
        
        plt.xlabel('Time')
        plt.ylabel('Number of models')
        plt.title('🧠 Loaded models')
        plt.grid(True, alpha=0.3)
        plt.legend()
        self._format_time_axis(plt.gca())
        
        plt.tight_layout()
        plt.savefig(f'{self.graph_dir}/models_loaded.png', dpi=100)
        plt.close()
    
    def _generate_memory_graph(self, data: Dict[str, List]):
        """Memory usage graph"""
        fig = self._setup_figure()
        
        plt.plot(data['timestamps'], data['memory_percent'],
                 color='#9C27B0', linewidth=2, label='System memory (%)')
        
        plt.fill_between(data['timestamps'], data['memory_percent'],
                         alpha=0.3, color='#9C27B0')
        
        plt.xlabel('Time')
        plt.ylabel('Usage (%)')
        plt.title('💾 Memory usage')
        plt.ylim(0, 100)
        plt.grid(True, alpha=0.3)
        plt.legend()
        self._format_time_axis(plt.gca())
        
        plt.tight_layout()
        plt.savefig(f'{self.graph_dir}/memory.png', dpi=100)
        plt.close()
    
    def _generate_combined_graph(self, data: Dict[str, List]):
        """Combined graph"""
        fig, axes = plt.subplots(2, 1, figsize=(14, 10))
        
        # Response time
        axes[0].plot(data['timestamps'], data['response_time'],
                     color='#2196F3', linewidth=2, label='Response time (sec)')
        axes[0].set_xlabel('Time')
        axes[0].set_ylabel('Response time (sec)')
        axes[0].set_title('⏱ Response time and model loading')
        axes[0].grid(True, alpha=0.3)
        axes[0].legend()
        self._format_time_axis(axes[0])
        
        # Models
        axes[1].plot(data['timestamps'], data['models_loaded'],
                     color='#4CAF50', linewidth=2, marker='o', markersize=3,
                     label='Models loaded')
        axes[1].fill_between(data['timestamps'], data['models_loaded'],
                              alpha=0.3, color='#4CAF50')
        axes[1].set_xlabel('Time')
        axes[1].set_ylabel('Number of models')
        axes[1].grid(True, alpha=0.3)
        axes[1].legend()
        self._format_time_axis(axes[1])
        
        plt.tight_layout()
        plt.savefig(f'{self.graph_dir}/combined.png', dpi=100)
        plt.close()


class WebServer:
    """Web server for displaying graphs"""
    
    HTML_TEMPLATE = """<!DOCTYPE html>
<html lang="ru">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>📊 Ollama Monitor</title>
    <style>
        * {
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }
        
        body {
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%);
            min-height: 100vh;
            color: #eee;
        }
        
        .container {
            max-width: 1400px;
            margin: 0 auto;
            padding: 20px;
        }
        
        header {
            text-align: center;
            padding: 30px 0;
            border-bottom: 1px solid #333;
            margin-bottom: 30px;
        }
        
        h1 {
            font-size: 2.5rem;
            margin-bottom: 10px;
            background: linear-gradient(90deg, #00d9ff, #00ff88);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
        }
        
        .status-bar {
            display: flex;
            justify-content: center;
            gap: 30px;
            flex-wrap: wrap;
            margin-bottom: 30px;
        }
        
        .status-card {
            background: rgba(255,255,255,0.05);
            border-radius: 12px;
            padding: 20px 30px;
            text-align: center;
            min-width: 150px;
            border: 1px solid rgba(255,255,255,0.1);
        }
        
        .status-card .label {
            font-size: 0.85rem;
            color: #888;
            margin-bottom: 5px;
        }
        
        .status-card .value {
            font-size: 1.8rem;
            font-weight: bold;
        }
        
        .status-card.up .value { color: #00ff88; }
        .status-card.down .value { color: #ff4444; }
        
        .status-card .unit {
            font-size: 0.9rem;
            color: #666;
        }
        
        .grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(600px, 1fr));
            gap: 20px;
            margin-bottom: 30px;
        }
        
        .graph-card {
            background: rgba(255,255,255,0.03);
            border-radius: 12px;
            padding: 20px;
            border: 1px solid rgba(255,255,255,0.1);
        }
        
        .graph-card h3 {
            margin-bottom: 15px;
            color: #00d9ff;
            font-size: 1.1rem;
        }
        
        .graph-card img {
            width: 100%;
            border-radius: 8px;
        }
        
        .footer {
            text-align: center;
            padding: 20px;
            color: #666;
            font-size: 0.9rem;
        }
        
        .refresh-info {
            text-align: center;
            color: #666;
            margin-bottom: 20px;
        }
        
        @media (max-width: 768px) {
            .grid {
                grid-template-columns: 1fr;
            }
            
            h1 {
                font-size: 1.8rem;
            }
            
            .status-bar {
                gap: 15px;
            }
            
            .status-card {
                min-width: 100px;
                padding: 15px;
            }
        }
    </style>
    <meta http-equiv="refresh" content="30">
</head>
<body>
    <div class="container">
        <header>
            <h1>📊 Ollama Monitor</h1>
            <p>Ollama service monitoring</p>
        </header>
        
        <div class="status-bar">
            <div class="status-card {{'up' if status.service_up else 'down'}}">
                <div class="label">Status</div>
                <div class="value">{{'✅ Running' if status.service_up else '❌ Unavailable'}}</div>
            </div>
            <div class="status-card">
                <div class="label">Response time</div>
                <div class="value">{{"%.3f"|format(status.response_time)}}</div>
                <div class="unit">sec</div>
            </div>
            <div class="status-card">
                <div class="label">Available models</div>
                <div class="value">{{status.models_available}}</div>
            </div>
            <div class="status-card">
                <div class="label">Loaded models</div>
                <div class="value">{{status.models_loaded}}</div>
            </div>
            <div class="status-card">
                <div class="label">System memory</div>
                <div class="value">{{"%.1f"|format(status.memory_percent)}}</div>
                <div class="unit">%</div>
            </div>
            <div class="status-card">
                <div class="label">Ollama memory</div>
                <div class="value">{{"%.0f"|format(status.ollama_memory_mb)}}</div>
                <div class="unit">MB</div>
            </div>
        </div>
        
        <p class="refresh-info">Page refreshes automatically every 30 seconds</p>
        
        <div class="grid">
            <div class="graph-card">
                <h3>📈 Response time</h3>
                <img src="/graphs/combined.png?t={{cache_bust}}" alt="Response time">
            </div>
            <div class="graph-card">
                <h3>🧠 Loaded models</h3>
                <img src="/graphs/models_loaded.png?t={{cache_bust}}" alt="Loaded models">
            </div>
            <div class="graph-card">
                <h3>💾 Memory usage</h3>
                <img src="/graphs/memory.png?t={{cache_bust}}" alt="Memory">
            </div>
            <div class="graph-card">
                <h3>📊 Overall statistics</h3>
                <img src="/graphs/combined.png?t={{cache_bust}}" alt="Statistics">
            </div>
        </div>
        
        <div class="footer">
            <p>Ollama Monitor • Updated: {{status.last_update}}</p>
        </div>
    </div>
</body>
</html>"""
    
    def __init__(self, config: dict, history: HistoryManager):
        self.config = config
        self.history = history
        self.graph_dir = config.get("graph_dir", "graphs")
        self.port = config.get("web_port", 8080)
        self.host = config.get("web_host", "0.0.0.0")
        self._server = None
    
    def _generate_html(self) -> str:
        """Generate HTML page"""
        status = self.history.get_latest_status()
        cache_bust = int(time.time())
        
        html = self.HTML_TEMPLATE
        html = html.replace('{{status.service_up}}', str(status['service_up']).lower())
        html = html.replace('{{status.response_time}}', str(status.get('response_time', 0)))
        html = html.replace('{{status.models_available}}', str(status.get('models_available', 0)))
        html = html.replace('{{status.models_loaded}}', str(status.get('models_loaded', 0)))
        html = html.replace('{{status.memory_percent}}', str(status.get('memory_percent', 0)))
        html = html.replace('{{status.ollama_memory_mb}}', str(status.get('ollama_memory_mb', 0)))
        html = html.replace('{{status.last_update}}', str(status.get('last_update', '-')))
        html = html.replace('{{cache_bust}}', str(cache_bust))
        
        return html
    
    def start(self):
        """Start web server"""
        from http.server import HTTPServer, SimpleHTTPRequestHandler
        import socketserver
        
        class Handler(SimpleHTTPRequestHandler):
            def __init__(self, *args, **kwargs):
                super().__init__(*args, directory=os.getcwd(), **kwargs)
            
            def do_GET(self):
                if self.path == '/' or self.path == '/index.html':
                    self.send_response(200)
                    self.send_header('Content-type', 'text/html; charset=utf-8')
                    self.end_headers()
                    html = web_server._generate_html()
                    self.wfile.write(html.encode('utf-8'))
                elif self.path.startswith('/graphs/'):
                    # Proxy graphs from graph_dir
                    graph_path = self.path.lstrip('/')
                    full_path = os.path.join(os.getcwd(), graph_path)
                    if os.path.exists(full_path) and os.path.isfile(full_path):
                        self.send_response(200)
                        if full_path.endswith('.png'):
                            self.send_header('Content-type', 'image/png')
                        self.send_header('Cache-Control', 'no-cache')
                        self.end_headers()
                        with open(full_path, 'rb') as f:
                            self.wfile.write(f.read())
                    else:
                        self.send_error(404)
                else:
                    self.send_error(404)
            
            def log_message(self, format, *args):
                pass  # Disable HTTP logging
        
        self._server = HTTPServer((self.host, self.port), Handler)
        print(f"🌐 Web interface available at: http://localhost:{self.port}")
        self._server.serve_forever()
    
    def stop(self):
        """Stop web server"""
        if self._server:
            self._server.shutdown()


# Global reference for Handler
web_server = None


class OllamaMonitor:
    def __init__(self, config: dict = CONFIG):
        self.config = config
        self.last_status: Optional[OllamaStatus] = None
        self.was_down = False
        
        # Initialize managers
        self.history = HistoryManager(config)
        self.graphs = GraphGenerator(config)
        
        # Configure logging
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s - %(levelname)s - %(message)s",
            handlers=[
                logging.FileHandler(config.get("log_file", "ollama_monitor.log")),
                logging.StreamHandler(),
            ],
        )
        self.logger = logging.getLogger(__name__)

    def get_system_memory_info(self) -> tuple:
        """Get system memory information"""
        try:
            memory = psutil.virtual_memory()
            # Find ollama process
            ollama_memory = 0
            for proc in psutil.process_iter(['name', 'memory_info']):
                try:
                    if 'ollama' in proc.info['name'].lower():
                        ollama_memory += proc.info['memory_info'].rss
                except (psutil.NoSuchProcess, psutil.AccessDenied):
                    pass
            return memory.percent, ollama_memory / (1024 * 1024)  # MB
        except Exception:
            return 0.0, 0.0

    def check_service(self) -> OllamaStatus:
        """Check Ollama service availability"""
        start_time = time.time()
        
        try:
            # Check base address
            response = requests.get(
                f"{self.config['ollama_host']}/",
                timeout=self.config["timeout"]
            )
            response_time = time.time() - start_time
            
            if response.status_code == 200:
                # Get list of models
                models_response = requests.get(
                    f"{self.config['ollama_host']}/api/tags",
                    timeout=self.config["timeout"]
                )
                
                # Get current tasks/models
                ps_response = requests.get(
                    f"{self.config['ollama_host']}/api/ps",
                    timeout=self.config["timeout"]
                )
                
                models: List[ModelInfo] = []
                running_tasks: List[RunningModel] = []
                loaded_models: List[str] = []
                total_size = 0
                
                if models_response.status_code == 200:
                    models_data = models_response.json().get("models", [])
                    for m in models_data:
                        model_info = ModelInfo(
                            name=m.get("name", ""),
                            size=m.get("size", 0),
                            modified_at=m.get("modified_at", ""),
                            digest=m.get("digest", "")
                        )
                        models.append(model_info)
                        total_size += model_info.size
                
                # Parse current tasks
                if ps_response.status_code == 200:
                    ps_data = ps_response.json()
                    loaded_model = ps_data.get("model")
                    if loaded_model:
                        loaded_models.append(loaded_model)
                    
                    for task in ps_data.get("tasks", []):
                        running_tasks.append(RunningModel(
                            model=task.get("model", ""),
                            digest=task.get("digest", ""),
                            duration=task.get("duration", 0),
                            done=task.get("done", True)
                        ))
                
                sys_mem_percent, ollama_mem_mb = self.get_system_memory_info()
                
                return OllamaStatus(
                    service_running=True,
                    api_reachable=True,
                    models_available=len(models),
                    models_loaded=loaded_models,
                    running_tasks=running_tasks,
                    total_models_size_gb=total_size / (1024 ** 3),
                    system_memory_percent=sys_mem_percent,
                    ollama_memory_mb=ollama_mem_mb,
                    response_time=response_time,
                )
            
            return OllamaStatus(
                service_running=False,
                api_reachable=False,
                models_available=0,
                response_time=response_time,
                error_message=f"HTTP {response.status_code}",
            )
            
        except requests.exceptions.ConnectionError:
            return OllamaStatus(
                service_running=False,
                api_reachable=False,
                models_available=0,
                response_time=time.time() - start_time,
                error_message="Connection refused - service not running",
            )
        except requests.exceptions.Timeout:
            return OllamaStatus(
                service_running=False,
                api_reachable=False,
                models_available=0,
                response_time=self.config["timeout"],
                error_message="Request timeout",
            )
        except Exception as e:
            return OllamaStatus(
                service_running=False,
                api_reachable=False,
                models_available=0,
                response_time=time.time() - start_time,
                error_message=str(e),
            )

    def send_telegram_alert(self, message: str):
        """Send notification to Telegram"""
        if not self.config["telegram"]["enabled"]:
            return
            
        url = f"https://api.telegram.org/bot{self.config['telegram']['bot_token']}/sendMessage"
        data = {
            "chat_id": self.config["telegram"]["chat_id"],
            "text": message,
            "parse_mode": "HTML",
        }
        
        try:
            requests.post(url, json=data, timeout=10)
        except Exception as e:
            self.logger.error(f"Failed to send Telegram alert: {e}")

    def send_email_alert(self, subject: str, body: str):
        """Send email notification"""
        if not self.config["email"]["enabled"]:
            return
            
        msg = MIMEText(body, "plain", "utf-8")
        msg["Subject"] = subject
        msg["From"] = self.config["email"]["from_addr"]
        msg["To"] = ", ".join(self.config["email"]["to_addrs"])
        
        try:
            with smtplib.SMTP(
                self.config["email"]["smtp_server"],
                self.config["email"]["smtp_port"]
            ) as server:
                server.starttls()
                server.login(
                    self.config["email"]["username"],
                    self.config["email"]["password"]
                )
                server.send_message(msg)
        except Exception as e:
            self.logger.error(f"Failed to send email alert: {e}")

    def send_alert(self, status: OllamaStatus):
        """Send problem notifications"""
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        message = f"🔴 <b>Ollama unavailable!</b>\n\n"
        message += f"⏰ Time: {timestamp}\n"
        message += f"❌ Error: {status.error_message}\n"
        message += f"⏱ Response time: {status.response_time:.2f}s"
        
        self.logger.warning(f"Ollama is DOWN: {status.error_message}")
        
        self.send_telegram_alert(message)
        self.send_email_alert("🚨 Ollama Alert!", message.replace("<b>", "").replace("</b>", ""))

    def send_recovery_alert(self):
        """Send recovery notification"""
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        message = f"✅ <b>Ollama recovered!</b>\n\n"
        message += f"⏰ Time: {timestamp}"
        
        self.logger.info("Ollama is back UP")
        
        self.send_telegram_alert(message)
        self.send_email_alert("✅ Ollama Recovered!", message.replace("<b>", "").replace("</b>", ""))

    def log_status(self, status: OllamaStatus):
        """Log status"""
        if status.service_running:
            loaded_info = ""
            if status.models_loaded:
                loaded_info = f" | Loaded: {', '.join(status.models_loaded)}"
            
            tasks_info = ""
            if status.running_tasks:
                active_tasks = [t.model for t in status.running_tasks if not t.done]
                if active_tasks:
                    tasks_info = f" | Tasks: {', '.join(set(active_tasks))}"
            
            self.logger.info(
                f"✓ Ollama is running | "
                f"Models: {status.models_available} | "
                f"Size: {status.total_models_size_gb:.2f} GB | "
                f"Memory: {status.system_memory_percent:.1f}%{loaded_info}{tasks_info} | "
                f"Response: {status.response_time:.2f}s"
            )
        else:
            self.logger.error(
                f"✗ Ollama unavailable | Error: {status.error_message}"
            )

    def run(self):
        """Start monitoring"""
        global web_server
        
        self.logger.info("=" * 60)
        self.logger.info("Starting Ollama monitoring")
        self.logger.info(f"Check interval: {self.config['check_interval']} sec")
        self.logger.info(f"Graph directory: {self.config['graph_dir']}")
        self.logger.info(f"Web interface: http://localhost:{self.config['web_port']}")
        self.logger.info("=" * 60)
        
        # Start web server in separate thread
        web_server = WebServer(self.config, self.history)
        web_thread = threading.Thread(target=web_server.start, daemon=True)
        web_thread.start()
        
        check_count = 0
        
        while True:
            status = self.check_service()
            self.log_status(status)
            
            # Save metric to history
            self.history.add_metric(status)
            
            # Generate graphs every 5 checks
            check_count += 1
            if check_count >= 5:
                self.graphs.generate_all(self.history)
                self.logger.info("📊 Graphs updated")
                check_count = 0
            
            # Send notifications when status changes
            if not status.service_running and not self.was_down:
                self.was_down = True
                self.send_alert(status)
            elif status.service_running and self.was_down:
                self.was_down = False
                self.send_recovery_alert()
            
            self.last_status = status
            time.sleep(self.config["check_interval"])


def quick_check():
    """Quick check without running the loop"""
    config = CONFIG.copy()
    monitor = OllamaMonitor(config)
    status = monitor.check_service()
    
    # Save metric
    monitor.history.add_metric(status)
    
    # Generate graphs
    monitor.graphs.generate_all(monitor.history)
    
    print("\n" + "=" * 50)
    print("📊 Ollama Status Check")
    print("=" * 50)
    
    if status.service_running:
        print(f"✅ Service: Running")
        print(f"✅ API: Available")
        print(f"📦 Available models: {status.models_available}")
        print(f"💾 Total model size: {status.total_models_size_gb:.2f} GB")
        print(f"🧠 System memory: {status.system_memory_percent:.1f}%")
        print(f"🐳 Ollama memory: {status.ollama_memory_mb:.1f} MB")
        
        if status.models_loaded:
            print(f"\n📌 Loaded models:")
            for model in status.models_loaded:
                print(f"   • {model}")
        else:
            print(f"\n📌 Loaded models: none")
        
        if status.running_tasks:
            active_tasks = [t for t in status.running_tasks if not t.done]
            if active_tasks:
                print(f"\n🔄 Active tasks:")
                for task in active_tasks:
                    print(f"   • {task.model} ({task.duration_sec:.1f} sec)")
        
        print(f"\n⏱ Response time: {status.response_time:.3f} sec")
        
        print(f"\n📈 Graphs saved to directory: {config['graph_dir']}/")
        print(f"   • response_time.png - response time")
        print(f"   • models_loaded.png - loaded models")
        print(f"   • memory.png - memory usage")
        print(f"   • combined.png - combined graph")
        
        print(f"\n🌐 Web interface: http://localhost:{config['web_port']}")
    else:
        print(f"❌ Service: Unavailable")
        print(f"❌ Error: {status.error_message}")
    
    print("=" * 50 + "\n")
    return status.service_running


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == "--check":
        quick_check()
    elif len(sys.argv) > 1 and sys.argv[1] == "--web":
        # Web server only
        config = CONFIG.copy()
        history = HistoryManager(config)
        graphs = GraphGenerator(config)
        
        print(f"Starting web server on port {config['web_port']}...")
        web_server = WebServer(config, history)
        web_server.start()
    else:
        monitor = OllamaMonitor()
        monitor.run()
