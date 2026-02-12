#!/usr/bin/env python3
"""
Ollama Chat Client v2 - Удалённый чат с сервером Ollama
Поддерживает потоковую передачу, историю, выбор моделей и переключение моделей в чате.
"""

import requests
import sys
import argparse
from typing import Iterator, List, Dict, Optional


class Colors:
    HEADER = '\033[95m'
    BLUE = '\033[94m'
    CYAN = '\033[96m'
    GREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'


class OllamaChat:
    """Клиент для чата с сервером Ollama"""
    
    def __init__(
        self,
        host: str = "http://localhost:11434",
        model: str = "llama3.2",
        system_prompt: str = "Ты полезный ассистент. Отвечай кратко и по существу."
    ):
        self.host = host.rstrip('/')
        self.model = model
        self.system_prompt = system_prompt
        self.history: List[Dict[str, str]] = []
        self.available_models: List[str] = []
        
    def check_connection(self) -> bool:
        """Проверка подключения к серверу Ollama"""
        try:
            response = requests.get(f"{self.host}/api/tags", timeout=5)
            if response.status_code == 200:
                data = response.json()
                self.available_models = [m['name'] for m in data.get('models', [])]
                print(f"{Colors.GREEN}✓ Подключение успешно!{Colors.ENDC}")
                self.show_models()
                return True
            else:
                print(f"{Colors.FAIL}✗ Ошибка подключения: {response.status_code}{Colors.ENDC}")
                return False
        except requests.exceptions.ConnectionError:
            print(f"{Colors.FAIL}✗ Не удалось подключиться к {self.host}{Colors.ENDC}")
            return False
        except Exception as e:
            print(f"{Colors.FAIL}✗ Ошибка: {e}{Colors.ENDC}")
            return False
    
    def get_available_models(self) -> List[str]:
        """Получение списка доступных моделей"""
        if self.available_models:
            return self.available_models
        try:
            response = requests.get(f"{self.host}/api/tags", timeout=5)
            if response.status_code == 200:
                self.available_models = [m['name'] for m in response.json().get('models', [])]
            return self.available_models
        except:
            return []
    
    def show_models(self):
        """Показать доступные модели"""
        models = self.get_available_models()
        if models:
            print(f"{Colors.CYAN}Доступные модели:{Colors.ENDC}")
            for m in models[:8]:
                if m == self.model:
                    print(f"  {Colors.GREEN}• {m} (текущая){Colors.ENDC}")
                else:
                    print(f"  • {m}")
            if len(models) > 8:
                print(f"  ... и ещё {len(models) - 8}")
        else:
            print(f"{Colors.WARNING}Не удалось получить список моделей{Colors.ENDC}")
    
    def switch_model(self, model_name: str) -> bool:
        """Переключение на другую модель"""
        available = self.get_available_models()
        
        if not available:
            self.model = model_name
            print(f"{Colors.CYAN}Переключено на: {model_name}{Colors.ENDC}")
            return True
        
        for model in available:
            if model_name.lower() in model.lower() or model.lower() in model_name.lower():
                self.model = model
                self.history = []  # Очищаем историю при смене модели
                print(f"{Colors.GREEN}✓ Модель переключена на: {model}{Colors.ENDC}")
                return True
        
        print(f"{Colors.WARNING}Модель '{model_name}' не найдена.{Colors.ENDC}")
        self.show_models()
        return False
    
    def _build_messages(self) -> List[Dict[str, str]]:
        """Построение сообщений для API"""
        messages = []
        messages.append({"role": "system", "content": self.system_prompt})
        for msg in self.history[-20:]:
            messages.append(msg)
        return messages
    
    def chat(self, message: str, stream: bool = True) -> str:
        """Отправка сообщения и получение ответа"""
        self.history.append({"role": "user", "content": message})
        
        payload = {
            "model": self.model,
            "messages": self._build_messages(),
            "stream": stream,
            "options": {"temperature": 0.7, "num_predict": 2048}
        }
        
        full_response = ""
        
        try:
            if stream:
                print(f"\n{Colors.CYAN}🤖 {self.model}:{Colors.ENDC} ", end="", flush=True)
                response = requests.post(f"{self.host}/api/chat", json=payload, stream=True, timeout=60)
                
                for line in response.iter_lines():
                    if line:
                        import json
                        chunk = json.loads(line.decode('utf-8'))
                        if 'message' in chunk:
                            content = chunk['message'].get('content', '')
                            full_response += content
                            print(content, end="", flush=True)
                        if chunk.get('done', False):
                            break
                print()
            else:
                print(f"\n{Colors.CYAN}🤖 {self.model}:{Colors.ENDC} ", end="")
                response = requests.post(f"{self.host}/api/chat", json=payload, timeout=60)
                if response.status_code == 200:
                    full_response = response.json()['message']['content']
                    print(full_response)
                else:
                    print(f"{Colors.FAIL}Ошибка: {response.status_code}{Colors.ENDC}")
                    return ""
            
            self.history.append({"role": "assistant", "content": full_response})
            return full_response
            
        except Exception as e:
            print(f"{Colors.FAIL}Ошибка: {e}{Colors.ENDC}")
            return ""
    
    def clear_history(self):
        """Очистка истории"""
        self.history = []
        print(f"{Colors.GREEN}История очищена!{Colors.ENDC}")
    
    def show_history(self):
        """Показать историю сообщений"""
        print(f"\n{Colors.HEADER}История ({len(self.history)} сообщений):{Colors.ENDC}")
        print("-" * 50)
        for msg in self.history:
            role = msg['role'].upper()
            color = Colors.GREEN if role == "USER" else Colors.CYAN
            print(f"{color}[{role}]{Colors.ENDC}")
            print(f"  {msg['content'][:100]}{'...' if len(msg['content']) > 100 else ''}\n")


def interactive_chat(host: str, model: str, system_prompt: str):
    """Интерактивный режим чата"""
    client = OllamaChat(host=host, model=model, system_prompt=system_prompt)
    
    print(f"\n{Colors.HEADER}{Colors.BOLD}")
    print("╔══════════════════════════════════════════╗")
    print("║     Ollama Chat Client v2                ║")
    print("╚══════════════════════════════════════════╝")
    print(f"{Colors.ENDC}")
    print(f"Подключение: {host}")
    print(f"{Colors.WARNING}Команды: /models, /use <модель>, /clear, /history, /quit{Colors.ENDC}\n")
    
    if not client.check_connection():
        return
    
    print(f"\n{Colors.GREEN}Начните чат!{Colors.ENDC}\n")
    
    while True:
        try:
            user_input = input(f"{Colors.GREEN}Вы:{Colors.ENDC} ").strip()
            
            if not user_input:
                continue
            
            if user_input.startswith('/'):
                cmd = user_input.lower().split()
                
                if cmd[0] in ['/quit', '/exit']:
                    print(f"{Colors.CYAN}До свидания!{Colors.ENDC}")
                    break
                elif cmd[0] == '/clear':
                    client.clear_history()
                elif cmd[0] == '/history':
                    client.show_history()
                elif cmd[0] == '/models':
                    client.show_models()
                elif cmd[0] == '/use' and len(cmd) > 1:
                    client.switch_model(cmd[1])
                else:
                    print(f"{Colors.WARNING}Команды: /models, /use <модель>, /clear, /history, /quit{Colors.ENDC}")
                continue
            
            client.chat(user_input)
            
        except KeyboardInterrupt:
            print(f"\n\n{Colors.CYAN}До свидания!{Colors.ENDC}")
            break
        except EOFError:
            break


def main():
    parser = argparse.ArgumentParser(
        description="Ollama Chat Client v2",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Команды в чате:
  /models          - показать доступные модели
  /use <модель>    - переключиться на другую модель
  /clear           - очистить историю
  /history         - показать историю
  /quit            - выход
        """
    )
    
    parser.add_argument('--host', '-H', default="http://10.10.10.2:11434",
                        help='Адрес сервера Ollama')
    parser.add_argument('--model', '-m', default="qwen3",
                        help='Модель по умолчанию')
    parser.add_argument('--prompt', '-p',
                        default="Ты полезный ассистент.",
                        help='System prompt')
    
    args = parser.parse_args()
    interactive_chat(host=args.host, model=args.model, system_prompt=args.prompt)


if __name__ == "__main__":
    main()

