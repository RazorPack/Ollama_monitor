#!/usr/bin/env python3
"""
Ollama Chat Client - Удалённый чат с сервером Ollama
Поддерживает потоковую передачу ответов, историю сообщений и различные модели.
"""

import requests
import sys
import argparse
from datetime import datetime
from typing import Iterator, List, Dict, Optional


# ANSI цветовые коды для терминала
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
        
    def check_connection(self) -> bool:
        """Проверка подключения к серверу Ollama"""
        try:
            response = requests.get(f"{self.host}/api/tags", timeout=5)
            if response.status_code == 200:
                models = response.json().get('models', [])
                print(f"{Colors.GREEN}✓ Подключение успешно!{Colors.ENDC}")
                print(f"{Colors.CYAN}Доступные модели:{Colors.ENDC}")
                for m in models[:5]:  # Показываем первые 5
                    print(f"  • {m['name']}")
                if len(models) > 5:
                    print(f"  ... и ещё {len(models) - 5}")
                return True
            else:
                print(f"{Colors.FAIL}✗ Ошибка подключения: {response.status_code}{Colors.ENDC}")
                return False
        except requests.exceptions.ConnectionError:
            print(f"{Colors.FAIL}✗ Не удалось подключиться к {self.host}{Colors.ENDC}")
            print(f"{Colors.WARNING}Убедитесь, что сервер Ollama запущен:{Colors.ENDC}")
            print(f"  1. Установите Ollama: https://ollama.com")
            print(f"  2. Запустите: ollama serve")
            return False
        except Exception as e:
            print(f"{Colors.FAIL}✗ Ошибка: {e}{Colors.ENDC}")
            return False
    
    def _build_messages(self) -> List[Dict[str, str]]:
        """Построение сообщений для API"""
        messages = []
        
        # Добавляем system prompt
        messages.append({"role": "system", "content": self.system_prompt})
        
        # Добавляем историю (ограничиваем последними 20 сообщениями)
        for msg in self.history[-20:]:
            messages.append(msg)
        
        return messages
    
    def chat(self, message: str, stream: bool = True) -> str:
        """Отправка сообщения и получение ответа"""
        
        # Добавляем сообщение пользователя в историю
        self.history.append({"role": "user", "content": message})
        
        # Подготовка данных для API
        payload = {
            "model": self.model,
            "messages": self._build_messages(),
            "stream": stream,
            "options": {
                "temperature": 0.7,
                "num_predict": 2048
            }
        }
        
        full_response = ""
        
        try:
            if stream:
                print(f"\n{Colors.CYAN}🤖 {self.model}:{Colors.ENDC} ", end="", flush=True)
                
                response = requests.post(
                    f"{self.host}/api/chat",
                    json=payload,
                    stream=True,
                    timeout=60
                )
                
                for line in response.iter_lines():
                    if line:
                        data = line.decode('utf-8')
                        import json as json_lib
                        chunk = json_lib.loads(data)
                        
                        if 'message' in chunk:
                            content = chunk['message'].get('content', '')
                            full_response += content
                            print(content, end="", flush=True)
                        
                        if chunk.get('done', False):
                            break
                
                print()  # Новая строка после ответа
                
            else:
                # Непотоковый режим
                print(f"\n{Colors.CYAN}🤖 {self.model}:{Colors.ENDC} ", end="")
                
                response = requests.post(
                    f"{self.host}/api/chat",
                    json=payload,
                    timeout=60
                )
                
                if response.status_code == 200:
                    data = response.json()
                    full_response = data['message']['content']
                    print(full_response)
                else:
                    error_msg = f"Ошибка: {response.status_code}"
                    print(f"{Colors.FAIL}{error_msg}{Colors.ENDC}")
                    return error_msg
            
            # Добавляем ответ ассистента в историю
            self.history.append({"role": "assistant", "content": full_response})
            
            return full_response
            
        except requests.exceptions.Timeout:
            error_msg = "Превышено время ожидания"
            print(f"{Colors.FAIL}{error_msg}{Colors.ENDC}")
            return error_msg
        except Exception as e:
            error_msg = f"Ошибка: {e}"
            print(f"{Colors.FAIL}{error_msg}{Colors.ENDC}")
            return error_msg
    
    def clear_history(self):
        """Очистка истории"""
        self.history = []
        print(f"{Colors.GREEN}История очищена!{Colors.ENDC}")
    
    def show_history(self):
        """Показать историю сообщений"""
        print(f"\n{Colors.HEADER}История чата ({len(self.history)} сообщений):{Colors.ENDC}")
        print("-" * 50)
        for i, msg in enumerate(self.history):
            role = msg['role'].upper()
            color = Colors.GREEN if role == "USER" else Colors.CYAN
            print(f"{color}[{role}]{Colors.ENDC}")
            print(f"  {msg['content'][:100]}{'...' if len(msg['content']) > 100 else ''}")
            print()


def interactive_chat(host: str, model: str, system_prompt: str):
    """Интерактивный режим чата"""
    client = OllamaChat(host=host, model=model, system_prompt=system_prompt)
    
    print(f"\n{Colors.HEADER}{Colors.BOLD}")
    print("╔══════════════════════════════════════════╗")
    print("║     Ollama Chat Client                   ║")
    print("╚══════════════════════════════════════════╝")
    print(f"{Colors.ENDC}")
    print(f"Подключение к: {host}")
    print(f"Модель: {model}")
    print(f"{Colors.WARNING}Команды: /clear - очистить историю, /history - показать историю, /quit - выход{Colors.ENDC}")
    print()
    
    if not client.check_connection():
        print(f"{Colors.FAIL}Не удалось подключиться к серверу.{Colors.ENDC}")
        return
    
    print(f"\n{Colors.GREEN}Начните чат! (Ctrl+C для выхода){Colors.ENDC}\n")
    
    while True:
        try:
            user_input = input(f"{Colors.GREEN}Вы:{Colors.ENDC} ").strip()
            
            if not user_input:
                continue
            
            if user_input.startswith('/'):
                command = user_input.lower()
                
                if command == '/quit' or command == '/exit':
                    print(f"{Colors.CYAN}До свидания!{Colors.ENDC}")
                    break
                elif command == '/clear':
                    client.clear_history()
                elif command == '/history':
                    client.show_history()
                else:
                    print(f"{Colors.WARNING}Неизвестная команда: {command}{Colors.ENDC}")
                
                continue
            
            client.chat(user_input)
            
        except KeyboardInterrupt:
            print(f"\n\n{Colors.CYAN}До свидания!{Colors.ENDC}")
            break
        except EOFError:
            break


def main():
    parser = argparse.ArgumentParser(
        description="Ollama Chat Client - Удалённый чат с сервером Ollama",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Примеры использования:
  %(prog)s                           # Запуск с настройками по умолчанию
  %(prog)s --host http://192.168.1.100:11434  # Подключение к удалённому серверу
  %(prog)s --model llama3.2          # Использование другой модели
  %(prog)s --prompt "Ты эксперт по Python"  # Кастомный system prompt

Команды в интерактивном режиме:
  /clear    - очистить историю чата
  /history  - показать историю сообщений
  /quit     - выйти из программы
        """
    )
    
    parser.add_argument(
        '--host', '-H',
        default="http://localhost:11434",
        help='Адрес сервера Ollama (по умолчанию: http://localhost:11434)'
    )
    parser.add_argument(
        '--model', '-m',
        default="llama3.2",
        help='Имя модели для чата (по умолчанию: llama3.2)'
    )
    parser.add_argument(
        '--prompt', '-p',
        default="Ты полезный ассистент. Отвечай кратко и по существу.",
        help='System prompt для модели'
    )
    
    args = parser.parse_args()
    
    interactive_chat(
        host=args.host,
        model=args.model,
        system_prompt=args.prompt
    )


if __name__ == "__main__":
    main()

