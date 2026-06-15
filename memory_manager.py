from langchain.memory import ConversationBufferMemory, ChatMessageHistory
from langchain_core.messages import HumanMessage, AIMessage
import json
import os

class PersistentMemory:
    def __init__(self, history_file="agent_history.json"):
        self.history_file = history_file
        self.history = ChatMessageHistory()
        self._load_history()
        self.memory = ConversationBufferMemory(chat_memory=self.history, return_messages=True)

    def _load_history(self):
        if os.path.exists(self.history_file):
            with open(self.history_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
                for message in data:
                    if message['type'] == 'human':
                        self.history.add_user_message(message['content'])
                    elif message['type'] == 'ai':
                        self.history.add_ai_message(message['content'])

    def save_history(self):
        data = []
        for message in self.history.messages:
            if isinstance(message, HumanMessage):
                data.append({'type': 'human', 'content': message.content})
            elif isinstance(message, AIMessage):
                data.append({'type': 'ai', 'content': message.content})
        with open(self.history_file, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=4)

    def get_memory(self):
        return self.memory
