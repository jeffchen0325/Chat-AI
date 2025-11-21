import os
from openai import OpenAI

class Deepseek:

    def __init__(self, token_file, base_url, model_id, pre_msg:str='', max_history=10):
        if not os.path.exists(token_file):
            raise FileNotFoundError(f"Token 文件不存在: {token_file}")

        try:
            with open(token_file, 'r', encoding='utf-8') as file:
                token = file.read().strip()
                if not token:
                    raise ValueError("Token 文件为空")
        except Exception as e:
            raise Exception(f"读取 token 文件失败: {e}")
        self.client = OpenAI(api_key=token, base_url=base_url)
        self.model = model_id
        self.msg = [{"role": "system", "content": pre_msg}] if pre_msg else []
        self.max_history = max_history
        print(f"Deepseek 模型设置成功")


    def chat(self, content, max_length=100, padding=True):
        self.msg.append({"role": "user", "content": content})
        response = self.client.chat.completions.create(
            model=self.model,
            messages=self.msg,
            max_tokens=max_length
        )
        reply = response.choices[0].message.content
        if padding:
            self.msg.append({"role": "assistant", "content": reply})
        else:
            _ = self.msg.pop()

        self._trim_history()

        return reply

    def _trim_history(self):
        """修剪对话历史，保留最新的消息"""
        if len(self.msg) > self.max_history:
            self.msg = self.msg[-10:]


if __name__ == "__main__":
    deepseek = Deepseek("../mytoken.txt", "https://api.deepseek.com", "deepseek-chat")

    messages = [
        "你好，Deepseek",
        "很高兴认识你",
        "我第一句话是什么来着？"
    ]

    for msg in messages:
        print(f"用户：{msg}")
        response = deepseek.chat(msg, max_length=50)
        print(f"用户：{response}")
        print('=' * 50)
