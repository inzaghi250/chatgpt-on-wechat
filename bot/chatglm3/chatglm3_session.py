from bot.zhipuai.zhipu_ai_session import ZhipuAISession
from common.log import logger


class ChatGLM3Session(ZhipuAISession):
    def __init__(self, session_id, system_prompt=None, model="glm-4"):
        super(ZhipuAISession, self).__init__(session_id, system_prompt=None)
        self.model = model
        self.reset()

    # 重置会话
    def reset(self):
        self.messages = []
