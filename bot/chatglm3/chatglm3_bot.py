# encoding:utf-8

from bot.bot import Bot
from bot.chatglm3.chatglm3_session import ChatGLM3Session
from bot.session_manager import SessionManager, Session
from bridge.context import ContextType
from bridge.reply import Reply, ReplyType
from common.log import logger
from config import conf, load_config

from typing import Union
from pathlib import Path
from peft import AutoPeftModelForCausalLM, PeftModelForCausalLM
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    PreTrainedModel,
    PreTrainedTokenizer,
    PreTrainedTokenizerFast,
    StoppingCriteria,
    StoppingCriteriaList,
)
import torch

ModelType = Union[PreTrainedModel, PeftModelForCausalLM]
TokenizerType = Union[PreTrainedTokenizer, PreTrainedTokenizerFast]


def _resolve_path(path: Union[str, Path]) -> Path:
    return Path(path).expanduser().resolve()


def load_model_and_tokenizer(
        model_dir: Union[str, Path], trust_remote_code: bool = True
) -> tuple[ModelType, TokenizerType]:
    model_dir = _resolve_path(model_dir)

    if (model_dir / 'adapter_config.json').exists():
        model = AutoPeftModelForCausalLM.from_pretrained(
            model_dir, trust_remote_code=trust_remote_code, device_map='auto'
        )
        tokenizer_dir = model.peft_config['default'].base_model_name_or_path
    else:
        model = AutoModelForCausalLM.from_pretrained(
            model_dir, trust_remote_code=trust_remote_code, device_map='auto'
        )
        tokenizer_dir = model_dir
    tokenizer = AutoTokenizer.from_pretrained(
        tokenizer_dir, trust_remote_code=trust_remote_code
    )
    return model, tokenizer

STOP_IDS = [64795, 64796, 64797] # user, assistant, observation
SKIP_IDS = STOP_IDS + [30910, 13, 0, 2]
class StopOnTokens(StoppingCriteria):
    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor, **kwargs) -> bool:
        stop_ids = STOP_IDS
        for stop_id in stop_ids:
            if input_ids[0][-1] == stop_id:
                return True
        return False

class ChatGLM3Bot(Bot):
    def __init__(self):
        super().__init__()
        self.sessions = SessionManager(ChatGLM3Session, model=conf().get("model") or "ZHIPU_AI")
        self.args = {
            "temperature": conf().get("temperature", 0.6),  # 值在(0,1)之间
            "top_p": conf().get("top_p", 0.8),  # 值在(0,1)之间
            "max_length": 128,
        }

        model_path = conf().get("model_path", "./models/checkpoint-3000")
        model, tokenizer = load_model_and_tokenizer(model_path, trust_remote_code=True)
        self.model = model
        self.tokenizer = tokenizer

    def predict(self, messages, max_length, top_p, temperature):
        stop = StopOnTokens()

        print("\n\n====conversation====\n", messages)
        model_inputs = self.tokenizer.apply_chat_template(messages,
                                                          add_generation_prompt=True,
                                                          tokenize=True,
                                                          return_tensors="pt").to(next(self.model.parameters()).device)

        generate_kwargs = {
            "input_ids": model_inputs,
            "max_new_tokens": max_length,
            "do_sample": True,
            "top_p": top_p,
            "temperature": temperature,
            "stopping_criteria": StoppingCriteriaList([stop]),
            "repetition_penalty": 1.2,
        }
        output_with_prefix = self.model.generate(**generate_kwargs).detach().cpu().numpy()[0].tolist()
        output = output_with_prefix[model_inputs.shape[1]:]
        output = [token_id for token_id in output if token_id not in SKIP_IDS]
        tokens = self.tokenizer.batch_decode(output, skip_special_tokens=True)

        return "".join(tokens)

    def reply(self, query, context=None):
        if context.type == ContextType.TEXT:
            logger.info("[ChatGLM3] query={}".format(query))

            session_id = context["session_id"]
            reply = None
            clear_memory_commands = conf().get("clear_memory_commands", ["#清除记忆"])
            if query in clear_memory_commands:
                self.sessions.clear_session(session_id)
                reply = Reply(ReplyType.INFO, "记忆已清除")
            elif query == "#清除所有":
                self.sessions.clear_all_session()
                reply = Reply(ReplyType.INFO, "所有人记忆已清除")
            elif query == "#更新配置":
                load_config()
                reply = Reply(ReplyType.INFO, "配置已更新")
            if reply:
                return reply
            session = self.sessions.session_query(query, session_id)
            logger.debug("[ChatGLM3] session query={}".format(session.messages))

            model = context.get("gpt_model")
            new_args = None
            if model:
                new_args = self.args.copy()
                new_args["model"] = model
            # if context.get('stream'):
            #     # reply in stream
            #     return self.reply_text_stream(query, new_query, session_id)

            reply_content = self.reply_text(session, args=new_args)
            logger.debug(
                "[ChatGLM3] new_query={}, session_id={}, reply_cont={}, completion_tokens={}".format(
                    session.messages,
                    session_id,
                    reply_content["content"],
                    reply_content["completion_tokens"],
                )
            )
            if reply_content["completion_tokens"] == 0 and len(reply_content["content"]) > 0:
                reply = Reply(ReplyType.ERROR, reply_content["content"])
            elif reply_content["completion_tokens"] > 0:
                self.sessions.session_reply(reply_content["content"], session_id, reply_content["total_tokens"])
                reply = Reply(ReplyType.TEXT, reply_content["content"])
            else:
                reply = Reply(ReplyType.ERROR, reply_content["content"])
                logger.debug("[ChatGLM3] reply {} used 0 tokens.".format(reply_content))
            return reply
            '''
            elif context.type == ContextType.IMAGE_CREATE:
                ok, retstring = self.create_img(query, 0)
                reply = None
                if ok:
                    reply = Reply(ReplyType.IMAGE_URL, retstring)
                else:
                    reply = Reply(ReplyType.ERROR, retstring)
                return reply
            '''
        else:
            reply = Reply(ReplyType.ERROR, "Bot不支持处理{}类型的消息".format(context.type))
            return reply

    def reply_text(self, session: Session, args=None, retry_count=0) -> dict:
        """
        call openai's ChatCompletion to get the answer
        :param session: a conversation session
        :param session_id: session id
        :param retry_count: retry count
        :return: {}
        """
        try:
            # if conf().get("rate_limit_chatgpt") and not self.tb4chatgpt.get_token():
            #     raise openai.error.RateLimitError("RateLimitError: rate limit exceeded")
            # if api_key == None, the default openai.api_key will be used
            if args is None:
                args = self.args
            response = self.predict(session.messages, **args)
            # logger.debug("[ZHIPU_AI] response={}".format(response))
            # logger.info("[ZHIPU_AI] reply={}, total_tokens={}".format(response.choices[0]['message']['content'], response["usage"]["total_tokens"]))

            return {
                "total_tokens": len(response),
                "completion_tokens": len(response),
                "content": response,
            }
        except Exception as e:
            need_retry = retry_count < 2
            result = {"completion_tokens": 0, "content": "我现在有点累了，等会再来吧"}
            '''
            if isinstance(e, openai.error.RateLimitError):
                logger.warn("[ZHIPU_AI] RateLimitError: {}".format(e))
                result["content"] = "提问太快啦，请休息一下再问我吧"
                if need_retry:
                    time.sleep(20)
            elif isinstance(e, openai.error.Timeout):
                logger.warn("[ChatGLM3] Timeout: {}".format(e))
                result["content"] = "我没有收到你的消息"
                if need_retry:
                    time.sleep(5)
            elif isinstance(e, openai.error.APIError):
                logger.warn("[ChatGLM3] Bad Gateway: {}".format(e))
                result["content"] = "请再问我一次"
                if need_retry:
                    time.sleep(10)
            elif isinstance(e, openai.error.APIConnectionError):
                logger.warn("[ChatGLM3] APIConnectionError: {}".format(e))
                result["content"] = "我连接不到你的网络"
                if need_retry:
                    time.sleep(5)
            else:
            '''
            if True:
                logger.exception("[ChatGLM3] Exception: {}".format(e), e)
                need_retry = False
                self.sessions.clear_session(session.session_id)

            if need_retry:
                logger.warn("[ChatGLM3] 第{}次重试".format(retry_count + 1))
                return self.reply_text(session, args, retry_count + 1)
            else:
                return result
