import os
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com' #使用 Hugging Face 国内镜像
import re
from typing import Optional, Dict, Any, List
import openai  # 需 pip install openai
import numpy as np
import torch
import torch.nn.functional as F

from transformers import AutoTokenizer, AutoModelForCausalLM, AutoModel
import chromadb # 新的依赖包, 需 pip install chromadb
from chromadb.config import Settings

class ChatBot_RAG:
    """RAG聊天机器人"""

    def __init__(
        self,
        api_configs={
            "deepseek":{ # 发布方: Deepseek 【付费】
                "api_key":"sk-730d31d5a8974d31a26e211ca093077e", # "your-own-api-key",
                "base_url":"https://api.deepseek.com", 
                "model":"deepseek-chat",
                "extra_body": {},  
                "model_embedding":None, #DeepSeek 的官方 API 不支持文本向量化功能 
            },
            "qianfan":{ # 发布方: 百度的文心一言 【付费】
                "api_key":"bce-v3/ALTAK-cuc0BJ5qo2BsuaMO4LlFS/36ab4682691b988c457957642f800de1e18a4c2a", # "your-own-api-key",
                "base_url":"https://qianfan.baidubce.com/v2", 
                "model":"ernie-5.0",
                "extra_body": {},   
                "model_embedding":"bge-large-zh",
            },
            "tongyi":{ # 发布方: 阿里的百炼平台 【免费】
                "api_key":"sk-cfcc5b6838984757a2867507b59f3b6c", # "your-own-api-key",
                "base_url":"https://dashscope.aliyuncs.com/compatible-mode/v1",
                "model":"qwen3-32b",
                "extra_body":{"enable_thinking": False},
                "model_embedding":None,
            },
            "tongyi2":{ # 发布方: 阿里的百炼平台 【百炼平台免费试用】
                "api_key":"sk-cfcc5b6838984757a2867507b59f3b6c", # "your-own-api-key",
                "base_url":"https://dashscope.aliyuncs.com/compatible-mode/v1",
                "model":"deepseek-v3",
                "extra_body": {"enable_thinking": False},
                "model_embedding":None, # DeepSeek 的官方 API 不支持文本向量化功能 
            },
            "minimax":{ # 发布方: Minimax【免费】
                "api_key":"sk-api-0fHzB1rA3n96tFqTkwW-fl6Kq7uK_AkrrAbHXrLxTK3XCGR9N3WMP90aOXQqKiAcFL24a-Ba-H0R8X5fg5BRIDJW9Bx5oCsFX-A5-EuT16qwwUeuSN18gNo", # "your-own-api-key",
                "base_url":"https://api.minimaxi.com/v1",
                "model":"MiniMax-M2.7",
                "extra_body":{"reasoning_split": True},
                "model_embedding":None,
            }
        },
        local_model_configs={
            "localhost": {
                "model_path": "./DeepSeek-Model"
            }
        },
        generation_params={
            "max_new_tokens": 512,
            "temperature": 0.6,
            "stream": False,
        },
        RAG={
            "use":True,
            "retrieveDataset":"./Dataset",
            "needSave2chroma":True,
            "top_k":2,
            "current_modelEmb":"localhost", #"qianfan", #
        },
        default_model = "qianfan"
    ):
        """
            api_configs: API 模型配置，键为模型标识，值为包含 api_key, base_url, model, extra_body 的字典。
            local_model_configs: 本地模型配置，键为模型标识，值为包含 model_path 的字典。
            generation_params: 生成参数，如 max_new_tokens, temperature, stream 等。
            default_model: 默认使用的模型标识。
        """
        self.api_models = api_configs
        self.local_models = local_model_configs
        self.gen_params = generation_params
        self.RAG = RAG

        # 本地嵌入模型配置信息（lazy load）
        self.local_embedding_model = "BAAI/bge-small-zh-v1.5" # local_embedding_model
        self.local_embedding_device = "cpu" # local_embedding_device
        self._local_embedding = None  # (tokenizer, model)

        self.current_model = default_model
        self.default_system_prompt = "请用简洁、简短的语言回答用户的问题"

        # 每个模型的对话历史
        self.histories: Dict[str, List[Dict[str, str]]] = {}
        all_models = set(self.api_models.keys()) | set(self.local_models.keys())
        for model_id in all_models:
            self.histories[model_id] = [
                {"role": "system", "content": self.default_system_prompt}
            ]

        # 已加载的本地模型对象
        self.loaded_local: Dict[str, Any] = {}

        # 加载本地模型（若路径存在）
        for model_id, cfg in self.local_models.items():
            if os.path.exists(cfg.get("model_path", "")):
                self._load_local_model(model_id)
        
        if self.RAG["use"]:
            # (1)创建 Chroma 客户端
            chroma_client = chromadb.PersistentClient(
                path="./chroma_db_"+self.RAG["current_modelEmb"],
                settings=Settings(anonymized_telemetry=False)
            )
            # (2)获取或创建集合，指定相似度度量方式（默认余弦相似度）
            self.collection = chroma_client.get_or_create_collection(
                name="my_lab3_docs",
                metadata={"hnsw:space": "cosine"}
            )
            if RAG['needSave2chroma']:
                retrieveDataset = RAG["retrieveDataset"]
                for fileName in os.listdir(retrieveDataset):
                #     self._save2chroma(filePath = retrieveDataset+"/"+fileName)        
                    if self.collection.count() == 0:# 仅在集合为空时执行初始化入库，避免重复写入
                        self._save2chroma(filePath = retrieveDataset+"/"+fileName)
                    else:
                        print(f"Chroma集合已有数据({self.collection.count()}条)，跳过初始化入库")

    def _load_local_model(self, model_id: str):
        """加载指定的本地模型到内存。"""
        cfg = self.local_models[model_id]
        model_path = cfg["model_path"]
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype="auto",
            device_map="cpu",  # 可改为 "cuda:0" 或 "auto"
        )
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.loaded_local[model_id] = {"model": model, "tokenizer": tokenizer}

    def switch_model(self, model_id: str) -> bool:
        """切换到指定模型。"""
        all_models = set(self.api_models.keys()) | set(self.local_models.keys())
        if model_id in all_models:
            self.current_model = model_id
            return True
        else:
            print(f"无法识别的模型: {model_id}，可用模型: {all_models}")
            return False

    def chat(self, question: str, model_id: Optional[str] = None) -> str:
        """
        向当前（或指定）模型发送提问并获取回复。

        Args:
            question: 用户问题。
            model_id: 临时指定模型，若为 None 则使用当前模型。

        Returns:
            模型回复文本。
        """
        if not question.strip():
            return "请输入有效的问题~"

        target_model = model_id or self.current_model
        history = self.histories.setdefault(target_model, [])
        history.append({"role": "user", "content": question})

        # RAG 增强
        enhanced_question = question
        if self.RAG["use"]:
            try:
                retrieved_docs = self.search_chroma(question,self.RAG["top_k"])
                if retrieved_docs:
                    context = "\n\n".join(retrieved_docs)
                    enhanced_question = (
                        f"请参考以下资料回答问题。\n"
                        f"【参考资料】\n{context}\n\n"
                        f"【问题】\n{question}"
                    )
            except Exception as e:
                print(f"检索增强失败: {e}")
            temp_history = history.copy() # 临时历史（只用于本次生成）
            if temp_history and temp_history[-1]["role"] == "user":
                temp_history[-1]["content"] = enhanced_question
        else:
            temp_history = history

        try:
            if target_model in self.local_models:
                response = self._call_local_model(target_model, temp_history)
            elif target_model in self.api_models:
                response = self._call_api_model(target_model, temp_history)
            else:
                raise ValueError(f"未知的模型标识: {target_model}")

            # 清理 DeepSeek-R1 的思维链标记（若存在）
            response = re.sub(r'.*?</think>\s*', '', response, flags=re.DOTALL).strip()

            history.append({"role": "assistant", "content": response})
            return response

        except Exception as e:
            error_msg = f"调用模型失败: {e}"
            # 移除已添加的用户消息，避免历史污染
            if history and history[-1]["role"] == "user":
                history.pop()
            return error_msg
        
    def _call_local_model(self, model_id: str, history: List[Dict[str, str]]) -> str:
        """调用本地 HuggingFace 模型生成回复。"""
        local = self.loaded_local.get(model_id)
        if not local:
            raise RuntimeError(f"本地模型 {model_id} 尚未加载，请先执行加载。")

        model = local["model"]
        tokenizer = local["tokenizer"]

        text = tokenizer.apply_chat_template(
            history,
            tokenize=False,
            add_generation_prompt=True,
        )
        inputs = tokenizer(text, return_tensors="pt").to(model.device)

        outputs = model.generate(
            **inputs,
            max_new_tokens=self.gen_params.get("max_new_tokens", 256),
            temperature=self.gen_params.get("temperature", 0.1),
            top_p=0.95,
            do_sample=True,
            repetition_penalty=1.1,
            pad_token_id=tokenizer.eos_token_id,
        )

        response = tokenizer.decode(
            outputs[0][inputs.input_ids.shape[1] :],
            skip_special_tokens=True,
        )
        return response

    def _call_api_model(self, model_id: str, history: List[Dict[str, str]]) -> str:
        """调用 OpenAI 兼容 API 生成回复。"""
        cfg = self.api_models[model_id]
        client = openai.OpenAI(
            api_key=cfg["api_key"],
            base_url=cfg["base_url"],
        )

        stream = self.gen_params.get("stream", False)
        message = client.chat.completions.create(
            model=cfg["model"],
            messages=history,
            extra_body=cfg.get("extra_body", {}),
            temperature=self.gen_params.get("temperature", 0.1),
            max_tokens=self.gen_params.get("max_new_tokens", 1000),
            stream=stream,
        )

        if stream:
            response = ""
            print(f" {model_id}:", end="", flush=True)
            for chunk in message:
                if chunk.choices[0].delta.content is not None:
                    content = chunk.choices[0].delta.content
                    response += content
                    print(content, end="", flush=True)
            print()  # 流式结束后换行
            return response
        else:
            response = message.choices[0].message.content
            return response

    def run(self):
        """启动命令行交互循环"""
        print(f"当前模型: {self.current_model}，输入 'switch-<模型ID>' 切换模型，Ctrl+C 退出。")
        try:
            while True:
                prompt = input("   our: ")
                if prompt.startswith("switch-"):
                    target = prompt.split("switch-", 1)[1].strip()
                    if target == "RAG":
                        self.RAG["use"]=True
                    elif target == "noRAG":
                        self.RAG["use"]=False
                    elif target == "gradio":
                        self.initUI_gradio()
                    elif target == "html":
                        self.initUI_html()
                    elif self.switch_model(target):
                        print(f"已切换模型为: {self.current_model}")
                    else:
                        print(f"切换失败，当前模型仍为: {self.current_model}")
                    continue

                response = self.chat(prompt)
                if not self.gen_params.get("stream", False):
                    print(f" {self.current_model}:", response)
        except KeyboardInterrupt:
            print("\n已退出。")

    def initUI_gradio(self):# Gradio UI
        def chat_with_model(question, model_id, useRAG):
            self.RAG["use"] = useRAG
            return self.chat(question, model_id)
        import gradio as gr # pip install gradio
        with gr.Blocks() as demo:
            gr.Markdown("## 💬 多模型 LLM 聊天界面")
            choices=list(self.api_models.keys()) + ["localhost"]
            model_selector = gr.Dropdown(choices=choices, value=self.current_model, label="选择模型")
            # 添加一个选择框（复选框），控制某个新参数
            box_useRAG = gr.Checkbox(label="启用RAG模式", value=self.RAG["use"])  # 默认 False
            with gr.Row():
                input_box = gr.Textbox(label="你的问题", placeholder="请输入你想问的问题", lines=1)
                send_button = gr.Button("发送")
            output_box = gr.Textbox(label="模型回答", interactive=False)
            send_button.click(chat_with_model, inputs=[input_box, model_selector, box_useRAG], outputs=output_box)
        demo.launch(debug=True)
    
    def initUI_html(self):
        from flask import Flask, send_from_directory # pip install flask
        from flask_socketio import SocketIO, emit # pip install flask-socketio
        app = Flask(__name__)
        socketio = SocketIO(app)
        @app.route("/")
        def index():
            return send_from_directory(".", "test.html")
        @socketio.on("chatMessage")
        def handle_chat_message(data):
            model_id = data["modelId"]
            question = data["question"]
            useRAG = data["useRAG"]
            self.RAG["use"] = useRAG
            try:
                response = self.chat(question, model_id)
            except Exception as e:
                response = str(e)
            print("response",response)
            emit("chatResponse", {"response": response})
        socketio.run(app, host="0.0.0.0", port=3000, allow_unsafe_werkzeug=True) # socketio.run(app, host="0.0.0.0", port=3000)
    
    def _load_local_embedding_model(self):
        """加载本地（HuggingFace）嵌入模型，例如 sentence-transformers/all-MiniLM-L6-v2"""
        if self.local_embedding_model is None:
            raise RuntimeError("未配置本地嵌入模型名称或路径")
        device = torch.device(self.local_embedding_device if (self.local_embedding_device == 'cpu' or not torch.cuda.is_available()) else self.local_embedding_device)
        tokenizer = AutoTokenizer.from_pretrained(self.local_embedding_model)
        model = AutoModel.from_pretrained(self.local_embedding_model)
        model.to(device)
        model.eval()
        self._local_embedding = (tokenizer, model)

    def embed_chunks(self, chunks: List[str]):
        if self.RAG["current_modelEmb"]=="localhost":
            # 延迟加载本地嵌入模型
            if self._local_embedding is None:
                self._load_local_embedding_model()
            tokenizer, model = self._local_embedding

            # 将输入分批处理以节省内存
            all_embeddings = []
            batch_size = 32
            device = torch.device(self.local_embedding_device if (self.local_embedding_device == 'cpu' or not torch.cuda.is_available()) else self.local_embedding_device)
            for i in range(0, len(chunks), batch_size):
                batch = chunks[i : i + batch_size]
                encoded_input = tokenizer(batch, padding=True, truncation=True, return_tensors="pt")
                encoded_input = {k: v.to(device) for k, v in encoded_input.items()}
                with torch.no_grad():
                    model_output = model(**encoded_input)
                # mean pooling
                token_embeddings = model_output[0]
                attention_mask = encoded_input.get("attention_mask")
                input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
                embeddings = torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)
                embeddings = F.normalize(embeddings, p=2, dim=1)
                all_embeddings.extend([emb.cpu().numpy() for emb in embeddings])
            return all_embeddings
        else:
            #将 chunk 转换成向量
            cfg = self.api_models[self.RAG["current_modelEmb"]]
            client = openai.OpenAI(
                api_key=cfg["api_key"],
                base_url=cfg["base_url"], 
            )
            response = client.embeddings.create(
                model=cfg["model_embedding"],#"deepseek-chat",#"text-embedding-3-small",
                input=chunks
            )
            embeddings = [item.embedding for item in response.data]
        return embeddings # return np.array(embeddings)
    
    def _save2chroma(
        self,
        filePath = "./Dataset/人工智能发展简史.txt",
        chunk_size = 500,
        overlap = 100,
        batch_size = 10 # 根据实际 API 限制设置
    ):
        print("预处理文档内容:",filePath)
        with open(filePath, "r", encoding="utf-8") as f:
            text = f.read()
        chunks = []
        start = 0
        text_len = len(text)
        while start < text_len:
            end = start + chunk_size
            chunk = text[start:end]
            chunks.append(chunk)
            start = end - overlap # 下一个起始位置：当前末尾 - 重叠量
            if start >= text_len: # 若已到末尾则退出
                break
        for i in range(0, len(chunks), batch_size):
            batch = chunks[i:i + batch_size]
            embeddings = self.embed_chunks(batch) #将每个 chunk 转换成向量
            self.collection.add(
                embeddings=embeddings,#[emb.tolist() for emb in embeddings],  # 文本块的特征向量
                documents=batch,#chunks,                                 # 文本快的字符串
                ids=[f"file:{filePath};chunk:{i};" for i in range(len(batch))] # 文本快的ID
            )

    def search_chroma(self, query, top_k=1) -> List[str]:
        """根据用户问题检索相关文本块"""
        query_embedding = np.array(self.embed_chunks([query]))[0]  # 获取问题的向量
        results = self.collection.query(
            query_embeddings=[query_embedding.tolist()],
            n_results=top_k,
            include=["documents", "distances"]
        ) # results["distances"], results["ids"]
        for i in range(len(results["distances"][0])):
            print(f'第{i}个检索到的文本块相关度是: {results["distances"][0][i]} \n编号是: {results["ids"][0][i]} \n内容是:{[results["documents"][0][i]]}')
        return results["documents"][0] # 返回文档列表

# 使用示例
if __name__ == "__main__":
    ChatBot_RAG().run()