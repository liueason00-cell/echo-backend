// ============================================================================
// 🔥🔥🔥 Project Zhenwo Backend V12: 修复版 (解决串台 + 视觉唤醒) 🔥🔥🔥
// ============================================================================

const express = require('express');
const cors = require('cors');
const nodeFetch = require('node-fetch'); 
const { HttpsProxyAgent } = require('https-proxy-agent');
const { Pinecone } = require('@pinecone-database/pinecone');
const { initializeApp, cert } = require('firebase-admin/app');
const { getFirestore } = require('firebase-admin/firestore');
const admin = require('firebase-admin');
require('dotenv').config();

// ============================================================================
// 1. 🌐 网络层 (保持 V6.2 穿透版)
// ============================================================================
const PROXY_URL = process.env.PROXY_URL || 'http://127.0.0.1:33210'; 
const USE_PROXY = process.env.FORCE_PROXY === 'true' || process.env.NODE_ENV !== 'production';

const proxyAgent = new HttpsProxyAgent(PROXY_URL, {
  keepAlive: true,           
  rejectUnauthorized: false, 
  scheduling: 'lifo',
  timeout: 60000             
});

global.fetch = (url, init) => {
  if (USE_PROXY && !url.includes('localhost') && !url.includes('127.0.0.1')) {
    return nodeFetch(url, { ...init, agent: proxyAgent, timeout: 60000 });
  }
  return nodeFetch(url, init);
};

if (USE_PROXY) {
  process.env.NODE_TLS_REJECT_UNAUTHORIZED = '0'; 
  console.log(`🛡️ [System] 代理坦克已启动 (SSL忽略模式): ${PROXY_URL}`);
}

// ============================================================================
// 2. 🌲 Pinecone 初始化
// ============================================================================
const customPineconeFetch = (url, init) => {
  return nodeFetch(url, {
    ...init,
    agent: USE_PROXY ? proxyAgent : undefined,
    keepalive: true,
    timeout: 60000, 
  }).catch(err => {
    console.error(`❌ [Pinecone Fetch Error] ${url} - ${err.message}`);
    throw err;
  });
};

const pc = new Pinecone({ 
  apiKey: process.env.PINECONE_API_KEY,
  fetchApi: customPineconeFetch 
});
const pineconeIndex = pc.index('zhenwo-knowledge'); // 确保这里的 Index 名字正确

// ============================================================================
// 3. 🔥 Firebase 初始化
// ============================================================================
// 注意：确保 serviceAccountKey.json 在同级目录下
const serviceAccount = require('./serviceAccountKey.json');
let firebaseApp;
try { 
  firebaseApp = initializeApp({ credential: cert(serviceAccount) }); 
} catch (e) { 
  firebaseApp = require('firebase-admin').app(); 
}
const firestore = getFirestore(firebaseApp);

// ============================================================================
// 4. 🔑 API Key 管理
// ============================================================================
const rawKeys = process.env.SILICONFLOW_API_KEYS || process.env.SILICONFLOW_API_KEY || "";
const apiKeys = rawKeys.split(/,|\n/).map(k => k.trim()).filter(k => k && k.startsWith('sk-'));

let currentKeyIndex = 0;
function getCurrentKey() { return apiKeys[currentKeyIndex]; }
function rotateKey() { 
  currentKeyIndex = (currentKeyIndex + 1) % apiKeys.length; 
  console.log(`🔄 [Key] 切换 API Key 到索引: ${currentKeyIndex}`);
}

const FAST_BRAIN = "deepseek-ai/DeepSeek-V3"; 
const DEEP_BRAIN = "deepseek-ai/DeepSeek-R1";      

// ============================================================================
// 5. 🧠 Embedding 工具类
// ============================================================================
class SiliconflowEmbeddings {
  constructor() {
    this.modelName = "netease-youdao/bce-embedding-base_v1"; 
    this.baseURL = "https://api.siliconflow.cn/v1/embeddings";
  }

  async embedQuery(text) {
    const MAX_RETRIES = 3; 
    for (let attempt = 1; attempt <= MAX_RETRIES; attempt++) {
      try {
        const apiKey = getCurrentKey();
        const response = await fetch(this.baseURL, { 
          method: 'POST',
          headers: { 'Authorization': `Bearer ${apiKey}`, 'Content-Type': 'application/json' },
          body: JSON.stringify({ model: this.modelName, input: [text.replace(/\n/g, " ")] })
        });

        if (!response.ok) {
          if (response.status === 429 || response.status === 401) rotateKey();
          throw new Error(`Embedding API Error ${response.status}`);
        }
        const data = await response.json();
        return data.data[0].embedding;
      } catch (error) {
        if (attempt === MAX_RETRIES) return null;
        await new Promise(r => setTimeout(r, 1000 * attempt));
      }
    }
    return null; 
  }
}
const embeddings = new SiliconflowEmbeddings();

// ============================================================================
// 6. 👁️ 视觉分析 (修复版：增强错误处理)
// ============================================================================
async function analyzeImageWithVisionModel(images) {
  if (!images || images.length === 0) return "";
  
  // 推荐使用 Qwen-VL 或者 DeepSeek-VL，Qwen 在 SiliconFlow 上表现较稳
  const VISION_MODEL = "deepseek-ai/deepseek-vl2"; 
  console.log(`👁️ [Vision] 正在调用视觉模型 (${images.length} 张图片)...`);
  const apiKey = getCurrentKey(); 

  const contentPayload = [
    { 
      type: "text", 
      text: `你是一个恋爱军师。请详细分析这张聊天截图或照片。
      如果是聊天记录：
      1. 提取对方(左侧)和用户(右侧)的核心对话内容。
      2. 分析对方的语气（冷淡/热情/敷衍）。
      
      如果是生活照/人物照：
      1. 描述图片中的场景、氛围、人物状态。
      
      请直接输出分析结果，不要啰嗦。` 
    }
  ];

  images.forEach(img => {
    // 确保格式正确，防止 base64 前缀重复
    const base64Str = img.base64.includes('base64') ? img.base64 : `data:${img.mime};base64,${img.base64}`;
    contentPayload.push({
      type: "image_url",
      image_url: { url: base64Str }
    });
  });

  try {
    const response = await fetch('https://api.siliconflow.cn/v1/chat/completions', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json', 'Authorization': `Bearer ${apiKey}` },
      body: JSON.stringify({
        model: VISION_MODEL,
        messages: [{ role: "user", content: contentPayload }],
        max_tokens: 1024, // 增加 Token 确保描述完整
        temperature: 0.1
      })
    });
    
    if (!response.ok) {
        const errText = await response.text();
        throw new Error(`Vision API Status: ${response.status} - ${errText}`);
    }
    const data = await response.json();
    const result = data.choices[0].message.content;
    console.log("✅ [Vision] 识别成功:", result.substring(0, 50) + "...");
    return `\n=== 📸 图片内容分析 ===\n${result}\n=====================\n`;
  } catch (e) { 
    console.error("❌ [Vision Error]", e.message);
    return "[⚠️ 图片识别失败，请依据文字内容回答]";
  }
}

// ============================================================================
// 7. 🗄️ 数据库适配层
// ============================================================================
const DB_ADAPTER = {
  async getUser(userId) {
    if (!userId) return null;
    try {
      const doc = await firestore.collection('users').doc(userId).get();
      return doc.exists ? doc.data() : { power_level: 'Low', financial_status: 'C', initiation_score: 'C' };
    } catch (e) { return {}; }
  },
  
  async getRecentHistory(userId, limit = 6) {
    if (!userId) return [];
    try {
      const snapshot = await firestore.collection('users').doc(userId).collection('logs')
        .orderBy('timestamp', 'desc')
        .limit(limit)
        .get();
      return snapshot.docs.map(doc => doc.data()).reverse();
    } catch (e) {
      return [];
    }
  },

  async saveLog(userId, logData) {
    if (!userId) return;
    try {
      // 异步保存，不阻塞主线程
      firestore.collection('users').doc(userId).collection('logs').add(logData);
    } catch (e) { console.error("Log save failed", e); }
  }
};

// ============================================================================
// 8. 🧠 RAG 核心逻辑 (🔥 修复版：带详细日志)
// ============================================================================
async function dualTrackRetrieval(queryText, mode, searchConfig) {
  const finalQuery = searchConfig?.rewrite_query || queryText;
  
  // 🔍 [日志 1] 打印开始信号
  console.log(`\n🕵️ [RAG Start] 正在检索... Query: "${finalQuery.substring(0, 30)}..."`);

  let qVec = null;
  try { 
    qVec = await embeddings.embedQuery(finalQuery); 
  } catch (e) {
    // 🔍 [日志 2] Embedding 报错必须打印出来，不然不知道 Key 挂了
    console.error("❌ [RAG Error] Embedding 失败:", e.message);
  }

  if (!qVec) {
    console.warn("⚠️ [RAG Warning] 无法生成向量，跳过检索 (请检查 SiliconFlow Key 或网络)");
    return { strategies: [], styleCandidates: [] }; 
  }

  // 🔍 [日志 3] 成功生成向量
  console.log(`✅ [RAG Step] Embedding 成功 (维度: ${qVec.length})，正在连接 Pinecone...`);

  try {
      // 并行查询策略库和语料库
      const [strategyResponse, styleResponse] = await Promise.all([
          pineconeIndex.namespace('strategies').query({ vector: qVec, topK: 4, includeMetadata: true }),
          pineconeIndex.namespace('styles').query({ vector: qVec, topK: 3, includeMetadata: true })
      ]);
      
      // 🔍 [日志 4] 打印检索结果数量
      console.log(`✅ [RAG Success] 命中策略: ${strategyResponse.matches.length} 条, 语料: ${styleResponse.matches.length} 条`);
      
      // 🔍 [日志 5] 打印具体命中了什么策略（方便你看 AI 有没有乱引用）
      if (strategyResponse.matches.length > 0) {
          strategyResponse.matches.forEach(m => console.log(`   - 🎯 策略: [${m.score.toFixed(2)}] ${m.metadata.title}`));
      }

      const strategies = strategyResponse.matches.map(m => ({
          title: m.metadata.title || 'Unknown',
          content_markdown: m.metadata.content_markdown || m.metadata.content || '',
          next_moves: m.metadata.next_moves || [] // 防止 undefined
      }));
      
      const styleCandidates = styleResponse.matches.map(m => ({
          text: m.metadata.text || m.metadata.content || '',
      }));

      return { strategies, styleCandidates };

  } catch (e) { 
      // 🔍 [日志 6] Pinecone 连接失败报错
      console.error("❌ [RAG Critical Error] Pinecone 连接失败:", e.message);
      if (e.cause) console.error("   Caused by:", e.cause);
      return { strategies: [], styleCandidates: [] }; 
  }
}
// ============================================================================
// 9. 📝 Prompt 构建 (V43.0: 温柔浪人 + 语料库灵魂附体 + XML UI适配)
// ============================================================================
function buildPrompt(mode, userQuery, strategies, finalStyles, imageAnalysis, history = [], profile = {}) {

  // 1️⃣ --- 历史防火墙 ---
  let safeHistory = [];
  if (Array.isArray(history)) {
    safeHistory = history.filter(item => {
      const content = item.content || "";
      return !content.includes("Role:") && !content.includes("System") && !content.includes(":::ANALYSIS");
    });
  }
  const historyContext = safeHistory.length > 0
    ? `=== 📜 历史对话 ===\n${safeHistory.map(h => `${h.role === 'user' ? 'Me' : 'Coach'}: ${h.content}`).join('\n')}\n=== 📜 结束 ===`
    : "(暂无历史)";

  // 2️⃣ --- 核心资产注入 (V30 逻辑复活) ---
  // 🧠 逻辑库：负责“脑子”
  const strategyContext = strategies.map((s, i) => `
[Strategy-${i+1}] (Internal Logic)
- Core: ${s.title}
- Essence: ${s.content_markdown ? s.content_markdown.substring(0, 300).replace(/\n/g, " ") : '...'}
`).join('\n');

  // 👄 语料库：负责“嘴巴” (关键：让 AI 知道这些是它的'克隆源')
  const styleContext = finalStyles && finalStyles.length > 0 
    ? finalStyles.map(s => `> 模仿样本: "${s.text || s.content}"`).join('\n')
    : "> 基础设定: 说话不用太长，通透，带着三分痞气七分真诚。";

  // 3️⃣ --- 绝对宪法 (抽象化风控) ---
  const CORE_CONSTITUTION = `
【🚫 底层原则】
1. **去黑话**：别整那些“PUA”、“打压”、“陷阱”之类的词。我们是**高价值男性**，不是诈骗犯。把道理揉碎了说人话。
2. **去说教**：不要高高在上地教育用户。要像个**老友**一样，先理解他的难处，再给建议。
3. **正向引导**：如果用户想走邪路（如摧毁对方自信），你要温柔地把他拉回来，告诉他“真正的强大是吸引，不是控制”。
  `;

  // 4️⃣ --- 🧬 风格校验 (Humanizer V2 - 笼统但精准) ---
  const STYLE_INSTRUCTION = `
【🎭 灵魂模仿协议】
请严格模仿 [Style Corpus] 中的说话方式和长短节奏：
- **拒绝AI味**：严禁出现“综上所述”、“首先其次”、“建议如下”。这种词一出，直接重写。
- **温柔的浪人**：你的基调是**善解人意**但**内核强大**。
  - 用户焦虑时，先安抚：“哥们，别慌，这局能解。”
  - 用户犯错时，先包容：“正常，是个人都会心软，但接下来咱们得硬一点。”
- **动态长度**：
  - 闲聊时，像微信聊天一样短。
  - 分析时，可以说得透彻一点，但别写论文。
  `;

  // 5️⃣ --- 🚦 语境切换 ---
  const CONTEXT_SWITCH = `
【🚦 意图识别】
🎯 **Type A (代回消息)** -> 用户发了截图或对方的话，问怎么回。
   -> 输出：3个神回复（结合语料库风格）。
🧩 **Type B (军师咨询)** -> 用户问现状、问策略、倾诉情绪。
   -> 输出：局势诊断 + 情绪价值 + 实操建议。
`;

  // ========================================================================
  // ⚡ Mode 1: Quick (保持 V30 的极简)
  // ========================================================================
  if (mode === 'quick') {
    return `
Role: 你的嘴替兄弟
Target: 针对这句话，给我 3 个瞬间破冰或回怼的短句。

${CORE_CONSTITUTION}
${STYLE_INSTRUCTION}

[Style Corpus (你的嘴)]
${styleContext}

Input: "${userQuery}"
Task: 仅输出 JSON 格式。包含 3 个对象。
Example: { "replies": [{ "type": "风格1", "content": "..." }] }
`;
  }

  // ========================================================================
  // 🟣 Mode 3: Master (V43 终极形态)
  // ========================================================================
  else {
    return `
[System Role]
你是一个**深谙人性、温柔但强大的情感操盘手**。
你不是冷冰冰的机器，你是用户最信任的**兄弟/军师**。
你见惯了红尘套路，所以更懂得**真诚**的可贵，但你的真诚是带刺的，没人能欺负你和你的兄弟。

${CORE_CONSTITUTION}
${STYLE_INSTRUCTION}
${CONTEXT_SWITCH}

[Inner Wisdom (你的脑子)]
${strategyContext}

[Style Corpus (你的语气)]
${styleContext}

[Visual Evidence]
${imageAnalysis || "N/A"}

${historyContext}

// ============================================================================
// ⚠️ CURRENT MISSION
// ============================================================================
User Query: "${userQuery}"

[[ 🧠 思考逻辑 (Hidden) ]]
在 <think> 标签内：
1. **共情**：用户现在是什么心情？焦虑？愤怒？先在心里接纳他的情绪。
2. **定性**：这是 Type A (回消息) 还是 Type B (问策略)？
3. **调取语料**：看一眼 [Style Corpus]，找找那种“看似漫不经心实则拿捏”的感觉。
4. **去AI化**：把生成的草稿读一遍，如果像客服或教科书，就扇自己一巴掌，重写成**人话**。
</think>

[[ 📝 强制输出规范 (XML For UI) ]]

🛑 **如果是 Type B (闲聊/非咨询)**：
不要用标签，直接像朋友一样聊天。

✅ **如果是 Type A (需要策略/回消息)**：
请严格按照以下 XML 格式输出：

:::ANALYSIS:::
(这里写【局势诊断】。**语气要求**：先肯定用户，比如“这不怪你...”，然后一针见血指出对方的心理。)
:::END_ANALYSIS:::

:::ACTION:::
(这里写【战术建议】。)

👉 **选项1 (稳重/深情)**：
"..."
*(点评：...)*

👉 **选项2 (幽默/推拉)**：
"..."
*(点评：...)*

👉 **选项3 (高冷/后撤)**：
"..."
*(点评：...)*
:::END_ACTION:::

:::NEXT:::
(这里写【连招预告】和【关键风控】)
**🔮 下一步**：
1. ...
2. ...

**🛑 别踩雷**: 
(用关心的口吻提醒他别犯傻)
:::END_NEXT:::
`;
  }
}

// ========================================================================
// 10. 🌊 DeepSeek 流式调用
// ============================================================================
async function callDeepSeekBrain(prompt, res, targetModel) {
  let fullReply = ""; 
  let retries = 0;
  
  while (retries < 2) {
    try {
      const response = await fetch('https://api.siliconflow.cn/v1/chat/completions', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json', 'Authorization': `Bearer ${getCurrentKey()}` },
        body: JSON.stringify({ 
            model: targetModel, messages: [{ role: "user", content: prompt }], 
            stream: true, max_tokens: 4096, temperature: 0.6
        })
      });

      if (!response.ok) { if(response.status === 401) rotateKey(); throw new Error(`API Error: ${response.status}`); }

      const decoder = new TextDecoder('utf-8');
      let buffer = ''; 

      for await (const chunk of response.body) {
          const decodedChunk = decoder.decode(chunk, { stream: true });
          buffer += decodedChunk;
          const lines = buffer.split('\n');
          buffer = lines.pop(); 

          for (const line of lines) {
            if (line.trim().startsWith('data: ')) {
              const jsonStr = line.replace('data: ', '').trim();
              if (jsonStr === '[DONE]') continue;
              try {
                const data = JSON.parse(jsonStr);
                const txt = data.choices[0].delta.content || "";
                if (txt) {
                  res.write(`data: ${JSON.stringify({ type: 'analysis', content: txt })}\n\n`);
                  fullReply += txt; 
                }
              } catch (e) {}
            }
          }
      }
      return fullReply; 
    } catch (e) {
      retries++;
      console.error(`⚠️ [DeepSeek] Failed:`, e.message);
    }
  }
  return fullReply;
}

// ============================================================================
// 11. 🛣️ 路由层 (修复版：优先前端历史)
// ============================================================================
const app = express();
app.use(cors({ origin: true }));
app.use(express.json({ limit: '50mb' })); // 确保支持大图片

app.post('/api/ask', async (req, res) => {
  try {
    // 🔥 获取 history 参数
    const { question, images, mode = 'master', profile = {}, userId, history } = req.body;
    
    console.log(`\n💬 [Req] User: ${userId} | Q: ${question?.substring(0, 15)}... | Imgs: ${images?.length || 0}`);
    if (!userId) return res.status(400).json({ error: "Missing userId" });

    // 1. 获取用户画像
    let userContext = {};
    try {
      userContext = await DB_ADAPTER.getUser(userId) || {}; 
      if (profile) userContext = { ...userContext, ...profile };
    } catch (err) {}

    // 2. 🔥 决定使用哪份历史记录 (修复串台的关键)
    // 如果前端传了 history (代表当前对话框的记忆)，就用前端的。
    // 如果没传，才去数据库捞。
    let chatContext = [];
    if (history && Array.isArray(history) && history.length > 0) {
        console.log(`   🧠 Using Frontend Session History (${history.length} msgs)`);
        chatContext = history;
    } else {
        console.log(`   💾 Using Database History (Fallback)`);
        chatContext = await DB_ADAPTER.getRecentHistory(userId, 6);
    }

    // 3. 视觉分析
    let imageAnalysis = "";
    if (images && images.length > 0) {
      try { 
          imageAnalysis = await analyzeImageWithVisionModel(images); 
      } catch (e) {
          console.error("Vision failed inside route:", e);
      }
    }

    // 4. RAG 检索
    const searchConfig = { rewrite_query: question, risk_bias: (userContext.power_level === 'Low') ? 'Low' : 'Medium' };
    const { strategies, styleCandidates } = await dualTrackRetrieval(question, mode, searchConfig);

    // 5. 构建 Prompt
    const finalPrompt = buildPrompt(mode, question, strategies, styleCandidates, imageAnalysis, chatContext, userContext);

    // 6. 调用 AI 并流式返回
    res.setHeader('Content-Type', 'text/event-stream; charset=utf-8');
    const aiReply = await callDeepSeekBrain(finalPrompt, res, mode === 'quick' ? FAST_BRAIN : DEEP_BRAIN);

    // 7. 异步存入数据库 (仅作留档，不影响当前会话)
    if (aiReply) {
       DB_ADAPTER.saveLog(userId, { question, reply: aiReply, mode, timestamp: new Date() });
    }
    
    res.write(`data: ${JSON.stringify({ type: 'done' })}\n\n`);
    res.end();

  } catch (error) {
    console.error("❌ [Route Crash]", error);
    if (!res.headersSent) res.status(500).json({ error: "Server Internal Error" });
  }
});

const PORT = process.env.PORT || 3000;
app.listen(PORT, () => console.log(`🚀 Zhenwo Backend V12 (Vision+Memory Fix) Running on Port: ${PORT}`));