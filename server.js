// server.js - OpenAI to NVIDIA NIM API Proxy
const express = require('express');
const cors = require('cors');
const axios = require('axios');

const app = express();
const PORT = process.env.PORT || 3000;

// Middleware
app.use(cors());
app.use(express.json());

// NVIDIA NIM API configuration
const NIM_API_BASE = process.env.NIM_API_BASE || 'https://integrate.api.nvidia.com/v1';
const NIM_API_KEY = process.env.NIM_API_KEY;

// 🔥 REASONING DISPLAY TOGGLE - Shows/hides reasoning in output
const SHOW_REASONING = false; // Set to true to show reasoning with <think> tags

// 🔥 THINKING MODE TOGGLE - Enables thinking for specific models that support it
const ENABLE_THINKING_MODE = false; // Set to true to enable chat_template_kwargs thinking parameter

// Complete NVIDIA NIM Model Mapping - 169 Models
// Replace the MODEL_MAPPING constant in server.js with this

const MODEL_MAPPING = {
// === REASONING & THINKING MODELS ===
'deepseek-ai/deepseek-r1': 'deepseek-ai/deepseek-r1',
'deepseek-ai/deepseek-r1-0528': 'deepseek-ai/deepseek-r1-0528',
'deepseek-ai/deepseek-r1-distill-llama-8b': 'deepseek-ai/deepseek-r1-distill-llama-8b',
'deepseek-ai/deepseek-r1-distill-qwen-7b': 'deepseek-ai/deepseek-r1-distill-qwen-7b',
'deepseek-ai/deepseek-r1-distill-qwen-14b': 'deepseek-ai/deepseek-r1-distill-qwen-14b',
'deepseek-ai/deepseek-r1-distill-qwen-32b': 'deepseek-ai/deepseek-r1-distill-qwen-32b',
'qwen/qwq-32b': 'qwen/qwq-32b',
'qwen/qwen3-next-80b-a3b-thinking': 'qwen/qwen3-next-80b-a3b-thinking',
'moonshotai/kimi-k2-thinking': 'moonshotai/kimi-k2-thinking',
'microsoft/phi-4-mini-flash-reasoning': 'microsoft/phi-4-mini-flash-reasoning',
'nvidia/cosmos-reason2-8b': 'nvidia/cosmos-reason2-8b',

// === PREMIUM LARGE MODELS (100B+) ===
'mistralai/mistral-large-3-675b-instruct-2512': 'mistralai/mistral-large-3-675b-instruct-2512',
'qwen/qwen3-coder-480b-a35b-instruct': 'qwen/qwen3-coder-480b-a35b-instruct',
'meta/llama-3.1-405b-instruct': 'meta/llama-3.1-405b-instruct',
'igenius/colosseum_355b_instruct_16k': 'igenius/colosseum_355b_instruct_16k',
'nvidia/llama-3.1-nemotron-ultra-253b-v1': 'nvidia/llama-3.1-nemotron-ultra-253b-v1',
'nvidia/nemotron-4-340b-instruct': 'nvidia/nemotron-4-340b-instruct',
'qwen/qwen3-235b-a22b': 'qwen/qwen3-235b-a22b',

// === HIGH-PERFORMANCE MODELS (70B-100B) ===
'deepseek-ai/deepseek-v3.2': 'deepseek-ai/deepseek-v3.2',
'deepseek-ai/deepseek-v3.1': 'deepseek-ai/deepseek-v3.1',
'deepseek-ai/deepseek-v3.1-terminus': 'deepseek-ai/deepseek-v3.1-terminus',
'z-ai/glm4.7': 'z-ai/glm4.7',
'meta/llama-3.3-70b-instruct': 'meta/llama-3.3-70b-instruct',
'meta/llama2-70b': 'meta/llama2-70b',
'nvidia/llama-3.1-nemotron-70b-instruct': 'nvidia/llama-3.1-nemotron-70b-instruct',
'moonshotai/kimi-k2-instruct': 'moonshotai/kimi-k2-instruct',
'moonshotai/kimi-k2-instruct-0905': 'moonshotai/kimi-k2-instruct-0905',
'qwen/qwen3-next-80b-a3b-instruct': 'qwen/qwen3-next-80b-a3b-instruct',
'writer/palmyra-creative-122b': 'writer/palmyra-creative-122b',
'openai/gpt-oss-120b': 'openai/gpt-oss-120b',
'writer/palmyra-fin-70b-32k': 'writer/palmyra-fin-70b-32k',
'writer/palmyra-med-70b': 'writer/palmyra-med-70b',
'writer/palmyra-med-70b-32k': 'writer/palmyra-med-70b-32k',
'abacusai/dracarys-llama-3.1-70b-instruct': 'abacusai/dracarys-llama-3.1-70b-instruct',
'institute-of-science-tokyo/llama-3.1-swallow-70b-instruct-v0.1': 'institute-of-science-tokyo/llama-3.1-swallow-70b-instruct-v0.1',
'tokyotech-llm/llama-3-swallow-70b-instruct-v0.1': 'tokyotech-llm/llama-3-swallow-70b-instruct-v0.1',
'yentinglin/llama-3-taiwan-70b-instruct': 'yentinglin/llama-3-taiwan-70b-instruct',
'nvidia/llama3-chatqa-1.5-70b': 'nvidia/llama3-chatqa-1.5-70b',

// === CODING SPECIALISTS ===
'mistralai/devstral-2-123b-instruct-2512': 'mistralai/devstral-2-123b-instruct-2512',
'mistralai/codestral-22b-instruct-v0.1': 'mistralai/codestral-22b-instruct-v0.1',
'qwen/qwen2.5-coder-32b-instruct': 'qwen/qwen2.5-coder-32b-instruct',
'qwen/qwen2.5-coder-7b-instruct': 'qwen/qwen2.5-coder-7b-instruct',
'deepseek-ai/deepseek-coder-6.7b-instruct': 'deepseek-ai/deepseek-coder-6.7b-instruct',
'bigcode/starcoder2-15b': 'bigcode/starcoder2-15b',
'bigcode/starcoder2-7b': 'bigcode/starcoder2-7b',
'meta/codellama-70b': 'meta/codellama-70b',
'google/codegemma-1.1-7b': 'google/codegemma-1.1-7b',
'google/codegemma-7b': 'google/codegemma-7b',
'ibm/granite-34b-code-instruct': 'ibm/granite-34b-code-instruct',
'ibm/granite-8b-code-instruct': 'ibm/granite-8b-code-instruct',
'nvidia/usdcode-llama-3.1-70b-instruct': 'nvidia/usdcode-llama-3.1-70b-instruct',
'nvidia/nv-embedcode-7b-v1': 'nvidia/nv-embedcode-7b-v1',

// === EFFICIENT MODELS (10B-50B) ===
'meta/llama-4-maverick-17b-128e-instruct': 'meta/llama-4-maverick-17b-128e-instruct',
'meta/llama-4-scout-17b-16e-instruct': 'meta/llama-4-scout-17b-16e-instruct',
'mistralai/magistral-small-2506': 'mistralai/magistral-small-2506',
'mistralai/ministral-14b-instruct-2512': 'mistralai/ministral-14b-instruct-2512',
'mistralai/mistral-small-3.1-24b-instruct-2503': 'mistralai/mistral-small-3.1-24b-instruct-2503',
'mistralai/mistral-small-24b-instruct': 'mistralai/mistral-small-24b-instruct',
'mistralai/mistral-nemotron': 'mistralai/mistral-nemotron',
'mistralai/mistral-large': 'mistralai/mistral-large',
'mistralai/mistral-large-2-instruct': 'mistralai/mistral-large-2-instruct',
'mistralai/mistral-medium-3-instruct': 'mistralai/mistral-medium-3-instruct',
'google/gemma-3-27b-it': 'google/gemma-3-27b-it',
'google/gemma-2-27b-it': 'google/gemma-2-27b-it',
'nvidia/llama-3.3-nemotron-super-49b-v1': 'nvidia/llama-3.3-nemotron-super-49b-v1',
'nvidia/llama-3.3-nemotron-super-49b-v1.5': 'nvidia/llama-3.3-nemotron-super-49b-v1.5',
'nvidia/llama-3.1-nemotron-51b-instruct': 'nvidia/llama-3.1-nemotron-51b-instruct',
'nv-mistralai/mistral-nemo-12b-instruct': 'nv-mistralai/mistral-nemo-12b-instruct',
'databricks/dbrx-instruct': 'databricks/dbrx-instruct',
'ai21labs/jamba-1.5-large-instruct': 'ai21labs/jamba-1.5-large-instruct',
'minimaxai/minimax-m2': 'minimaxai/minimax-m2',
'minimaxai/minimax-m2.1': 'minimaxai/minimax-m2.1',
'baichuan-inc/baichuan2-13b-chat': 'baichuan-inc/baichuan2-13b-chat',
'bytedance/seed-oss-36b-instruct': 'bytedance/seed-oss-36b-instruct',
'nvidia/nemotron-3-nano-30b-a3b': 'nvidia/nemotron-3-nano-30b-a3b',
'nvidia/nemotron-nano-3-30b-a3b': 'nvidia/nemotron-nano-3-30b-a3b',
'openai/gpt-oss-20b': 'openai/gpt-oss-20b',
'mistralai/mixtral-8x22b-instruct-v0.1': 'mistralai/mixtral-8x22b-instruct-v0.1',
'mistralai/mixtral-8x22b-v0.1': 'mistralai/mixtral-8x22b-v0.1',
'nvidia/neva-22b': 'nvidia/neva-22b',
'meta/llama-guard-4-12b': 'meta/llama-guard-4-12b',
'speakleash/bielik-11b-v2.3-instruct': 'speakleash/bielik-11b-v2.3-instruct',
'speakleash/bielik-11b-v2.6-instruct': 'speakleash/bielik-11b-v2.6-instruct',
'igenius/italia_10b_instruct_16k': 'igenius/italia_10b_instruct_16k',
'upstage/solar-10.7b-instruct': 'upstage/solar-10.7b-instruct',
'stockmark/stockmark-2-100b-instruct': 'stockmark/stockmark-2-100b-instruct',
'nvidia/nemotron-nano-12b-v2-vl': 'nvidia/nemotron-nano-12b-v2-vl',

// === SMALL & EFFICIENT (1B-9B) ===
'meta/llama-3.2-3b-instruct': 'meta/llama-3.2-3b-instruct',
'meta/llama-3.2-1b-instruct': 'meta/llama-3.2-1b-instruct',
'meta/llama-3.1-8b-instruct': 'meta/llama-3.1-8b-instruct',
'meta/llama3-8b-instruct': 'meta/llama3-8b-instruct',
'nvidia/nemotron-mini-4b-instruct': 'nvidia/nemotron-mini-4b-instruct',
'nvidia/nemotron-4-mini-hindi-4b-instruct': 'nvidia/nemotron-4-mini-hindi-4b-instruct',
'nvidia/llama-3.1-nemotron-nano-4b-v1.1': 'nvidia/llama-3.1-nemotron-nano-4b-v1.1',
'nvidia/llama-3.1-nemotron-nano-8b-v1': 'nvidia/llama-3.1-nemotron-nano-8b-v1',
'nvidia/nvidia-nemotron-nano-9b-v2': 'nvidia/nvidia-nemotron-nano-9b-v2',
'microsoft/phi-4-mini-instruct': 'microsoft/phi-4-mini-instruct',
'microsoft/phi-3.5-mini-instruct': 'microsoft/phi-3.5-mini-instruct',
'microsoft/phi-3.5-moe-instruct': 'microsoft/phi-3.5-moe-instruct',
'microsoft/phi-3-mini-128k-instruct': 'microsoft/phi-3-mini-128k-instruct',
'microsoft/phi-3-mini-4k-instruct': 'microsoft/phi-3-mini-4k-instruct',
'microsoft/phi-3-small-128k-instruct': 'microsoft/phi-3-small-128k-instruct',
'microsoft/phi-3-small-8k-instruct': 'microsoft/phi-3-small-8k-instruct',
'microsoft/phi-3-medium-128k-instruct': 'microsoft/phi-3-medium-128k-instruct',
'microsoft/phi-3-medium-4k-instruct': 'microsoft/phi-3-medium-4k-instruct',
'google/gemma-3-4b-it': 'google/gemma-3-4b-it',
'google/gemma-3-1b-it': 'google/gemma-3-1b-it',
'google/gemma-3-12b-it': 'google/gemma-3-12b-it',
'google/gemma-3n-e2b-it': 'google/gemma-3n-e2b-it',
'google/gemma-3n-e4b-it': 'google/gemma-3n-e4b-it',
'google/gemma-2-9b-it': 'google/gemma-2-9b-it',
'google/gemma-2-2b-it': 'google/gemma-2-2b-it',
'google/gemma-2b': 'google/gemma-2b',
'google/gemma-7b': 'google/gemma-7b',
'google/recurrentgemma-2b': 'google/recurrentgemma-2b',
'google/shieldgemma-9b': 'google/shieldgemma-9b',
'mistralai/mistral-7b-instruct-v0.2': 'mistralai/mistral-7b-instruct-v0.2',
'mistralai/mistral-7b-instruct-v0.3': 'mistralai/mistral-7b-instruct-v0.3',
'mistralai/mixtral-8x7b-instruct-v0.1': 'mistralai/mixtral-8x7b-instruct-v0.1',
'mistralai/mamba-codestral-7b-v0.1': 'mistralai/mamba-codestral-7b-v0.1',
'mistralai/mathstral-7b-v0.1': 'mistralai/mathstral-7b-v0.1',
'qwen/qwen2.5-7b-instruct': 'qwen/qwen2.5-7b-instruct',
'qwen/qwen2-7b-instruct': 'qwen/qwen2-7b-instruct',
'tiiuae/falcon3-7b-instruct': 'tiiuae/falcon3-7b-instruct',
'ibm/granite-3.3-8b-instruct': 'ibm/granite-3.3-8b-instruct',
'ibm/granite-3.0-8b-instruct': 'ibm/granite-3.0-8b-instruct',
'ibm/granite-3.0-3b-a800m-instruct': 'ibm/granite-3.0-3b-a800m-instruct',
'ibm/granite-guardian-3.0-8b': 'ibm/granite-guardian-3.0-8b',
'marin/marin-8b-instruct': 'marin/marin-8b-instruct',
'nvidia/mistral-nemo-minitron-8b-8k-instruct': 'nvidia/mistral-nemo-minitron-8b-8k-instruct',
'nvidia/mistral-nemo-minitron-8b-base': 'nvidia/mistral-nemo-minitron-8b-base',
'nvidia/llama-3.1-nemotron-nano-vl-8b-v1': 'nvidia/llama-3.1-nemotron-nano-vl-8b-v1',
'nvidia/llama-3.1-nemoguard-8b-content-safety': 'nvidia/llama-3.1-nemoguard-8b-content-safety',
'nvidia/llama-3.1-nemoguard-8b-topic-control': 'nvidia/llama-3.1-nemoguard-8b-topic-control',
'nvidia/llama-3.1-nemotron-safety-guard-8b-v3': 'nvidia/llama-3.1-nemotron-safety-guard-8b-v3',
'nvidia/llama3-chatqa-1.5-8b': 'nvidia/llama3-chatqa-1.5-8b',
'adept/fuyu-8b': 'adept/fuyu-8b',
'aisingapore/sea-lion-7b-instruct': 'aisingapore/sea-lion-7b-instruct',
'mediatek/breeze-7b-instruct': 'mediatek/breeze-7b-instruct',
'rakuten/rakutenai-7b-chat': 'rakuten/rakutenai-7b-chat',
'rakuten/rakutenai-7b-instruct': 'rakuten/rakutenai-7b-instruct',
'opengpt-x/teuken-7b-instruct-commercial-v0.4': 'opengpt-x/teuken-7b-instruct-commercial-v0.4',
'utter-project/eurollm-9b-instruct': 'utter-project/eurollm-9b-instruct',
'gotocompany/gemma-2-9b-cpt-sahabatai-instruct': 'gotocompany/gemma-2-9b-cpt-sahabatai-instruct',
'institute-of-science-tokyo/llama-3.1-swallow-8b-instruct-v0.1': 'institute-of-science-tokyo/llama-3.1-swallow-8b-instruct-v0.1',
'zyphra/zamba2-7b-instruct': 'zyphra/zamba2-7b-instruct',
'thudm/chatglm3-6b': 'thudm/chatglm3-6b',
'sarvamai/sarvam-m': 'sarvamai/sarvam-m',
'ai21labs/jamba-1.5-mini-instruct': 'ai21labs/jamba-1.5-mini-instruct',
'01-ai/yi-large': '01-ai/yi-large',
'nvidia/riva-translate-4b-instruct': 'nvidia/riva-translate-4b-instruct',
'nvidia/riva-translate-4b-instruct-v1.1': 'nvidia/riva-translate-4b-instruct-v1.1',

// === VISION & MULTIMODAL MODELS ===
'microsoft/phi-4-multimodal-instruct': 'microsoft/phi-4-multimodal-instruct',
'microsoft/phi-3.5-vision-instruct': 'microsoft/phi-3.5-vision-instruct',
'microsoft/phi-3-vision-128k-instruct': 'microsoft/phi-3-vision-128k-instruct',
'meta/llama-3.2-90b-vision-instruct': 'meta/llama-3.2-90b-vision-instruct',
'meta/llama-3.2-11b-vision-instruct': 'meta/llama-3.2-11b-vision-instruct',
'nvidia/vila': 'nvidia/vila',
'google/paligemma': 'google/paligemma',
'microsoft/kosmos-2': 'microsoft/kosmos-2',
'google/deplot': 'google/deplot',

// === EMBEDDING MODELS ===
'nvidia/nv-embed-v1': 'nvidia/nv-embed-v1',
'nvidia/embed-qa-4': 'nvidia/embed-qa-4',
'nvidia/nv-embedqa-mistral-7b-v2': 'nvidia/nv-embedqa-mistral-7b-v2',
'nvidia/nv-embedqa-e5-v5': 'nvidia/nv-embedqa-e5-v5',
'nvidia/llama-3.2-nv-embedqa-1b-v1': 'nvidia/llama-3.2-nv-embedqa-1b-v1',
'nvidia/llama-3.2-nv-embedqa-1b-v2': 'nvidia/llama-3.2-nv-embedqa-1b-v2',
'nvidia/llama-3.2-nemoretriever-300m-embed-v1': 'nvidia/llama-3.2-nemoretriever-300m-embed-v1',
'nvidia/llama-3.2-nemoretriever-300m-embed-v2': 'nvidia/llama-3.2-nemoretriever-300m-embed-v2',
'nvidia/llama-3.2-nemoretriever-1b-vlm-embed-v1': 'nvidia/llama-3.2-nemoretriever-1b-vlm-embed-v1',
'baai/bge-m3': 'baai/bge-m3',
'snowflake/arctic-embed-l': 'snowflake/arctic-embed-l',
'nvidia/nvclip': 'nvidia/nvclip',

// === SPECIALIZED MODELS ===
'nvidia/nemotron-4-340b-reward': 'nvidia/nemotron-4-340b-reward',
'nvidia/llama-3.1-nemotron-70b-reward': 'nvidia/llama-3.1-nemotron-70b-reward',
'nvidia/nemoretriever-parse': 'nvidia/nemoretriever-parse',
'nvidia/nemotron-parse': 'nvidia/nemotron-parse',
'nvidia/streampetr': 'nvidia/streampetr'
};
// Health check endpoint
app.get('/health', (req, res) => {
  res.json({ 
    status: 'ok', 
    service: 'OpenAI to NVIDIA NIM Proxy', 
    reasoning_display: SHOW_REASONING,
    thinking_mode: ENABLE_THINKING_MODE
  });
});

// List models endpoint (OpenAI compatible)
app.get('/v1/models', (req, res) => {
  const models = Object.keys(MODEL_MAPPING).map(model => ({
    id: model,
    object: 'model',
    created: Date.now(),
    owned_by: 'nvidia-nim-proxy'
  }));
  
  res.json({
    object: 'list',
    data: models
  });
});

// Chat completions endpoint (main proxy)
app.post('/v1/chat/completions', async (req, res) => {
  try {
    const { model, messages, temperature, max_tokens, stream } = req.body;
    
    // Smart model selection with fallback
    let nimModel = MODEL_MAPPING[model];
    if (!nimModel) {
      try {
        await axios.post(`${NIM_API_BASE}/chat/completions`, {
          model: model,
          messages: [{ role: 'user', content: 'test' }],
          max_tokens: 1
        }, {
          headers: { 'Authorization': `Bearer ${NIM_API_KEY}`, 'Content-Type': 'application/json' },
          validateStatus: (status) => status < 500
        }).then(res => {
          if (res.status >= 200 && res.status < 300) {
            nimModel = model;
          }
        });
      } catch (e) {}
      
      if (!nimModel) {
        const modelLower = model.toLowerCase();
        if (modelLower.includes('gpt-4') || modelLower.includes('claude-opus') || modelLower.includes('405b')) {
          nimModel = 'meta/llama-3.1-405b-instruct';
        } else if (modelLower.includes('claude') || modelLower.includes('gemini') || modelLower.includes('70b')) {
          nimModel = 'meta/llama-3.1-70b-instruct';
        } else {
          nimModel = 'meta/llama-3.1-8b-instruct';
        }
      }
    }
    
    // Transform OpenAI request to NIM format
    const nimRequest = {
      model: nimModel,
      messages: messages,
      temperature: temperature || 0.6,
      max_tokens: max_tokens || 9024,
      extra_body: ENABLE_THINKING_MODE ? { chat_template_kwargs: { thinking: true } } : undefined,
      stream: stream || false
    };
    
    // Make request to NVIDIA NIM API
    const response = await axios.post(`${NIM_API_BASE}/chat/completions`, nimRequest, {
      headers: {
        'Authorization': `Bearer ${NIM_API_KEY}`,
        'Content-Type': 'application/json'
      },
      responseType: stream ? 'stream' : 'json'
    });
    
    if (stream) {
      // Handle streaming response with reasoning
      res.setHeader('Content-Type', 'text/event-stream');
      res.setHeader('Cache-Control', 'no-cache');
      res.setHeader('Connection', 'keep-alive');
      
      let buffer = '';
      let reasoningStarted = false;
      
      response.data.on('data', (chunk) => {
        buffer += chunk.toString();
        const lines = buffer.split('\\n');
        buffer = lines.pop() || '';
        
        lines.forEach(line => {
          if (line.startsWith('data: ')) {
            if (line.includes('[DONE]')) {
              res.write(line + '\\n');
              return;
            }
            
            try {
              const data = JSON.parse(line.slice(6));
              if (data.choices?.[0]?.delta) {
                const reasoning = data.choices[0].delta.reasoning_content;
                const content = data.choices[0].delta.content;
                
                if (SHOW_REASONING) {
                  let combinedContent = '';
                  
                  if (reasoning && !reasoningStarted) {
                    combinedContent = '<think>\\n' + reasoning;
                    reasoningStarted = true;
                  } else if (reasoning) {
                    combinedContent = reasoning;
                  }
                  
                  if (content && reasoningStarted) {
                    combinedContent += '</think>\\n\\n' + content;
                    reasoningStarted = false;
                  } else if (content) {
                    combinedContent += content;
                  }
                  
                  if (combinedContent) {
                    data.choices[0].delta.content = combinedContent;
                    delete data.choices[0].delta.reasoning_content;
                  }
                } else {
                  if (content) {
                    data.choices[0].delta.content = content;
                  } else {
                    data.choices[0].delta.content = '';
                  }
                  delete data.choices[0].delta.reasoning_content;
                }
              }
              res.write(`data: ${JSON.stringify(data)}\\n\\n`);
            } catch (e) {
              res.write(line + '\\n');
            }
          }
        });
      });
      
      response.data.on('end', () => res.end());
      response.data.on('error', (err) => {
        console.error('Stream error:', err);
        res.end();
      });
    } else {
      // Transform NIM response to OpenAI format with reasoning
      const openaiResponse = {
        id: `chatcmpl-${Date.now()}`,
        object: 'chat.completion',
        created: Math.floor(Date.now() / 1000),
        model: model,
        choices: response.data.choices.map(choice => {
          let fullContent = choice.message?.content || '';
          
          if (SHOW_REASONING && choice.message?.reasoning_content) {
            fullContent = '<think>\\n' + choice.message.reasoning_content + '\\n</think>\\n\\n' + fullContent;
          }
          
          return {
            index: choice.index,
            message: {
              role: choice.message.role,
              content: fullContent
            },
            finish_reason: choice.finish_reason
          };
        }),
        usage: response.data.usage || {
          prompt_tokens: 0,
          completion_tokens: 0,
          total_tokens: 0
        }
      };
      
      res.json(openaiResponse);
    }
    
  } catch (error) {
    console.error('Proxy error:', error.message);
    
    res.status(error.response?.status || 500).json({
      error: {
        message: error.message || 'Internal server error',
        type: 'invalid_request_error',
        code: error.response?.status || 500
      }
    });
  }
});

// Catch-all for unsupported endpoints
app.all('*', (req, res) => {
  res.status(404).json({
    error: {
      message: `Endpoint ${req.path} not found`,
      type: 'invalid_request_error',
      code: 404
    }
  });
});

app.listen(PORT, () => {
  console.log(`OpenAI to NVIDIA NIM Proxy running on port ${PORT}`);
  console.log(`Health check: http://localhost:${PORT}/health`);
  console.log(`Reasoning display: ${SHOW_REASONING ? 'ENABLED' : 'DISABLED'}`);
  console.log(`Thinking mode: ${ENABLE_THINKING_MODE ? 'ENABLED' : 'DISABLED'}`);
});
