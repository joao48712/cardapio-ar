# 📖 IA Paginada

App desktop (Electron + Node.js + Python/PyTorch) que roda modelos de IA **muito maiores que a
memória da sua GPU** — texto, imagem e vídeo — lendo os pesos **uma página por vez**, como uma
impressora que imprime um livro inteiro página por página: pega uma, imprime, joga fora, pega a
próxima.

> Projeto independente dentro deste repositório. Não altera nada do cardápio (`src/`, `public/`…).

## Como funciona

1. O modelo é montado **vazio** (dispositivo `meta` do PyTorch): a arquitetura existe, mas não
   ocupa nem 1 byte de RAM/VRAM.
2. Cada bloco grande (camada de transformer, bloco da UNet, bloco do VAE…) vira uma **página**.
3. Um *hook* do PyTorch roda logo antes de cada página: lê só os pesos dela do `.safetensors` e
   coloca na GPU/CPU. Ao terminar o forward, os pesos são **descartados** e a próxima página entra.
4. Enquanto uma página “imprime”, a próxima já está sendo lida do disco em segundo plano
   (*prefetch*). Opcionalmente, páginas lidas ficam num **cache em RAM** (útil quando o modelo
   cabe na RAM mas não na VRAM).
5. Peças pequenas (embeddings, normas, cabeça de saída) ficam residentes o tempo todo.

Pico de memória ≈ peças residentes + 1–2 páginas + ativações, em vez do modelo inteiro.
Funciona com qualquer arquitetura que o `transformers`/`diffusers` saibam montar, inclusive com
KV-cache na geração de texto. O código do núcleo está em
[`backend/paged/pager.py`](backend/paged/pager.py).

**Troca:** fica bem mais lento (cada token de texto ou passo de difusão relê o modelo inteiro do
disco). Um SSD NVMe ajuda muito; o cache em RAM ajuda mais ainda.

## Recursos

- ⬇️ **Baixar do Hugging Face** dentro do app: busca por tipo (texto, texto→imagem,
  imagem→imagem, texto→vídeo, imagem→vídeo), ID direto, ou **a partir de um Space** (o app descobre
  quais modelos o Space usa). Mostra os arquivos e o tamanho antes de baixar, baixa só
  `.safetensors` + configs, pode preferir a variante fp16, aceita token para modelos restritos.
- 📚 **Meus modelos**: carregar / descarregar / apagar; escolher GPU (CUDA), Apple (MPS) ou CPU,
  precisão, cache em RAM, prefetch e a partir de que tamanho paginar.
- ✨ **Gerar**: texto com streaming (respeita o chat template), imagens e vídeos (MP4) via
  pipelines do `diffusers`, com imagem de entrada opcional (img2img / imagem→vídeo), semente,
  passos, guidance, tamanho, quadros, FPS e cancelamento.
- 🖨️ **Impressora de páginas**: visualização ao vivo de qual página de cada componente está
  carregada, quantas leituras e quantos MB já passaram.

## Requisitos

- Node.js 20+ e Python 3.10+
- Opcional: GPU NVIDIA com CUDA ou Mac com Apple Silicon

## Instalação

```bash
cd ia-paginada
npm install                    # instala o Electron
npm run setup                  # cria backend/.venv e instala torch, transformers, diffusers…
# variações:
npm run setup -- --cpu         # torch só CPU (download menor)
npm run setup -- --cuda cu128  # torch para CUDA 12.8
npm start
```

Variáveis opcionais:

| Variável       | Para quê                                                 |
|----------------|----------------------------------------------------------|
| `IAPAG_HOME`   | Pasta dos modelos e saídas (padrão `~/IA-Paginada`)      |
| `IAPAG_PYTHON` | Python a usar (padrão: `backend/.venv` ou `python3`)      |
| `HF_TOKEN`     | Token do Hugging Face (também pode ser digitado no app)  |

## Formatos suportados

- **Texto:** repositórios Transformers com `.safetensors` (Llama, Qwen, Mistral, Gemma, Phi…,
  qualquer `AutoModelForCausalLM`).
- **Imagem e vídeo:** repositórios no formato **Diffusers** (`model_index.json`), ex.: FLUX,
  Stable Diffusion/SDXL/SD3, Wan (`*-Diffusers`), LTX-Video, CogVideoX, HunyuanVideo.
- Não suportados: GGUF, `.bin`/`.ckpt` antigos e checkpoints “single file” (ex.: o repositório
  `Wan-AI/Wan2.1-T2V-14B` original — use `Wan-AI/Wan2.1-T2V-14B-Diffusers`). O app avisa antes de
  baixar.

## Estrutura

```
ia-paginada/
├── electron/main.mjs        # janela + inicia o backend Python numa porta livre (com chave secreta)
├── renderer/                # interface (HTML/CSS/JS puro)
├── backend/
│   ├── server.py            # API local (FastAPI) — tarefas, download, carregar, gerar
│   ├── paged/pager.py       # ⭐ o carregador página-por-página
│   ├── paged/loaders.py     # monta modelos de texto (transformers) e imagem/vídeo (diffusers)
│   ├── paged/hub.py         # busca/planos/download no Hugging Face (modelos e Spaces)
│   └── tests/               # prova que o resultado paginado = resultado do modelo inteiro
└── scripts/                 # setup do venv Python
```

## Testes

```bash
npm run test:python
```

Os testes montam modelos minúsculos (Llama, Stable Diffusion) e verificam que a saída paginada é
idêntica à do modelo carregado inteiro, que nenhuma camada fica na memória depois do forward, e
que a geração de texto com KV-cache dá os mesmos tokens.

## Limites conhecidos

- As **ativações** não são paginadas: vídeos longos em alta resolução ainda podem precisar de
  bastante VRAM. Reduza quadros/resolução se faltar memória.
- Modelos quantizados (bitsandbytes, GGUF) não são suportados pelo paginador.
- Uma geração por vez (há uma fila simples com trava).
