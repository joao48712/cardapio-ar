# 📖 IA Paginada

App desktop (Electron + Node.js + Python/PyTorch) que roda modelos de IA **muito maiores que a
memória da sua GPU** — texto, imagem e vídeo — lendo os pesos **uma página por vez**, como uma
impressora que imprime um livro inteiro página por página: pega uma, imprime, joga fora, pega a
próxima.

> Projeto independente dentro deste repositório. Não altera nada do cardápio (`src/`, `public/`…).

## Como funciona

1. O modelo é montado **vazio** (dispositivo `meta` do PyTorch): a arquitetura existe, mas não
   ocupa nem 1 byte de RAM/VRAM.
2. **Todos os componentes são divididos em páginas**: cada camada de transformer, bloco da UNet,
   bloco do VAE, camada dos codificadores de texto — e também as peças soltas grandes
   (embeddings, cabeça de saída, convoluções de entrada/saída).
3. Um *hook* do PyTorch roda logo antes de cada página e põe os pesos dela no lugar. Ao terminar o
   forward, os pesos são **descartados** e a próxima página entra.
4. Só peças minúsculas (normas) ou compartilhadas entre módulos ficam residentes.

Pico de memória ≈ 1–3 páginas + ativações, em vez do modelo inteiro.

### Troca de página rápida (só CPU)

| Técnica | O que faz |
|---|---|
| **Sem cópia (mmap)** | Na CPU, a página vira uma *janela* direto no arquivo (`torch.frombuffer` sobre `mmap`). “Carregar” custa ~0,3 ms; o sistema operacional lê do disco só o que é tocado e guarda no cache dele. |
| **Encadernação** | Se o tipo no disco difere do tipo de cálculo (ex.: fp16 → fp32), converte **uma única vez** para um arquivo `.ia-paginada-pack/` alinhado e na ordem das páginas. Depois disso nunca mais há conversão. |
| **Ordem aprendida + leitura antecipada** | Registra a ordem real em que as páginas rodam (UNets e VAEs não seguem a numeração) e pede ao sistema (`madvise WILLNEED`) as próximas N páginas enquanto a atual calcula. |
| **bfloat16 automático** | Em CPUs com AMX/AVX512-BF16, calcula em bf16: metade dos bytes para mover. |
| **Todos os núcleos** | `torch.set_num_threads(nº de núcleos)`. |

Medido com `python backend/bench.py` (Llama de 16 camadas, checkpoint em fp16, CPU de 4 núcleos):

| Modo | ms/passada | troca média | |
|---|---:|---:|---:|
| Paginado, convertendo a cada leitura (versão antiga) | 186,8 | 2,04 ms | 1,0× |
| Paginado, encadernado sem cópia (fp32) | 65,2 | 0,31 ms | 2,9× |
| **Paginado, sem cópia em bfloat16 (AMX)** | **34,7** | 0,26 ms | **5,4×** |
| Modelo inteiro carregado na RAM (fp32), referência | 56,7 | — | |

Quando o modelo é maior que a RAM, o limite passa a ser a velocidade do disco: a leitura antecipada
mantém o SSD ocupado enquanto a CPU calcula.

### ⚡ Variável “Velocidade” (imagem e vídeo)

A troca de página já custa < 1 ms; numa geração de 205 s, as trocas somam ~1 s. O tempo está no
**cálculo** dentro das páginas. Por isso a variável **Velocidade** (número ≥ 1, **sem limite
superior**) controla quantas vezes as páginas pesadas são percorridas: com velocidade N, o modelo
principal (UNet/transformer) só é recalculado em ~1 de cada N passos; nos outros, a saída do último
cálculo é reaproveitada e nenhuma página é carregada. Os 2 primeiros passos e o último são sempre
calculados. `1` = idêntico ao normal.

- No app: controle ⚡ Velocidade na tela **Gerar** (campo livre + barra).
- Na API: campo `speed` em `POST /api/generate`.
- Padrão global: variável de ambiente `IAPAG_SPEED` (ex.: `IAPAG_SPEED=2 npm start`).

Medido com `ali-vilab/text-to-video-ms-1.7b`, 256×256, 16 quadros, 25 passos, só CPU (4 núcleos):

| Velocidade | Tempo | Passos calculados | Qualidade |
|---:|---:|---:|---|
| 1× | 205 s | 25 | referência |
| 2× | 120 s | 14 | praticamente igual |
| 4× | 70 s | 8 | perde detalhe no fundo |
Funciona com qualquer arquitetura que o `transformers`/`diffusers` saibam montar, inclusive com
KV-cache na geração de texto. O código do núcleo está em
[`backend/paged/pager.py`](backend/paged/pager.py).

**Troca:** quando o modelo não cabe na RAM, cada token de texto ou passo de difusão relê o modelo
do disco — um SSD NVMe ajuda muito. A encadernação ocupa espaço extra em disco (ex.: fp16 → fp32
dobra o tamanho); desligue-a na tela “Meus modelos” se faltar espaço.

## Recursos

- ⬇️ **Baixar do Hugging Face** dentro do app: busca por tipo (texto, texto→imagem,
  imagem→imagem, texto→vídeo, imagem→vídeo), ID direto, ou **a partir de um Space** (o app descobre
  quais modelos o Space usa). Mostra os arquivos e o tamanho antes de baixar, baixa só
  `.safetensors` + configs, pode preferir a variante fp16, aceita token para modelos restritos.
- 📚 **Meus modelos**: carregar / descarregar / apagar; escolher GPU (CUDA), Apple (MPS) ou CPU,
  precisão, quantas páginas ler adiante, encadernação e o tamanho mínimo de uma página.
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
| `IAPAG_SPEED`  | Velocidade padrão das gerações (padrão `1`, sem limite)  |

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
│   ├── bench.py             # mede a velocidade da troca de páginas
│   ├── server.py            # API local (FastAPI) — tarefas, download, carregar, gerar
│   ├── paged/pager.py       # ⭐ o carregador página-por-página
│   ├── paged/loaders.py     # monta modelos de texto (transformers) e imagem/vídeo (diffusers)
│   ├── paged/speed.py       # variável Velocidade (pula passos reaproveitando saídas)
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
