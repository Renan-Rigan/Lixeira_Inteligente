# Lixeira inteligente — software

Visão computacional (OpenCV + YOLO) e controle dos servos para o protótipo de
separação automática de resíduos recicláveis. A lixeira identifica o material,
gira a base até o compartimento certo e solta o resíduo nele, o mais próximo
possível do tempo real.

| Compartimento | Material | Exemplos |
|---|---|---|
| `metal` | metal | latas de bebida e de alimento |
| `plastico` | garrafa PET | garrafas de refrigerante, água, suco |
| `resto` | todo o resto (na prática, orgânicos) | restos de comida, cascas, guardanapo, e qualquer item que o modelo não reconheça com confiança |

```
câmera ──► OpenCV: tem objeto parado? ──► YOLO-cls (CNN): qual material? ──► servos: base gira, escotilha abre
(Pi/celular) (presence.py)                 (classifier.py, 3 frames)          (actuators.py)
                                                    │
                                    logs/events.csv + data/inbox/  ──► revisão ──► novo treino
```

## Decisões de projeto

- **Metal, garrafa PET e resto.** Latas e garrafas PET são os recicláveis mais
  comuns e têm aparência bem definida (brilho metálico; plástico transparente
  com rótulo e tampa), o que facilita chegar a uma acurácia alta. Todo o resto
  (orgânicos, guardanapo, vidro...) vai para `resto`. `plastico` é **só PET**:
  sacolas e potes de bancos públicos ficam de fora do treino para o modelo não
  aprender que "plástico transparente" pode ser `resto`.
- **OpenCV detecta presença, YOLO classifica.** A comparação com o fundo vazio é
  quase gratuita no Pi; o modelo só roda quando há um objeto **parado** na
  plataforma (a mão do usuário já saiu). Isso economiza CPU e evita classificar
  a mão.
- **YOLO no modo *classify*, não *detect*.** A lixeira recebe um objeto por vez
  numa área controlada, então classificar a imagem inteira basta. É mais rápido,
  mais preciso e o dataset é só "uma pasta por classe", sem desenhar caixas.
- **Na dúvida, `resto`.** Se a confiança ficar abaixo do limiar
  (`classifier.conf_threshold`), o resíduo vai para `resto` (`reject_label`) em
  vez de contaminar os recicláveis. Errar para `resto` custa menos que pôr
  orgânico junto das latas. Nos relatórios, "rejeição" conta só o que foi para
  `resto` por falta de confiança, não o `resto` previsto com certeza.
- **Rede neural convolucional (CNN).** O YOLO-cls é uma CNN; veja
  [Por que CNN](#por-que-uma-rede-neural-convolucional-cnn).
- **Votação em 3 frames.** As probabilidades de 3 frames são somadas, o que reduz
  erro por reflexo ou borrão.
- **Escotilha nunca abre girando.** Há uma trava no código (`Sorter._rotate`).
- **Tudo calibrável no `config.yaml`**: ângulos, pinos, limiares e ROI. O código
  não muda quando o protótipo muda.

## Começando (no PC, sem hardware)

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python -m unittest discover tests -v          # 34 testes, sem câmera/modelo/servo
python run.py --fake-model --show             # webcam + classificador falso + servos simulados
```

No Windows (PowerShell), só a ativação muda:

```powershell
python -m venv .venv; .venv\Scripts\Activate.ps1
```

Rode todos os comandos **a partir da raiz do repositório** (a pasta com `run.py`
e `tests/`): os testes, o `config.yaml` e as pastas `logs/`, `models/` e `data/`
são procurados a partir do diretório atual.

### PC sem webcam: celular como câmera

1. Instale no celular o **IP Webcam** (Android) ou o **DroidCam** (Android/iPhone)
   e inicie a transmissão. PC e celular precisam estar na **mesma rede Wi-Fi**.
2. Coloque o IP que o app mostrar em `config.celular.yaml`
   (`http://<IP>:8080/video` no IP Webcam, `http://<IP>:4747/video` no DroidCam).
3. Prenda o celular **fixo, apontando de cima para a plataforma** (um suporte ou
   uma caixa servem). Se ele mexer, o fundo aprendido deixa de valer. No app,
   trave a exposição e o balanço de branco, se houver a opção.
4. Use `--config config.celular.yaml` em qualquer script que abre câmera:

```bash
python scripts/calibrate_roi.py --config config.celular.yaml   # recorta só a plataforma
python scripts/collect.py --config config.celular.yaml         # fotos para o dataset
python scripts/live.py --config config.celular.yaml            # identificação ao vivo (sem plataforma)
python run.py --config config.celular.yaml --fake-model --show # sistema completo, sem modelo
python run.py --config config.celular.yaml --show              # com o modelo treinado
```

A leitura do stream roda numa thread que guarda só o frame mais recente (sem
atraso acumulado) e reconecta sozinha se o Wi-Fi cair.

## Treino com banco de imagens + teste ao vivo pelo celular

Dá para treinar o modelo antes de ter o protótipo, com um **banco de imagens**
(pronto ou montado por você), e usar o **celular** para simular a identificação
em tempo real.

```
banco de imagens ──► check_images.py ──► prepare_dataset.py ──► train.py ──► live.py (celular)
(data/bancos/)       qualidade (régua:     mapeia pastas →        CNN           classifica ao vivo,
                     câmera do Pi)         metal/plastico/resto                 salva frames de teste
```

### 1. Escolher o banco

Pronto ou próprio? **Os dois, em etapas.** Um banco pronto dá volume logo no
começo (milhares de imagens); fotos próprias, tiradas com a câmera e a luz da
lixeira, são o que mais pesa na acurácia do protótipo. Comece pelo pronto, teste
com o celular e vá acrescentando fotos suas onde o modelo errar.

| Banco | O que tem | Serve para |
|---|---|---|
| Drinking Waste Classification (Kaggle) | latas de alumínio, garrafas PET, vidro, HDPE | `metal` e `plastico` (PET) |
| RealWaste (UCI, CC BY 4.0) | 4.752 fotos 524×524 em 9 classes, incl. *Food Organics* e *Vegetation* | `resto` (orgânicos) e `metal` |
| TrashNet | 2.527 fotos 512×384, fundo branco, 6 classes | complemento |
| Garbage Classification, 12 classes (Kaggle) | inclui *biological* (orgânico) | complemento para `resto` |

Confira a licença de cada um antes de usar na IC. Baixe cada banco para
`data/bancos/<nome>/` (a pasta `data/` não vai para o git), mantendo uma
subpasta por classe, do jeito que eles vêm.

### 2. Verificar a qualidade (régua: câmera do Raspberry Pi)

```bash
python scripts/check_images.py data/bancos/drinking_waste data/bancos/realwaste
```

A câmera do Pi tem sensor de 5 a 12 MP (v1: 2592×1944, v2: 3280×2464, Module 3:
4608×2592), mas **não é isso que o modelo vê**: o sistema captura o stream em
640×480 (`camera.width/height`), recorta a ROI e o YOLO reduz para 224×224
(`classifier.imgsz`). Por isso o `check_images.py` usa o `config.pi.yaml` como
régua e marca:

| Verificação | Erro (fica de fora) | Aviso |
|---|---|---|
| resolução | lado menor < 224 px (seria ampliada: mais borrada que o Pi) | menor que o frame do Pi (640×480) |
| nitidez | borrada (Laplaciano < `blur_threshold`, medido na escala do Pi) | — |
| brilho | escura (< 40) ou estourada (> 215) | pouco contraste |
| repetidas | mesma imagem em classes diferentes ou em treino e teste | cópias na mesma classe |
| arquivo | não abre | proporção > 2,5:1 (o corte central perde o objeto) |

Imagens maiores que o frame do Pi não são problema: use `--max-side 640` no
`prepare_dataset.py` para reduzi-las (menos disco, treino mais rápido). O
relatório (`reports/qualidade_*.md` e `.csv`) traz as medianas de nitidez e
brilho por classe; rode também nas suas fotos (`data/raw`) e compare: se o banco
for muito mais nítido ou claro que a sua câmera, ajuste os limiares em
`quality:` no `config.yaml`. `--quarantine data/reprovadas` move as imagens com
erro para revisão.

### 3. Montar o dataset

Os bancos usam nomes próprios de pasta; `--map` traduz para as classes do
projeto. Pastas que já se chamam `metal` (ou `Metal`) não precisam de `--map`;
`--map "*=resto"` manda o que sobrou para `resto`; `--map Pasta=` descarta.

```bash
python scripts/prepare_dataset.py \
    --source data/bancos/drinking_waste --source data/bancos/realwaste \
    --map PET=plastico --map AluCans=metal --map HDPEM= --map Plastic= \
    --map "*=resto" --max-per-class 1500 --max-side 640 --check-quality
```

- `--max-per-class` evita que `resto` (que junta várias pastas) fique muito
  maior que as outras classes. Fotos suas nunca são cortadas, só as do banco.
- `--check-quality` deixa de fora as imagens com erro no `check_images.py`.
- Com só bancos em `--source`, eles são divididos em treino/val/teste. A
  acurácia nesse teste sai **otimista** (mesma fonte, mesma luz). Quando tiver
  fotos suas, passe os bancos para `--train-only` e use as suas em `--source`.

### 4. Treinar e testar ao vivo com o celular

```bash
python scripts/train.py                                     # gera models/best.pt
python scripts/live.py --config config.celular.yaml         # aponte o celular para os resíduos
```

O `live.py` classifica cada quadro, mostra a classe e as probabilidades
(suavizadas entre quadros), avisa se a imagem está borrada ou escura e desenha o
quadrado central, que é o que o modelo recebe. O stream do celular é reduzido
para 640×480, a mesma resolução do Pi. Não precisa de fundo fixo nem de suporte.
Aperte `s` para salvar o quadro em `data/live/<classe prevista>/`: revise
(mova os erros para a pasta certa) e use como teste de verdade:
`--source data/live`.

## Por que uma rede neural convolucional (CNN)

"Rede convolucional" e "CNN" são a mesma coisa: CNN é a sigla em inglês de
*Convolutional Neural Network* (rede neural convolucional). O projeto já usa uma:
o **YOLO11n-cls** é uma CNN de classificação. As camadas de convolução aprendem
filtros que respondem a bordas, brilho, texturas e formas, que é exatamente o
que separa lata (reflexo metálico), PET (transparência, rótulo, tampa) e
orgânico (textura irregular, cores de alimento).

Para este caso, a melhor escolha é uma **CNN leve com transfer learning**:

| Opção | Por que (não) |
|---|---|
| **YOLO11n-cls** (usado) | CNN pequena (~1,6 M parâmetros), pré-treinada no ImageNet, treina com uma pasta por classe e exporta para NCNN, que é o formato mais rápido no Pi. Dezenas de ms por imagem no Pi 4/5. |
| MobileNetV3 / EfficientNet-Lite | CNNs leves equivalentes; dariam resultado parecido, mas exigem escrever o treino e a exportação à mão. |
| CNN treinada do zero | Precisaria de muito mais imagens; o transfer learning aproveita o que a rede já sabe de milhões de fotos. |
| Vision Transformer (ViT) | Não é convolucional; precisa de mais dados e é pesado demais para o Pi. |
| YOLO *detect* (caixas) | Também é CNN, mas exige desenhar caixas no dataset; só vale se houver vários objetos ao mesmo tempo. |

Se precisar de mais acurácia e o Pi aguentar a latência, teste
`--model yolo11s-cls.pt` no `train.py` (mesma CNN, maior).

## Ciclo de trabalho (o que você vai repetir ao longo da IC)

1. **Coletar fotos** na plataforma real, com os LEDs ligados:
   `python scripts/collect.py` (sem webcam: `--config config.celular.yaml`). Aperte `n` a cada objeto novo e `r` para tirar
   uma rajada enquanto gira o objeto. Mire em **objetos variados** (latas e
   garrafas amassadas, sujas, com e sem rótulo; vários tipos de orgânico), não
   em muitas fotos do mesmo.
2. **Verificar e montar o dataset**: `python scripts/check_images.py data/raw` e
   `python scripts/prepare_dataset.py --source data/raw` (exemplos com banco de
   imagens no `--help` e na seção acima).
3. **Treinar** (PC com GPU ou Google Colab): `python scripts/train.py`. Sem GPU
   no PC, o treino funciona, mas é lento: leve `data/dataset/` para o Colab e
   traga de volta o `models/best.pt`.
4. **Avaliar** no teste: `python scripts/evaluate.py`. O relatório em `reports/`
   traz a matriz de confusão, o IC 95%, os erros mais confiantes e a tabela de
   limiares para você escolher `conf_threshold`.
5. **Exportar** para o Pi: `python scripts/export.py` (gera NCNN).
6. **Usar e coletar**: cada descarte salva a imagem em `data/inbox/<previsto>/`.
   Revise as pastas, mova o que estiver errado para a pasta certa e treine de
   novo com `--source data/raw --source data/inbox`.

### Para chegar nos 90% de forma honesta

- O split é feito **por objeto** (prefixo `objXXXX__`). Fotos do mesmo objeto
  nunca ficam em treino e teste ao mesmo tempo. Sem isso, a acurácia fica
  inflada e cai no protótipo.
- Quando tiver fotos suas, bancos públicos entram **só no treino**
  (`--train-only`). Eles ajudam no começo, mas o teste precisa refletir a sua
  câmera e a sua luz.
- Com poucos itens de teste, 90% é incerto: com 50 itens e 45 acertos, o IC 95%
  vai de ~79% a ~96%. Reporte o intervalo, não só o número.
- Comece por volta de **30+ objetos diferentes por classe** e cresça a partir dos
  erros que o `evaluate.py` lista.

## No Raspberry Pi

```bash
python3 -m venv --system-site-packages .venv && source .venv/bin/activate
pip install -r requirements-pi.txt
# copie models/best_ncnn_model/ do PC para cá
python scripts/calibrate_roi.py --config config.pi.yaml   # cole o roi no config.yaml
python scripts/servo_test.py --config config.pi.yaml      # ache os ângulos de cada compartimento
python scripts/evaluate.py --config config.pi.yaml        # latência real no Pi
python run.py --config config.pi.yaml
```

`config.pi.yaml` herda de `config.yaml` e troca só câmera (Picamera2), modelo
(NCNN) e servos (gpiozero). **Alimente os servos por uma fonte própria** (5 V) com
GND comum ao Pi. Puxar a corrente dos servos pelo Pi causa reinicializações.

## Testes experimentais (seção 5.5)

```bash
python run.py --config config.pi.yaml --true-label plastico   # descarte só garrafas PET nesta sessão
python run.py --config config.pi.yaml --true-label metal
python scripts/analyze_events.py                           # acurácia, matriz, tempos
```

A taxa de acerto do **direcionamento** (o objeto caiu no compartimento certo?) é
mecânica e a câmera não vê. Anote à parte ou use `scripts/servo_test.py`
(`ciclo 20`) para os ensaios repetitivos da seção 6.4.

Antes de ter o protótipo, dá para fazer as mesmas sessões no PC com o celular e
servos simulados: `python run.py --config config.celular.yaml --show --true-label plastico`.

### Tempo de resposta

Cada descarte grava em `logs/events.csv` três tempos: da detecção à decisão
(`detect_to_decision_ms`), do mecanismo (`actuation_ms`) e o total (`total_ms`).
A visão leva menos de 1 s: esperar o objeto parar (`presence.stable_frames`)
mais 3 inferências do modelo nano. O que mais pesa é o mecanismo: girar
(`rotation.settle_s`), abrir e esperar cair (`hatch.open_s`), fechar e voltar,
cerca de 3–4 s com os valores padrão. Para ficar mais perto do tempo real,
reduza esses tempos no `config.yaml` até o limite que o protótipo aguentar.

## Mapa do código × projeto

| Seção do projeto | Arquivo |
|---|---|
| 5.1 Captura de imagens | `lixeira/camera.py`, `scripts/collect.py`, `scripts/calibrate_roi.py`, `config.celular.yaml` |
| Qualidade das imagens (régua: câmera do Pi) | `lixeira/quality.py`, `scripts/check_images.py` |
| 5.2 Modelo (CNN, transfer learning, aumento de dados) | `lixeira/dataset.py`, `scripts/prepare_dataset.py`, `scripts/train.py` |
| Identificação ao vivo pelo celular | `scripts/live.py` |
| 5.3 Mecanismo (base + escotilha) | `lixeira/actuators.py`, `scripts/servo_test.py` |
| 5.4 Integração (detecção → decisão → ação) | `lixeira/pipeline.py`, `run.py` |
| 5.5 Testes e validação | `scripts/evaluate.py`, `scripts/analyze_events.py`, `logs/events.csv` |
| 6.2 Otimização para o Pi (quantização) | `scripts/export.py` (`--int8`) |

## Estrutura

```
config.yaml                    parâmetros (PC com webcam, servos simulados)
config.celular.yaml            PC sem webcam: celular como câmera pelo Wi-Fi
config.pi.yaml                 Raspberry Pi (Picamera2, NCNN, servos reais)
run.py                         programa principal
lixeira/                       câmera, presença, classificador, servos, pipeline, log, métricas, dataset, qualidade
scripts/                       coleta, qualidade, dataset, treino, ao vivo, avaliação, exportação, calibração
tests/                         testes com imagens sintéticas (sem hardware)
```

## Fluxo de trabalho com branches

```
main  ← versão estável: só recebe PR vindo da dev
 └─ dev  ← integração e testes (branch padrão)
     ├─ feature/<nome>   novidade      → PR para dev
     └─ fix/<nome>       correção      → PR para dev
```

1. **Criar a branch a partir da `dev` atualizada**
   ```bash
   git checkout dev
   git pull origin dev
   git checkout -b feature/minha-mudanca
   ```
2. **Trabalhar e testar localmente**
   ```bash
   python -m unittest discover tests -v
   git add <arquivos>
   git commit -m "Descreve a mudança"
   git push -u origin feature/minha-mudanca
   ```
3. **Abrir PR `feature/minha-mudanca` → `dev`** no GitHub. O CI roda os testes
   automaticamente; com tudo verde, faça o merge e apague a branch.
4. **Testar a `dev`** (no PC e, quando fizer sentido, no Raspberry Pi).
5. **Abrir PR `dev` → `main`** quando a `dev` estiver estável. A `main` é a versão
   que vai para o protótipo e para os resultados da IC.

Nunca faça commit direto na `main`.
