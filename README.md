# Lixeira inteligente — software

Visão computacional (OpenCV + YOLO) e controle dos servos para o protótipo de
separação automática de resíduos recicláveis. A lixeira identifica o material,
gira a base até o compartimento certo e solta o resíduo nele, o mais próximo
possível do tempo real.

| Compartimento | Material | Exemplos |
|---|---|---|
| `metal` | metal | latas de bebida e de alimento, tampinhas |
| `papel` | papel e papelão | folhas, jornal, caixas, embalagens de papelão |
| `plastico` | plástico | garrafas PET, potes, embalagens, copos |
| `indefinido` | rejeito | qualquer item que o modelo não reconheça com confiança |

```
câmera ──► OpenCV: tem objeto parado? ──► YOLO-cls: qual material? ──► servos: base gira, escotilha abre
(Pi/celular) (presence.py)                 (classifier.py, 3 frames)    (actuators.py)
                                                    │
                                    logs/events.csv + data/inbox/  ──► revisão ──► novo treino
```

## Decisões de projeto

- **Metal, papel e plástico, sem vidro.** São os recicláveis mais comuns no lixo
  doméstico, têm aparência bem diferente entre si (brilho, textura fosca,
  transparência ou cores fortes) e aparecem em datasets públicos como o TrashNet.
  O vidro ficou de fora porque pode quebrar no mecanismo e se confunde com
  plástico transparente.
- **OpenCV detecta presença, YOLO classifica.** A comparação com o fundo vazio é
  quase gratuita no Pi; o modelo só roda quando há um objeto **parado** na
  plataforma (a mão do usuário já saiu). Isso economiza CPU e evita classificar
  a mão.
- **YOLO no modo *classify*, não *detect*.** A lixeira recebe um objeto por vez
  numa área controlada, então classificar a imagem inteira basta. É mais rápido,
  mais preciso e o dataset é só "uma pasta por classe", sem desenhar caixas.
- **Compartimento `indefinido`.** Se a confiança ficar abaixo do limiar
  (`classifier.conf_threshold`), o resíduo vai para um 4º compartimento em vez de
  contaminar os recicláveis. Errar para "indefinido" custa menos que errar de
  material.
- **Votação em 3 frames.** As probabilidades de 3 frames são somadas, o que reduz
  erro por reflexo ou borrão.
- **Escotilha nunca abre girando.** Há uma trava no código (`Sorter._rotate`).
- **Tudo calibrável no `config.yaml`**: ângulos, pinos, limiares e ROI. O código
  não muda quando o protótipo muda.

## Começando (no PC, sem hardware)

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python -m unittest discover tests -v          # 20 testes, sem câmera/modelo/servo
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
python run.py --config config.celular.yaml --fake-model --show # sistema completo, sem modelo
python run.py --config config.celular.yaml --show              # com o modelo treinado
```

A leitura do stream roda numa thread que guarda só o frame mais recente (sem
atraso acumulado) e reconecta sozinha se o Wi-Fi cair.

## Ciclo de trabalho (o que você vai repetir ao longo da IC)

1. **Coletar fotos** na plataforma real, com os LEDs ligados:
   `python scripts/collect.py` (sem webcam: `--config config.celular.yaml`). Aperte `n` a cada objeto novo e `r` para tirar
   uma rajada enquanto gira o objeto. Mire em **objetos variados** (amassados,
   sujos, com rótulo, transparentes), não em muitas fotos do mesmo.
2. **Montar o dataset**: `python scripts/prepare_dataset.py --source data/raw`
   (exemplos com dataset público no `--help`).
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
- Datasets públicos entram **só no treino** (`--train-only`). Eles ajudam no
  começo, mas o teste precisa refletir a sua câmera e a sua luz. Um bom ponto de
  partida é o TrashNet (classes glass, paper, cardboard, plastic, metal, trash).
  Mapeie `cardboard` para `papel`.
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
python run.py --config config.pi.yaml --true-label papel   # descarte só papéis nesta sessão
python run.py --config config.pi.yaml --true-label metal
python scripts/analyze_events.py                           # acurácia, matriz, tempos
```

A taxa de acerto do **direcionamento** (o objeto caiu no compartimento certo?) é
mecânica e a câmera não vê. Anote à parte ou use `scripts/servo_test.py`
(`ciclo 20`) para os ensaios repetitivos da seção 6.4.

Antes de ter o protótipo, dá para fazer as mesmas sessões no PC com o celular e
servos simulados: `python run.py --config config.celular.yaml --show --true-label papel`.

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
| 5.2 Modelo (transfer learning, aumento de dados) | `scripts/prepare_dataset.py`, `scripts/train.py` |
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
lixeira/                       câmera, presença, classificador, servos, pipeline, log, métricas, dataset
scripts/                       coleta, dataset, treino, avaliação, exportação, calibração
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
