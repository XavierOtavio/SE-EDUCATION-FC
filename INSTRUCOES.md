# Estadio Interativo - Guia de Execucao (Raspberry Pi)

## Requisitos
- Raspberry Pi com Raspberry Pi OS
- Sensor de ruido analogico ligado a MCP3008 (canal CH0 por defeito)
- Botao em GPIO17 (BCM) com resistor pull-down (ou ajuste para pull-up no codigo)
- LED em GPIO27 (BCM) com resistor
- Camera Pi v2 (opcional, para capturas em GOLO/VAIA)

## Preparacao do sistema
1) Ativar SPI: `sudo raspi-config` > Interface Options > SPI > Enable, depois reiniciar.
2) Instalar dependencias:
```bash
sudo apt update
sudo apt install -y python3-gpiozero python3-spidev python3-picamera2  # picamera2 so se quiser usar camera
```

## Instalar dependencias Python
O script usa apenas gpiozero (e picamera2 se camera ativa). Opcionalmente:
```bash
pip install -r requirements.txt  # se criar um requirements, nao obrigatorio aqui
```

## Executar
```bash
python stadium_interactive.py --interval 0.5
```
Argumentos uteis:
- `--noise-channel` (padrao 0) para escolher o canal do MCP3008.
- `--button-pin` (padrao 17) para definir o GPIO do botao (BCM).
- `--led-pin` (padrao 27) para definir o GPIO do LED (BCM).
- `--camera` para ativar capturas com a Camera Pi v2 em transicoes GOLO/VAIA.
- `--camera-dir` (padrao `capturas`) diretorio onde gravar fotos quando `--camera` esta ativo.

## O que vera no terminal
Linhas como:
```
Ruido= 850 | Pressao=True  | LED=ON  | GOLO
Ruido= 900 | Pressao=False | LED=OFF | VAIA
Ruido= 200 | Pressao=False | LED=OFF | Entusiasmo normal
```

## Criterios de aceitacao cobertos
- Leitura real do sensor de ruido (0-1023) via MCP3008 e do botao (True/False).
- Logica:
  - Ruido > 512 e botao pressionado: LED ON e mensagem `GOLO`.
  - Ruido > 512 e botao nao pressionado: LED OFF e mensagem `VAIA`.
  - Ruido <= 512: LED OFF e mensagem `Entusiasmo normal`.
- Mensagens e estado do LED refletem os sensores em tempo real; se `--camera` estiver ativo, fotos sao guardadas em cada mudanca para GOLO/VAIA.
