# Testes recomendados (Raspberry Pi)

## Preparacao
- Confirmar SPI ativo (`sudo raspi-config` > Interface Options > SPI).
- Confirmar wiring: MCP3008 ligado a 3.3V/GND/SPI, sensor de ruido no CH0, botao no GPIO17 (pull-down), LED no GPIO27 com resistor.
- Instalar dependencias: `sudo apt update && sudo apt install -y python3-gpiozero python3-picamera2` (se usar camera) e `pip install -r requirements.txt` se existir.

## Checklist funcional
1. **Arranque**: `python stadium_interactive.py --interval 0.5` (adicionar `--camera` para testar capturas). Ver se a app imprime leituras sem erros.
2. **Ruido baixo**: Com sensor em silencio, confirmar linhas com `Ruido` < 512 e mensagem `Entusiasmo normal`, LED apagado.
3. **Ruido alto + botao pressionado**: Fazer ruido elevado (bater palmas/assobiar) e pressionar o botao -> LED acende e mensagem `GOLO`.
4. **Ruido alto + botao solto**: Manter ruido alto e soltar o botao -> LED apaga e mensagem `VAIA`.
5. **Alternancia rapida**: Alternar pressao do botao durante ruido alto e confirmar que o LED segue imediatamente a logica.
6. **Camera (se ativada)**: Com `--camera`, provocar `GOLO` e `VAIA` e verificar se surgem ficheiros `.jpg` no diretório `capturas/` com timestamps.
7. **Threshold**: Verificar que por volta de ~512 o LED so liga quando ruido ultrapassa o valor e o botao esta pressionado.
8. **Paragem limpa**: Premir `Ctrl+C` e confirmar que a app termina sem stack trace e liberta GPIO.

## Diagnostico rapido
- Se `Sem leitura do MCP3008`: confirmar SPI ativo e ligações MISO/MOSI/SCLK/CE0.
- Se o botao nunca muda: rever pull-down/pull-up e pino configurado.
- Se o LED fica sempre ON/OFF: testar com `--led-pin` correto e rever resistor/ligacao.
