
# GUIA DE ESTUDO — Sistemas Embebidos  

*(MATÉRIA TEÓRICA COMPLETA + Exercícios estilo exame)*

---

# ⭐ 1. Fundamentos de Sistemas Embebidos

*(Aula Teórica 1 e 2 — ver slides 9; 2–6; 9–14)*

## 1.1 Definição
>
> “Um sistema embebido é qualquer dispositivo que inclui um computador programável, mas que não é destinado a ser um computador de uso geral.”  
> *(ver Aula Teórica 2, slide 2 — Marilyn Wolf)*

Um sistema embebido integra **hardware e software dedicados** para executar uma função específica dentro de um dispositivo maior, geralmente com **restrições de energia, memória e desempenho**.

### Propriedades fundamentais

- **Finalidade específica:** desenvolvido para um conjunto limitado de tarefas, ao contrário de computadores de uso geral.  
- **Determinismo temporal:** em muitos casos, é necessário garantir tempos de resposta previsíveis.  
- **Eficiência energética:** muitos dispositivos são alimentados a bateria ou devem operar com baixo consumo.  
- **Recursos limitados:** pouca RAM, pouca Flash/ROM, processadores simples.  
- **Robustez e fiabilidade:** funcionamento contínuo, muitas vezes em condições adversas.  
- **Interação física:** recebe sinais do ambiente (sensores) e atua sobre ele (atuadores).  
- **Baixo custo:** otimização económica é uma prioridade.  
- **Concorrência:** várias tarefas simultaneamente (ex.: leitura de sensores + comunicação + controlo).

---

## 1.2 Exemplos

*(ver Aula Teórica 2, slides 11–29)*

### Consumo e dia‑a‑dia

- smartphones (componentes embebidos internos)  
- dispositivos wearables e dispositivos de fitness
- routers, televisores, set‑top boxes  
- eletrodomésticos inteligentes  

### Automóvel

- ABS  
- Airbag  
- ECU de motor  
- ADAS (radares, câmaras, sensores de proximidade)  

### Indústria

- PLCs  
- sistemas de controlo distribuído  
- robots industriais  
- sensores e atuadores conectados por buses industriais  

### Especializados

- drones (controladora com IMU e GPS)  
- dispositivos médicos (bombas de insulina, pacemakers)  
- aeroespacial  

---

## 1.3 Gerações tecnológicas

*(ver Aula Teórica 2, slides 36–39)*

**1ª geração (8 bits):**  

- computação muito limitada  
- funções simples e determinísticas  
- microcontroladores básicos

**2ª geração (16 bits):**  

- mais memória e periféricos  
- aplicações mais ricas

**3ª geração (32 bits + DSP):**  

- processamento de sinais em tempo real  
- multimédia e controlos mais complexos

**4ª geração (64 bits + multicore):**  

- computação comparável a sistemas modernos  
- elevada integração

**5ª geração (IoT, Edge, IA/TinyML):**  

- autonomia, conectividade e inteligência local  
- sensores inteligentes com pré-processamento  

---

## 1.4 Arquitetura típica

### Componentes de hardware

- CPU / microcontrolador  
- RAM  
- ROM/Flash (firmware)  
- periféricos: timers, ADC, DAC, interfaces de comunicação  
- GPIO para sensores e atuadores  

### Software

- firmware dedicado  
- drivers  
- bibliotecas  
- (opcional) RTOS para multitarefa determinística  

---

## 1.5 Classificação dos sistemas embebidos

### Por objetivo

- **Monitorização** (ex.: sensores ambientais)  
- **Controlo** (ex.: robótica, automação)  
- **Comunicação** (gateways, hubs IoT)  

### Por complexidade

- simples (8/16 bits)  
- médios (32 bits)  
- avançados (64 bits, IA local)

---

## 💬 *Ênfase do professor*

- A missão do sistema embebido é **mais importante que a potência do hardware**.  
- Um sistema embebido deve ser desenhado para **não falhar**, mesmo com poucos recursos.  
- A interação com o mundo físico é uma das **partes mais difíceis** da disciplina.

---

## ⚠️ Armadilhas comuns

- confundir “rápido” com “tempo real” → são conceitos diferentes  
- assumir que todos os sistemas embebidos têm sistema operativo  
- acreditar que um sensor dá “valores perfeitos” — na prática, há ruído  
- pensar que aumentar bits do microcontrolador resolve tudo  

---

## 📝 Exercícios — Fundamentos (Estilo Exame)

1. **Explique, com as suas palavras, o que distingue um sistema embebido de um computador de uso geral.**  
   Identifique pelo menos **três características fundamentais**.

2. **Classifique os dispositivos seguintes** como embebidos ou não, justificando:  
   a) Câmera de estacionamento  
   b) Smartwatch  
   c) Bomba de insulina  
   d) Router Wi‑Fi  

3. Um microcontrolador tem:  
   - 32 KB RAM  
   - 256 KB Flash  
   - CPU ARM Cortex‑M4  
   - ADC 12 bits  
   Identifique **a qual geração tecnológica pertence** e justifique.

4. Um sistema industrial exige fiabilidade extrema. Explique:  
   - o que significa “fiabilidade” neste contexto  
   - consequências potenciais de uma falha  
   - duas técnicas para aumentar a fiabilidade  

---

### Gabarito resumido — Fundamentos
1. Sistema com função específica, recursos limitados e requisitos de previsibilidade/tempo real (ex.: determinismo, baixo consumo, robustez).  
2. a) embebido, b) embebido, c) embebido, d) embebido.  
3. 3ª geração (32 bits + DSP) pelo Cortex‑M4 e recursos associados.  
4. Fiabilidade = funcionamento correto contínuo; falhas podem causar paragens, danos ou riscos; técnicas: redundância, watchdog, testes rigorosos/ECC.


# ⭐ 2. I/O Digital e Analógica + Sensores e Atuadores
*(Aula Teórica 3 — slides 3–4; Aula Teórica 4 — slides 7–15)*

## 2.1 Sinal Digital (ON/OFF)

Um sinal digital assume apenas dois estados possíveis:
- **0 / LOW** → normalmente 0 V  
- **1 / HIGH** → 3.3 V ou 5 V, consoante o microcontrolador  

### Características essenciais:
- elevada **imunidade ao ruído** (pequenas flutuações não alteram o valor lógico)  
- processamento simples (comparações diretas)  
- tempo de resposta rápido  
- ideal para **botões, interruptores, relés, LEDs**, comunicação binária  

### Curiosidade importante:
A deteção de níveis lógicos depende de **limiares internos** (thresholds).  
Um ruído que não ultrapasse esses limiares **não altera** o estado lógico.

*(ver Aula Teórica 4, slides 10–11)*

---

## 2.2 Sinal Analógico

Um sinal analógico varia de forma **contínua**, podendo assumir infinitos valores dentro da sua gama.

### Exemplos:
- temperatura (termistor, LM35)  
- luminosidade (LDR)  
- pressão, humidade  
- sinais biomédicos  

### Necessidade de conversão:
Para que o microcontrolador interprete um valor analógico, é necessário um **conversor analógico‑digital (ADC)**.

*(ver Aula Teórica 4, slides 8–9)*

### Características relevantes:
- sensível ao ruído  
- limitada pela resolução do ADC  
- pode ser filtrado (analógico + digital)  
- depende da gama de entrada (0–5 V, ±2.5 V, 0–20 mA, etc.)

---

## 2.3 Sensores, Atuadores e Transdutores
*(ver Aula Teórica 3, slides 3–4)*

### Transdutor
Dispositivo que converte um tipo de energia noutro.  
Ex.:  
- microfone (acústica → elétrica)  
- altifalante (elétrica → acústica)  

### Sensor (entrada)
Capta informação do ambiente:
- temperatura  
- IMU (acelerómetro, giroscópio)  
- GPS  
- sensores de proximidade  

### Atuador (saída)
Produz efeito físico no ambiente:
- motores DC / servo / passo  
- buzzers / altifalantes  
- relés  
- válvulas  

### Tipos de sinal produzidos:
- sensores **digitais**: interruptores magnéticos, sensores Hall digitais, módulos I2C/SPI  
- sensores **analógicos**: LDR, termistores, potenciómetros  
- atuadores **digitais**: relés, LEDs  
- atuadores **analógicos/PWM**: velocidade de motor, brilho de LED, servo‑motores  

---

## 2.4 Condicionamento de sinal (importantíssimo)
*(conteúdo inferido da prática habitual + alinhado com slides)*

Antes de um sinal chegar ao ADC/Digital é frequente usar:
- **Amplificação** (op‑amps)  
- **Filtros analógicos** (passa‑baixo → remover ruído)  
- **Divisores de tensão** (adequar tensões ao ADC)  
- **Isolamento elétrico** (optoacopladores em ambiente industrial)  

Falhas neste condicionamento produzem medições incorretas → **tema muito frequente em exame**.

---

## 2.5 Debouncing (botões e sensores digitais)
Quando um botão é pressionado, há vibração mecânica e o sinal oscila entre 0 e 1 durante alguns milissegundos.

Solução:
- **debouncing por hardware** (RC filter, Schmitt trigger)  
- **debouncing por software** (tempo morto de leitura)

Exames adoram perguntas sobre isto.

---

## 💬 *Ênfase do professor*
- **Nunca ligar diretamente atuadores potentes ao microcontrolador**.  
  → usar transístores, MOSFETs ou drivers (L298, ULN2803).  
- Identificar corretamente o tipo de sinal que um sensor gera é fundamental para **escolher ADC, filtros e técnicas de leitura**.

---

## ⚠️ Armadilhas comuns
- Confundir **entrada analógica** (ADC) com **saída PWM** (pseudo-analógica).  
- Assumir que sensores analógicos são automaticamente mais precisos — **não são**.  
- Ligar sensores de 5V a microcontroladores de 3.3V → pode destruir o MCU.  
- Esquecer que sinais analógicos precisam de **filtros** (ruído, aliasing).

---

## 📝 Exercícios — I/O e Sensores (Estilo Exame)

1. **Explique por que razão sinais digitais são mais robustos ao ruído do que sinais analógicos.**  
   Inclua no raciocínio os thresholds internos do microcontrolador.

2. **Indique três sensores: um digital, um analógico e um digital serial (ex.: I2C/SPI).**  
   Para cada um, descreva a forma de leitura.

3. Um botão produz múltiplas transições rápidas ao ser pressionado.  
   **a)** Explique o fenómeno.  
   **b)** Apresente duas soluções práticas para o eliminar.

4. Um sensor analógico com gama 0–5 V envia sinal para um ADC de 3.3 V.  
   **Descreva o que deve ser feito** para evitar danos no microcontrolador.

5. Identifique dois atuadores que exigem **PWM** para controlo e justifique porquê.

6. Um sensor lê 1.4 V e está ligado a uma entrada digital cuja threshold HIGH é 2.0 V.  
   Indique o valor lógico lido e explique.

---



### Gabarito resumido — I/O e Sensores
1. Digital é robusto por margens de ruído/thresholds internos; pequenas variações não mudam o estado lógico.  
2. Ex.: digital (botão, leitura HIGH/LOW), analógico (LDR, leitura via ADC), serial I2C/SPI (IMU, leitura por registos).  
3. a) bounce mecânico; b) RC/Schmitt trigger ou atraso por software.  
4. Usar divisor de tensão/level shifter e proteção para limitar a 3.3 V.  
5. Ex.: motor DC e LED (brilho/velocidade exigem controlo por PWM).  
6. Lê LOW (0), pois 1.4 V < 2.0 V (threshold HIGH).

# ⭐ 3. PWM, ADC, Amostragem, Quantização e Aliasing
*(Aula Teórica 3 — slides 7–28)*

## 3.1 PWM — Pulse Width Modulation
*(ver slides 7–8; 9–10)*

A Modulação por Largura de Pulso (PWM) é uma técnica digital utilizada para simular um sinal analógico variando o **duty‑cycle** (percentagem de tempo em que o sinal permanece HIGH durante um período fixo).

### Conceitos fundamentais
- o sinal é **binário** (0 ou 1), mas a média ao longo do tempo pode assumir qualquer valor entre 0 e $V_{\text{fonte}}$  
- aumenta a eficiência energética, especialmente no controlo de motores  
- a frequência do PWM é fixa; apenas o **duty‑cycle** muda  
- quanto maior o duty‑cycle → maior a energia média fornecida ao atuador  

### Exemplos clássicos de utilização:
- controlo de velocidade de motores DC  
- posição de servomotores (PWM especial, 50 Hz, pulsos 1–2 ms)  
- controlo de brilho de LEDs  
- controlo térmico em resistências aquecedoras  

### Valor médio:
```math
V_{médio} = duty\% \times V_{fonte}
```

### Observações importantes:
- PWM não é DAC (conversor analógico‑digital); é apenas uma **aproximação temporal**  
- o efeito analógico só se verifica quando o sistema tem **inércia** (ex.: motor) ou quando existe filtragem (RC)

---

## 3.2 ADC — Conversão Analógico → Digital
*(ver slide 14)*

O ADC converte uma tensão analógica contínua num valor discreto, representado por **n bits**.

### Características principais
- número de níveis:  
```math
2^n
```

- resolução (step):  
```math
step = \frac{V_{ref}}{2^n}
```

- a resolução define o **incremento mínimo detetável**

### Exemplo rápido:
ADC de **10 bits**, Vref = 5 V:  
```math
step = \frac{5}{1024} = 4.88 \text{ mV}
```


### Tipos de ADC mais comuns:
- **SAR (Successive Approximation Register)**  
  - rápido, preciso, muito comum em microcontroladores  
- **Flash**  
  - extremamente rápido, mas caro, usado em aplicações de alta velocidade  
- **Sigma‑Delta**  
  - muito preciso para sinais de baixa frequência  

### Erros associados ao ADC:
- offset  
- ganho  
- ruído de quantização  
- saturação (ultrapassar Vref)

---

## 3.3 Teorema de Nyquist–Shannon
*(ver slides 15–16)*

O teorema afirma que a frequência de amostragem deve ser **pelo menos 2× a frequência máxima do sinal**:
```math
f_s \geq 2 f_{max}
```

Quando isto não acontece, perde‑se informação e ocorrem fenómenos de aliasing.

### Intuição:
- se amostrares demasiado devagar, o sinal “parece” ter uma frequência diferente da real  
- isto acontece porque a amostragem não consegue seguir o ritmo da variação

### Exemplo:
Para um sinal de **1 kHz**, a amostragem deve ser no mínimo **2 kHz**, sendo recomendável valores muito superiores (ex.: 5–10×).

---

## 3.4 Quantização e Aliasing
*(ver slides 17–23)*

### Quantização
Conversão de um valor analógico contínuo para o nível mais próximo representável pelo ADC.

- é **inevitável**  
- produz **erro de quantização**  
- melhora ao aumentar o número de bits do ADC

### Aliasing
Fenómeno em que um sinal de frequência alta parece ter uma frequência mais baixa devido a amostragem insuficiente.

O aliasing distorce completamente a leitura e torna a reconstrução impossível sem filtragem prévia.

### Filtro Anti‑Aliasing
Um **filtro passa‑baixo analógico**, colocado antes do ADC, garante que a máxima frequência contém apenas componentes amostráveis.

- obrigatório em sistemas reais  
- evita amostragens ambíguas  
- limita ruído de alta frequência

---

## 💬 *Ênfase do professor*
- “Antes de medir sinais reais, passa SEMPRE por um **filtro passa‑baixo**.”
- “PWM não é analógico, mas pode parecer analógico se o sistema tiver inércia.”
- “Um ADC sem filtragem dá leituras bonitas... mas erradas.”

---

## ⚠️ Armadilhas comuns
- Confundir **resolução** (tamanho do step) com **precisão** (quão correto está o valor).  
- Acreditar que aumentar bits resolve todos os problemas — **sem filtro, o ADC continua a medir aliasing**.  
- Esquecer que Vref afeta diretamente a resolução.

---

## 📝 Exercícios — PWM/ADC (Estilo Exame)

1. Um ADC de **12 bits** com **Vref = 3.3 V** tem que resolução?  
   Mostra o cálculo passo a passo.

2. Um PWM a 5 V tem **duty‑cycle de 60%**.  
   a) Calcula o valor médio.  
   b) Indica dois exemplos de sistemas onde esta tensão média não corresponde ao comportamento instantâneo.

3. Um sinal de **10 Hz** é amostrado a **12 Hz**.  
   a) Ocorre aliasing? Explica.  
   b) Qual é a frequência “falsa” (alias) observada?

4. Um ADC satura quando o sinal ultrapassa 3.3 V.  
   Se o sinal real for 4.1 V, que valor o ADC lê?  
   Explica as consequências.

5. Um motor DC ligado a PWM parece tremer em duty‑cycles muito baixos.  
   Explica o fenómeno em termos de inércia mecânica e frequência de PWM.

---




### Gabarito resumido — PWM/ADC
1. \( step = 3.3/4096 \approx 0.000805 \text{ V} \) (≈0.805 mV).  
2. a) \( V_{médio} = 0.6 \times 5 = 3.0 \text{ V} \). b) Em motores/LEDs, o instantâneo é 0/5 V; a média só faz sentido com inércia/filtragem.  
3. a) Sim, há aliasing (fs < 2f). b) \( f_{alias} = |10 - 12| = 2 \text{ Hz} \).  
4. Lê o valor máximo (saturação); ocorre clipping e perda de informação.  
5. Pulsos curtos não vencem inércia/atrito; torque médio baixo causa tremores.

# ⭐ 4. Interfaces de Comunicação
*(Aula Teórica 4 — slides 16–44)*

As interfaces de comunicação permitem que o microcontrolador **troque informação com sensores, atuadores e outros dispositivos**.  
Dividem‑se em **série assíncrona**, **série síncrona** e **paralela**, cada uma com diferentes compromissos entre velocidade, cablagem e complexidade.

---

## 4.1 UART — Assíncrona
*(ver slides 16–25)*

A UART (Universal Asynchronous Receiver and Transmitter) é uma comunicação **série assíncrona**, ou seja, **não usa clock partilhado** entre emissor e recetor.

## Características gerais
- comunicação ponto‑a‑ponto  
- sem clock → sincronização através de bits especiais  
- robusto para longas distâncias  
- muito simples de implementar  
- apenas duas linhas: **TX** e **RX**

## Estrutura do frame UART
Um frame típico inclui:
- **Start bit** (força uma transição HIGH → LOW para indicar início)  
- **Data bits** (geralmente 8)  
- **Optional parity** (par/impar, ou ausente)  
- **Stop bit(s)** (1 ou 2 bits HIGH)

Exemplo: **8E1** (8 data, Even parity, 1 stop).

### Temporização — Baud rate
O *baud rate* define **quantos símbolos por segundo** são enviados.  
A duração de cada bit é:

```math
T_{bit} = \frac{1}{baud}
```

Ex.: baud = 9600 → $T_{bit}$ ≈ 104 μs.

### Eficiência
Para 8 data bits, 1 paridade e 1 stop:

```math
Ef = \frac{8}{8+1+1} = \frac{8}{11} \approx 73\%
```

Quanto mais bits de controlo, menor a eficiência.

## Vantagens
- simples  
- ideal para longas distâncias  
- muito compatível entre dispositivos

## Desvantagens
- apenas **1 emissor ↔ 1 recetor** (não é bus)  
- menos eficiente devido a bits de controlo  
- limitado em velocidade quando comparado com SPI

---

## 4.2 SPI — Síncrona
*(ver slides 27–29)*

SPI (Serial Peripheral Interface) é uma comunicação **série síncrona**, rápida e full‑duplex.

## Linhas:
- **MOSI** — Master Out, Slave In  
- **MISO** — Master In, Slave Out  
- **SCK** — clock enviado pelo master  
- **SS / CS** — selecionar o slave ativo

## Características principais
- **full‑duplex** (emite e recebe simultaneamente)  
- muito **rápido** (MHz)  
- cada slave tem uma linha SS/CS dedicada  
- adequado para sensores de alta velocidade, LCDs, memórias Flash

## Modos SPI
A comunicação depende de duas propriedades:
- **CPOL** (polarity)  
- **CPHA** (phase)

Há 4 modos: MODE0, MODE1, MODE2, MODE3.

Exame pode pedir identificação do modo com base em diagramas.

## Vantagens
- extremamente rápido  
- simples e determinístico  
- ideal para throughput elevado

## Desvantagens
- consome muitos pinos (1 SS por slave)  
- não suporta endereços nativamente  
- má escolha quando há muitos periféricos

---

## 4.3 I2C — Síncrona, dois fios
*(ver slides 30–39)*

I2C (Inter‑Integrated Circuit) é comunicação **série síncrona**, master/slave, baseada em **endereçamento**, ideal para muitos dispositivos.

## Linhas:
- **SDA** — dados  
- **SCL** — clock

## Características principais
- **2 fios apenas**, independentemente do número de dispositivos  
- **vários masters** e **vários slaves**  
- cada slave tem um **endereço único**  
- protocolo inclui **ACK/NACK**  
- half‑duplex  
- velocidades típicas: 100 kHz, 400 kHz, 1 MHz (Fast‑Mode+)

## Estrutura das mensagens
- condição **START**  
- endereço + bit R/W  
- ACK/NACK  
- bytes de dados  
- condição **STOP**

Se o slave reconhecer o endereço → envia ACK.

## Vantagens
- poupança de pinos  
- ideal para ligar **muitos sensores**  
- protocolo simples e muito usado em sensores modernos

## Desvantagens
- mais lento que SPI  
- sensível a ruído (linhas abertas com resistores pull‑up)  
- conflito entre masters exige deteção de colisões

---

## 4.4 Paralela
*(ver slides 42–43)*

Comunicação paralela transmite **vários bits em simultâneo**.

## Características
- muito **rápida**  
- muitos pinos (4, 8, 16, 32 bits)  
- ideal quando se precisa de throughput extremo  
- usada frequentemente em **LCDs mais antigos** ou buses como PCI, ISA (contexto genérico)

## Vantagens
- altíssimo débito de dados  
- latência mínima

## Desvantagens
- muito consumo de pinos  
- maior suscetibilidade a ruído em cablagem longa

---

## 💬 *Ênfase do professor*
- “A escolha da interface depende sempre do compromisso entre **velocidade**, **número de fios**, **complexidade** e **número de dispositivos**.”  
- “SPI é o mais rápido, I2C é o mais escalável, UART é o mais simples.”

---

## ⚠️ Armadilhas comuns
- tentar usar UART como bus de vários dispositivos  
- escolher SPI em sistemas com poucos pinos disponíveis  
- esquecer os **resistores pull‑up** obrigatórios no I2C  
- confundir endereçamento (I2C) com seleção por linha SS (SPI)  
- não perceber que UART é **assíncrona**, SPI/I2C são **síncronas**

---

## 📝 Exercícios — Comunicação (Estilo Exame)

1. **Compare UART, SPI e I2C** em termos de:  
   - velocidade  
   - número de fios  
   - escalabilidade (nº de dispositivos)  
   - robustez e complexidade  

2. Numa UART **8N2**, calcule a eficiência:  
   - 8 data bits  
   - 0 paridade  
   - 2 stop  
   Mostre o raciocínio.

3. Um sistema com 6 sensores deve comunicar com apenas 2 pinos.  
   **Qual a interface mais adequada? Porquê?**

4. Desenhe (em texto) um frame UART **8E1** (inclua start, data, paridade e stop).

5. Um SPI com três slaves precisa de quantas linhas SS?  
   Justifique.

6. Num sistema I2C, explique o papel do **ACK** e dê um exemplo de quando um slave envia **NACK**.

---




### Gabarito resumido — Comunicação
1. UART: simples, 2 fios, baixa/média velocidade, ponto‑a‑ponto; SPI: muito rápido, mais fios, baixa escalabilidade; I2C: 2 fios, endereçado, velocidade média, alta escalabilidade.  
2. Eficiência = \( 8/(8+0+2) = 80\% \).  
3. I2C, por suportar vários dispositivos com 2 fios.  
4. Start(0) + 8 data + paridade even + stop(1).  
5. 3 linhas SS (uma por slave).  
6. ACK confirma receção/endereço; NACK quando não reconhece o endereço ou está ocupado.

# ⭐ 5. Sistemas de Tempo Real
*(Aula Teórica 5 — slides 3–6)*

Sistemas de tempo real são sistemas cujo **valor** de uma resposta depende não apenas do resultado correto, mas também de **quando** esse resultado é produzido.  
Em muitas aplicações embebidas, cumprir deadlines é tão importante como a lógica do programa.

---

## 5.1 Tipos de tempo real
*(ver Aula Teórica 5, slide 6)*

### ⭐ Soft Real‑Time  
- Deadlines **podem ser ultrapassados** ocasionalmente.  
- O sistema continua funcional, mas com perda de qualidade.  
- Exemplos: streaming de vídeo, áudio, videojogos.

### ⭐ Firm Real‑Time  
- O resultado **não tem utilidade** se ultrapassar o deadline.  
- No entanto, ultrapassar deadlines **não destrói o sistema**.  
- Exemplos: sistemas de recolha periódica de dados, comunicação em rede com janelas temporais definidas.

### ⭐ Hard Real‑Time  
- Falhar um deadline implica **falha catastrófica**.  
- Tolerância a atrasos é zero → deve ser *provado* que deadlines são sempre cumpridos.  
- Exemplos: ABS, airbags, ventiladores médicos, controlo de voo.

---

## 5.2 Atrasos e jitter
*(ver Aula Teórica 5, slides 3–5)*

Um sistema de tempo real interage com processos físicos (sensores, atuadores).  
Esta interação introduz **atrasos**, que podem comprometer o controlo.

### Tipos de atraso:
- **Atraso de observação (input delay):** tempo entre o instante real e a leitura do sensor.  
- **Atraso de computação:** tempo necessário para o sistema calcular a resposta.  
- **Atraso de atuação (output delay):** tempo desde o comando até ao atuador reagir.

### Jitter
```math
\text{Jitter} = \text{variação não determinística do atraso}
```

- Pode causar instabilidade em sistemas de controlo.  
- Comuns causas: interrupções, multitarefa, latência de comunicação.

### Exemplo típico (simplificado)
Um controlador tenta manter o nível de água num depósito (ver slides sobre controlo de nível).  
- Se o atraso entre medir e atuar for grande → o sistema “dispara” tarde → overshoot.  
- Se além disso existir jitter → comportamento imprevisível → instabilidade.

---

## 5.3 Determinismo e previsibilidade

Num sistema de tempo real, nem sempre importa ser “rápido”, mas sim:

- **previsível**,  
- **determinístico**,  
- com **tempos de pior caso conhecidos (WCET)**.

Determinismo é a base para provar escalonabilidade, essencial para Hard RT.

### WCET — Worst Case Execution Time
Para garantir deadlines, é necessário saber:
- tempo mínimo (BCET)  
- tempo típico  
- **tempo máximo de execução (WCET)**

Sem WCET → impossível garantir hard real‑time.

---

## 5.4 Ciclos de controlo e periodicidade

Muitos sistemas de tempo real são **periódicos**, executando leituras e ações em ciclos:
1. ler sensores  
2. calcular controlo  
3. atuar  
4. esperar até ao próximo período

Se o ciclo não terminar antes do próximo período → falha.

Exemplos:
- controlo PID a 100 Hz  
- leitura de sensores IMU a 1 kHz  
- malhas industriais de 5 ms  

---

## 💬 *Ênfase do professor*
- “**Falhar um deadline em Hard RT não é aceitável.** Não interessa se é uma vez em mil.”  
- “Tempo real não significa *rápido*, significa *a tempo*.”  
- “Sistemas físicos têm atrasos inevitáveis — o truque é torná‑los previsíveis.”  

---

## ⚠️ Armadilhas comuns
- confundir *velocidade* com *tempo real*  
- ignorar jitter na análise de estabilidade  
- assumir que tarefas esporádicas são simples de encaixar no escalonamento  
- não contabilizar interrupções e atrasos de comunicação  
- esquecer que leitura+processamento+atuação **contam para o deadline total**

---

## 📝 Exercícios — Tempo Real (Estilo Exame)

1. **Classifique como Soft, Firm ou Hard RT**, justificando:  
   - ABS  
   - media player  
   - robot cirúrgico  
   - monitor de glicose contínuo  

2. Considere um sistema de controlo de temperatura com período de 50 ms.  
   O sensor demora 12 ms a responder, o cálculo demora 20 ms e o atuador introduz 8 ms de atraso.  
   - O deadline é cumprido?  
   - Onde está o gargalo?  
   - Como mitigarias o atraso?

3. Explique, em termos de estabilidade, o efeito de **jitter elevado** numa malha de controlo.  
   Ilustre com um exemplo realista.

4. Suponha que um sistema recebe dados de um sensor a 100 Hz mas processa a 80 Hz.  
   - O que acontece?  
   - É um problema de tempo real ou de throughput?  
   - Como o corrigir?

5. Dê um exemplo onde atrasos de observação e de atuação combinados causam **overshoot** num sistema de controlo. Explique o mecanismo.

---



### Gabarito resumido — Tempo Real
1. ABS: Hard; media player: Soft; robot cirúrgico: Hard; monitor de glicose contínuo: Firm.  
2. 12 + 20 + 8 = 40 ms → cumpre 50 ms; gargalo no cálculo; mitigar com otimização/MCU mais rápido ou reduzir atrasos de sensor/atuador.  
3. Jitter alto causa atraso variável → instabilidade/oscilações (ex.: controlo de velocidade com amostragem irregular).  
4. Acumula backlog ou perde amostras; é problema de throughput que afeta RT; corrigir com processamento mais rápido, redução da taxa ou filtragem/skip.  
5. Atraso leitura+ação grande gera overshoot (ex.: aquecimento com reação tardia).

# ⭐ 6. Escalonamento
*(Aula Teórica 5 — slides 8–26; Aula Teórica 6 — slides 2–19)*

O escalonamento define **qual tarefa executa em cada instante** no processador.  
O objetivo é garantir que tarefas **completam antes dos deadlines**, respeitando prioridades, períodos, e restrições temporais.

---

## 6.1 Conceitos Fundamentais
*(ver slides 8–11)*

### Parâmetros principais de uma tarefa
- **C** — tempo de computação (worst‑case).  
- **T** — período (tempo entre ativações sucessivas).  
- **D** — deadline (instante limite de conclusão).  
- **aᵢ** — instante de chegada (release time).  

### Utilização
```math
U_i = \frac{C_i}{T_i}
```
A utilização total do processador é:
```math
U = \sum U_i
```

### Escalonamento Praticável
Um escalonamento é **praticável** (feasible schedule) se **todas** as tarefas cumprem **todos os deadlines**, em todas as instâncias.

### Conjunto Escalonável
Um conjunto de tarefas é escalonável se **existe pelo menos um algoritmo** que gera um escalonamento praticável para ele.

### Preemptivo vs Não‑preemptivo
- **Preemptivo** — uma tarefa pode ser interrompida para outra de maior prioridade.  
- **Não‑preemptivo** — uma tarefa, uma vez iniciada, só termina quando acabar.

### Diagramas de Gantt
Ferramenta visual fundamental para análises em exame.

---

## 6.2 Algoritmos Clássicos

### ⭐ FCFS — First Come First Served
*(ver slides 20–21)*

- Não preemptivo.  
- As tarefas são executadas pela **ordem de chegada**.  
- Simples, mas sujeito ao **convoy effect**: uma tarefa longa atrasa todas as outras.

### ⭐ SJF — Shortest Job First
*(ver slides 22–23)*

- Escolhe a tarefa com **menor tempo de execução C**.  
- Minimiza o **tempo médio de espera**.  
- Não preemptivo.

### ⭐ SRTF — Shortest Remaining Time First
*(ver slides 24–25)*

- Versão preemptiva do SJF.  
- Se chega uma tarefa com **C restante menor**, a tarefa atual é interrompida.  
- Efetivamente minimiza o tempo médio de espera em sistemas preemptivos.

### ⭐ Escalonamento por Prioridades
*(Aula Teórica 6 — slides 2–7)*

- Cada tarefa tem prioridade fixa (quanto **menor o número**, maior a prioridade).  
- Preemptivo ou não preemptivo.  
- Problema: **starvation** → algumas tarefas de baixa prioridade podem *nunca* executar.  
- Solução: **aging** (aumenta gradualmente a prioridade de tarefas que esperam demasiado).

### ⭐ Round Robin (RR)
*(Aula Teórica 6 — slides 8–9)*

- Preemptivo.  
- Cada tarefa recebe um **quantum** fixo.  
- Ideal para fairness e sistemas time‑sharing.  
- Quantum demasiado pequeno → overhead.  
- Quantum demasiado grande → baixa responsividade.

---

## 6.3 Algoritmos de Tempo Real

### ⭐ EDF — Earliest Deadline First
*(ver slides 10–11)*

- Prioridade **dinâmica**: tarefa com deadline mais próximo é executada primeiro.  
- Com preempção e tarefas independentes:
```math
U \le 100\% \quad \Rightarrow \quad \text{escalonável}
```
- Potencialmente muitas preempções.  
- Ideal para sistemas aperiódicos/esporádicos.

### ⭐ RM — Rate Monotonic
*(ver slides 12–14)*

- Prioridade **estática**: menor período → maior prioridade.  
- Decisão baseada apenas nos períodos.  
- Condição de Liu & Layland:
```math
U \le n(2^{1/n}-1)
```
Para grandes n:
```math
\lim_{n\to\infty} n(2^{1/n}-1) \approx 0.693
```
Ou seja, RM garante escalonamento abaixo de **69%** de utilização.

### Comparação EDF vs RM
| Propriedade  | EDF           | RM       |
| ------------ | ------------- | -------- |
| prioridade   | dinâmica      | estática |
| limite de U  | 100%          | ~69%     |
| simplicidade | mais complexo | simples  |
| preempções   | muitas        | menos    |

---

## 💬 *Ênfase do professor*
- “EDF é ótimo teoricamente, mas **pode preemptar demasiado**.”  
- “Em RM, **prova‑se escalonabilidade** com base no limite de utilização.”  
- “A escolha do quantum em Round Robin é **crítica**.”  
- “Para Hard RT, mais importante que a média é o **pior caso**.”

---

## ⚠️ Armadilhas comuns
- confundir prioridade fixa (RM, prioridades estáticas) com deadlines (EDF).  
- pensar que RM funciona a 100% de utilização — **não funciona**.  
- esquecer preempções nos cálculos de EDF.  
- escolher quantum demasiado pequeno no RR → overhead destrói desempenho.  
- assumir que FCFS é aceitável em sistemas críticos — raramente é.

---

## 📝 Exercícios — Escalonamento (Estilo Exame)

### 1) SJF / SRTF

Tarefas:  
P1=7, P2=5, P3=1, P4=2, P5=8.  
Desenha os **Gantt** para:

- SJF
- SRTF

Compara os tempos médios de espera.

---

### 2) EDF — Verificação de escalonabilidade
T1: C=6, D=27  
T2: C=7, D=22  
T3: C=5, D=14  
- Calcula U total.  
- Verifica se \( U \le 1 \).  
- Desenha a ordem de execução segundo EDF.

---

### 3) RM — Limite de utilização
Tarefas:  
P1: C=0.5, T=2  
P2: C=2, T=6  
P3: C=1.75, T=6  
- Calcula \( U \).  
- Calcula o limite para n=3.  
- Determina se o conjunto é garantidamente escalonável por RM.

---

### 4) Round Robin
Quantum = 4 ms  
P1=10, P2=4, P3=6  
- Desenha o diagrama de Gantt.  
- Indica o tempo de resposta de cada tarefa.

---

### 5) Prioridades — Starvation
Considere três tarefas:  
- T1 (prioridade 1) chega constantemente  
- T2 (prioridade 2) é periódica  
- T3 (prioridade 5) é rara  
Explique:  
- porque pode ocorrer starvation  
- como aplicar **aging** para evitar esse problema.

---

### 6) Hard Real‑Time — Deadline falhado
Dado um sistema aeroportuário que monitoriza velocidade do vento a cada 200 ms:  
- Se a tarefa levar 250 ms, o que acontece?  
- É Soft, Firm ou Hard RT?  
- Como garantir que o pior caso nunca ultrapassa o deadline?

---


### Gabarito resumido — Escalonamento
1. SJF: P3 → P4 → P2 → P1 → P5. SRTF: igual (assumindo todas as tarefas disponíveis em t=0).  
2. \( U \approx 0.897 \le 1 \); ordem EDF por deadline: T3 → T2 → T1.  
3. \( U = 0.875 \); limite n=3 ≈ 0.779 → não garantido por RM.  
4. Gantt: P1(0–4) → P2(4–8) → P3(8–12) → P1(12–16) → P3(16–18) → P1(18–20). Tempos de resposta: P2=8, P3=18, P1=20.  
5. Starvation quando tarefas de maior prioridade dominam; aging aumenta prioridade das tarefas à espera.  
6. 250 ms > 200 ms → deadline falhado; em Hard RT é inaceitável; garantir via WCET, otimização ou hardware mais rápido.

# ✔ FIM DO GUIA
