// Ponte simples Johnny-Five: lê sensores e aceita comandos de LED via stdin.
const five = require("johnny-five");

const board = new five.Board();

board.on("ready", () => {
  const noiseSensor = new five.Sensor({ pin: "A0", freq: 250 });
  const pressureButton = new five.Button(2);
  const led = new five.Led(13);

  let pressureState = false;

  pressureButton.on("press", () => {
    pressureState = true;
  });

  pressureButton.on("release", () => {
    pressureState = false;
  });

  noiseSensor.on("data", () => {
    const payload = { noise: noiseSensor.value, pressure: pressureState };
    process.stdout.write(`${JSON.stringify(payload)}\n`);
  });

  process.stdin.setEncoding("utf8");
  process.stdin.on("data", (chunk) => {
    chunk
      .split(/\r?\n/)
      .map((piece) => piece.trim())
      .filter(Boolean)
      .forEach((command) => {
        if (command === "LED_ON") {
          led.on();
        } else if (command === "LED_OFF") {
          led.off();
        }
      });
  });

  const shutdown = () => {
    led.off();
    process.exit(0);
  };

  process.on("SIGINT", shutdown);
  process.on("SIGTERM", shutdown);
});

board.on("error", (err) => {
  console.error("Erro na ligação Johnny-Five:", err.message);
  process.exit(1);
});
