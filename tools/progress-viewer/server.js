const http = require("http");
const fs = require("fs");
const path = require("path");

const PORT = 4321;
const LOG_PATH = process.argv[2] || path.resolve(__dirname, "..", "..", "engine", "run.log");
const INDEX_PATH = path.join(__dirname, "index.html");

const server = http.createServer((req, res) => {
  if (req.url === "/log") {
    fs.readFile(LOG_PATH, "utf8", (err, data) => {
      if (err) {
        res.writeHead(200, { "Content-Type": "text/plain" });
        res.end("");
        return;
      }
      res.writeHead(200, { "Content-Type": "text/plain" });
      res.end(data);
    });
    return;
  }

  fs.readFile(INDEX_PATH, "utf8", (err, data) => {
    if (err) {
      res.writeHead(500);
      res.end("index.html missing");
      return;
    }
    res.writeHead(200, { "Content-Type": "text/html" });
    res.end(data);
  });
});

server.listen(PORT, () => {
  console.log(`Progress viewer watching ${LOG_PATH}`);
  console.log(`Open http://localhost:${PORT}`);
});
