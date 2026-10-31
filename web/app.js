// HNRS web GUI: everything the page does. web_app.py does the recognising and training.
const $ = (selector) => document.querySelector(selector);
let last = null;         // {image, expected} of the last input, so a settings change can re-run it
let lastResult = null;   // {result, image} of the last recognition, so the picture can be redrawn

// ---------- talking to web_app.py ----------

async function api(path, body) {
  const options = body === undefined ? {} :
    {method: "POST", headers: {"Content-Type": "application/json"}, body: JSON.stringify(body)};
  const response = await fetch(path, options);
  const data = await response.json();
  if (!response.ok) throw new Error(data.error || response.statusText);
  return data;
}

function settings() {
  return {model: $("#model").value, threshold: $("#threshold").value, segmentation: $("#segmentation").value};
}

function setStatus(text, isError = false) {
  $("#status").textContent = text;
  $("#status").classList.toggle("error", isError);
}

async function loadModels(select) {
  const data = await api("/api/models");
  const keep = select || $("#model").value;
  $("#model").innerHTML = data.models.map((name) => `<option>${name}</option>`).join("");
  $("#model").value = data.models.includes(keep) ? keep : (data.default || "");
  if (!$("#train-model").options.length) {
    $("#train-model").innerHTML = data.trainable.map((name) => `<option>${name}</option>`).join("");
    $("#train-model").value = "cnn";
  }
  setStatus(data.models.length ? `${data.models.length} trained model(s) found`
                               : "No trained models yet: train one in the Train tab", !data.models.length);
}

async function recognise(image, expected = null) {
  last = {image, expected};
  setStatus("Recognising...");
  try {
    showResult(await api("/api/recognise", {image, expected, ...settings()}), image);
    setStatus("Done");
  } catch (error) {
    setStatus(error.message, true);
  }
}

// ---------- the Result panel ----------

function showResult(result, image) {
  lastResult = {result, image};
  $("#number").textContent = (result.number || "–") + (result.answer !== null ? ` = ${result.answer}` : "");

  const detail = $("#detail");
  detail.replaceChildren();
  if (result.correct !== null) {
    const verdict = document.createElement("span");
    verdict.className = result.correct ? "good" : "bad";
    verdict.textContent = result.correct ? "Correct" : "Wrong";
    detail.append(verdict, ` (expected ${result.expected}) · `);
  } else {
   detail.append(result.number ? `${result.labels.length} character(s) · ` : "No digits found · ");
  }
  detail.append(`${result.model} · ${result.ms} ms`);

  drawPicture();
  $("#digits").replaceChildren(...result.digits.map((src, i) => {
    const cell = document.createElement("div");
    cell.className = "digit " + (result.confidences[i] >= 0.8 ? "sure" : "unsure");
    cell.innerHTML = `<img src="${src}" alt="">${result.labels[i]}  ${Math.round(result.confidences[i] * 100)}%`;
    return cell;
  }));
}

function drawPicture() {
  if (!lastResult) return;
  const {result, image} = lastResult;
  $("#picture").src = $("#show-binary").checked ? result.binary : image;
  // The boxes are in the original picture's pixels; a viewBox of the same size scales them with it
  const svg = $("#boxes");
  svg.setAttribute("viewBox", `0 0 ${result.width} ${result.height}`);
  const size = Math.max(result.width, result.height) / 22;
  svg.innerHTML = result.boxes.map(([x, y, w, h], i) =>
    `<rect x="${x}" y="${y}" width="${w}" height="${h}"/>` +
    `<text x="${x}" y="${y - size * 0.25}" font-size="${size}">${result.labels[i]}</text>`).join("");
}

// ---------- tabs and settings ----------

document.querySelectorAll(".tab").forEach((tab) => tab.addEventListener("click", () => {
  document.querySelectorAll(".tab, .panel").forEach((el) => el.classList.remove("active"));
  tab.classList.add("active");
  $(`#${tab.dataset.tab}`).classList.add("active");
}));

// Changing a setting re-runs the last input, so methods are easy to compare
["#model", "#threshold", "#segmentation"].forEach((id) => $(id).addEventListener("change", () => {
  if (last) recognise(last.image, last.expected);
}));
$("#show-binary").addEventListener("change", drawPicture);

// ---------- tab 1: draw ----------

const canvas = $("#canvas");
const ctx = canvas.getContext("2d");
let drawing = false, lastPoint = null, hasInk = false;

function clearCanvas() {
  // A canvas is transparent by default and OpenCV turns transparent pixels black,
  // so always paint it white first
  ctx.fillStyle = "white";
  ctx.fillRect(0, 0, canvas.width, canvas.height);
  hasInk = false;
}

function point(event) {
  const box = canvas.getBoundingClientRect();     // the canvas may be shown smaller than 620 x 260
  return [(event.clientX - box.left) * canvas.width / box.width,
          (event.clientY - box.top) * canvas.height / box.height];
}

function stroke([x, y]) {
  ctx.strokeStyle = "black";
  ctx.lineWidth = $("#pen").value;
  ctx.lineCap = ctx.lineJoin = "round";
  ctx.beginPath();
  ctx.moveTo(...lastPoint);
  ctx.lineTo(x, y);
  ctx.stroke();
  lastPoint = [x, y];
  hasInk = true;
}

canvas.addEventListener("pointerdown", (event) => {
  drawing = true;
  canvas.setPointerCapture(event.pointerId);
  lastPoint = point(event);
  stroke(lastPoint);
});
canvas.addEventListener("pointermove", (event) => { if (drawing) stroke(point(event)); });
canvas.addEventListener("pointerup", () => {
  if (!drawing) return;
  drawing = false;
  if ($("#live").checked) recogniseDrawing();
});
canvas.addEventListener("pointercancel", () => { drawing = false; });

function recogniseDrawing() {
  if (!hasInk) return setStatus("Draw something first");
  // Copy the drawing onto a white canvas 20 px bigger on every side: segmentation ignores
  // marks touching the picture's edge (in photos that's the table), so strokes mustn't touch it
  const padded = document.createElement("canvas");
  padded.width = canvas.width + 40;
  padded.height = canvas.height + 40;
  const paddedCtx = padded.getContext("2d");
  paddedCtx.fillStyle = "white";
  paddedCtx.fillRect(0, 0, padded.width, padded.height);
  paddedCtx.drawImage(canvas, 20, 20);
  recognise(padded.toDataURL("image/png"));
}

$("#recognise-drawing").addEventListener("click", recogniseDrawing);
$("#clear").addEventListener("click", clearCanvas);

// ---------- tab 2: photo ----------

function openPhoto(file) {
  if (!file) return;
  const reader = new FileReader();
  reader.onload = () => {
    $("#photo-preview").src = reader.result;
    const stem = file.name.replace(/\.[^.]+$/, "");
    recognise(reader.result, /^\d+$/.test(stem) ? stem : null);
  };
  reader.readAsDataURL(file);
}

$("#file").addEventListener("change", (event) => {
  openPhoto(event.target.files[0]);
  event.target.value = "";                        // so choosing the same file again still works
});
const drop = $("#drop");
drop.addEventListener("dragover", (event) => { event.preventDefault(); drop.classList.add("over"); });
drop.addEventListener("dragleave", () => drop.classList.remove("over"));
drop.addEventListener("drop", (event) => {
  event.preventDefault();
  drop.classList.remove("over");
  openPhoto(event.dataTransfer.files[0]);
});

// ---------- tab 3: compose ----------

$("#random").addEventListener("click", () => {
  $("#target").value = String(10 + Math.floor(Math.random() * 999990));
});

$("#compose-button").addEventListener("click", async () => {
  const number = $("#target").value.trim();
  setStatus("Composing...");
  try {
    const source = document.querySelector("input[name=source]:checked").value;
    const result = await api("/api/compose", {number, source, folder: $("#folder").value, ...settings()});
    $("#compose-preview").src = result.image;
    last = {image: result.image, expected: number};      // re-runs use this picture, not new digits
    showResult(result, result.image);
    setStatus("Done");
  } catch (error) {
    setStatus(error.message, true);
  }
});

// ---------- tab 4: train ----------

$("#start-training").addEventListener("click", async () => {
  const name = $("#train-model").value;
  const trained = [...$("#model").options].map((option) => option.value);
  if (trained.includes(name) && !confirm(`checkpoints/${name}.pt already exists. Train again and replace it?`)) return;
  $("#train-log").textContent = "Starting...";      // clear the last run's log straight away
  $("#progress").value = 0;
  try {
    await api("/api/train", {model: name, epochs: $("#epochs").value, batch_size: $("#batch").value, lr: $("#lr").value});
    watchTraining(true);
  } catch (error) {
    setStatus(error.message, true);
  }
});

async function watchTraining(selectWhenDone = false) {
  const job = await api("/api/train/status");
  if (!job.name) return;                          // nothing trained since the server started
  $("#start-training").disabled = job.running;
  $("#progress").value = job.rows.length / job.epochs;
  let text = `Training ${job.name} for ${job.epochs} epochs. The first epoch can take a while without a GPU.\n` +
    job.rows.map((r) => `epoch ${r.epoch}/${job.epochs}   train acc ${r.train_acc}%   test acc ${r.test_acc}%   (${r.seconds}s)`).join("\n");
  if (job.running) {
    setTimeout(() => watchTraining(selectWhenDone).catch((error) => setStatus(error.message, true)), 1000);
  } else if (job.error) {
    text += `\n\nTraining failed: ${job.error}`;
  } else {
    text += `\n\nDone. Final test accuracy ${job.accuracy.toFixed(2)}%. It's now in the Model list.`;
    await loadModels(selectWhenDone ? job.name : undefined);
  }
  $("#train-log").textContent = text;
}

// ---------- start ----------

clearCanvas();
loadModels().catch((error) => setStatus(error.message, true));
watchTraining().catch(() => {});                  // shows a run that was going before a page reload