const MAX_FILE_BYTES = Number(document.body.dataset.maxFileBytes);

const healthElement = document.getElementById("health");
const healthStatus = document.getElementById("health-status");
const healthVersion = document.getElementById("health-version");
const form = document.getElementById("prediction-form");
const textOption = document.getElementById("text-option");
const textSourceState = document.getElementById("text-source-state");
const fileSourceState = document.getElementById("file-source-state");
const subjectInput = document.getElementById("subject");
const textInput = document.getElementById("email-text");
const textError = document.getElementById("text-error");
const fileInput = document.getElementById("eml-file");
const fileError = document.getElementById("file-error");
const dropZone = document.getElementById("drop-zone");
const fileSummary = document.getElementById("file-summary");
const fileName = document.getElementById("file-name");
const fileSize = document.getElementById("file-size");
const removeFileButton = document.getElementById("remove-file");
const submitButton = document.getElementById("submit-button");
const submitLabel = document.getElementById("submit-label");
const clearButton = document.getElementById("clear-button");
const result = document.getElementById("result");
const requestError = document.getElementById("request-error");

let selectedFile = null;
let dragDepth = 0;

function formatBytes(bytes) {
  if (bytes < 1024) return `${bytes} bytes`;
  if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(1)} KB`;
  return `${(bytes / (1024 * 1024)).toFixed(2)} MB`;
}

function setFieldError(element, message = "") {
  element.textContent = message;
  element.hidden = !message;
}

function showRequestError(message) {
  result.hidden = true;
  requestError.textContent = message;
  requestError.hidden = false;
  requestError.scrollIntoView({ behavior: "smooth", block: "nearest" });
}

function clearErrors() {
  setFieldError(textError);
  setFieldError(fileError);
  requestError.hidden = true;
  requestError.textContent = "";
  textInput.removeAttribute("aria-invalid");
  dropZone.removeAttribute("aria-invalid");
}

function isEmlFile(file) {
  return file && (
    file.name.toLowerCase().endsWith(".eml")
    || file.type === "message/rfc822"
  );
}

function updateSourceState() {
  const hasFile = selectedFile !== null;
  textOption.classList.toggle("is-inactive", hasFile);
  textOption.setAttribute("aria-disabled", String(hasFile));
  textSourceState.textContent = hasFile ? "Saved, not active" : "Ready to use";
  fileSourceState.textContent = hasFile ? "Active source" : "No file selected";
  dropZone.classList.toggle("has-file", hasFile);
}

function setSelectedFile(file) {
  clearErrors();
  if (!isEmlFile(file)) {
    clearSelectedFile();
    setFieldError(fileError, "Choose an RFC 822 email file with the .eml extension.");
    dropZone.setAttribute("aria-invalid", "true");
    return false;
  }
  if (file.size > MAX_FILE_BYTES) {
    clearSelectedFile();
    setFieldError(
      fileError,
      `This file exceeds the ${formatBytes(MAX_FILE_BYTES)} request limit.`,
    );
    dropZone.setAttribute("aria-invalid", "true");
    return false;
  }

  selectedFile = file;
  fileName.textContent = file.name;
  fileSize.textContent = formatBytes(file.size);
  fileSummary.hidden = false;
  dropZone.querySelector(".upload-icon").hidden = true;
  dropZone.querySelector(".drop-copy").hidden = true;
  dropZone.querySelector(".file-button").hidden = true;
  updateSourceState();
  return true;
}

function clearSelectedFile({ focus = false } = {}) {
  selectedFile = null;
  fileInput.value = "";
  fileName.textContent = "";
  fileSize.textContent = "";
  fileSummary.hidden = true;
  dropZone.querySelector(".upload-icon").hidden = false;
  dropZone.querySelector(".drop-copy").hidden = false;
  dropZone.querySelector(".file-button").hidden = false;
  dropZone.classList.remove("is-dragging");
  dragDepth = 0;
  updateSourceState();
  if (focus) dropZone.focus();
}

function showResult(prediction) {
  clearErrors();
  const probability = Math.max(0, Math.min(1, Number(prediction.probability)));
  const threshold = Math.max(0, Math.min(1, Number(prediction.threshold)));
  result.classList.toggle("is-high-risk", prediction.is_phishing);
  result.classList.toggle("is-low-risk", !prediction.is_phishing);
  document.getElementById("result-label").textContent = prediction.is_phishing
    ? "Elevated phishing risk"
    : "No elevated risk detected";
  document.getElementById("result-decision").textContent = prediction.is_phishing
    ? "Review carefully"
    : "Remain cautious";
  document.getElementById("result-probability").textContent =
    `${(probability * 100).toFixed(2)}% phishing probability`;
  document.getElementById("result-classification").textContent = prediction.label;
  document.getElementById("result-threshold").textContent =
    `${(threshold * 100).toFixed(2)}%`;
  document.getElementById("result-version").textContent = prediction.model_version;
  document.getElementById("risk-fill").style.width = `${probability * 100}%`;
  document.getElementById("threshold-marker").style.left = `${threshold * 100}%`;
  document.getElementById("threshold-label").textContent =
    `Threshold ${(threshold * 100).toFixed(1)}%`;
  result.hidden = false;
  result.scrollIntoView({ behavior: "smooth", block: "nearest" });
}

async function loadHealth() {
  try {
    const response = await fetch("/health");
    const payload = await response.json();
    if (!response.ok) throw new Error(payload.error || "Model unavailable");
    healthStatus.textContent = "Available";
    healthVersion.textContent = `Version ${payload.model_version}`;
    healthElement.classList.add("is-available");
  } catch (error) {
    healthStatus.textContent = "Unavailable";
    healthVersion.textContent = error.message;
    healthElement.classList.add("is-unavailable");
  }
}

fileInput.addEventListener("change", () => {
  if (fileInput.files.length) setSelectedFile(fileInput.files[0]);
});

removeFileButton.addEventListener("click", (event) => {
  event.stopPropagation();
  clearErrors();
  clearSelectedFile({ focus: true });
});

dropZone.addEventListener("click", (event) => {
  if (selectedFile || event.target.closest(".remove-file")) return;
  if (event.target !== fileInput && !event.target.closest(".file-button")) fileInput.click();
});

dropZone.addEventListener("keydown", (event) => {
  if (!selectedFile && (event.key === "Enter" || event.key === " ")) {
    event.preventDefault();
    fileInput.click();
  }
});

dropZone.addEventListener("dragenter", (event) => {
  event.preventDefault();
  dragDepth += 1;
  dropZone.classList.add("is-dragging");
});

dropZone.addEventListener("dragover", (event) => {
  event.preventDefault();
  event.dataTransfer.dropEffect = "copy";
});

dropZone.addEventListener("dragleave", (event) => {
  event.preventDefault();
  dragDepth = Math.max(0, dragDepth - 1);
  if (dragDepth === 0) dropZone.classList.remove("is-dragging");
});

dropZone.addEventListener("drop", (event) => {
  event.preventDefault();
  dragDepth = 0;
  dropZone.classList.remove("is-dragging");
  const [file] = event.dataTransfer.files;
  if (file) setSelectedFile(file);
});

document.addEventListener("dragover", (event) => event.preventDefault());
document.addEventListener("drop", (event) => {
  if (!dropZone.contains(event.target)) event.preventDefault();
});

clearButton.addEventListener("click", () => {
  form.reset();
  clearSelectedFile();
  result.hidden = true;
  clearErrors();
  subjectInput.focus();
});

form.addEventListener("submit", async (event) => {
  event.preventDefault();
  clearErrors();
  const subject = subjectInput.value.trim();
  const text = textInput.value.trim();
  if (!selectedFile && !text) {
    setFieldError(textError, "Paste the email body, or choose an EML file below.");
    textInput.setAttribute("aria-invalid", "true");
    textInput.focus();
    return;
  }

  submitButton.disabled = true;
  submitButton.setAttribute("aria-busy", "true");
  submitLabel.textContent = "Reviewing email...";
  try {
    const requestOptions = selectedFile
      ? {
          method: "POST",
          headers: { "Content-Type": "message/rfc822" },
          body: await selectedFile.arrayBuffer(),
        }
      : {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ subject, text }),
        };
    const response = await fetch("/api/v1/predict", requestOptions);
    const payload = await response.json();
    if (!response.ok) {
      throw new Error(payload.error || `Prediction failed with status ${response.status}.`);
    }
    showResult(payload);
  } catch (error) {
    showRequestError(error.message);
  } finally {
    submitButton.disabled = false;
    submitButton.removeAttribute("aria-busy");
    submitLabel.textContent = "Review email";
  }
});

updateSourceState();
loadHealth();
