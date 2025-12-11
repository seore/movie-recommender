// Change this if your backend runs elsewhere
const API_BASE_URL = "http://localhost:8000";

const movieSelect = document.getElementById("movieSelect");
const recommendBtn = document.getElementById("recommendBtn");
const statusMessage = document.getElementById("statusMessage");
const resultsGrid = document.getElementById("resultsGrid");
const baseMovieTitleEl = document.getElementById("baseMovieTitle");
const topKInput = document.getElementById("topK");
const alphaSlider = document.getElementById("alpha");
const alphaValue = document.getElementById("alphaValue");

async function fetchJSON(url, options = {}) {
  const res = await fetch(url, {
    headers: { "Content-Type": "application/json" },
    ...options,
  });

  if (!res.ok) {
    const detail = await res.text();
    throw new Error(detail || `Request failed with status ${res.status}`);
  }

  return res.json();
}

function setStatus(message, type = "info") {
  statusMessage.textContent = message || "";
  statusMessage.classList.remove("error", "success");

  if (type === "error") statusMessage.classList.add("error");
  if (type === "success") statusMessage.classList.add("success");
}

function clearResults() {
  resultsGrid.innerHTML = "";
  baseMovieTitleEl.textContent = "";
}

function renderMoviesDropdown(movies) {
  movieSelect.innerHTML = "";

  if (!movies || movies.length === 0) {
    const opt = document.createElement("option");
    opt.value = "";
    opt.textContent = "No movies available";
    movieSelect.appendChild(opt);
    movieSelect.disabled = true;
    return;
  }

  const placeholder = document.createElement("option");
  placeholder.value = "";
  placeholder.textContent = "Select a movie…";
  movieSelect.appendChild(placeholder);

  movies.forEach((m) => {
    const opt = document.createElement("option");
    opt.value = m.movie_id;
    opt.textContent = `${m.title} ${m.genres ? `(${m.genres})` : ""}`;
    movieSelect.appendChild(opt);
  });
}

function renderRecommendations(baseTitle, recommendations) {
  clearResults();

  if (!recommendations || recommendations.length === 0) {
    baseMovieTitleEl.textContent = `No recommendations found for "${baseTitle}".`;
    return;
  }

  baseMovieTitleEl.textContent = `Because you selected "${baseTitle}", you might also enjoy:`;

  recommendations.forEach((rec) => {
    const card = document.createElement("div");
    card.className = "movie-card";

    const title = document.createElement("h3");
    title.textContent = rec.title;

    const score = document.createElement("div");
    score.className = "score";
    score.textContent = `Similarity score: ${rec.score.toFixed(3)}`;

    card.appendChild(title);
    card.appendChild(score);
    resultsGrid.appendChild(card);
  });
}

async function loadMovies() {
  try {
    setStatus("Loading movies…");
    const data = await fetchJSON(`${API_BASE_URL}/movies?limit=30`);
    renderMoviesDropdown(data.movies);
    setStatus("Movies loaded. Select one to get started.", "success");
  } catch (err) {
    console.error(err);
    renderMoviesDropdown([]);
    setStatus("Failed to load movies. Check that the backend is running.", "error");
  }
}

async function getRecommendations() {
  clearResults();

  const selectedMovieId = movieSelect.value;
  if (!selectedMovieId) {
    setStatus("Please select a movie first.", "error");
    return;
  }

  const topK = parseInt(topKInput.value, 10) || 10;
  let alpha = parseFloat(alphaSlider.value);
  if (Number.isNaN(alpha)) alpha = 0.5;

  const payload = {
    movie_id: Number(selectedMovieId),
    top_k: topK,
    alpha,
  };

  try {
    setStatus("Fetching recommendations…");
    recommendBtn.disabled = true;

    const data = await fetchJSON(`${API_BASE_URL}/recommend`, {
      method: "POST",
      body: JSON.stringify(payload),
    });

    renderRecommendations(data.base_title, data.recommendations);
    setStatus(`Found ${data.recommendations.length} recommendations.`, "success");
  } catch (err) {
    console.error(err);
    setStatus(
      err.message || "Something went wrong while fetching recommendations.",
      "error"
    );
  } finally {
    recommendBtn.disabled = false;
  }
}

function initUI() {
  // Alpha slider display
  alphaValue.textContent = alphaSlider.value;
  alphaSlider.addEventListener("input", () => {
    alphaValue.textContent = alphaSlider.value;
  });

  recommendBtn.addEventListener("click", (e) => {
    e.preventDefault();
    getRecommendations();
  });
}

async function initApp() {
  initUI();
  await loadMovies();
}

// Init when DOM is ready
document.addEventListener("DOMContentLoaded", () => {
  initApp();
});
