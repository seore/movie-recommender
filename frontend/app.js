const API_BASE_URL = "http://localhost:8000";

const movieSelect = document.getElementById("movieSelect");
const recommendBtn = document.getElementById("recommendBtn");
const statusMessage = document.getElementById("statusMessage");
const resultsGrid = document.getElementById("resultsGrid");
const baseMovieTitleEl = document.getElementById("baseMovieTitle");
const topKInput = document.getElementById("topK");
const alphaSlider = document.getElementById("alpha");
const alphaValue = document.getElementById("alphaValue");
const modeTabs = document.querySelectorAll(".mode-tab");

// Modal + Search elements
const modalEl = document.getElementById("movieModal");
const modalCloseBtn = document.getElementById("modalCloseBtn");
const modalPoster = document.getElementById("modalPoster");
const modalTitle = document.getElementById("modalTitle");
const modalMeta = document.getElementById("modalMeta");
const modalOverview = document.getElementById("modalOverview");
const movieSearchInput = document.getElementById("movieSearchInput");
const searchSuggestionsContainer = document.getElementById("searchSuggestions");

const exploreGenreInput = document.getElementById("exploreGenre");
const exploreMinRatingInput = document.getElementById("exploreMinRating");
const exploreGrid = document.getElementById("exploreGrid");
const exploreApplyBtn = document.getElementById("exploreApplyFilters");
const exploreLoadMoreBtn = document.getElementById("exploreLoadMore");

let explorePage = 1;
let exploreHasMore = true;
let searchTimeout = null;
let currentMode = "hybrid";
const favorites = new Set(JSON.parse(localStorage.getItem("favorites") || "[]"));

// --- helpers ---
async function fetchJSON(url, options = {}) {
  const res = await fetch(url, {
    headers: { "Content-Type": "application/json" },
    ...options,
  });

  if (!res.ok) {
    const text = await res.text();
    throw new Error(text || `Request failed: ${res.status}`);
  }
  return res.json();
}

function setStatus(msg, type = "info") {
  statusMessage.textContent = msg || "";
  statusMessage.classList.remove("error", "success");
  if (type === "error") statusMessage.classList.add("error");
  if (type === "success") statusMessage.classList.add("success");
}

function clearResults() {
  resultsGrid.innerHTML = "";
  baseMovieTitleEl.textContent = "";
}

function splitTitleAndYear(fullTitle) {
  const match = fullTitle.match(/^(.*)\s+\((\d{4})\)$/);
  if (!match) return { title: fullTitle, year: null };
  return { title: match[1], year: match[2] };
}

async function fetchPosterUrl(fullTitle) {
  const { title, year } = splitTitleAndYear(fullTitle);
  const params = new URLSearchParams({ title });
  if (year) params.append("year", year);

  try {
    const res = await fetch(
      `${API_BASE_URL}/poster?${params.toString()}`
    );
    if (!res.ok) throw new Error("poster endpoint error");
    const data = await res.json();
    return data.poster_url || null;
  } catch (err) {
    console.warn("Poster fetch failed:", fullTitle, err);
    return null;
  }
}

async function fetchMovieDetails(fullTitle) {
  const { title, year } = splitTitleAndYear(fullTitle);
  const params = new URLSearchParams({ title });
  if (year) params.append("year", year);

  try {
    const res = await fetch(
      `${API_BASE_URL}/movie_details?${params.toString()}`
    );
    if (!res.ok) throw new Error("details endpoint error");
    const data = await res.json();
    return data.details || null;
  } catch (err) {
    console.warn("Details fetch failed:", fullTitle, err);
    return null;
  }
}

function clearSuggestions() {
  searchSuggestionsContainer.innerHTML = "";
}

function renderSearchSuggestions(movies) {
  clearSuggestions();
  if (!movies || movies.length === 0) return;

  const list = document.createElement("div");
  list.className = "search-suggestions-list";

  movies.forEach((m) => {
    const item = document.createElement("div");
    item.className = "search-suggestion-item";
    item.textContent = `${m.title}${m.genres ? ` (${m.genres})` : ""}`;
    item.addEventListener("click", () => {
      // Set dropdown to selected movie
      movieSelect.value = m.movie_id;
      movieSearchInput.value = m.title;
      clearSuggestions();
    });
    list.appendChild(item);
  });

  searchSuggestionsContainer.appendChild(list);
}

async function handleSearchInput() {
  const query = movieSearchInput.value.trim();
  if (!query) {
    clearSuggestions();
    return;
  }
  try {
    const data = await fetchJSON(
      `${API_BASE_URL}/search_movies?` +
        new URLSearchParams({ query, limit: 10 }).toString()
    );
    renderSearchSuggestions(data.movies);
  } catch (err) {
    console.error(err);
  }
}

function clearExplore() {
  exploreGrid.innerHTML = "";
  explorePage = 1;
  exploreHasMore = true;
}

function renderExploreMovies(movies) {
  movies.forEach((m) => {
    const card = document.createElement("div");
    card.className = "movie-card";

    const posterWrapper = document.createElement("div");
    posterWrapper.className = "movie-poster-wrapper skeleton";

    const posterImg = document.createElement("img");
    posterImg.className = "movie-poster";
    posterImg.alt = m.title;
    posterImg.loading = "lazy";
    posterWrapper.appendChild(posterImg);

    const content = document.createElement("div");
    content.className = "movie-content";

    const title = document.createElement("h3");
    title.textContent = m.title;

    const meta = document.createElement("div");
    meta.className = "score";
    const ratingText =
      m.vote_average != null ? `Rating: ${m.vote_average.toFixed(1)}` : "Rating: N/A";
    meta.textContent = `${ratingText}${m.genres ? ` • ${m.genres}` : ""}`;

    content.appendChild(title);
    content.appendChild(meta);

    card.appendChild(posterWrapper);
    card.appendChild(content);
    exploreGrid.appendChild(card);

    // Click opens modal
    card.addEventListener("click", async () => {
      await openMovieModal(m.title);
    });

    // load poster
    fetchPosterUrl(m.title).then((url) => {
      posterWrapper.classList.remove("skeleton");
      if (url) posterImg.src = url;
    });
  });
}

async function loadExplorePage() {
  if (!exploreHasMore) return;

  const genre = exploreGenreInput.value.trim();
  const minRatingValue = exploreMinRatingInput.value.trim();
  const params = new URLSearchParams({
    page: explorePage.toString(),
    page_size: "20",
  });
  if (genre) params.append("genre", genre);
  if (minRatingValue) params.append("min_rating", minRatingValue);

  try {
    const data = await fetchJSON(
      `${API_BASE_URL}/explore_movies?${params.toString()}`
    );
    renderExploreMovies(data.movies);
    const shown = explorePage * 20;
    if (shown >= data.total || data.movies.length === 0) {
      exploreHasMore = false;
      exploreLoadMoreBtn.disabled = true;
      exploreLoadMoreBtn.textContent = "No more movies";
    } else {
      explorePage += 1;
    }
  } catch (err) {
    console.error(err);
  }
}

// --- favorites ---
function saveFavorites() {
  localStorage.setItem("favorites", JSON.stringify([...favorites]));
}

function toggleFavorite(movieId) {
  if (favorites.has(movieId)) {
    favorites.delete(movieId);
  } else {
    favorites.add(movieId);
  }
  saveFavorites();
}

// --- UI rendering ---
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
    opt.textContent = `${m.title}${m.genres ? ` (${m.genres})` : ""}`;
    movieSelect.appendChild(opt);
  });
}

function renderRecommendations(baseTitle, recommendations) {
  clearResults();

  if (!recommendations || recommendations.length === 0) {
    baseMovieTitleEl.textContent = `No recommendations found for "${baseTitle}".`;
    return;
  }

  const modeLabel =
    currentMode === "content"
      ? "content-based similarity"
      : currentMode === "collab"
      ? "collaborative filtering"
      : "a hybrid of content and collaborative signals";

  baseMovieTitleEl.textContent = `Because you selected "${baseTitle}", you might also enjoy (using ${modeLabel}):`;

  recommendations.forEach((rec) => {
    const card = document.createElement("div");
    card.className = "movie-card";

    const posterWrapper = document.createElement("div");
    posterWrapper.className = "movie-poster-wrapper skeleton";

    const posterImg = document.createElement("img");
    posterImg.className = "movie-poster";
    posterImg.alt = rec.title;
    posterImg.loading = "lazy";
    posterWrapper.appendChild(posterImg);

    const content = document.createElement("div");
    content.className = "movie-content";

    const title = document.createElement("h3");
    title.textContent = rec.title;

    const score = document.createElement("div");
    score.className = "score";
    score.textContent = `Similarity score: ${rec.score.toFixed(3)}`;

    content.appendChild(title);
    content.appendChild(score);

    // Favorite button
    const favBtn = document.createElement("button");
    favBtn.className = "favorite-btn";
    favBtn.innerHTML = "★";
    if (favorites.has(rec.movie_id)) {
      favBtn.classList.add("active");
    }

    favBtn.addEventListener("click", (e) => {
      e.stopPropagation();
      toggleFavorite(rec.movie_id);
      favBtn.classList.toggle("active");
    });

    // Open modal on card click
    card.addEventListener("click", async () => {
      await openMovieModal(rec.title);
    });

    card.appendChild(posterWrapper);
    card.appendChild(content);
    card.appendChild(favBtn);
    resultsGrid.appendChild(card);

    // async load poster
    fetchPosterUrl(rec.title).then((url) => {
      posterWrapper.classList.remove("skeleton");
      if (url) {
        posterImg.src = url;
      }
    });
  });
}

// --- Modals ---
async function openMovieModal(fullTitle) {
  modalTitle.textContent = fullTitle;
  modalMeta.textContent = "Loading details…";
  modalOverview.textContent = "";
  modalPoster.src = "";
  modalEl.classList.remove("hidden");

  const details = await fetchMovieDetails(fullTitle);
  if (!details) {
    modalMeta.textContent = "No additional details found.";
    return;
  }

  modalTitle.textContent = details.title || fullTitle;
  const metaParts = [];
  if (details.release_date) metaParts.push(details.release_date);
  if (details.vote_average)
    metaParts.push(`Rating: ${details.vote_average.toFixed(1)} (${details.vote_count} votes)`);
  modalMeta.textContent = metaParts.join(" • ");
  modalOverview.textContent = details.overview || "No overview available.";
  if (details.poster_url) {
    modalPoster.src = details.poster_url;
  }
}

function closeModal() {
  modalEl.classList.add("hidden");
}

// --- data flows ---
async function loadMovies() {
  try {
    setStatus("Loading movies…");
    const data = await fetchJSON(`${API_BASE_URL}/movies?limit=30`);
    renderMoviesDropdown(data.movies);
    setStatus("Movies loaded. Select one to get started.", "success");
  } catch (err) {
    console.error(err);
    renderMoviesDropdown([]);
    setStatus("Failed to load movies. Is the backend running?", "error");
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
    mode: currentMode,
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
    setStatus(err.message || "Failed to fetch recommendations.", "error");
  } finally {
    recommendBtn.disabled = false;
  }
}

// --- init ---
function initUI() {
  alphaValue.textContent = alphaSlider.value;
  alphaSlider.addEventListener("input", () => {
    alphaValue.textContent = alphaSlider.value;
  });

  recommendBtn.addEventListener("click", (e) => {
    e.preventDefault();
    getRecommendations();
  });

  modeTabs.forEach((tab) => {
    tab.addEventListener("click", () => {
      modeTabs.forEach((t) => t.classList.remove("active"));
      tab.classList.add("active");
      currentMode = tab.dataset.mode;
    });
  });

  // modal closing
  modalCloseBtn.addEventListener("click", closeModal);
  modalEl.addEventListener("click", (e) => {
    if (e.target.classList.contains("modal-backdrop")) {
      closeModal();
    }
  });

  // movie search
  movieSearchInput.addEventListener("input", () => {
    clearSuggestions();
    if (searchTimeout) clearTimeout(searchTimeout);
    searchTimeout = setTimeout(handleSearchInput, 250);
  });

  document.addEventListener("click", (e) => {
    if (!searchSuggestionsContainer.contains(e.target) 
      && e.target !== movieSearchInput) {
    clearSuggestions();
      }
  });

  // explore options
    exploreApplyBtn.addEventListener("click", (e) => {
    e.preventDefault();
    clearExplore();
    exploreLoadMoreBtn.disabled = false;
    exploreLoadMoreBtn.textContent = "Load more";
    loadExplorePage();
  });

  exploreLoadMoreBtn.addEventListener("click", (e) => {
    e.preventDefault();
    loadExplorePage();
  });
}

async function initApp() {
  initUI();
  await loadMovies();
  loadExplorePage();
}

document.addEventListener("DOMContentLoaded", initApp);
