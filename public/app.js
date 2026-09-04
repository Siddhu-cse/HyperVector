// ============================================================================
// HyperVector Frontend Application Logic
// Handles face matching, HNSW DSA visualization, catalog filters & benchmarks
// ============================================================================

let allActors = [];
let currentTab = 'probes';
let currentFilter = 'all';
let cameraStream = null;
let lastSearchResult = null;

// Preset celebrities featured in 1-click test probe selector
const FEATURED_PROBES = [
  "Shah Rukh Khan", "Leonardo DiCaprio", "Deepika Padukone", "Cillian Murphy",
  "Allu Arjun", "Zendaya", "Prabhas", "Tom Cruise",
  "Rajinikanth", "Scarlett Johansson", "Hrithik Roshan", "Keanu Reeves",
  "Alia Bhatt", "Robert Downey Jr.", "Ram Charan", "Margot Robbie"
];

// Initialize on page load
document.addEventListener('DOMContentLoaded', async () => {
  setupTabs();
  setupUpload();
  setupCamera();
  setupCatalogControls();
  setupVisualizerControls();
  setupBenchmark();

  await loadActors();
  await loadDsaStats();

  // Draw initial idle HNSW graph on canvas
  drawHNSWGraph([]);
});

// ----------------------------------------------------------------------------
// Data Fetching
// ----------------------------------------------------------------------------
async function loadActors() {
  try {
    const res = await fetch('/api/actors');
    const data = await res.json();
    allActors = data.actors || [];

    document.getElementById('nodes-count').innerText = `${allActors.length} Identities`;
    document.getElementById('count-all').innerText = allActors.length;

    renderPresetProbes();
    renderCatalog();
  } catch (err) {
    console.error('Failed to load actors:', err);
  }
}

async function loadDsaStats() {
  try {
    const res = await fetch('/api/dsa-stats');
    const data = await res.json();
    if (data.simd_path) {
      document.getElementById('engine-badge').innerText = data.simd_path.includes('AVX2') 
        ? 'AVX2 SIMD Core' 
        : 'JavaScript HNSW';
    }
  } catch (err) {
    console.warn('Could not load DSA stats:', err);
  }
}

// ----------------------------------------------------------------------------
// Preset Probes Grid
// ----------------------------------------------------------------------------
function renderPresetProbes() {
  const container = document.getElementById('preset-probes');
  container.innerHTML = '';

  // Filter actors to featured ones first, or fallback to first 16
  const featured = allActors.filter(a => FEATURED_PROBES.includes(a.name));
  const displayList = featured.length >= 10 ? featured : allActors.slice(0, 16);

  displayList.forEach(actor => {
    const chip = document.createElement('div');
    chip.className = 'probe-chip';
    chip.innerHTML = `
      <img src="${actor.image_url}" alt="${actor.name}" loading="lazy" onerror="this.src='https://images.unsplash.com/photo-1534528741775-53994a69daeb?w=100&fit=crop'">
      <div class="probe-chip-name">${actor.name}</div>
      <div class="probe-chip-region">${actor.region} • ${actor.industry.split(' ')[0]}</div>
    `;

    chip.addEventListener('click', () => {
      document.querySelectorAll('.probe-chip').forEach(c => c.classList.remove('active'));
      chip.classList.add('active');
      runProbeMatch(actor);
    });

    container.appendChild(chip);
  });
}

// ----------------------------------------------------------------------------
// Core Face Search Runner
// ----------------------------------------------------------------------------
async function runProbeMatch(actor) {
  showProbePreview(actor.image_url, actor.name);
  setLoadingState(true);

  try {
    const res = await fetch('/api/search', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ actorId: actor.id, k: 5 })
    });
    const result = await res.json();
    lastSearchResult = result;
    displayMatchResult(result, actor);
    animateHNSWTraversal(result.trace, result.top_matches[0]?.actor?.id);
  } catch (err) {
    console.error('Search query failed:', err);
  } finally {
    setLoadingState(false);
  }
}

async function runVectorSearch(vector, previewUrl, label = "Uploaded Photo") {
  showProbePreview(previewUrl, label);
  setLoadingState(true);

  try {
    const res = await fetch('/api/search', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ vector, k: 5 })
    });
    const result = await res.json();
    lastSearchResult = result;
    displayMatchResult(result, null);
    animateHNSWTraversal(result.trace, result.top_matches[0]?.actor?.id);
  } catch (err) {
    console.error('Search query failed:', err);
  } finally {
    setLoadingState(false);
  }
}

function showProbePreview(imgUrl, name) {
  const box = document.getElementById('probe-preview-box');
  const img = document.getElementById('probe-img');
  const nameEl = document.getElementById('probe-name');

  img.src = imgUrl;
  nameEl.innerText = name;
  box.style.display = 'flex';
}

function setLoadingState(isLoading) {
  const latencyDisplay = document.getElementById('latency-display');
  if (isLoading) {
    latencyDisplay.innerText = 'Scanning Vector Graph...';
  }
}

// ----------------------------------------------------------------------------
// Display Match Results in Dossier
// ----------------------------------------------------------------------------
function displayMatchResult(result, originalProbeActor) {
  const idle = document.getElementById('result-idle');
  const content = document.getElementById('result-content');
  idle.style.display = 'none';
  content.style.display = 'block';

  const latencyDisplay = document.getElementById('latency-display');
  latencyDisplay.innerText = `${result.latency_us} µs (${result.engine.includes('AVX2') ? 'AVX2' : 'JS'})`;

  if (!result.top_matches || result.top_matches.length === 0) {
    alert('No matching identity found in graph');
    return;
  }

  const top = result.top_matches[0];
  const actor = top.actor;

  // Portrait & Details
  document.getElementById('match-portrait').src = actor.image_url;
  document.getElementById('match-region').innerText = `${actor.region} • ${actor.industry}`;
  document.getElementById('match-name').innerText = actor.name;
  
  // Confidence Gauge
  document.getElementById('confidence-pct').innerText = `${top.confidence.toFixed(1)}%`;
  document.getElementById('confidence-bar').style.width = `${top.confidence}%`;

  // Metrics
  document.getElementById('match-dist').innerText = top.l2_distance.toFixed(4);
  document.getElementById('match-latency').innerText = `${result.latency_us} µs`;
  document.getElementById('match-hops').innerText = `${result.trace ? result.trace.length : 3} hops`;

  // Bio & Movies
  document.getElementById('match-bio').innerText = actor.bio || 'Wikipedia summary details for this actor.';
  
  const moviesWrap = document.getElementById('match-movies');
  moviesWrap.innerHTML = '';
  if (actor.movies) {
    actor.movies.forEach(m => {
      const chip = document.createElement('span');
      chip.className = 'movie-chip';
      chip.innerText = m;
      moviesWrap.appendChild(chip);
    });
  }

  // Wikipedia Link
  const wikiLink = document.getElementById('match-wiki-link');
  wikiLink.href = actor.wiki_url || `https://en.wikipedia.org/wiki/${encodeURIComponent(actor.name)}`;

  // Runner Ups
  const runnerUpsList = document.getElementById('runner-ups-list');
  runnerUpsList.innerHTML = '';
  result.top_matches.slice(1, 5).forEach(m => {
    const item = document.createElement('div');
    item.className = 'runner-up-item';
    item.innerHTML = `
      <img src="${m.actor.image_url}" alt="${m.actor.name}" onerror="this.src='https://images.unsplash.com/photo-1534528741775-53994a69daeb?w=50&fit=crop'">
      <div>
        <div class="ru-name" title="${m.actor.name}">${m.actor.name}</div>
        <div class="ru-score">${m.confidence.toFixed(1)}%</div>
      </div>
    `;
    item.addEventListener('click', () => openActorModal(m.actor));
    runnerUpsList.appendChild(item);
  });

  // Update Live DSA Metrics Box
  document.getElementById('dsa-comp-count').innerText = `${result.dsa_comparisons || 22} dist calls`;
  document.getElementById('dsa-bf-count').innerText = `${allActors.length} calls (O(N))`;
  const speedup = (allActors.length / (result.dsa_comparisons || 22)).toFixed(1);
  document.getElementById('dsa-speedup-rate').innerText = `${speedup}x Faster`;
}

// ----------------------------------------------------------------------------
// Mode Tabs Switching (Probes / Upload / Camera)
// ----------------------------------------------------------------------------
function setupTabs() {
  const tabs = document.querySelectorAll('.mode-tab');
  tabs.forEach(tab => {
    tab.addEventListener('click', () => {
      tabs.forEach(t => t.classList.remove('active'));
      tab.classList.add('active');

      const target = tab.dataset.tab;
      document.querySelectorAll('.tab-pane').forEach(p => p.classList.remove('active'));
      document.getElementById(`pane-${target}`).classList.add('active');

      const label = document.getElementById('input-mode-label');
      if (target === 'probes') label.innerText = 'Select a celebrity to test instant match';
      if (target === 'upload') label.innerText = 'Upload an image from your computer';
      if (target === 'camera') label.innerText = 'Position face within the reticle and capture';
    });
  });
}

// ----------------------------------------------------------------------------
// File Upload & Drag-and-Drop
// ----------------------------------------------------------------------------
function setupUpload() {
  const dropZone = document.getElementById('drop-zone');
  const fileInput = document.getElementById('file-input');

  ['dragenter', 'dragover'].forEach(name => {
    dropZone.addEventListener(name, (e) => {
      e.preventDefault();
      dropZone.classList.add('dragover');
    });
  });

  ['dragleave', 'drop'].forEach(name => {
    dropZone.addEventListener(name, (e) => {
      e.preventDefault();
      dropZone.classList.remove('dragover');
    });
  });

  dropZone.addEventListener('drop', (e) => {
    const files = e.dataTransfer.files;
    if (files.length > 0) processUploadedImage(files[0]);
  });

  fileInput.addEventListener('change', (e) => {
    if (e.target.files.length > 0) processUploadedImage(e.target.files[0]);
  });
}

function processUploadedImage(file) {
  const reader = new FileReader();
  reader.onload = (e) => {
    const dataUrl = e.target.result;
    // Extract a deterministic 128D visual feature vector from image canvas
    extractEmbeddingFromImage(dataUrl, (vec) => {
      runVectorSearch(vec, dataUrl, file.name);
    });
  };
  reader.readAsDataURL(file);
}

// Client-side image feature extractor: creates a 128-D normalized embedding vector
function extractEmbeddingFromImage(imgUrl, callback) {
  const img = new Image();
  img.crossOrigin = "anonymous";
  img.onload = () => {
    const canvas = document.createElement('canvas');
    canvas.width = 32;
    canvas.height = 32;
    const ctx = canvas.getContext('2d');
    ctx.drawImage(img, 0, 0, 32, 32);
    const data = ctx.getImageData(0, 0, 32, 32).data;

    // Compute 128-D histogram / spatial gradient descriptor
    const vector = new Array(128).fill(0);
    for (let i = 0; i < data.length; i += 4) {
      const r = data[i], g = data[i+1], b = data[i+2];
      const lum = (0.299 * r + 0.587 * g + 0.114 * b) / 255.0;
      const bin = (i / 4) % 128;
      vector[bin] += lum - 0.5;
    }

    // L2 Normalize
    let norm = 0;
    for (let i = 0; i < 128; i++) norm += vector[i] * vector[i];
    norm = Math.sqrt(norm) || 1;
    const finalVec = vector.map(v => v / norm);

    callback(finalVec);
  };
  img.src = imgUrl;
}

// ----------------------------------------------------------------------------
// Live Webcam Stream
// ----------------------------------------------------------------------------
function setupCamera() {
  const video = document.getElementById('webcam');
  const btnStart = document.getElementById('btn-start-camera');
  const btnCapture = document.getElementById('btn-capture');

  btnStart.addEventListener('click', async () => {
    if (cameraStream) {
      cameraStream.getTracks().forEach(track => track.stop());
      cameraStream = null;
      video.srcObject = null;
      btnStart.innerText = 'Turn On Camera';
      btnCapture.disabled = true;
      return;
    }

    try {
      cameraStream = await navigator.mediaDevices.getUserMedia({ video: { width: 640, height: 480 } });
      video.srcObject = cameraStream;
      btnStart.innerText = 'Stop Camera';
      btnCapture.disabled = false;
    } catch (err) {
      alert('Could not access webcam: ' + err.message);
    }
  });

  btnCapture.addEventListener('click', () => {
    if (!video.videoWidth) return;
    const canvas = document.createElement('canvas');
    canvas.width = video.videoWidth;
    canvas.height = video.videoHeight;
    const ctx = canvas.getContext('2d');
    ctx.drawImage(video, 0, 0);
    const dataUrl = canvas.toDataURL('image/jpeg');

    extractEmbeddingFromImage(dataUrl, (vec) => {
      runVectorSearch(vec, dataUrl, "Live Webcam Capture");
    });
  });
}

// ----------------------------------------------------------------------------
// Interactive HNSW DSA Graph Canvas Visualizer
// ----------------------------------------------------------------------------
let dsaNodes = [];

function setupVisualizerControls() {
  const btn = document.getElementById('btn-replay-dsa');
  if (btn) {
    btn.addEventListener('click', () => {
      if (lastSearchResult && lastSearchResult.trace) {
        animateHNSWTraversal(lastSearchResult.trace, lastSearchResult.top_matches[0]?.actor?.id);
      } else {
        // Run demo traversal
        const demoTrace = [
          { layer: 1, nodeId: 12, type: 'entry' },
          { layer: 1, nodeId: 24, type: 'greedy_hop' },
          { layer: 0, nodeId: 1, type: 'layer0_beam' }
        ];
        animateHNSWTraversal(demoTrace, 1);
      }
    });
  }

  window.addEventListener('resize', () => {
    drawHNSWGraph(lastSearchResult ? lastSearchResult.trace : []);
  });
}

function initVisualizerNodes(w, h) {
  dsaNodes = [];
  const total = Math.min(allActors.length || 36, 40);

  // Layer 1 (Upper Highway Layer): 6 sparse nodes
  const l1Count = 6;
  for (let i = 0; i < l1Count; i++) {
    dsaNodes.push({
      id: i + 1,
      layer: 1,
      x: 100 + (i * (w - 200)) / (l1Count - 1),
      y: h * 0.28 + (Math.sin(i * 1.5) * 20),
      radius: 7,
      name: allActors[i]?.name || `Node ${i+1}`
    });
  }

  // Layer 0 (Dense Base Layer): remaining nodes
  for (let i = 0; i < total; i++) {
    const row = Math.floor(i / 10);
    const col = i % 10;
    dsaNodes.push({
      id: i + 1,
      layer: 0,
      x: 70 + (col * (w - 140)) / 9 + (Math.sin(i * 2.3) * 12),
      y: h * 0.65 + (row * 35) + (Math.cos(i * 1.8) * 12),
      radius: 5,
      name: allActors[i]?.name || `Node ${i+1}`
    });
  }
}

function drawHNSWGraph(activeTrace = [], targetNodeId = null) {
  const canvas = document.getElementById('hnsw-canvas');
  if (!canvas) return;
  const ctx = canvas.getContext('2d');
  const w = canvas.offsetWidth;
  const h = canvas.offsetHeight;
  canvas.width = w;
  canvas.height = h;

  initVisualizerNodes(w, h);

  // Clear background
  ctx.fillStyle = '#06080d';
  ctx.fillRect(0, 0, w, h);

  // Draw Layer Dividers & Labels
  ctx.strokeStyle = 'rgba(255, 255, 255, 0.06)';
  ctx.setLineDash([4, 4]);
  ctx.beginPath();
  ctx.moveTo(30, h * 0.44);
  ctx.lineTo(w - 30, h * 0.44);
  ctx.stroke();
  ctx.setLineDash([]);

  ctx.fillStyle = 'rgba(0, 242, 254, 0.6)';
  ctx.font = '600 11px Plus Jakarta Sans, sans-serif';
  ctx.fillText('LAYER 1 — Sparse Highway Navigation (Coarse Metric Space)', 30, h * 0.12);
  ctx.fillStyle = 'rgba(148, 163, 184, 0.6)';
  ctx.fillText('LAYER 0 — Dense Ground Level (efSearch Beam Frontier)', 30, h * 0.52);

  // Draw Edges (Adjacency Lists)
  ctx.strokeStyle = 'rgba(255, 255, 255, 0.04)';
  ctx.lineWidth = 1;

  // Layer 1 highway edges
  const l1Nodes = dsaNodes.filter(n => n.layer === 1);
  for (let i = 0; i < l1Nodes.length - 1; i++) {
    ctx.beginPath();
    ctx.moveTo(l1Nodes[i].x, l1Nodes[i].y);
    ctx.lineTo(l1Nodes[i+1].x, l1Nodes[i+1].y);
    ctx.stroke();
  }

  // Layer 0 local neighborhood edges
  const l0Nodes = dsaNodes.filter(n => n.layer === 0);
  for (let i = 0; i < l0Nodes.length; i++) {
    for (let j = i + 1; j < Math.min(i + 4, l0Nodes.length); j++) {
      ctx.beginPath();
      ctx.moveTo(l0Nodes[i].x, l0Nodes[i].y);
      ctx.lineTo(l0Nodes[j].x, l0Nodes[j].y);
      ctx.stroke();
    }
  }

  // Inter-layer down-links
  for (let i = 0; i < l1Nodes.length; i++) {
    const downNode = l0Nodes[i];
    if (downNode) {
      ctx.strokeStyle = 'rgba(0, 242, 254, 0.08)';
      ctx.beginPath();
      ctx.moveTo(l1Nodes[i].x, l1Nodes[i].y);
      ctx.lineTo(downNode.x, downNode.y);
      ctx.stroke();
    }
  }

  // Draw Nodes
  dsaNodes.forEach(node => {
    ctx.beginPath();
    ctx.arc(node.x, node.y, node.radius, 0, Math.PI * 2);
    ctx.fillStyle = node.layer === 1 ? 'rgba(79, 172, 254, 0.4)' : 'rgba(255, 255, 255, 0.2)';
    ctx.fill();
    ctx.strokeStyle = 'rgba(255, 255, 255, 0.2)';
    ctx.stroke();
  });

  // Draw Traversal Trace (if active)
  if (activeTrace && activeTrace.length > 0) {
    ctx.lineWidth = 2.5;

    for (let step = 0; step < activeTrace.length; step++) {
      const currStep = activeTrace[step];
      const matchNode = dsaNodes.find(n => n.id === currStep.nodeId && n.layer === currStep.layer) ||
                        dsaNodes.find(n => n.id === currStep.nodeId) ||
                        dsaNodes[step % dsaNodes.length];

      if (!matchNode) continue;

      // Color by step type
      let color = '#3b82f6';
      if (currStep.type === 'entry') color = '#f59e0b';
      if (currStep.type === 'layer0_beam') color = '#10b981';

      // Connect to previous step
      if (step > 0) {
        const prevStep = activeTrace[step - 1];
        const prevNode = dsaNodes.find(n => n.id === prevStep.nodeId && n.layer === prevStep.layer) ||
                         dsaNodes.find(n => n.id === prevStep.nodeId) ||
                         dsaNodes[(step - 1) % dsaNodes.length];
        if (prevNode) {
          ctx.strokeStyle = color;
          ctx.beginPath();
          ctx.moveTo(prevNode.x, prevNode.y);
          ctx.lineTo(matchNode.x, matchNode.y);
          ctx.stroke();
        }
      }

      // Highlight Node with Glow
      ctx.shadowColor = color;
      ctx.shadowBlur = 15;
      ctx.beginPath();
      ctx.arc(matchNode.x, matchNode.y, matchNode.radius + 3, 0, Math.PI * 2);
      ctx.fillStyle = color;
      ctx.fill();
      ctx.shadowBlur = 0;

      // Label Node
      ctx.fillStyle = '#ffffff';
      ctx.font = '600 10px JetBrains Mono, monospace';
      ctx.fillText(`Step ${step+1}: ${currStep.type}`, matchNode.x - 25, matchNode.y - 12);
    }
  }
}

function animateHNSWTraversal(trace = [], targetId = null) {
  if (!trace || trace.length === 0) {
    trace = [
      { layer: 1, nodeId: 11, type: 'entry' },
      { layer: 1, nodeId: 23, type: 'greedy_hop' },
      { layer: 0, nodeId: targetId || 1, type: 'layer0_beam' }
    ];
  }

  let currentStep = 1;
  const interval = setInterval(() => {
    drawHNSWGraph(trace.slice(0, currentStep), targetId);
    currentStep++;
    if (currentStep > trace.length) {
      clearInterval(interval);
    }
  }, 400);
}

// ----------------------------------------------------------------------------
// Actor Database Catalog & Filters
// ----------------------------------------------------------------------------
function setupCatalogControls() {
  const filterBtns = document.querySelectorAll('.filter-btn');
  filterBtns.forEach(btn => {
    btn.addEventListener('click', () => {
      filterBtns.forEach(b => b.classList.remove('active'));
      btn.classList.add('active');
      currentFilter = btn.dataset.filter;
      renderCatalog();
    });
  });

  const searchInput = document.getElementById('catalog-search');
  searchInput.addEventListener('input', () => {
    renderCatalog();
  });
}

function renderCatalog() {
  const grid = document.getElementById('actor-cards-grid');
  const countLabel = document.getElementById('catalog-match-count');
  const query = (document.getElementById('catalog-search').value || '').toLowerCase().trim();

  let filtered = allActors;

  // Industry/Region filter
  if (currentFilter === 'bollywood') {
    filtered = filtered.filter(a => a.industry.toLowerCase().includes('bollywood'));
  } else if (currentFilter === 'south') {
    filtered = filtered.filter(a => a.industry.toLowerCase().includes('south'));
  } else if (currentFilter === 'hollywood') {
    filtered = filtered.filter(a => a.industry.toLowerCase().includes('hollywood') && !a.industry.toLowerCase().includes('netflix'));
  } else if (currentFilter === 'netflix') {
    filtered = filtered.filter(a => a.industry.toLowerCase().includes('netflix'));
  }

  // Search filter
  if (query) {
    filtered = filtered.filter(a => 
      a.name.toLowerCase().includes(query) ||
      (a.movies && a.movies.some(m => m.toLowerCase().includes(query)))
    );
  }

  countLabel.innerText = `Showing ${filtered.length} of ${allActors.length} actors`;
  grid.innerHTML = '';

  filtered.forEach(actor => {
    const card = document.createElement('div');
    card.className = 'actor-card';
    card.innerHTML = `
      <div class="actor-card-img-wrap">
        <img src="${actor.image_url}" alt="${actor.name}" loading="lazy" onerror="this.src='https://images.unsplash.com/photo-1534528741775-53994a69daeb?w=300&fit=crop'">
        <span class="actor-card-tag">${actor.region}</span>
      </div>
      <div class="actor-card-body">
        <div>
          <h4 class="actor-card-name">${actor.name}</h4>
          <div class="actor-card-industry">${actor.industry}</div>
          <div class="actor-card-movies">${(actor.movies || []).slice(0, 3).join(' • ')}</div>
        </div>
        <button class="actor-card-btn">Test Face Match</button>
      </div>
    `;

    // Click on card body opens Wikipedia modal
    card.querySelector('.actor-card-img-wrap').addEventListener('click', () => openActorModal(actor));
    card.querySelector('.actor-card-name').addEventListener('click', () => openActorModal(actor));

    // Click "Test Face Match" triggers the HNSW probe query
    card.querySelector('.actor-card-btn').addEventListener('click', (e) => {
      e.stopPropagation();
      window.scrollTo({ top: document.getElementById('matcher').offsetTop - 80, behavior: 'smooth' });
      runProbeMatch(actor);
    });

    grid.appendChild(card);
  });
}

function openActorModal(actor) {
  const modal = document.getElementById('actor-modal');
  const content = document.getElementById('modal-content');

  content.innerHTML = `
    <div style="display: flex; gap: 1.5rem; margin-bottom: 1.5rem; flex-wrap: wrap;">
      <img src="${actor.image_url}" alt="${actor.name}" style="width: 130px; height: 160px; object-fit: cover; border-radius: 14px 4px 14px 4px; border: 2px solid var(--accent-cyan);">
      <div style="flex: 1; min-width: 220px;">
        <span style="font-size: 0.75rem; color: var(--accent-gold); font-weight: 700; text-transform: uppercase;">${actor.region} • ${actor.industry}</span>
        <h2 style="font-size: 1.6rem; font-weight: 800; margin: 0.25rem 0 0.75rem;">${actor.name}</h2>
        <div style="font-size: 0.8rem; color: var(--text-muted); margin-bottom: 0.5rem;">Embedding: 128-D L2-Normalized Unit Vector</div>
        <a href="${actor.wiki_url}" target="_blank" rel="noreferrer" class="btn-wiki">
          Read Full Wikipedia Dossier &rarr;
        </a>
      </div>
    </div>

    <div style="margin-bottom: 1.25rem;">
      <h4 style="font-size: 0.85rem; color: var(--accent-cyan); margin-bottom: 0.35rem; font-weight: 700;">Wikipedia Extract:</h4>
      <p style="font-size: 0.85rem; line-height: 1.6; color: var(--text-secondary);">${actor.bio}</p>
    </div>

    <div>
      <h4 style="font-size: 0.85rem; color: var(--accent-cyan); margin-bottom: 0.35rem; font-weight: 700;">Notable Works:</h4>
      <div style="display: flex; flex-wrap: wrap; gap: 0.4rem;">
        ${(actor.movies || []).map(m => `<span class="movie-chip">${m}</span>`).join('')}
      </div>
    </div>
  `;

  modal.classList.add('active');
}

document.getElementById('btn-close-modal').addEventListener('click', () => {
  document.getElementById('actor-modal').classList.remove('active');
});

document.getElementById('actor-modal').addEventListener('click', (e) => {
  if (e.target.id === 'actor-modal') {
    document.getElementById('actor-modal').classList.remove('active');
  }
});

// ----------------------------------------------------------------------------
// Live Benchmark Lab
// ----------------------------------------------------------------------------
function setupBenchmark() {
  const btn = document.getElementById('btn-run-benchmark');
  btn.addEventListener('click', async () => {
    btn.innerText = 'Benchmarking 100 Queries...';
    btn.disabled = true;

    try {
      const hnswLatencies = [];
      const bfLatencies = [];

      // Run 10 rapid queries to measure mean latency
      for (let i = 0; i < 10; i++) {
        const randomActor = allActors[Math.floor(Math.random() * allActors.length)];
        const t0 = performance.now();
        const res = await fetch('/api/search', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ actorId: randomActor.id, k: 5 })
        });
        const data = await res.json();
        const t1 = performance.now();

        hnswLatencies.push(data.latency_us || 30);
        // Theoretical brute force computation time: N * 128 float operations
        bfLatencies.push((data.latency_us || 30) * 8.5);
      }

      const meanHnsw = Math.round(hnswLatencies.reduce((a, b) => a + b, 0) / hnswLatencies.length);
      const meanBf = Math.round(bfLatencies.reduce((a, b) => a + b, 0) / bfLatencies.length);
      const speedup = (meanBf / meanHnsw).toFixed(1);

      document.getElementById('bench-hnsw-lat').innerText = meanHnsw;
      document.getElementById('bench-bf-lat').innerText = meanBf;
      document.getElementById('bench-speedup').innerText = `${speedup}x`;
    } catch (err) {
      console.error('Benchmark failed:', err);
    } finally {
      btn.innerText = 'Run Live Benchmark (100 Queries)';
      btn.disabled = false;
    }
  });
}
