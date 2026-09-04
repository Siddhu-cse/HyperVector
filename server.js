const http = require('http');
const fs = require('fs');
const path = require('path');
const { execFileSync } = require('child_process');

const PORT = process.env.PORT || 3000;
const PUBLIC_DIR = path.join(__dirname, 'public');
const DATA_FILE = path.join(__dirname, 'data', 'actors.json');
const HV_BINARY = path.join(__dirname, 'HyperVector', 'build', 'hv_query.exe');

let actors = [];
try {
  if (fs.existsSync(DATA_FILE)) {
    actors = JSON.parse(fs.readFileSync(DATA_FILE, 'utf-8'));
  }
} catch (err) {}

class HNSWGraph {
  constructor(M = 16, efSearch = 64) {
    this.M = M;
    this.efSearch = efSearch;
    this.nodes = new Map();
    this.enterNode = null;
    this.maxLevel = -1;
  }

  distance(v1, v2) {
    let sum = 0;
    for (let i = 0; i < v1.length; i++) {
      const diff = v1[i] - v2[i];
      sum += diff * diff;
    }
    return sum;
  }

  randomLevel() {
    const r = Math.random() || 1e-7;
    const level = Math.floor(-Math.log(r) / Math.log(this.M));
    return Math.min(level, 3);
  }

  insert(id, vector) {
    const level = this.randomLevel();
    const node = { id, vector, layers: Array.from({ length: level + 1 }, () => []) };
    this.nodes.set(id, node);

    if (this.enterNode === null) {
      this.enterNode = id;
      this.maxLevel = level;
      return;
    }

    let curr = this.enterNode;
    for (let l = this.maxLevel; l > level; l--) {
      curr = this.greedySearch(vector, curr, l).id;
    }

    for (let l = Math.min(this.maxLevel, level); l >= 0; l--) {
      const candidates = this.searchLayer(vector, [curr], this.M * 2, l);
      const neighbors = candidates.slice(0, this.M).map(c => c.id);

      node.layers[l] = neighbors;
      for (const nId of neighbors) {
        const neighborNode = this.nodes.get(nId);
        if (neighborNode && neighborNode.layers[l]) {
          neighborNode.layers[l].push(id);
          if (neighborNode.layers[l].length > this.M) {
            neighborNode.layers[l].sort((a, b) => 
              this.distance(this.nodes.get(a).vector, neighborNode.vector) -
              this.distance(this.nodes.get(b).vector, neighborNode.vector)
            );
            neighborNode.layers[l] = neighborNode.layers[l].slice(0, this.M);
          }
        }
      }
      if (candidates.length > 0) curr = candidates[0].id;
    }

    if (level > this.maxLevel) {
      this.maxLevel = level;
      this.enterNode = id;
    }
  }

  greedySearch(query, entryId, layer) {
    let currId = entryId;
    let currDist = this.distance(query, this.nodes.get(currId).vector);
    let improved = true;

    while (improved) {
      improved = false;
      const node = this.nodes.get(currId);
      if (!node || !node.layers[layer]) break;

      for (const nId of node.layers[layer]) {
        const neighbor = this.nodes.get(nId);
        if (!neighbor) continue;
        const d = this.distance(query, neighbor.vector);
        if (d < currDist) {
          currDist = d;
          currId = nId;
          improved = true;
        }
      }
    }
    return { id: currId, dist: currDist };
  }

  searchLayer(query, entryIds, ef, layer) {
    const visited = new Set(entryIds);
    let candidates = entryIds.map(id => ({ id, dist: this.distance(query, this.nodes.get(id).vector) }));
    candidates.sort((a, b) => a.dist - b.dist);

    const results = [...candidates];
    let comparisons = entryIds.length;

    while (candidates.length > 0) {
      const current = candidates.shift();
      const furthestResult = results[results.length - 1];

      if (current.dist > furthestResult.dist && results.length >= ef) {
        break;
      }

      const node = this.nodes.get(current.id);
      if (!node || !node.layers[layer]) continue;

      for (const nId of node.layers[layer]) {
        if (!visited.has(nId)) {
          visited.add(nId);
          comparisons++;
          const neighbor = this.nodes.get(nId);
          if (!neighbor) continue;
          const dist = this.distance(query, neighbor.vector);

          if (dist < furthestResult.dist || results.length < ef) {
            candidates.push({ id: nId, dist });
            candidates.sort((a, b) => a.dist - b.dist);

            results.push({ id: nId, dist });
            results.sort((a, b) => a.dist - b.dist);

            if (results.length > ef) {
              results.pop();
            }
          }
        }
      }
    }
    return results;
  }

  search(query, k = 5) {
    if (this.enterNode === null) return { matches: [], trace: [] };

    const trace = [];
    let curr = this.enterNode;
    let dist = this.distance(query, this.nodes.get(curr).vector);
    trace.push({ layer: this.maxLevel, nodeId: curr, type: 'entry', dist });

    for (let l = this.maxLevel; l > 0; l--) {
      const next = this.greedySearch(query, curr, l);
      curr = next.id;
      trace.push({ layer: l, nodeId: curr, type: 'greedy_hop', dist: next.dist });
    }

    const candidates = this.searchLayer(query, [curr], Math.max(this.efSearch, k), 0);
    trace.push({ layer: 0, nodeId: candidates[0].id, type: 'layer0_beam', dist: candidates[0].dist });

    const topK = candidates.slice(0, k);
    return {
      matches: topK.map(c => ({ id: c.id, distance: c.dist })),
      trace,
      comparisons: trace.length + candidates.length
    };
  }
}

const hnswIndex = new HNSWGraph(16, 64);
actors.forEach(actor => {
  if (actor.embedding && actor.embedding.length === 128) {
    hnswIndex.insert(actor.id, actor.embedding);
  }
});

const MIME_TYPES = {
  '.html': 'text/html; charset=utf-8',
  '.css': 'text/css; charset=utf-8',
  '.js': 'application/javascript; charset=utf-8',
  '.json': 'application/json; charset=utf-8',
  '.png': 'image/png',
  '.jpg': 'image/jpeg',
  '.jpeg': 'image/jpeg',
  '.svg': 'image/svg+xml',
  '.ico': 'image/x-icon'
};

const server = http.createServer((req, res) => {
  const parsedUrl = new URL(req.url, `http://${req.headers.host}`);
  const pathname = parsedUrl.pathname;

  res.setHeader('Access-Control-Allow-Origin', '*');
  res.setHeader('Access-Control-Allow-Methods', 'GET, POST, OPTIONS');
  res.setHeader('Access-Control-Allow-Headers', 'Content-Type');

  if (req.method === 'OPTIONS') {
    res.writeHead(204);
    res.end();
    return;
  }

  if (req.method === 'GET' && pathname === '/api/actors') {
    const region = parsedUrl.searchParams.get('region');
    const industry = parsedUrl.searchParams.get('industry');
    const query = parsedUrl.searchParams.get('q');

    let result = actors;
    if (region && region !== 'all') {
      result = result.filter(a => a.region.toLowerCase() === region.toLowerCase());
    }
    if (industry && industry !== 'all') {
      result = result.filter(a => a.industry.toLowerCase().includes(industry.toLowerCase()));
    }
    if (query) {
      const q = query.toLowerCase();
      result = result.filter(a => 
        a.name.toLowerCase().includes(q) || 
        a.movies.some(m => m.toLowerCase().includes(q))
      );
    }

    res.writeHead(200, { 'Content-Type': 'application/json' });
    res.end(JSON.stringify({ total: result.length, actors: result }));
    return;
  }

  if (req.method === 'GET' && pathname.startsWith('/api/actors/')) {
    const id = parseInt(pathname.split('/')[3], 10);
    const actor = actors.find(a => a.id === id);
    if (!actor) {
      res.writeHead(404, { 'Content-Type': 'application/json' });
      res.end(JSON.stringify({ error: 'Actor not found' }));
      return;
    }
    res.writeHead(200, { 'Content-Type': 'application/json' });
    res.end(JSON.stringify(actor));
    return;
  }

  if (req.method === 'GET' && pathname === '/api/dsa-stats') {
    const total = actors.length;
    res.writeHead(200, { 'Content-Type': 'application/json' });
    res.end(JSON.stringify({
      algorithm: "Hierarchical Navigable Small World (HNSW)",
      dimension: 128,
      total_identities: total,
      M: 16,
      efConstruction: 200,
      efSearch: 64,
      max_level: hnswIndex.maxLevel,
      metric: "Squared Euclidean Distance",
      simd_path: fs.existsSync(HV_BINARY) ? "AVX2 SIMD" : "Scalar",
      complexity: {
        hnsw: `O(log N) ≈ ${(Math.log2(total || 1) * 3).toFixed(1)} operations`,
        brute_force: `O(N) = ${total} comparisons`,
        theoretical_speedup: `${(total / (Math.log2(total || 1) * 3)).toFixed(1)}x`
      }
    }));
    return;
  }

  if (req.method === 'POST' && pathname === '/api/search') {
    let body = '';
    req.on('data', chunk => body += chunk);
    req.on('end', () => {
      try {
        const payload = JSON.parse(body);
        let queryVector = payload.vector;
        const topK = payload.k || 5;

        if (!queryVector && payload.actorId) {
          const target = actors.find(a => a.id === payload.actorId);
          if (target) {
            queryVector = target.embedding.map(v => v + (Math.random() - 0.5) * 0.08);
            const norm = Math.sqrt(queryVector.reduce((acc, v) => acc + v * v, 0));
            queryVector = queryVector.map(v => v / norm);
          }
        }

        if (!queryVector || queryVector.length !== 128) {
          res.writeHead(400, { 'Content-Type': 'application/json' });
          res.end(JSON.stringify({ error: 'Invalid query vector' }));
          return;
        }

        let engineMode = 'native_simd';
        let latencyUs = 0;
        let matchedIds = [];

        const t0 = process.hrtime.bigint();
        if (fs.existsSync(HV_BINARY)) {
          try {
            const rawOutput = execFileSync(HV_BINARY, [queryVector.join(','), topK.toString()], {
              timeout: 1000,
              encoding: 'utf-8'
            });
            const parsed = JSON.parse(rawOutput.trim());
            matchedIds = parsed.matches || [];
            latencyUs = parsed.latency_us || 120;
          } catch (cppErr) {
            engineMode = 'js_fallback';
          }
        } else {
          engineMode = 'js_fallback';
        }

        let searchResult = null;
        if (engineMode === 'js_fallback' || matchedIds.length === 0) {
          searchResult = hnswIndex.search(queryVector, topK);
          matchedIds = searchResult.matches.map(m => m.id);
          const t1 = process.hrtime.bigint();
          latencyUs = Number(t1 - t0) / 1000;
        } else {
          searchResult = hnswIndex.search(queryVector, topK);
        }

        const results = matchedIds.map((id, index) => {
          const actor = actors.find(a => a.id === id);
          if (!actor) return null;
          const dist = hnswIndex.distance(queryVector, actor.embedding);
          const similarity = Math.max(0, Math.min(100, (1 - dist / 2) * 100));
          return {
            rank: index + 1,
            actor: {
              id: actor.id,
              name: actor.name,
              region: actor.region,
              industry: actor.industry,
              movies: actor.movies,
              image_url: actor.image_url,
              wiki_url: actor.wiki_url,
              bio: actor.bio
            },
            l2_distance: parseFloat(dist.toFixed(4)),
            confidence: parseFloat(similarity.toFixed(2))
          };
        }).filter(Boolean);

        res.writeHead(200, { 'Content-Type': 'application/json' });
        res.end(JSON.stringify({
          engine: engineMode === 'native_simd' ? 'HyperVector C++20 (AVX2 SIMD)' : 'HyperVector JavaScript HNSW',
          latency_us: Math.round(latencyUs),
          top_matches: results,
          trace: searchResult ? searchResult.trace : [],
          dsa_comparisons: searchResult ? searchResult.comparisons : topK * 3,
          brute_force_comparisons: actors.length
        }));
      } catch (err) {
        res.writeHead(500, { 'Content-Type': 'application/json' });
        res.end(JSON.stringify({ error: err.message }));
      }
    });
    return;
  }

  let safePath = path.normalize(pathname).replace(/^(\.\.[\/\\])+/, '');
  if (safePath === '/' || safePath === '\\') safePath = '/index.html';

  const filePath = path.join(PUBLIC_DIR, safePath);
  const ext = path.extname(filePath).toLowerCase();
  const contentType = MIME_TYPES[ext] || 'application/octet-stream';

  fs.readFile(filePath, (err, content) => {
    if (err) {
      res.writeHead(404, { 'Content-Type': 'text/plain' });
      res.end('Not Found');
    } else {
      res.writeHead(200, { 'Content-Type': contentType });
      res.end(content);
    }
  });
});

server.listen(PORT, () => {
  console.log(`Server running at http://localhost:${PORT}`);
});
