// ---------------------------------------------------------------------------
// Tab switching
// ---------------------------------------------------------------------------
document.querySelectorAll(".tab").forEach(tab => {
    tab.addEventListener("click", () => {
        document.querySelectorAll(".tab").forEach(t => t.classList.remove("active"));
        document.querySelectorAll(".tab-content").forEach(c => c.classList.remove("active"));
        tab.classList.add("active");
        document.getElementById("tab-" + tab.dataset.tab).classList.add("active");
    });
});

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------
function show(el) { el.classList.remove("hidden"); }
function hide(el) { el.classList.add("hidden"); }

function renderIngestResult(container, data) {
    container.innerHTML = `
        <div class="result-card">
            <h4>Ingestion Complete</h4>
            <div class="stat"><span class="stat-value">${data.entities}</span><br><span class="stat-label">Entities</span></div>
            <div class="stat"><span class="stat-value">${data.relations}</span><br><span class="stat-label">Relations</span></div>
            <div class="stat"><span class="stat-value">${data.chunks}</span><br><span class="stat-label">Chunks</span></div>
        </div>`;
    show(container);
    refreshDocuments();
}

async function refreshDocuments() {
    const res = await fetch("/api/documents");
    const data = await res.json();
    const list = document.getElementById("doc-list");
    if (!data.documents || data.documents.length === 0) {
        list.innerHTML = '<li class="empty">No documents yet</li>';
        return;
    }
    list.innerHTML = data.documents.map(d => {
        const src = d.source.startsWith("seed:") ? "seed" : "text_input";
        const label = d.source.startsWith("seed:") ? "Seed data" : "Text input";
        return `<li>
            <span class="source-tag ${src}">${src}</span>
            ${label}
            <br><small style="color:var(--text-dim)">${d.entities} entities, ${d.relations} relations</small>
        </li>`;
    }).join("");
}

// ---------------------------------------------------------------------------
// Text input
// ---------------------------------------------------------------------------
document.getElementById("btn-ingest-text").addEventListener("click", async () => {
    const text = document.getElementById("text-input").value.trim();
    if (!text) return;
    const btn = document.getElementById("btn-ingest-text");
    const resultEl = document.getElementById("text-result");
    btn.disabled = true;
    btn.innerHTML = 'Processing<span class="loading"></span>';
    hide(resultEl);

    try {
        const form = new FormData();
        form.append("text", text);
        const res = await fetch("/api/ingest/text", { method: "POST", body: form });
        const data = await res.json();
        if (data.error) {
            resultEl.innerHTML = `<p style="color:var(--accent-red)">${data.error}</p>`;
            show(resultEl);
        } else {
            renderIngestResult(resultEl, data);
            document.getElementById("text-input").value = "";
        }
    } catch (err) {
        resultEl.innerHTML = `<p style="color:var(--accent-red)">Error: ${err.message}</p>`;
        show(resultEl);
    } finally {
        btn.disabled = false;
        btn.textContent = "Add to Knowledge Graph";
    }
});

// ---------------------------------------------------------------------------
// Search
// ---------------------------------------------------------------------------
const searchInput = document.getElementById("search-input");
const btnSearch = document.getElementById("btn-search");
const modeHint = document.getElementById("search-mode-hint");
const llmToggle = document.getElementById("llm-toggle");
let currentMode = "fused";

const MODE_HINTS = {
    fused: "Entity + Semantic + Path retrieval merged via Reciprocal Rank Fusion",
    entity: "Entity lookup in the graph + 2-hop neighborhood expansion",
    semantic: "Vector similarity search over entity embeddings",
    path: "Shortest paths between parsed entities in the query",
    graph: "Fuzzy text match on the knowledge graph — instant, no pipeline",
};

const MODE_PLACEHOLDERS = {
    fused: "e.g., What technologies does Knowledgator develop?",
    entity: "e.g., What is GLiNER?",
    semantic: "e.g., zero-shot named entity recognition",
    path: "e.g., How are Knowledgator and PyTorch connected?",
    graph: "e.g., retrico",
};

document.querySelectorAll(".mode-btn").forEach(btn => {
    btn.addEventListener("click", () => {
        document.querySelectorAll(".mode-btn").forEach(b => b.classList.remove("active"));
        btn.classList.add("active");
        currentMode = btn.dataset.mode;
        modeHint.textContent = MODE_HINTS[currentMode] || "";
        searchInput.placeholder = MODE_PLACEHOLDERS[currentMode] || "";
        const wrapper = document.getElementById("llm-toggle-wrapper");
        wrapper.style.display = currentMode === "graph" ? "none" : "";
    });
});
modeHint.textContent = MODE_HINTS[currentMode];
searchInput.placeholder = MODE_PLACEHOLDERS[currentMode];

searchInput.addEventListener("keydown", e => { if (e.key === "Enter") btnSearch.click(); });

btnSearch.addEventListener("click", async () => {
    const query = searchInput.value.trim();
    if (!query) return;

    const resultEl = document.getElementById("search-result");
    const answerEl = document.getElementById("search-answer");
    const entitiesEl = document.getElementById("search-entities");
    const relationsEl = document.getElementById("search-relations");
    const chunksEl = document.getElementById("search-chunks");
    const graphContainerEl = document.getElementById("search-graph-container");

    btnSearch.disabled = true;
    btnSearch.innerHTML = 'Searching<span class="loading"></span>';
    hide(resultEl); hide(answerEl); hide(entitiesEl); hide(relationsEl); hide(chunksEl);
    hide(graphContainerEl);

    try {
        const form = new FormData();
        form.append("query", query);
        form.append("mode", currentMode);
        form.append("use_llm", llmToggle.checked ? "true" : "false");
        const res = await fetch("/api/search", { method: "POST", body: form });
        const data = await res.json();

        if (data.error) {
            resultEl.innerHTML = `<p style="color:var(--accent-red)">${data.error}</p>`;
            show(resultEl);
            return;
        }

        show(resultEl);

        if (currentMode === "graph") {
            if (data.nodes && data.nodes.length > 0) {
                renderGraphSearchResult("search-graph", data.nodes, data.edges || []);
                show(graphContainerEl);
            } else {
                resultEl.innerHTML = '<p style="color:var(--text-dim)">No matching entities found.</p>';
            }
            return;
        }

        if (data.answer) {
            answerEl.innerHTML = `<h4 style="margin-bottom:0.4rem;font-size:0.8rem;color:var(--text-dim)">ANSWER</h4>${data.answer}`;
            show(answerEl);
        }

        renderSearchGraph(data.entities, data.relations);

        if (data.entities && data.entities.length > 0) {
            entitiesEl.innerHTML = `
                <div class="result-card">
                    <h4>Entities (${data.entities.length})</h4>
                    ${data.entities.map(e => `<span class="entity-tag">${e.label}<span class="type">${e.type || ""}</span></span>`).join("")}
                </div>`;
            show(entitiesEl);
        }

        if (data.relations && data.relations.length > 0) {
            relationsEl.innerHTML = `
                <div class="result-card">
                    <h4>Relations (${data.relations.length})</h4>
                    ${data.relations.map(r => `
                        <div class="relation-row">
                            <span class="head">${r.head}</span>
                            <span class="arrow">&rarr;</span>
                            <span class="rel-type">${r.type}</span>
                            <span class="arrow">&rarr;</span>
                            <span class="tail">${r.tail}</span>
                        </div>`).join("")}
                </div>`;
            show(relationsEl);
        }

        if (data.chunks && data.chunks.length > 0) {
            chunksEl.innerHTML = `
                <div class="result-card">
                    <h4>Source Chunks (${data.chunks.length})</h4>
                    ${data.chunks.map(c => `<div class="chunk-text">${c.text}</div>`).join("")}
                </div>`;
            show(chunksEl);
        }

        if (!data.answer && (!data.entities || data.entities.length === 0)) {
            resultEl.innerHTML = '<p style="color:var(--text-dim)">No results found. Try adding some text first.</p>';
        }
    } catch (err) {
        resultEl.innerHTML = `<p style="color:var(--accent-red)">Error: ${err.message}</p>`;
        show(resultEl);
    } finally {
        btnSearch.disabled = false;
        btnSearch.textContent = "Search";
    }
});

// ---------------------------------------------------------------------------
// Graph visualization (vis-network)
// ---------------------------------------------------------------------------

const TYPE_COLORS = {
    person: "#6c63ff",
    organization: "#f87171",
    location: "#34d399",
    event: "#fbbf24",
    date: "#a78bfa",
    technology: "#60a5fa",
    product: "#fb923c",
    concept: "#e879f9",
    country: "#2dd4bf",
    city: "#4ade80",
};

function getNodeColor(type) {
    return TYPE_COLORS[(type || "").toLowerCase()] || "#8b90a0";
}

const VIS_OPTIONS = {
    nodes: {
        shape: "dot",
        font: { color: "#e1e4ed", size: 13, face: "system-ui, sans-serif" },
        borderWidth: 2,
        shadow: { enabled: true, color: "rgba(0,0,0,0.3)", size: 6 },
    },
    edges: {
        color: { color: "#3e4460", highlight: "#6c63ff", hover: "#6c63ff" },
        font: { color: "#8b90a0", size: 10, face: "system-ui, sans-serif", strokeWidth: 3, strokeColor: "#0f1117" },
        arrows: { to: { enabled: true, scaleFactor: 0.6 } },
        smooth: { type: "continuous", roundness: 0.3 },
        width: 1.5,
    },
    physics: {
        solver: "forceAtlas2Based",
        forceAtlas2Based: { gravitationalConstant: -40, centralGravity: 0.008, springLength: 140, springConstant: 0.04 },
        stabilization: { iterations: 120, fit: true },
    },
    interaction: {
        hover: true,
        tooltipDelay: 150,
        zoomView: true,
        dragView: true,
    },
    layout: { improvedLayout: true },
};

function buildVisData(nodes, edges) {
    const visNodes = nodes.map(n => ({
        id: n.id,
        label: n.label,
        title: `${n.label}\nType: ${n.type || "unknown"}\nMentions: ${n.mentions || 0}`,
        color: { background: getNodeColor(n.type), border: getNodeColor(n.type) },
        size: Math.max(12, Math.min(35, 12 + (n.mentions || 1) * 3)),
        _type: n.type,
        _mentions: n.mentions,
    }));
    const nodeIds = new Set(nodes.map(n => n.id));
    const visEdges = edges
        .filter(e => nodeIds.has(e.from) && nodeIds.has(e.to))
        .map((e, i) => ({
            id: `e${i}`,
            from: e.from,
            to: e.to,
            label: e.type.replace(/_/g, " ").toLowerCase(),
            title: `${e.type} (score: ${(e.score || 0).toFixed(2)})`,
        }));
    return { nodes: new vis.DataSet(visNodes), edges: new vis.DataSet(visEdges) };
}

function renderGraph(containerId, nodes, edges, onSelectNode) {
    const container = document.getElementById(containerId);
    if (!container) return null;
    if (!nodes || nodes.length === 0) {
        container.innerHTML = '<p style="color:var(--text-dim);text-align:center;padding-top:4rem">No graph data yet</p>';
        return null;
    }
    container.innerHTML = "";
    const data = buildVisData(nodes, edges);
    const network = new vis.Network(container, data, VIS_OPTIONS);
    if (onSelectNode) {
        network.on("click", (params) => {
            if (params.nodes.length > 0) {
                onSelectNode(params.nodes[0]);
            }
        });
    }
    return network;
}

function renderGraphSearchResult(containerId, nodes, edges) {
    const container = document.getElementById(containerId);
    if (!container || !nodes || nodes.length === 0) return;
    container.innerHTML = "";

    const visNodes = nodes.map(n => {
        const isMatched = n.matched !== false;
        const color = getNodeColor(n.type);
        return {
            id: n.id,
            label: n.label,
            title: `${n.label}\nType: ${n.type || "unknown"}\nMentions: ${n.mentions || 0}${isMatched ? "\n(matched)" : "\n(neighbor)"}`,
            color: isMatched
                ? { background: color, border: "#34d399" }
                : { background: "rgba(35,39,51,0.8)", border: "#3e4460" },
            borderWidth: isMatched ? 3 : 1,
            size: isMatched ? Math.max(16, Math.min(40, 16 + (n.mentions || 1) * 3)) : 10,
            font: { color: isMatched ? "#e1e4ed" : "#8b90a0", size: isMatched ? 13 : 11 },
            _type: n.type,
        };
    });

    const nodeIds = new Set(nodes.map(n => n.id));
    const visEdges = edges
        .filter(e => nodeIds.has(e.from) && nodeIds.has(e.to))
        .map((e, i) => ({
            id: `e${i}`,
            from: e.from,
            to: e.to,
            label: e.type.replace(/_/g, " ").toLowerCase(),
            title: `${e.type} (score: ${(e.score || 0).toFixed(2)})`,
        }));

    const network = new vis.Network(
        container,
        { nodes: new vis.DataSet(visNodes), edges: new vis.DataSet(visEdges) },
        VIS_OPTIONS,
    );

    network.on("click", (params) => {
        if (params.nodes.length > 0) {
            const nodeId = params.nodes[0];
            const node = nodes.find(n => n.id === nodeId);
            if (node) {
                searchInput.value = node.label;
            }
        }
    });

    const parent = container.parentElement;
    let legend = parent.querySelector(".graph-legend");
    if (!legend) {
        legend = document.createElement("div");
        legend.className = "graph-legend";
        legend.innerHTML = `
            <span class="graph-legend-item"><span class="legend-dot matched"></span> Matched</span>
            <span class="graph-legend-item"><span class="legend-dot neighbor"></span> Neighbor</span>
            <span style="margin-left:auto;font-size:0.75rem;color:var(--text-dim)">${nodes.length} nodes, ${edges.length} edges</span>
        `;
        parent.appendChild(legend);
    } else {
        legend.querySelector("span:last-child").textContent = `${nodes.length} nodes, ${edges.length} edges`;
    }
}

function renderSearchGraph(entities, relations) {
    const graphContainer = document.getElementById("search-graph-container");
    if ((!entities || entities.length === 0) && (!relations || relations.length === 0)) {
        hide(graphContainer);
        return;
    }
    const nodeMap = {};
    (entities || []).forEach(e => {
        const id = e.label.toLowerCase();
        nodeMap[id] = { id, label: e.label, type: e.type || "", mentions: 1 };
    });
    const edgeList = (relations || []).map(r => {
        const fromId = r.head.toLowerCase();
        const toId = r.tail.toLowerCase();
        if (!nodeMap[fromId]) nodeMap[fromId] = { id: fromId, label: r.head, type: "", mentions: 1 };
        if (!nodeMap[toId]) nodeMap[toId] = { id: toId, label: r.tail, type: "", mentions: 1 };
        return { from: fromId, to: toId, type: r.type, score: r.score || 0 };
    });
    show(graphContainer);
    renderGraph("search-graph", Object.values(nodeMap), edgeList);
}

// ---------------------------------------------------------------------------
// Explore tab
// ---------------------------------------------------------------------------
let exploreNetwork = null;
let exploreData = null;

async function loadExploreGraph(neighborOf) {
    const statsEl = document.getElementById("graph-stats");
    statsEl.innerHTML = 'Loading<span class="loading"></span>';
    try {
        const url = neighborOf ? `/api/graph/neighbors/${encodeURIComponent(neighborOf)}` : "/api/graph";
        const res = await fetch(url);
        const data = await res.json();
        exploreData = data;
        statsEl.textContent = `${data.nodes.length} nodes, ${data.edges.length} edges`;
        exploreNetwork = renderGraph("explore-graph", data.nodes, data.edges, (nodeId) => {
            showNodeDetail(nodeId, data);
            loadExploreGraph(nodeId);
        });
        if (neighborOf) {
            showNodeDetail(neighborOf, data);
        } else {
            hide(document.getElementById("node-detail"));
        }
    } catch (err) {
        statsEl.textContent = "Error loading graph";
    }
}

async function exploreSearch(query) {
    if (!query.trim()) { loadExploreGraph(); return; }
    const statsEl = document.getElementById("graph-stats");
    statsEl.innerHTML = 'Searching<span class="loading"></span>';
    try {
        const form = new FormData();
        form.append("query", query.trim());
        form.append("mode", "graph");
        form.append("use_llm", "false");
        const res = await fetch("/api/search", { method: "POST", body: form });
        const data = await res.json();
        exploreData = data;
        statsEl.textContent = `${(data.nodes || []).length} nodes, ${(data.edges || []).length} edges`;
        if (!data.nodes || data.nodes.length === 0) {
            document.getElementById("explore-graph").innerHTML =
                '<p style="color:var(--text-dim);text-align:center;padding-top:4rem">No matching entities found</p>';
            hide(document.getElementById("node-detail"));
            return;
        }
        renderGraphSearchResult("explore-graph", data.nodes, data.edges || []);
        hide(document.getElementById("node-detail"));
    } catch (err) {
        statsEl.textContent = "Error searching graph";
    }
}

function showNodeDetail(nodeId, data) {
    const node = data.nodes.find(n => n.id === nodeId);
    const panel = document.getElementById("node-detail");
    if (!node) { hide(panel); return; }

    document.getElementById("node-detail-title").textContent = node.label;
    document.getElementById("node-detail-type").textContent = node.type || "unknown type";
    document.getElementById("node-detail-mentions").textContent = `${node.mentions || 0} mentions`;

    const rels = data.edges.filter(e => e.from === nodeId || e.to === nodeId);
    const relHtml = rels.length > 0
        ? rels.map(r => {
            const isHead = r.from === nodeId;
            const other = isHead ? r.to : r.from;
            const otherNode = data.nodes.find(n => n.id === other);
            const otherLabel = otherNode ? otherNode.label : other;
            return `<div class="relation-row">
                <span class="head">${isHead ? node.label : otherLabel}</span>
                <span class="arrow">&rarr;</span>
                <span class="rel-type">${r.type.replace(/_/g, " ").toLowerCase()}</span>
                <span class="arrow">&rarr;</span>
                <span class="tail">${isHead ? otherLabel : node.label}</span>
            </div>`;
        }).join("")
        : '<p style="color:var(--text-dim);font-size:0.85rem">No relations</p>';

    document.getElementById("node-detail-relations").innerHTML = relHtml;
    show(panel);
}

// Explore search bar
const exploreSearchInput = document.getElementById("explore-search-input");
const btnExploreSearch = document.getElementById("btn-explore-search");
btnExploreSearch.addEventListener("click", () => exploreSearch(exploreSearchInput.value));
exploreSearchInput.addEventListener("keydown", e => { if (e.key === "Enter") exploreSearch(exploreSearchInput.value); });

document.getElementById("btn-refresh-graph").addEventListener("click", () => {
    exploreSearchInput.value = "";
    hide(document.getElementById("node-detail"));
    loadExploreGraph();
});

// Load documents on start
refreshDocuments();