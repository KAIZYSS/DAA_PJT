/**
 * static/js/api.js
 * ================
 * Client-side API abstraction for the ROUTEWISE REST backend.
 * Implements in-memory caching for the static road network (Spec Section 46).
 */

const Api = (function () {
  let cachedRoads = null;
  let cachedLocations = null;

  // Determine API base: If running on Live Server (e.g. port 5500) or file://, direct to Flask default port 5000
  const isCustomDevServer = (
    window.location.protocol === 'file:' ||
    (window.location.port && window.location.port !== '5000')
  );
  const API_BASE = isCustomDevServer ? 'http://127.0.0.1:5000' : '';

  async function request(endpoint, options = {}) {
    const url = `${API_BASE}${endpoint}`;
    try {
      const response = await fetch(url, {
        headers: {
          'Accept': 'application/json',
          'Content-Type': 'application/json'
        },
        ...options
      });

      const data = await response.json();
      if (!response.ok || data.success === false) {
        throw new Error(data.error || `HTTP ${response.status}: Request failed`);
      }
      return data;
    } catch (err) {
      console.error(`[API Error] ${url}:`, err);
      // Transform raw browser network errors into actionable troubleshooting guidance
      if (err.name === 'TypeError' && (err.message === 'Failed to fetch' || err.message.includes('fetch') || err.message.includes('NetworkError'))) {
        throw new Error(`Unable to reach backend server (${API_BASE || 'http://127.0.0.1:5000'}). Please ensure 'python app.py' is running.`);
      }
      throw err;
    }
  }


  return {
    /**
     * Fetch predefined locations for search and autocomplete.
     */
    async getLocations() {
      if (cachedLocations) return cachedLocations;
      const res = await request('/api/locations');
      cachedLocations = res.locations;
      return cachedLocations;
    },

    /**
     * Fetch the full road network layer. Caches result in memory.
     */
    async getRoads() {
      if (cachedRoads) return cachedRoads;
      const res = await request('/api/roads');
      cachedRoads = res.roads;
      return cachedRoads;
    },

    /**
     * Fetch specific road metadata, reliability %, and anomaly history.
     */
    async getRoadDetail(roadId) {
      const res = await request(`/api/roads/${encodeURIComponent(roadId)}`);
      return res.road;
    },

    /**
     * Request Dijkstra optimal route between two node IDs.
     */
    async getRoute(startNodeId, destNodeId, mode = 'reliable') {
      const params = new URLSearchParams({
        start: startNodeId,
        destination: destNodeId,
        mode: mode
      });
      return await request(`/api/route?${params.toString()}`);
    },

    /**
     * Fetch dynamic summary statistics for the network status card.
     */
    async getNetworkSummary() {
      const res = await request('/api/network-summary');
      return res.summary;
    },

    /**
     * Verify backend API and SQLite database linkage health.
     */
    async getHealth() {
      return await request('/api/health');
    }
  };
})();

