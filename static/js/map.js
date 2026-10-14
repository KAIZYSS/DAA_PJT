/**
 * static/js/map.js
 * ================
 * Leaflet map orchestration, risk-tier overlays, marker drop-in animations,
 * progressive polyline drawing, and route cross-fades.
 */

const MapController = (function () {
  let map = null;
  let roadLayerGroup = null;
  let activeRouteLayer = null;
  let activeRouteOutlineLayer = null;
  let startMarker = null;
  let destMarker = null;
  let roadPolylinesByTier = { low: [], moderate: [], high: [] };
  let roadLayersById = {};
  let onRoadClickCallback = null;
  let onMapClickCallback = null;
  let isMapPickMode = false;

  // Standard coordinates for Chennai center
  const CHENNAI_CENTER = [12.965, 80.190];
  const DEFAULT_ZOOM = 11;

  function getTierColor(riskScore) {
    if (riskScore <= 0.33) return '#10b981';
    if (riskScore <= 0.66) return '#f59e0b';
    return '#ef4444';
  }

  function getTierName(riskScore) {
    if (riskScore <= 0.33) return 'low';
    if (riskScore <= 0.66) return 'moderate';
    return 'high';
  }

  function createMarkerIcon(letter, type) {
    const isDest = type === 'dest';
    return L.divIcon({
      className: 'custom-map-marker',
      html: `
        <div class="marker-pin-wrapper marker-drop-in">
          <div class="marker-pin ${isDest ? 'dest' : 'start'}">
            <span class="marker-letter">${letter}</span>
          </div>
        </div>
      `,
      iconSize: [32, 42],
      iconAnchor: [16, 42]
    });
  }

  return {
    /**
     * Initialize Leaflet map instance.
     */
    init(elementId = 'map') {
      map = L.map(elementId, {
        center: CHENNAI_CENTER,
        zoom: DEFAULT_ZOOM,
        zoomControl: true,
        attributionControl: true
      });

      // Standard OSM Tile Layer with mandatory attribution
      L.tileLayer('https://tile.openstreetmap.org/{z}/{x}/{y}.png', {
        maxZoom: 18,
        attribution: '&copy; <a href="https://www.openstreetmap.org/copyright" target="_blank">OpenStreetMap</a> contributors'
      }).addTo(map);

      roadLayerGroup = L.layerGroup().addTo(map);

      // Handle map clicks
      map.on('click', (e) => {
        if (isMapPickMode && onMapClickCallback) {
          onMapClickCallback(e.latlng);
        }
      });

      return map;
    },

    /**
     * Set callback for road click.
     */
    setOnRoadClick(cb) {
      onRoadClickCallback = cb;
    },

    /**
     * Set callback for map click when in location pick mode.
     */
    setOnMapClick(cb) {
      onMapClickCallback = cb;
    },

    setMapPickMode(active) {
      isMapPickMode = active;
      if (map) {
        map.getContainer().style.cursor = active ? 'crosshair' : '';
      }
    },

    /**
     * Render the full road network with risk tier colors and staggered fade-in.
     */
    renderRoadNetwork(roads) {
      if (!roadLayerGroup) return;
      roadLayerGroup.clearLayers();
      roadPolylinesByTier = { low: [], moderate: [], high: [] };
      roadLayersById = {};

      roads.forEach((road) => {
        if (!road.geometry || road.geometry.length < 2) return;

        const tier = getTierName(road.risk_score);
        const color = getTierColor(road.risk_score);

        // Render base polyline
        const polyline = L.polyline(road.geometry, {
          color: color,
          weight: 4,
          opacity: 0.65,
          lineCap: 'round',
          lineJoin: 'round',
          className: `road-poly road-tier-${tier}`
        });

        // Hover & Click interactions
        polyline.on('mouseover', function () {
          this.setStyle({ weight: 6, opacity: 0.95 });
        });

        polyline.on('mouseout', function () {
          this.setStyle({ weight: 4, opacity: 0.65 });
        });

        polyline.on('click', (e) => {
          L.DomEvent.stopPropagation(e);
          if (onRoadClickCallback) {
            onRoadClickCallback(road);
          }
        });

        roadLayerGroup.addLayer(polyline);
        roadPolylinesByTier[tier].push(polyline);
        roadLayersById[road.id] = polyline;
      });
    },

    /**
     * Animate a pulse effect on all roads in a given risk tier (Legend Hover).
     */
    pulseRiskTier(tier, active) {
      const polylines = roadPolylinesByTier[tier] || [];
      polylines.forEach((poly) => {
        if (active) {
          poly.setStyle({ weight: 6, opacity: 1.0 });
        } else {
          poly.setStyle({ weight: 4, opacity: 0.65 });
        }
      });
    },

    /**
     * Set start (A) or destination (B) marker with drop-in bounce animation.
     */
    setMarker(type, latlng, label = '') {
      if (!map) return;
      const letter = type === 'start' ? 'A' : 'B';

      if (type === 'start') {
        if (startMarker) map.removeLayer(startMarker);
        startMarker = L.marker(latlng, {
          icon: createMarkerIcon(letter, type),
          zIndexOffset: 1000
        }).addTo(map);
        if (label) startMarker.bindTooltip(label, { permanent: false, direction: 'top' });
      } else {
        if (destMarker) map.removeLayer(destMarker);
        destMarker = L.marker(latlng, {
          icon: createMarkerIcon(letter, type),
          zIndexOffset: 1000
        }).addTo(map);
        if (label) destMarker.bindTooltip(label, { permanent: false, direction: 'top' });
      }
    },

    clearMarkers() {
      if (startMarker && map) map.removeLayer(startMarker);
      if (destMarker && map) map.removeLayer(destMarker);
      startMarker = null;
      destMarker = null;
    },

    /**
     * Progressive Polyline Draw-On (Spec Section 4).
     * Progressively reveals the polyline along the geometry from A to B
     * over duration ms using requestAnimationFrame.
     */
    drawPolylineProgressively(latlngs, duration = 900, options = {}) {
      if (!map || !latlngs || latlngs.length < 2) return Promise.resolve();

      // Clean up previous routes
      this.clearRoute();

      const color = options.color || '#3b82f6';
      const weight = options.weight || 6;

      // Dark background casing for high contrast
      activeRouteOutlineLayer = L.polyline([], {
        color: '#0b0f19',
        weight: weight + 3,
        opacity: 0.8,
        lineCap: 'round',
        lineJoin: 'round'
      }).addTo(map);

      activeRouteLayer = L.polyline([], {
        color: color,
        weight: weight,
        opacity: 0.95,
        lineCap: 'round',
        lineJoin: 'round',
        className: 'route-polyline-active'
      }).addTo(map);

      // Fit map bounds to encompass the full route smoothly
      const fullBounds = L.latLngBounds(latlngs);
      map.fitBounds(fullBounds, {
        padding: [60, 60],
        maxZoom: 14,
        animate: true,
        duration: 0.6
      });

      return new Promise((resolve) => {
        // Check for prefers-reduced-motion
        const prefersReducedMotion = window.matchMedia('(prefers-reduced-motion: reduce)').matches;
        if (prefersReducedMotion || duration <= 0) {
          activeRouteOutlineLayer.setLatLngs(latlngs);
          activeRouteLayer.setLatLngs(latlngs);
          resolve();
          return;
        }

        const startTime = performance.now();
        const totalPoints = latlngs.length;

        function animateFrame(currentTime) {
          const elapsed = currentTime - startTime;
          const progress = Math.min(1.0, elapsed / duration);
          // Ease-out cubic: 1 - pow(1 - p, 3)
          const easeProgress = 1 - Math.pow(1 - progress, 3);

          const targetIndex = Math.max(2, Math.floor(easeProgress * totalPoints));
          const currentPoints = latlngs.slice(0, targetIndex);

          activeRouteOutlineLayer.setLatLngs(currentPoints);
          activeRouteLayer.setLatLngs(currentPoints);

          if (progress < 1.0) {
            requestAnimationFrame(animateFrame);
          } else {
            // Ensure complete path is rendered
            activeRouteOutlineLayer.setLatLngs(latlngs);
            activeRouteLayer.setLatLngs(latlngs);
            resolve();
          }
        }

        requestAnimationFrame(animateFrame);
      });
    },

    /**
     * Cross-fade route swap (Spec Section 4).
     * Smoothly cross-fades old polyline out (150ms) and new polyline in (150ms).
     */
    crossFadeRoute(newLatlngs, color = '#3b82f6') {
      if (!map || !newLatlngs || newLatlngs.length < 2) return;

      const oldLayer = activeRouteLayer;
      const oldOutline = activeRouteOutlineLayer;

      // Check prefers-reduced-motion
      if (window.matchMedia('(prefers-reduced-motion: reduce)').matches) {
        this.clearRoute();
        this.drawPolylineProgressively(newLatlngs, 0, { color });
        return;
      }

      // Step 1: Fade out old polyline over 150ms
      let fadeOutProgress = 1.0;
      const fadeOutStart = performance.now();

      function fadeOut(time) {
        const elapsed = time - fadeOutStart;
        fadeOutProgress = Math.max(0, 1.0 - (elapsed / 150));
        if (oldLayer) oldLayer.setStyle({ opacity: fadeOutProgress * 0.95 });
        if (oldOutline) oldOutline.setStyle({ opacity: fadeOutProgress * 0.8 });

        if (elapsed < 150) {
          requestAnimationFrame(fadeOut);
        } else {
          // Remove old layers
          if (oldLayer) map.removeLayer(oldLayer);
          if (oldOutline) map.removeLayer(oldOutline);

          // Step 2: Draw and fade in new polyline
          activeRouteOutlineLayer = L.polyline(newLatlngs, {
            color: '#0b0f19',
            weight: 9,
            opacity: 0.0,
            lineCap: 'round',
            lineJoin: 'round'
          }).addTo(map);

          activeRouteLayer = L.polyline(newLatlngs, {
            color: color,
            weight: 6,
            opacity: 0.0,
            lineCap: 'round',
            lineJoin: 'round'
          }).addTo(map);

          const fadeInStart = performance.now();
          function fadeIn(fTime) {
            const fElapsed = fTime - fadeInStart;
            const fProg = Math.min(1.0, fElapsed / 150);
            activeRouteLayer.setStyle({ opacity: fProg * 0.95 });
            activeRouteOutlineLayer.setStyle({ opacity: fProg * 0.8 });
            if (fElapsed < 150) {
              requestAnimationFrame(fadeIn);
            }
          }
          requestAnimationFrame(fadeIn);
        }
      }

      requestAnimationFrame(fadeOut);
    },

    clearRoute() {
      if (activeRouteLayer && map) {
        map.removeLayer(activeRouteLayer);
        activeRouteLayer = null;
      }
      if (activeRouteOutlineLayer && map) {
        map.removeLayer(activeRouteOutlineLayer);
        activeRouteOutlineLayer = null;
      }
    },

    highlightRoad(roadId) {
      const layer = roadLayersById[roadId];
      if (layer) {
        layer.setStyle({ weight: 7, opacity: 1.0 });
        setTimeout(() => {
          layer.setStyle({ weight: 4, opacity: 0.65 });
        }, 1500);
      }
    }
  };
})();
