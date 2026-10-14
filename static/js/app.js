/**
 * static/js/app.js
 * ================
 * Main UI orchestrator for ROUTEWISE.
 * Implements rolling number count-ups, sliding segmented control,
 * button state machines, staggered bullet reveals, and toast notifications.
 */

document.addEventListener('DOMContentLoaded', async () => {
  // --------------------------------------------------------------------------
  // State
  // --------------------------------------------------------------------------
  let locations = [];
  let startLocation = null;
  let destLocation = null;
  let currentMode = 'reliable';
  let activeRouteData = null;
  let pickTarget = null; // 'start' or 'dest' when picking from map

  // --------------------------------------------------------------------------
  // DOM Elements
  // --------------------------------------------------------------------------
  const startInput = document.getElementById('start-input');
  const destInput = document.getElementById('dest-input');
  const startDropdown = document.getElementById('start-dropdown');
  const destDropdown = document.getElementById('dest-dropdown');
  const swapBtn = document.getElementById('swap-btn');
  const mapPickBtn = document.getElementById('map-pick-btn');
  const mapClickBanner = document.getElementById('map-click-banner');
  const cancelMapPickBtn = document.getElementById('cancel-map-pick-btn');

  const segmentedIndicator = document.getElementById('segmented-indicator');
  const segmentedOptions = document.querySelectorAll('.segmented-option');
  const findRouteBtn = document.getElementById('find-route-btn');
  const findRouteBtnText = document.getElementById('find-route-btn-text');
  const findRouteSpinner = document.getElementById('find-route-spinner');

  // Network Status elements
  const statSegments = document.getElementById('stat-segments');
  const statOpen = document.getElementById('stat-open');
  const statFlagged = document.getElementById('stat-flagged');
  const statHighRisk = document.getElementById('stat-high-risk');

  // Route Analysis Panel
  const routeAnalysisPanel = document.getElementById('route-analysis-panel');
  const closeAnalysisBtn = document.getElementById('close-analysis-btn');
  const metricDistance = document.getElementById('metric-distance');
  const metricTime = document.getElementById('metric-time');
  const metricSegments = document.getElementById('metric-segments');
  const metricReliability = document.getElementById('metric-reliability');
  const gaugeCircle = document.getElementById('gauge-circle');
  const gaugePercent = document.getElementById('gauge-percent');
  const routeTierBadge = document.getElementById('route-tier-badge');
  const routeModeDesc = document.getElementById('route-mode-desc');
  const whyRouteList = document.getElementById('why-route-list');
  const comparisonRows = document.getElementById('comparison-rows');

  // Road Info Panel
  const roadInfoPanel = document.getElementById('road-info-panel');
  const closeRoadInfoBtn = document.getElementById('close-road-info-btn');
  const roadName = document.getElementById('road-name');
  const roadReliability = document.getElementById('road-reliability');
  const roadRiskTier = document.getElementById('road-risk-tier');
  const roadStatus = document.getElementById('road-status');
  const roadDistance = document.getElementById('road-distance');
  const roadSpeed = document.getElementById('road-speed');
  const roadAnomaliesList = document.getElementById('road-anomalies-list');

  // Nav & Mobile
  const sidebar = document.getElementById('sidebar');
  const toggleSidebarBtn = document.getElementById('toggle-sidebar-btn');
  const toastContainer = document.getElementById('toast-container');
  const mapSkeletonOverlay = document.getElementById('map-skeleton-overlay');
  const dbStatusIndicator = document.getElementById('db-status-indicator');
  const dbStatusDot = document.getElementById('db-status-dot');
  const dbStatusText = document.getElementById('db-status-text');

  // --------------------------------------------------------------------------
  // Helper: Rolling Number Counter (Spec Section 2)
  // --------------------------------------------------------------------------
  function animateValue(el, targetVal, duration = 700, formatFn = (v) => Math.round(v)) {
    if (!el) return;
    const currentVal = parseFloat(el.dataset.val || '0');
    if (currentVal === targetVal && el.textContent.trim() !== '') return;

    // Check prefers-reduced-motion
    if (window.matchMedia('(prefers-reduced-motion: reduce)').matches) {
      el.textContent = formatFn(targetVal);
      el.dataset.val = targetVal;
      return;
    }

    const startVal = currentVal;
    const startTime = performance.now();

    function updateNum(currentTime) {
      const elapsed = currentTime - startTime;
      const progress = Math.min(1.0, elapsed / duration);
      // Ease-out cubic: 1 - pow(1 - p, 3)
      const ease = 1 - Math.pow(1 - progress, 3);
      const val = startVal + (targetVal - startVal) * ease;
      el.textContent = formatFn(val);

      if (progress < 1.0) {
        requestAnimationFrame(updateNum);
      } else {
        el.textContent = formatFn(targetVal);
        el.dataset.val = targetVal;
      }
    }

    requestAnimationFrame(updateNum);
  }

  // --------------------------------------------------------------------------
  // Helper: Toast Notifications (Spec Section 5)
  // --------------------------------------------------------------------------
  function showToast(message, type = 'error') {
    const toast = document.createElement('div');
    toast.className = `toast-message ${type}`;

    const icon = type === 'error'
      ? '<i data-lucide="alert-circle" style="color:var(--danger)"></i>'
      : type === 'success'
      ? '<i data-lucide="check-circle-2" style="color:var(--success)"></i>'
      : '<i data-lucide="info" style="color:var(--accent)"></i>';

    toast.innerHTML = `
      ${icon}
      <span>${message}</span>
    `;

    toastContainer.appendChild(toast);
    if (window.lucide) lucide.createIcons();

    // Auto dismiss after 4s with fade animation
    setTimeout(() => {
      toast.classList.add('dismissing');
      setTimeout(() => {
        if (toast.parentNode) toast.parentNode.removeChild(toast);
      }, 200);
    }, 4000);
  }

  // --------------------------------------------------------------------------
  // 1. Initial Map and App Shell Load Sequence (Spec Section 1)
  // --------------------------------------------------------------------------
  MapController.init('map');

  // Road click handler
  MapController.setOnRoadClick(async (road) => {
    try {
      const detail = await Api.getRoadDetail(road.id);
      renderRoadInfoPanel(detail);
      MapController.highlightRoad(road.id);
    } catch (err) {
      showToast(`Unable to load details for ${road.name}`);
    }
  });

  // Map location click handler
  MapController.setOnMapClick((latlng) => {
    handleMapCoordinatePicked(latlng);
  });

  // --------------------------------------------------------------------------
  // Backend & Database Linkage Verification
  // --------------------------------------------------------------------------
  async function checkBackendLink() {
    if (!dbStatusIndicator || !dbStatusDot || !dbStatusText) return;
    try {
      const health = await Api.getHealth();
      if (health.success && health.status === 'healthy') {
        dbStatusDot.className = 'status-dot';
        dbStatusDot.style.background = '';
        dbStatusDot.style.boxShadow = '';
        dbStatusText.textContent = 'API & DB LINKED';
        const c = health.database.counts;
        dbStatusIndicator.title = `SQLite Connected: ${c.roads} roads, ${c.nodes} nodes, ${c.anomalies} hazards, ${c.locations} locations`;
      } else {
        throw new Error(health.error || 'Database unhealthy');
      }
    } catch (err) {
      dbStatusDot.className = 'status-dot danger';
      dbStatusText.textContent = 'API OFFLINE';
      dbStatusIndicator.title = 'Flask server not reachable. Please run: python app.py on port 5000.';
    }
  }

  if (dbStatusIndicator) {
    dbStatusIndicator.addEventListener('click', () => {
      checkBackendLink();
      if (dbStatusText.textContent.includes('LINKED')) {
        showToast(dbStatusIndicator.title, 'success');
      } else {
        showToast('Flask server is offline. Please run "python app.py" on port 5000.', 'error');
      }
    });
  }

  // Initial link check & keep-alive polling every 10s
  checkBackendLink();
  setInterval(checkBackendLink, 10000);

  // Load road network and locations concurrently
  try {
    const [roadsData, locationsData, summaryData] = await Promise.all([
      Api.getRoads(),
      Api.getLocations(),
      Api.getNetworkSummary()
    ]);

    locations = locationsData;

    // Render road layer with risk tiers
    MapController.renderRoadNetwork(roadsData);

    // Fade out map skeleton overlay
    if (mapSkeletonOverlay) {
      mapSkeletonOverlay.classList.add('hidden');
    }

    // Populate Network Status card with rolling numbers
    updateNetworkStatus(summaryData);

    // Pre-fill Chennai Central -> VIT Chennai for instant demo readiness
    const central = locations.find(l => l.name.toLowerCase().includes('central'));
    const vit = locations.find(l => l.name.toLowerCase().includes('vit'));
    if (central) selectLocation('start', central);
    if (vit) selectLocation('dest', vit);

  } catch (err) {
    console.error('Initialization error:', err);
    showToast(err.message || 'Failed to initialize road network. Please refresh.');
    checkBackendLink();
  }

  // --------------------------------------------------------------------------
  // 2. Network Status Card Population with Rolling Numbers
  // --------------------------------------------------------------------------
  function updateNetworkStatus(summary) {
    const removeSkeleton = (el) => el.classList.remove('skeleton');

    removeSkeleton(statSegments);
    animateValue(statSegments, summary.total_segments, 800);

    removeSkeleton(statOpen);
    animateValue(statOpen, summary.open_roads, 800);

    removeSkeleton(statFlagged);
    animateValue(statFlagged, summary.flagged_roads, 800);

    removeSkeleton(statHighRisk);
    animateValue(statHighRisk, summary.high_risk_roads, 800);
  }

  // --------------------------------------------------------------------------
  // 3. Sliding Pill Segmented Control (Route Preference) (Spec Section 3)
  // --------------------------------------------------------------------------
  segmentedOptions.forEach((opt, idx) => {
    opt.addEventListener('click', () => {
      segmentedOptions.forEach(o => o.classList.remove('active'));
      opt.classList.add('active');
      currentMode = opt.dataset.mode;

      // Glide the sliding indicator pill
      const percent = idx * 100;
      segmentedIndicator.style.transform = `translateX(${percent}%)`;

      // If a route is already calculated, recalculate or highlight matching mode
      if (startLocation && destLocation && activeRouteData) {
        calculateRoute();
      }
    });
  });

  // --------------------------------------------------------------------------
  // 4. Autocomplete & Location Search (Spec Section 5)
  // --------------------------------------------------------------------------
  function setupAutocomplete(inputEl, dropdownEl, type) {
    inputEl.addEventListener('input', () => {
      const query = inputEl.value.trim().toLowerCase();
      if (!query) {
        dropdownEl.classList.remove('open');
        return;
      }

      const matches = locations.filter(loc =>
        loc.name.toLowerCase().includes(query) ||
        (loc.description && loc.description.toLowerCase().includes(query))
      );

      if (matches.length === 0) {
        dropdownEl.innerHTML = '<div class="autocomplete-item"><span class="autocomplete-item-desc">No matching locations</span></div>';
        dropdownEl.classList.add('open');
        return;
      }

      dropdownEl.innerHTML = matches.map(loc => `
        <div class="autocomplete-item" data-id="${loc.id}">
          <strong>${loc.name}</strong>
          <span class="autocomplete-item-desc">${loc.description || ''}</span>
        </div>
      `).join('');

      dropdownEl.classList.add('open');

      dropdownEl.querySelectorAll('.autocomplete-item').forEach((item, index) => {
        item.addEventListener('click', () => {
          const loc = locations.find(l => l.id === item.dataset.id);
          if (loc) selectLocation(type, loc);
          dropdownEl.classList.remove('open');
        });
      });
    });

    inputEl.addEventListener('focus', () => {
      if (inputEl.value.trim().length > 0) {
        dropdownEl.classList.add('open');
      }
    });

    // Close on click outside
    document.addEventListener('click', (e) => {
      if (!inputEl.contains(e.target) && !dropdownEl.contains(e.target)) {
        dropdownEl.classList.remove('open');
      }
    });
  }

  setupAutocomplete(startInput, startDropdown, 'start');
  setupAutocomplete(destInput, destDropdown, 'dest');

  function selectLocation(type, loc) {
    if (type === 'start') {
      startLocation = loc;
      startInput.value = loc.name;
      MapController.setMarker('start', [loc.lat, loc.lng], loc.name);
    } else {
      destLocation = loc;
      destInput.value = loc.name;
      MapController.setMarker('dest', [loc.lat, loc.lng], loc.name);
    }
  }

  // Swap Locations Button
  swapBtn.addEventListener('click', () => {
    const tempLoc = startLocation;
    const tempVal = startInput.value;

    startLocation = destLocation;
    startInput.value = destInput.value;

    destLocation = tempLoc;
    destInput.value = tempVal;

    if (startLocation) MapController.setMarker('start', [startLocation.lat, startLocation.lng], startLocation.name);
    else MapController.setMarker('start', null);

    if (destLocation) MapController.setMarker('dest', [destLocation.lat, destLocation.lng], destLocation.name);
    else MapController.setMarker('dest', null);

    if (startLocation && destLocation && activeRouteData) {
      calculateRoute();
    }
  });

  // Map Location Picker Mode
  mapPickBtn.addEventListener('click', () => {
    pickTarget = !startLocation ? 'start' : 'dest';
    MapController.setMapPickMode(true);
    mapClickBanner.classList.add('active');
    mapClickBanner.querySelector('span').textContent = `Click on map to choose ${pickTarget === 'start' ? 'Start' : 'Destination'} point`;
  });

  cancelMapPickBtn.addEventListener('click', () => {
    MapController.setMapPickMode(false);
    mapClickBanner.classList.remove('active');
    pickTarget = null;
  });

  function handleMapCoordinatePicked(latlng) {
    // Find closest location by euclidean distance
    let nearest = null;
    let minDist = Infinity;

    locations.forEach(loc => {
      const d = Math.hypot(loc.lat - latlng.lat, loc.lng - latlng.lng);
      if (d < minDist) {
        minDist = d;
        nearest = loc;
      }
    });

    if (nearest) {
      selectLocation(pickTarget || 'start', nearest);
      showToast(`Selected nearest hub: ${nearest.name}`, 'info');
    }

    MapController.setMapPickMode(false);
    mapClickBanner.classList.remove('active');
    pickTarget = null;
  }

  // --------------------------------------------------------------------------
  // 5. Route Calculation & Button State Machine (Spec Section 3 & 4)
  // --------------------------------------------------------------------------
  findRouteBtn.addEventListener('click', () => {
    calculateRoute();
  });

  async function calculateRoute() {
    if (!startLocation || !destLocation) {
      showToast('Please select both a start and destination location.');
      return;
    }

    if (startLocation.node_id === destLocation.node_id) {
      showToast('Start and destination locations cannot be identical.');
      return;
    }

    // Button loading state
    findRouteBtn.disabled = true;
    findRouteBtnText.textContent = 'CALCULATING ROUTE...';
    findRouteSpinner.style.display = 'block';

    try {
      const result = await Api.getRoute(
        startLocation.node_id,
        destLocation.node_id,
        currentMode
      );

      activeRouteData = result;

      // Button Success Flash State
      findRouteBtnText.textContent = 'ROUTE FOUND';
      findRouteSpinner.style.display = 'none';
      findRouteBtn.classList.add('success-flash');

      // Draw polyline progressively from A to B (800ms - 1.2s)
      await MapController.drawPolylineProgressively(result.route.geometry, 950);

      // Render Route Analysis Panel with rolling numbers
      renderRouteAnalysisPanel(result);

      // Revert button to normal state after brief success feedback
      setTimeout(() => {
        findRouteBtn.disabled = false;
        findRouteBtnText.textContent = 'FIND BEST ROUTE';
        findRouteBtn.classList.remove('success-flash');
      }, 600);

    } catch (err) {
      console.error('Route error:', err);
      showToast(err.message || 'Route calculation failed');
      checkBackendLink();
      findRouteBtn.disabled = false;
      findRouteBtnText.textContent = 'FIND BEST ROUTE';
      findRouteSpinner.style.display = 'none';
    }

  }

  // --------------------------------------------------------------------------
  // 6. Route Analysis Panel & Radial Gauge Animation (Spec Section 2 & 6)
  // --------------------------------------------------------------------------
  function renderRouteAnalysisPanel(data) {
    const route = data.route;

    // Show panel with slide/fade up
    routeAnalysisPanel.classList.add('visible');

    // Rolling numbers
    animateValue(metricDistance, route.distance_km, 700, v => v.toFixed(1));
    animateValue(metricTime, route.travel_time_minutes, 700, v => Math.round(v));
    animateValue(metricSegments, route.segments_count, 700, v => Math.round(v));
    animateValue(metricReliability, route.reliability_pct, 700, v => `${Math.round(v)}%`);

    // Animate radial gauge fill
    animateRadialGauge(route.reliability_pct);

    // Tier badge
    routeTierBadge.textContent = route.risk_tier;
    routeTierBadge.className = `tier-badge ${route.risk_tier.toLowerCase()}`;
    routeModeDesc.textContent = route.mode_description;

    // Staggered reveal of "Why This Route?" lines (Spec Section 6)
    whyRouteList.innerHTML = '';
    (data.explanations || []).forEach((bullet, idx) => {
      const item = document.createElement('div');
      item.className = 'why-route-item';
      item.style.setProperty('--stagger-index', idx);
      item.innerHTML = `
        <i data-lucide="check" class="why-route-icon" style="width:14px;height:14px"></i>
        <span>${bullet}</span>
      `;
      whyRouteList.appendChild(item);
    });

    // Populate Comparison rows (Spec Section 3 & 4)
    renderComparisonRows(data.comparison, route.mode);

    if (window.lucide) lucide.createIcons();
  }

  function animateRadialGauge(percent) {
    const circumference = 138; // 2 * PI * 22
    const offset = circumference - (percent / 100) * circumference;
    gaugeCircle.style.strokeDashoffset = offset;

    // Color code gauge circle by reliability
    if (percent >= 75) {
      gaugeCircle.style.stroke = 'var(--success)';
    } else if (percent >= 50) {
      gaugeCircle.style.stroke = 'var(--warning)';
    } else {
      gaugeCircle.style.stroke = 'var(--danger)';
    }

    animateValue(gaugePercent, percent, 800, v => `${Math.round(v)}%`);
  }

  // --------------------------------------------------------------------------
  // 7. Route Comparison Rows & Cross-Fade Swap (Spec Section 3 & 4)
  // --------------------------------------------------------------------------
  function renderComparisonRows(comparison, activeMode) {
    if (!comparison) return;
    comparisonRows.innerHTML = '';

    const options = [
      { key: 'fastest', data: comparison.fastest, label: 'Fastest', tag: 'Speed' },
      { key: 'reliable', data: comparison.recommended, label: 'Recommended', tag: 'Reliable' }
    ];

    options.forEach(opt => {
      if (!opt.data) return;
      const row = document.createElement('div');
      const isActive = (activeMode === opt.key) || (activeMode === 'reliable' && opt.key === 'reliable');
      row.className = `comparison-row ${isActive ? 'active' : ''}`;
      row.dataset.mode = opt.key;

      row.innerHTML = `
        <div class="comparison-name-group">
          <div class="comparison-title">
            <span>${opt.label}</span>
            <span class="comparison-tag">${opt.tag}</span>
          </div>
          <span style="font-size:10px;color:var(--muted)">${opt.data.distance_km} km</span>
        </div>
        <div class="comparison-stats">
          <div class="comparison-stat-item">
            <span class="comparison-stat-num">${Math.round(opt.data.travel_time_minutes)} min</span>
            <span class="comparison-stat-label">Duration</span>
          </div>
          <div class="comparison-stat-item">
            <span class="comparison-stat-num" style="color:${opt.data.reliability_pct >= 75 ? 'var(--success)' : 'var(--warning)'}">${opt.data.reliability_pct}%</span>
            <span class="comparison-stat-label">Reliability</span>
          </div>
        </div>
      `;

      // Click to cross-fade swap route
      row.addEventListener('click', () => {
        document.querySelectorAll('.comparison-row').forEach(r => r.classList.remove('active'));
        row.classList.add('active');

        // Cross fade polylines on map (150ms fade out -> 150ms fade in)
        const lineColor = opt.key === 'reliable' ? '#3b82f6' : '#64748b';
        MapController.crossFadeRoute(opt.data.geometry, lineColor);

        // Update radial gauge
        animateRadialGauge(opt.data.reliability_pct);
      });

      comparisonRows.appendChild(row);
    });
  }

  closeAnalysisBtn.addEventListener('click', () => {
    routeAnalysisPanel.classList.remove('visible');
    MapController.clearRoute();
  });

  // --------------------------------------------------------------------------
  // 8. Road Info Panel (Spec Section 3 & 14)
  // --------------------------------------------------------------------------
  function renderRoadInfoPanel(road) {
    roadInfoPanel.classList.add('visible');

    roadName.textContent = road.name;
    roadReliability.textContent = `${road.reliability_pct}%`;
    roadRiskTier.textContent = road.risk_tier;
    roadRiskTier.className = `tier-badge ${road.risk_tier.toLowerCase()}`;
    roadStatus.textContent = (road.status || 'open').toUpperCase();
    roadDistance.textContent = `${road.distance_km} km`;
    roadSpeed.textContent = `${road.speed_kmph} km/h`;

    if (!road.anomalies || road.anomalies.length === 0) {
      roadAnomaliesList.innerHTML = `
        <div style="font-size:11px;color:var(--muted);padding:8px 0;">
          No incident history or active road flags recorded for this corridor.
        </div>
      `;
    } else {
      roadAnomaliesList.innerHTML = road.anomalies.map(a => `
        <div class="road-anomaly-item ${a.type}">
          <div class="road-anomaly-header">
            <span>${a.type} (Sev: ${a.severity}/10)</span>
            <span>${a.duration_minutes} min</span>
          </div>
          <div class="road-anomaly-meta">
            ${a.description || 'Monitored traffic restriction'}
          </div>
        </div>
      `).join('');
    }

    if (window.lucide) lucide.createIcons();
  }

  closeRoadInfoBtn.addEventListener('click', () => {
    roadInfoPanel.classList.remove('visible');
  });

  // --------------------------------------------------------------------------
  // 9. Legend Swatch Hover Interaction (Spec Section 5)
  // --------------------------------------------------------------------------
  document.querySelectorAll('.legend-item[data-tier]').forEach(item => {
    const tier = item.dataset.tier;
    item.addEventListener('mouseenter', () => {
      MapController.pulseRiskTier(tier, true);
    });
    item.addEventListener('mouseleave', () => {
      MapController.pulseRiskTier(tier, false);
    });
  });

  // --------------------------------------------------------------------------
  // 10. Responsive Sidebar Toggle (Spec Section 7)
  // --------------------------------------------------------------------------
  if (toggleSidebarBtn) {
    toggleSidebarBtn.addEventListener('click', () => {
      sidebar.classList.toggle('collapsed');
      const isCollapsed = sidebar.classList.contains('collapsed');
      toggleSidebarBtn.innerHTML = isCollapsed
        ? '<i data-lucide="menu"></i>'
        : '<i data-lucide="chevron-left"></i>';
      if (window.lucide) lucide.createIcons();
    });
  }

  // Initialize Lucide icons
  if (window.lucide) {
    lucide.createIcons();
  }
});
