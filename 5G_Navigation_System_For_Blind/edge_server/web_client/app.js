    let ws = null;
    let isStreaming = false;
    let currentSpeedKmh = 0.0;

    // Determine WebSocket URL from current host (HTTP -> WS, HTTPS -> WSS)
    function getWsUrl() {
      const loc = window.location;
      const protocol = (loc.protocol === 'https:') ? 'wss:' : 'ws:';
      return `${protocol}//${loc.host}/ws`;
    }

    let reconnectAttempts = 0;
    let connLostSpoken = false, everConnected = false;

    // Connect WebSocket to Edge Server
    function connectWebSocket() {
      const wsUrl = getWsUrl();
      console.log("Connecting WebSocket to:", wsUrl);
      ws = new WebSocket(wsUrl);

      ws.onopen = () => {
        reconnectAttempts = 0;
        if (connLostSpoken) { say("Guidance reconnected.", 3); connLostSpoken = false; }
        everConnected = true;
        console.log("✅ WebSocket connected to Edge Server!");
        ws.send(JSON.stringify({ type: "hello", "role": "phone" }));
        const badge = document.getElementById('netBadge');
        if (badge) badge.className = "badge active";
        const netTxt = document.getElementById('netText');
        if (netTxt) netTxt.textContent = "5G LIVE LINK ACTIVE";
        
        // Request wake lock so iOS does not sleep screen while walking
        requestScreenWakeLock();

        // Sync active Cloudflare URL to Cloud Sync bus for Arduino auto-discovery (from mobile cellular IP)
        if (window.location.origin.includes("trycloudflare.com")) {
          try {
            fetch("https://ntfy.sh/iitg_5g_wearable_nav_sync_9169957", {
              method: "POST",
              headers: { "Title": "Active5GTunnelURL" },
              body: window.location.origin
            }).catch(() => {});
          } catch(e) {}
        }

        // Immediately send GPS fix to Edge Server if already available
        if (lastLat != null && lastLon != null) {
          sendGpsTelemetry(lastLat, lastLon, lastAcc, lastSpd);
        }
        requestImmediateGpsFix();
      };

      ws.onclose = () => {
        if (everConnected && !connLostSpoken) { say("Connection to guidance lost. Please stop and wait.", 1); connLostSpoken = true; }
        const badge = document.getElementById('netBadge');
        if (badge) badge.className = "badge";
        const netTxt = document.getElementById('netText');
        if (netTxt) netTxt.textContent = "5G RECONNECTING...";
        ws = null;
        // Aggressive exponential backoff for fast 5G cell tower handoffs (400ms -> 800ms -> max 2500ms)
        const delay = Math.min(2500, Math.round(400 * Math.pow(1.5, reconnectAttempts)));
        reconnectAttempts++;
        setTimeout(() => {
          if (!ws || ws.readyState === WebSocket.CLOSED) {
            connectWebSocket();
          }
        }, delay);
      };

      ws.onerror = (err) => {
        console.error("WebSocket error:", err);
      };

      ws.onmessage = (event) => {
        try {
          const msg = JSON.parse(event.data);
          handleDownlinkMessage(msg);
        } catch (e) {
          console.error("Error parsing downlink message:", e);
        }
      };
    }

    // Global Route and Audio State
    let routeData = null;
    let currentStepIndex = 0;
    let totalSteps = 0;
    let audioUnlocked = false;
    let currentUtterance = null;
    let chimeAudioCtx = null;

    // Primes and caches an active AudioContext for zero-latency earcon chimes
    function getWarmAudioContext() {
      try {
        const AudioCtx = window.AudioContext || window.webkitAudioContext;
        if (!chimeAudioCtx && AudioCtx) {
          chimeAudioCtx = new AudioCtx();
        }
        if (chimeAudioCtx && chimeAudioCtx.state === 'suspended') {
          chimeAudioCtx.resume().catch(() => {});
        }
        return chimeAudioCtx;
      } catch (e) {
        return null;
      }
    }

    let keepAliveOsc = null;
    let keepAliveGain = null;

    // Continuous near-zero gain background audio loop
    // Tells iOS CoreAudio and Android AudioService the tab is an active media app,
    // completely preventing background sleep while Google Maps is navigating
    function startBackgroundAudioKeepAlive() {
      try {
        const ctx = getWarmAudioContext();
        if (!ctx || keepAliveOsc) return;
        keepAliveOsc = ctx.createOscillator();
        keepAliveGain = ctx.createGain();
        keepAliveOsc.frequency.setValueAtTime(55, ctx.currentTime);
        keepAliveGain.gain.setValueAtTime(0.00005, ctx.currentTime); // Inaudible to human ear
        keepAliveOsc.connect(keepAliveGain);
        keepAliveGain.connect(ctx.destination);
        keepAliveOsc.start();
        console.log("🛡️ [AUDIO KEEP-ALIVE ACTIVE]: Prevents background OS suspension while in Google Maps!");
      } catch (e) {
        console.warn("Audio keep-alive notice:", e);
      }
    }

    // Primes and unlocks iOS Safari SpeechSynthesis & Web Audio API
    function unlockAudio() {
      getWarmAudioContext();
      startBackgroundAudioKeepAlive();
      if ('speechSynthesis' in window) {
        try {
          const silent = new SpeechSynthesisUtterance(' ');
          silent.volume = 0.01;
          window.speechSynthesis.speak(silent);
        } catch(e) {}
      }
      if (!audioUnlocked) {
        audioUnlocked = true;
        const badge = document.getElementById('audioBadge');
        if (badge) {
          badge.className = "badge active";
          const txt = document.getElementById('audioText');
          if (txt) txt.textContent = "🎧 EARPHONES ACTIVE";
        }
        console.log("🔊 5G Earphone Audio Engine Unlocked!");
      }
    }

    // --- Priority Speech Queue (Item 2: Obstacle alerts never cut off) ---
    // Priority levels:
    // 0: Critical obstacle
    // 1: Obstacle warning
    // 2: "Turn now"
    // 3: Step, prepare, route messages
    // 4: Reassurance
    const speechQueue = [];
    let speakingPriority = null;
    let speechWatchdog = null;

    function say(text, priority = 3, ttlMs = 6000) {
      if (!text) return;
      unlockAudio();
      const item = { text: String(text), priority, expires: Date.now() + ttlMs };
      // Priority 0 (critical warning) always replaces ongoing speech immediately
      if (speakingPriority === null || priority < speakingPriority || priority === 0) {
        if ('speechSynthesis' in window) {
          try { window.speechSynthesis.cancel(); } catch(e) {}
        }
        playNow(item);
      } else {
        speechQueue.push(item);
        speechQueue.sort((a, b) => a.priority - b.priority);
      }
    }

    function playNow(item) {
      if ((isRecordingAudio || isListening) && item.priority > 0) {
        speechQueue.unshift(item);
        return;
      }
      speakingPriority = item.priority;
      if (item.priority >= 2) {
        playNavChime();
      }
      if ('speechSynthesis' in window) {
        try {
          const u = new SpeechSynthesisUtterance(item.text);
          u.lang = 'en-IN';
          u.rate = item.priority === 0 ? 1.30 : (item.priority === 1 ? 1.15 : 1.0);
          u.pitch = 1.0;
          currentUtterance = u;

          // Bug 5c: iPhone speech completion watchdog
          const words = item.text.split(/\s+/).length;
          clearTimeout(speechWatchdog);
          speechWatchdog = setTimeout(() => {
            if (currentUtterance === u) {
              console.warn("⚠️ Speech synthesis watchdog fired. Resetting speech queue.");
              speakingPriority = null;
              currentUtterance = null;
              playNext();
            }
          }, words * 400 + 2000);

          u.onend = u.onerror = (e) => {
            // Bug 5a: Ignore late callbacks from previously interrupted/cancelled utterances
            if (u !== currentUtterance) return;
            clearTimeout(speechWatchdog);
            speakingPriority = null;
            currentUtterance = null;
            playNext();
          };
          window.speechSynthesis.speak(u);
        } catch (e) {
          clearTimeout(speechWatchdog);
          speakingPriority = null;
          currentUtterance = null;
          playNext();
        }
      } else {
        speakingPriority = null;
      }
    }

    function playNext() {
      while (speechQueue.length > 0 && speechQueue[0].expires < Date.now()) {
        speechQueue.shift();
      }
      if (speechQueue.length > 0) {
        playNow(speechQueue.shift());
      }
    }

    // Backward-compatible voice dispatcher
    function speakAlert(text, isFast = false) {
      say(text, isFast ? 0 : 3);
    }

    // Zero-latency acoustic earcon chime for Push-to-Talk and activation cues
    function playChime(freq = 880, duration = 0.12) {
      try {
        const ctx = getWarmAudioContext();
        if (!ctx) return;
        const now = ctx.currentTime;
        const osc = ctx.createOscillator();
        const gain = ctx.createGain();
        osc.type = 'sine';
        osc.frequency.setValueAtTime(freq, now);
        gain.gain.setValueAtTime(0.3, now);
        gain.gain.exponentialRampToValueAtTime(0.001, now + duration);
        osc.connect(gain);
        gain.connect(ctx.destination);
        osc.start(now);
        osc.stop(now + duration);
      } catch (e) {
        // Safe no-op on restricted/unlocked audio context
      }
    }

    // Zero-latency (<10ms) earcon chimes for immediate obstacle reaction time
    function playWarningChime(urgency = 'warning') {
      try {
        const ctx = getWarmAudioContext();
        if (!ctx) return;

        const now = ctx.currentTime;
        const osc = ctx.createOscillator();
        const gain = ctx.createGain();
        osc.connect(gain);
        gain.connect(ctx.destination);

        if (urgency === 'critical') {
          // Urgent TCAS-style rapid double-beep: 1050Hz -> 1350Hz at boosted gain (0.85)
          // Pierces cleanly through Google Maps voice guidance
          osc.type = 'sawtooth';
          osc.frequency.setValueAtTime(1050, now);
          osc.frequency.setValueAtTime(1350, now + 0.05);
          osc.frequency.setValueAtTime(1050, now + 0.11);
          osc.frequency.setValueAtTime(1400, now + 0.16);
          gain.gain.setValueAtTime(0.85, now);
          gain.gain.exponentialRampToValueAtTime(0.01, now + 0.24);
          osc.start(now);
          osc.stop(now + 0.25);
        } else {
          // Standard obstacle warning tone: 680Hz -> 440Hz
          osc.type = 'triangle';
          osc.frequency.setValueAtTime(680, now);
          osc.frequency.setValueAtTime(440, now + 0.08);
          gain.gain.setValueAtTime(0.40, now);
          gain.gain.exponentialRampToValueAtTime(0.01, now + 0.22);
          osc.start(now);
          osc.stop(now + 0.24);
        }
      } catch(e) {}
    }

    let proxTimer = null, proxLevel = 'CLEAR', proxSide = null, proxLastMsg = 0, proxMuted = false;
    const PROX_INTERVAL = { ADVISORY: 1000, WARNING: 330, CRITICAL: 125 };

    function proxBeep() {
      const ctx = getWarmAudioContext();
      if (!ctx || proxMuted || speakingPriority !== null) return;          // pause while speech plays
      const osc = ctx.createOscillator(), gain = ctx.createGain();
      const pan = ctx.createStereoPanner ? ctx.createStereoPanner() : null;
      osc.frequency.value = proxLevel === 'CRITICAL' ? 1200 : (proxLevel === 'WARNING' ? 900 : 600);
      if (pan) pan.pan.value = proxSide === 'left' ? -0.8 : (proxSide === 'right' ? 0.8 : 0);
      const t = ctx.currentTime;
      gain.gain.setValueAtTime(0.25, t); gain.gain.exponentialRampToValueAtTime(0.001, t + 0.08);
      osc.connect(gain);
      if (pan) { gain.connect(pan); pan.connect(ctx.destination); } else { gain.connect(ctx.destination); }
      osc.start(t); osc.stop(t + 0.09);
    }

    function setProximity(level, side) {
      proxLastMsg = Date.now(); proxSide = side;
      if (level === proxLevel && (proxTimer || level === 'CLEAR')) return;   // same level: keep rhythm
      proxLevel = level;
      if (proxTimer) { clearInterval(proxTimer); proxTimer = null; }
      if (level === 'CLEAR') return;
      proxTimer = setInterval(() => {
        if (Date.now() - proxLastMsg > 1000) { clearInterval(proxTimer); proxTimer = null; proxLevel = 'CLEAR'; return; }
        proxBeep();
      }, PROX_INTERVAL[level] || 1000);
    }

    // Handle incoming AI alerts and navigation cues from Laptop Edge Server
    function handleDownlinkMessage(msg) {
      if (msg.type === "threat_alert") {
        const isCrit = (msg.urgency === 'critical' || (msg.distance_m != null && msg.distance_m <= 1.5));

        // If user is actively speaking/recording, silence non-critical warnings but allow critical ones through
        if ((isRecordingAudio || isListening) && !isCrit) {
          const banner = document.getElementById('threatBanner');
          if (banner) {
            banner.textContent = `🚨 ${msg.text.toUpperCase()}`;
            banner.style.display = 'block';
          }
          return;
        }

        // Bug 5b: Do not cancel speech unconditionally here; say() will prioritize critical/warning alerts
        const banner = document.getElementById('threatBanner');
        banner.textContent = `🚨 ${msg.text.toUpperCase()}`;
        banner.style.display = 'block';

        // 1. Instant zero-latency earcon sound (<10ms playback)
        playWarningChime(isCrit ? 'critical' : 'warning');

        // 2. Immediate tactile haptic feedback
        if (navigator.vibrate) {
          try {
            if (msg.haptic === 'left') navigator.vibrate([300, 100, 100]);
            else if (msg.haptic === 'right') navigator.vibrate([100, 100, 300]);
            else navigator.vibrate(isCrit ? [220, 70, 220, 70, 220] : [160, 60, 160]);
          } catch(e) {}
        }

        // 3. Spoken voice alert via priority queue (0: critical, 1: hazard warning)
        say(msg.text, isCrit ? 0 : 1);

        const heroCard = document.getElementById('statusHeroCard');
        const heroIcon = document.getElementById('statusHeroIcon');
        const heroMain = document.getElementById('statusHeroMain');
        const heroSub = document.getElementById('statusHeroSub');
        if (heroCard) heroCard.className = 'status-hero-card threat-active';
        if (heroIcon) heroIcon.textContent = isCrit ? '🚨' : '⚠️';
        if (heroMain) heroMain.textContent = isCrit ? 'CRITICAL OBSTACLE STOP' : 'HAZARD CAUTION';
        if (heroSub) heroSub.textContent = msg.text;

        // Immediately update wearable distance card if distance is provided
        if (msg.distance_m != null) {
          const ultraEl = document.getElementById('ultraVal');
          if (ultraEl) ultraEl.textContent = `${(msg.distance_m * 100).toFixed(0)} cm`;
        }

        setTimeout(() => {
          banner.style.display = 'none';
          if (heroCard) heroCard.className = 'status-hero-card';
          if (heroIcon) heroIcon.textContent = '🟢';
          if (heroMain) heroMain.textContent = 'PATHWAY CLEAR';
          if (heroSub) heroSub.textContent = 'Wearable Vision & Ultrasonic Active';
        }, isCrit ? 2000 : 2500);
      }
      else if (msg.type === "proximity") {
        setProximity(msg.level || 'CLEAR', msg.side || null);
      }
      else if (msg.type === "telemetry_update") {
        // Live sensor telemetry stream from Edge Server (Wearable Ultrasonic + Speed)
        if (msg.ultra_cm != null && msg.ultra_cm < 400.0) {
          const ultraEl = document.getElementById('ultraVal');
          if (ultraEl) ultraEl.textContent = `${msg.ultra_cm.toFixed(0)} cm`;
        }
        if (msg.speed_kmh != null) {
          const speedEl = document.getElementById('speedVal');
          if (speedEl && (speedEl.textContent === '0.0 km/h' || speedEl.textContent === '--')) {
            speedEl.textContent = `${msg.speed_kmh.toFixed(1)} km/h`;
          }
        }
      }
      else if (msg.type === "speech_prompt") {
        // Problem 11: Accessible tactile haptics for blind navigation guidance
        if (msg.haptic && navigator.vibrate) {
          try {
            if (msg.haptic === "left") navigator.vibrate([300, 100, 100]);
            else if (msg.haptic === "right") navigator.vibrate([100, 100, 300]);
            else if (msg.haptic === "turn_now") navigator.vibrate([200, 80, 200, 80, 200]);
            else if (msg.haptic === "arrive") navigator.vibrate([300, 100, 300, 100, 300]);
          } catch(e) {}
        }
        // Direct spoken guidance from Edge Server via priority queue (2: turn now, 3: general/route/step)
        if (msg.text) {
          say(msg.text, msg.priority === "high" ? 2 : 3);
          const lowText = msg.text.toLowerCase();
          if (lowText.includes("navigating") || lowText.includes("calculating") || lowText.includes("route to")) {
            setPttState('success', `🧭 ${msg.text.toUpperCase()}`);
          } else if (lowText.includes("no destination") || lowText.includes("failed") || lowText.includes("again")) {
            setPttState('error', `⚠️ ${msg.text.toUpperCase()}`);
          } else {
            setPttState('standby');
          }
        } else {
          setPttState('standby');
        }
      }
      else if (msg.type === "nav_route") {
        // Full structured route received from Edge Server
        setPttState('standby');
        routeData = msg;
        currentStepIndex = 0;
        totalSteps = msg.total_steps || (msg.steps ? msg.steps.length : 1);
        
        const controls = document.getElementById('stepControls');
        const progress = document.getElementById('routeProgress');
        if (controls) controls.style.display = 'flex';
        if (progress) progress.style.display = 'block';
        
        const distEl = document.getElementById('routeDistLeft');
        if (distEl) {
          distEl.textContent = `Total Distance: ${msg.total_distance_m}m • Destination: ${msg.destination}`;
        }

        // Build dynamic Google Maps links strictly pinned to phone's live hardware GPS coordinates
        const sLat = (lastLat != null) ? lastLat : (msg.start_coords ? msg.start_coords[0] : null);
        const sLon = (lastLon != null) ? lastLon : (msg.start_coords ? msg.start_coords[1] : null);
        const dCoords = msg.dest_coords || (routeData && routeData.dest_coords);
        
        let liveMapUrl = msg.google_maps_url || "";
        let nativeScheme = "";
        
        if (dCoords) {
          const isIOS = /iPad|iPhone|iPod/.test(navigator.userAgent) && !window.MSStream;
          if (isIOS) {
            // Modern iOS Google Maps deep link protocol with explicit dir_action=navigate for direct turn-by-turn launch
            nativeScheme = `comgooglemapsurl://www.google.com/maps/dir/?api=1&destination=${dCoords[0].toFixed(6)},${dCoords[1].toFixed(6)}&travelmode=walking&dir_action=navigate`;
          } else {
            // Android Google Navigation native intent
            nativeScheme = `google.navigation:q=${dCoords[0].toFixed(6)},${dCoords[1].toFixed(6)}&mode=w`;
          }
        }
        
        if (dCoords) {
          // AUTO-START NAVIGATION: Omitting the 'origin' parameter allows Google Maps to bind directly
          // to the device's live GPS hardware, immediately initiating turn-by-turn navigation.
          liveMapUrl = `https://www.google.com/maps/dir/?api=1&destination=${dCoords[0].toFixed(6)},${dCoords[1].toFixed(6)}&travelmode=walking&dir_action=navigate`;

          // [FALLBACK PREVIEW MODE]: Kept commented out for quick recovery if explicit lat/lon is needed:
          // if (sLat != null && sLon != null && dCoords) {
          //   liveMapUrl = `https://www.google.com/maps/dir/?api=1&origin=${sLat.toFixed(6)},${sLon.toFixed(6)}&destination=${dCoords[0].toFixed(6)},${dCoords[1].toFixed(6)}&travelmode=walking&dir_action=navigate`;
          // }
        }

        cachedMapsUniversalUrl = liveMapUrl;
        cachedMapsNativeScheme = nativeScheme;
        pendingMapsLaunch = true;

        pendingMapsLaunch = false;
        const gmapsBtn = document.getElementById('gmapsBtn');
        if (gmapsBtn) {
          gmapsBtn.href = cachedMapsUniversalUrl || "#";
          gmapsBtn.target = "_blank";
          gmapsBtn.onclick = null;
          gmapsBtn.textContent = "🗺️ GOOGLE MAPS (OPTIONAL)";
          gmapsBtn.style.background = "rgba(0, 230, 118, 0.2)";
          gmapsBtn.style.color = "#00e676";
          gmapsBtn.style.fontWeight = "700";
          gmapsBtn.style.boxShadow = "none";
          gmapsBtn.classList.remove('flashing');
        }

        // Autonomous audio navigation: server announces route start and Step 1.
        // Update screen display without double-speaking.
        updateStepDisplay(0, false, false);
      }
      else if (msg.type === "nav_step_update") {
        // Sequential step progression from Edge Server (automatic GPS waypoint advancement)
        currentStepIndex = msg.step_index;
        totalSteps = msg.total_steps || totalSteps;
        updateStepDisplay(currentStepIndex, Boolean(msg.speak), false);
        
        if (msg.speak) {
          if (navigator.vibrate) { try { navigator.vibrate([100, 50, 100]); } catch(e) {} }
        }
      }
      else if (msg.type === "nav_step") {
        // Backward-compatible single step message
        const stepEl = document.getElementById('navStepText');
        if (stepEl) stepEl.textContent = msg.step;
        if (msg.speak) {
          speakAlert(msg.step);
        }
      }
    }

    // Updates navigation card and visually updates Prev/Next button states
    function updateStepDisplay(index, speakIt = true, notifyServer = false) {
      if (!routeData || !routeData.steps || routeData.steps.length === 0) return;
      const count = routeData.steps.length;
      totalSteps = count;
      
      if (index < 0) index = 0;
      if (index >= count) index = count - 1;
      
      currentStepIndex = index;
      const stepObj = routeData.steps[index];
      const stepText = (typeof stepObj === 'string') ? stepObj : (stepObj.instruction || stepObj.text || '');
      const formatted = `Step ${index + 1} of ${count}: ${stepText}`;
      
      const stepEl = document.getElementById('navStepText');
      const counterEl = document.getElementById('stepCounter');
      if (stepEl) stepEl.textContent = formatted;
      if (counterEl) counterEl.textContent = `STEP ${index + 1} OF ${count}`;
      
      // Update Prev / Next button state (both disabled attribute and CSS classes)
      const prevBtn = document.getElementById('prevStepBtn');
      const nextBtn = document.getElementById('nextStepBtn');
      const isFirst = (index <= 0);
      const isLast = (index >= count - 1);
      
      if (prevBtn) {
        prevBtn.disabled = isFirst;
        prevBtn.classList.toggle('disabled', isFirst);
      }
      if (nextBtn) {
        nextBtn.disabled = isLast;
        nextBtn.classList.toggle('disabled', isLast);
      }
      
      if (speakIt) {
        speakAlert(formatted);
      }
      
      // Notify edge server of step change over 5G WebSocket ONLY if manually requested by user buttons
      if (notifyServer && ws && ws.readyState === WebSocket.OPEN) {
        ws.send(JSON.stringify({
          type: "step_change",
          step_index: index,
          timestamp: Date.now() / 1000
        }));
      }
    }

    // Advances or rewinds steps manually with button active feedback
    function navigateStep(delta, event) {
      if (event) {
        event.preventDefault();
        event.stopPropagation();
      }
      unlockAudio();
      if (!routeData || !routeData.steps || routeData.steps.length === 0) return;
      const count = routeData.steps.length;
      const newIndex = currentStepIndex + delta;
      if (newIndex >= 0 && newIndex < count) {
        updateStepDisplay(newIndex, true, true);
        if (navigator.vibrate) { try { navigator.vibrate(50); } catch(e) {} }
      }
    }

    let pendingMapsLaunch = false;
    let cachedMapsUniversalUrl = "";
    let cachedMapsNativeScheme = "";

    // High-reliability multi-platform Google Maps launcher
    // Runs inside a direct user gesture to guarantee 100% popup bypass
    function launchGoogleMapsNavigation(event) {
      if (event) {
        try { event.preventDefault(); event.stopPropagation(); } catch(e) {}
      }
      unlockAudio();
      startBackgroundAudioKeepAlive();
      
      const isIOS = /iPad|iPhone|iPod/.test(navigator.userAgent) && !window.MSStream;
      const targetScheme = cachedMapsNativeScheme;
      const targetUniversal = cachedMapsUniversalUrl;
      
      console.log("🗺️ [LAUNCHING GOOGLE MAPS]:", { isIOS, targetScheme, targetUniversal });
      
      // 1. Attempt Native App Scheme (hands off to native app without page reload)
      if (targetScheme) {
        try {
          window.location.href = targetScheme;
        } catch (e) {
          console.warn("Native scheme attempt:", e);
          if (targetUniversal) {
            window.open(targetUniversal, '_blank');
          }
        }
      } else if (targetUniversal) {
        // 2. Fallback to Universal URL if no native scheme
        try {
          window.open(targetUniversal, '_blank');
        } catch(e) {
          window.location.href = targetUniversal;
        }
      }
      
      pendingMapsLaunch = false;
      const gmapsBtn = document.getElementById('gmapsBtn');
      if (gmapsBtn) {
        gmapsBtn.textContent = "🗺️ RUNNING IN GOOGLE MAPS";
        gmapsBtn.style.background = "rgba(0, 230, 118, 0.25)";
        gmapsBtn.style.color = "#00e676";
        gmapsBtn.style.boxShadow = "none";
        gmapsBtn.classList.remove('flashing');
      }
      if (navigator.vibrate) { try { navigator.vibrate(60); } catch(e) {} }
    }

    // Repeats current step in user's earphones with visual button feedback
    function repeatCurrentStep(event) {
      if (event) {
        event.preventDefault();
        event.stopPropagation();
      }
      unlockAudio();
      
      const btn = document.getElementById('repeatStepBtn');
      if (btn) {
        btn.classList.add('flashing');
        setTimeout(() => btn.classList.remove('flashing'), 220);
      }
      
      if (routeData && routeData.steps && routeData.steps.length > 0) {
        updateStepDisplay(currentStepIndex, true, false);
      } else {
        const stepEl = document.getElementById('navStepText');
        const text = stepEl ? stepEl.textContent : "Ready. Tap Push to Talk to begin.";
        speakAlert(text);
      }
      if (navigator.vibrate) { try { navigator.vibrate(50); } catch(e) {} }
    }

    // Accessible full-card tap handler (repeats current step instruction aloud)
    function handleCardTap() {
      repeatCurrentStep();
    }

    // Request iOS Screen WakeLock
    async function requestScreenWakeLock() {
      if ('wakeLock' in navigator) {
        try {
          await navigator.wakeLock.request('screen');
          console.log("🔒 Screen WakeLock active.");
        } catch (e) {
          console.log("WakeLock warning:", e);
        }
      }
    }

    document.addEventListener('visibilitychange', () => {
      if (document.visibilityState === 'visible') {
        requestScreenWakeLock();
        if (!ws || ws.readyState === WebSocket.CLOSED) connectWebSocket();
        requestImmediateGpsFix();
      }
    });

    // --- Web Bluetooth (BLE) Manager for Arduino UNO Q ---
    let bleDevice = null;
    let bleServer = null;
    let bleCharacteristic = null;
    let lastBleUltrasonicCm = 400.0;
    let lastBleButtonState = 0;
    let blePacketCount = 0;

    const BLE_SERVICE_UUID = "19b10000-e8f2-537e-4f6c-d104768a1214";
    const BLE_CHARACTERISTIC_UUID = "19b10001-e8f2-537e-4f6c-d104768a1214";

    async function connectBleWearable() {
      const badge = document.getElementById('bleBadge');
      const text = document.getElementById('bleText');

      if (!navigator.bluetooth) {
        alert("Web Bluetooth is not supported on this browser.\n• Android: Open in Google Chrome\n• iPhone: Open in Bluefy or WebBLE browser");
        return;
      }

      try {
        if (text) text.textContent = "PAIRING...";
        console.log("Searching for Arduino UNO Q BLE...");

        bleDevice = await navigator.bluetooth.requestDevice({
          acceptAllDevices: true,
          optionalServices: [BLE_SERVICE_UUID]
        });

        bleDevice.addEventListener('gattserverdisconnected', onBleDisconnected);

        console.log("Connecting to GATT Server...");
        bleServer = await bleDevice.gatt.connect();

        console.log("Getting BLE Service...");
        const service = await bleServer.getPrimaryService(BLE_SERVICE_UUID);

        console.log("Getting BLE Characteristic...");
        bleCharacteristic = await service.getCharacteristic(BLE_CHARACTERISTIC_UUID);

        await bleCharacteristic.startNotifications();
        bleCharacteristic.addEventListener('characteristicvaluechanged', handleBleNotification);

        blePacketCount = 0;
        if (badge) badge.className = "badge active";
        if (text) text.textContent = "🟢 UNO Q LINKED (0 pkts)";
        speakAlert("Wearable connected over Bluetooth");
        if (navigator.vibrate) { try { navigator.vibrate([100, 50, 100]); } catch(e) {} }

      } catch (err) {
        console.warn("BLE Connection notice:", err.message);
        onBleDisconnected();
      }
    }

    function onBleDisconnected() {
      const badge = document.getElementById('bleBadge');
      const text = document.getElementById('bleText');
      if (badge) badge.className = "badge";
      if (text) text.textContent = "🔗 BLE PAIR";
      blePacketCount = 0;
    }

    function handleBleNotification(event) {
      const value = event.target.value;
      const decoder = new TextDecoder('utf-8');
      const jsonStr = decoder.decode(value);

      try {
        const data = JSON.parse(jsonStr);
        blePacketCount++;

        if (data.dist != null) {
          lastBleUltrasonicCm = parseFloat(data.dist);
        }

        const text = document.getElementById('bleText');
        if (text) {
          text.textContent = `🟢 UNO Q (${blePacketCount} pkts | ${lastBleUltrasonicCm.toFixed(0)}cm)`;
        }

        if (data.btn != null) {
          const newBtn = parseInt(data.btn);
          if (newBtn === 1 && lastBleButtonState === 0) {
            console.log("🔘 [PHYSICAL BLE PTT PRESSED]: Triggering Push-to-Talk!");
            if (text) text.textContent = `🔘 PTT ACTIVE (${blePacketCount} pkts)`;
            triggerPushToTalk(null);
          }
          lastBleButtonState = newBtn;
        }
        if (data.event === "ptt_press") {
          console.log("🔘 [BLE PTT DOWN]: Starting voice capture!");
          if (text) text.textContent = `🔘 PTT ACTIVE (${blePacketCount} pkts)`;
          if (!isRecordingAudio) {
            triggerPushToTalk(null);
          }
        } else if (data.event === "ptt_release" || data.event === "ptt_click") {
          if (isRecordingAudio) {
            stopAndSendAudio();
          }
        }
      } catch (e) {}
    }

    let lastLat = null;
    let lastLon = null;
    let lastAcc = null;
    let lastSpd = 0.0;
    let lastGpsTime = 0;
    let lastFixTime = 0;
    let currentHeading = null;
    let gpsCourse = null;
    let lastHeadingLat = null;
    let lastHeadingLon = null;
    let isWalkingState = false;
    let gpsWatchId = null;
    let gpsLocked = false;
    let phoneStepCount = 0;
    let lastStepTimestamp = 0;

    // Step counting via device motion (Problem 14: peak > 1.25g, >= 300ms gap)
    if (window.DeviceMotionEvent) {
      window.addEventListener('devicemotion', (e) => {
        const acc = e.accelerationIncludingGravity;
        if (!acc) return;
        const mag = Math.sqrt((acc.x || 0)**2 + (acc.y || 0)**2 + (acc.z || 0)**2) / 9.80665;
        const nowMs = Date.now();
        if (mag > 1.25 && (nowMs - lastStepTimestamp) > 300) {
          lastStepTimestamp = nowMs;
          phoneStepCount++;
        }
      }, true);
    }

    // Direction of travel from two GPS readings (Problem 8)
    function bearing(lat1, lon1, lat2, lon2) {
      const toR = Math.PI / 180.0;
      const y = Math.sin((lon2 - lon1) * toR) * Math.cos(lat2 * toR);
      const x = Math.cos(lat1 * toR) * Math.sin(lat2 * toR) - Math.sin(lat1 * toR) * Math.cos(lat2 * toR) * Math.cos((lon2 - lon1) * toR);
      return (Math.atan2(y, x) / toR + 360.0) % 360.0;
    }

    // Compass orientation listener (Problem 8: true north on Android, absolute fallback)
    function handleOrientation(e) {
      if (e.webkitCompassHeading != null) {
        currentHeading = e.webkitCompassHeading;
      } else if (e.absolute === true && e.alpha != null) {
        currentHeading = (360 - e.alpha) % 360;
      } else if (e.alpha != null && currentHeading == null) {
        currentHeading = (360 - e.alpha) % 360;
      }
    }
    if ('ondeviceorientationabsolute' in window) {
      window.addEventListener('deviceorientationabsolute', handleOrientation, true);
    } else if (window.DeviceOrientationEvent) {
      window.addEventListener('deviceorientation', handleOrientation, true);
    }

    // Two-note accessible chime for routine navigation cues (Problem 11)
    function playNavChime() {
      try {
        const ctx = getWarmAudioContext();
        if (!ctx) return;
        const now = ctx.currentTime;
        const osc = ctx.createOscillator();
        const gain = ctx.createGain();
        osc.type = 'sine';
        osc.frequency.setValueAtTime(520, now);
        osc.frequency.setValueAtTime(660, now + 0.08);
        gain.gain.setValueAtTime(0.20, now);
        gain.gain.exponentialRampToValueAtTime(0.001, now + 0.22);
        osc.connect(gain);
        gain.connect(ctx.destination);
        osc.start(now);
        osc.stop(now + 0.24);
      } catch (e) {}
    }

    // Dispatches GPS telemetry packet over WebSocket
    function sendGpsTelemetry(lat, lon, acc = 5.0, spd = 0.0, source = "live", fixTime = null) {
      if (lat == null || lon == null) return;
      // Item 7: Drop degraded updates with accuracy > 50m when live to prevent false triggers
      if (acc > 50.0 && (source === "live" || source === "live_watch")) {
        console.warn(`📡 [GPS DROPPED]: Poor accuracy (±${acc.toFixed(1)}m > 50m)`);
        return;
      }
      const now = Date.now() / 1000.0;
      const fixTs = fixTime || lastFixTime || now;
      const fixAge = (lastFixTime > 0) ? Math.max(0.0, now - lastFixTime) : 0.0;
      
      // Calculate derived pedestrian speed & walking motion if browser reports spd = 0
      if (lastLat != null && lastLon != null && lastGpsTime > 0) {
        const dt = now - lastGpsTime;
        if (dt >= 0.5) {
          const R = 6371000.0;
          const dLat = (lat - lastLat) * Math.PI / 180.0;
          const dLon = (lon - lastLon) * Math.PI / 180.0;
          const a = Math.sin(dLat/2)*Math.sin(dLat/2) +
                    Math.cos(lastLat*Math.PI/180.0)*Math.cos(lat*Math.PI/180.0)*
                    Math.sin(dLon/2)*Math.sin(dLon/2);
          const c = 2 * Math.atan2(Math.sqrt(a), Math.sqrt(1-a));
          const distMeters = R * c;
          
          if (spd <= 0.1 && distMeters >= 1.5 && dt <= 5.0) {
            spd = (distMeters / dt) * 3.6;
          }
          // Genuine walking speed is typically 1.2 to 5.0 km/h
          if (spd >= 1.2 && distMeters >= 1.2) {
            isWalkingState = true;
          } else {
            isWalkingState = false;
          }

          // Problem 8: Direction of travel from GPS when moving at least 3m
          if (lastHeadingLat != null && lastHeadingLon != null) {
            const hDist = haversineM(lastHeadingLat, lastHeadingLon, lat, lon);
            if (hDist >= 3.0) {
              gpsCourse = bearing(lastHeadingLat, lastHeadingLon, lat, lon);
              lastHeadingLat = lat;
              lastHeadingLon = lon;
            }
          } else {
            lastHeadingLat = lat;
            lastHeadingLon = lon;
          }
        }
      } else {
        lastHeadingLat = lat;
        lastHeadingLon = lon;
      }
      
      lastLat = lat;
      lastLon = lon;
      lastAcc = acc;
      lastSpd = spd;
      lastGpsTime = now;
      gpsLocked = true;

      // Problem 8: While walking use GPS direction of travel; use compass when standing still
      const effectiveHeading = (isWalkingState && gpsCourse != null) ? gpsCourse : currentHeading;
      const headingSource = (isWalkingState && gpsCourse != null) ? "gps" : "compass";

      const el = document.getElementById('coordsVal');
      if (el) {
        el.textContent = `${lat.toFixed(5)}, ${lon.toFixed(5)}`;
        el.style.color = "#00e676";
      }
      const spdEl = document.getElementById('speedVal');
      if (spdEl) spdEl.textContent = `${spd.toFixed(1)} km/h`;
      const accEl = document.getElementById('accVal');
      if (accEl) accEl.textContent = `±${acc.toFixed(1)}m`;

      if (ws && ws.readyState === WebSocket.OPEN) {
        ws.send(JSON.stringify({
          type: "gps",
          lat: lat,
          lon: lon,
          speed_kmh: spd,
          accuracy: acc,
          moving: isWalkingState,
          ultrasonic_cm: lastBleUltrasonicCm,
          hardware_button: lastBleButtonState,
          source: source,
          fix_time: fixTs,
          fix_age: fixAge,
          heading: effectiveHeading,
          heading_source: headingSource,
          steps: phoneStepCount,
          timestamp: now
        }));
      }
    }

    // Helper for local distance calculation
    function haversineM(lat1, lon1, lat2, lon2) {
      const R = 6371000.0;
      const dLat = (lat2 - lat1) * Math.PI / 180.0;
      const dLon = (lon2 - lon1) * Math.PI / 180.0;
      const a = Math.sin(dLat/2)**2 + Math.cos(lat1 * Math.PI/180.0) * Math.cos(lat2 * Math.PI/180.0) * Math.sin(dLon/2)**2;
      return R * 2 * Math.atan2(Math.sqrt(a), Math.sqrt(1 - a));
    }

    // Fast 1-tap campus calibration for indoor testing (Problem 2: fresh fix_age=0 timestamp)
    function setManualPresetGps(lat, lon, name, event) {
      console.log(`📍 Campus preset selected: ${name} (${lat}, ${lon})`);
      document.querySelectorAll('.gps-preset-btn').forEach(b => b.classList.remove('active'));
      if (event && event.currentTarget) {
        event.currentTarget.classList.add('active');
      }
      lastFixTime = Date.now() / 1000.0;
      sendGpsTelemetry(lat, lon, 3.0, 0.0, "preset", lastFixTime);
      const el = document.getElementById('coordsVal');
      if (el) {
        el.textContent = `${name}: ${lat.toFixed(4)}, ${lon.toFixed(4)}`;
        el.style.color = "#00e5ff";
      }
      if (navigator.vibrate) { try { navigator.vibrate(40); } catch(e) {} }
    }

    // Multi-tier geolocation: Fast network fix first (<500ms), followed by fine GNSS satellite lock
    function requestImmediateGpsFix(event) {
      document.querySelectorAll('.gps-preset-btn').forEach(b => b.classList.remove('active'));
      const liveBtn = document.getElementById('liveGnssBtn') || (event ? event.currentTarget : null);
      if (liveBtn) liveBtn.classList.add('active');

      if (!("geolocation" in navigator)) {
        const el = document.getElementById('coordsVal');
        if (el) el.textContent = "GPS Not Supported";
        return;
      }

      const el = document.getElementById('coordsVal');
      if (el && !gpsLocked) el.textContent = "Acquiring GPS...";

      // Tier 1: Fast Cell/Wi-Fi positioning (works INDOORS, resolves in <500ms)
      navigator.geolocation.getCurrentPosition(
        (pos) => {
          if (pos && pos.coords) {
            const lat = pos.coords.latitude;
            const lon = pos.coords.longitude;
            const acc = (pos.coords.accuracy != null) ? pos.coords.accuracy : 15.0;
            const spd = (pos.coords.speed != null && pos.coords.speed >= 0) ? (pos.coords.speed * 3.6) : 0.0;
            lastFixTime = (pos.timestamp ? pos.timestamp / 1000.0 : Date.now() / 1000.0);
            console.log(`📡 Fast network GPS lock: (${lat}, ${lon}) ±${acc}m`);
            sendGpsTelemetry(lat, lon, acc, spd, "network", lastFixTime);
          }
        },
        (err) => {
          console.warn("Fast GPS lock notice:", err ? err.message : err);
          if (el && !gpsLocked) {
            if (err && err.code === 1) {
              el.textContent = "⚠️ Safari Blocked (Using IITG Fix)";
              el.style.color = "#ff9100";
              setTimeout(() => {
                if (!gpsLocked) {
                  setManualPresetGps(26.1885, 91.6917, "Hostel Area");
                }
              }, 1200);
            } else if (err && err.code === 3) {
              el.textContent = "📡 Indoors (Using IITG Fix)";
              el.style.color = "#ff9100";
              setTimeout(() => {
                if (!gpsLocked) {
                  setManualPresetGps(26.1885, 91.6917, "Hostel Area");
                }
              }, 1200);
            }
          }
        },
        { enableHighAccuracy: true, timeout: 10000, maximumAge: 5000 }
      );

      // Tier 2: Fine GNSS hardware satellite lock (outdoors / window)
      navigator.geolocation.getCurrentPosition(
        (pos) => {
          if (pos && pos.coords) {
            const lat = pos.coords.latitude;
            const lon = pos.coords.longitude;
            const acc = (pos.coords.accuracy != null) ? pos.coords.accuracy : 5.0;
            const spd = (pos.coords.speed != null && pos.coords.speed >= 0) ? (pos.coords.speed * 3.6) : 0.0;
            lastFixTime = (pos.timestamp ? pos.timestamp / 1000.0 : Date.now() / 1000.0);
            console.log(`🛰️ Fine GNSS satellite lock: (${lat}, ${lon}) ±${acc}m`);
            sendGpsTelemetry(lat, lon, acc, spd, "satellite", lastFixTime);
          }
        },
        (err) => {
          console.warn("Fine GNSS lock notice:", err ? err.message : err);
        },
        { enableHighAccuracy: true, timeout: 20000, maximumAge: 10000 }
      );
    }

    // Continuous watchPosition for live walking updates
    function startGpsTracking() {
      requestImmediateGpsFix();

      if (!("geolocation" in navigator)) return;
      if (gpsWatchId !== null) {
        try { navigator.geolocation.clearWatch(gpsWatchId); } catch(e) {}
        gpsWatchId = null;
      }

      gpsWatchId = navigator.geolocation.watchPosition(
        (pos) => {
          if (!pos || !pos.coords) return;
          const lat = pos.coords.latitude;
          const lon = pos.coords.longitude;
          const acc = (pos.coords.accuracy != null) ? pos.coords.accuracy : 5.0;
          const spd = (pos.coords.speed != null && pos.coords.speed >= 0) ? (pos.coords.speed * 3.6) : 0.0;
          lastFixTime = (pos.timestamp ? pos.timestamp / 1000.0 : Date.now() / 1000.0);
          sendGpsTelemetry(lat, lon, acc, spd, "live_watch", lastFixTime);
        },
        (err) => {
          console.warn("GPS watch notice:", err ? err.message : err);
        },
        { enableHighAccuracy: true, timeout: 25000, maximumAge: 5000 }
      );
    }

    // Problem 3: Independent WebSocket heartbeat ping (does NOT touch GPS variables or velocity)
    setInterval(() => {
      if (ws && ws.readyState === WebSocket.OPEN) {
        ws.send(JSON.stringify({ type: "ping", t: Date.now() / 1000 }));
      }
    }, 1500);

    let micStream = null;

    // Activate Accessible Audio Navigation & High-Accuracy GNSS Tracking
    async function activateAudioAndGps(e) {
      if (e) e.preventDefault();
      unlockAudio();
      
      const startBtn = document.getElementById('startBtn');
      if (startBtn) startBtn.textContent = "⏳ INITIALIZING AUDIO & GPS...";

      try {
        startGpsTracking();

        // Problem 9 & 14: Request compass & motion permissions on iOS Safari
        if (typeof DeviceOrientationEvent !== 'undefined' && typeof DeviceOrientationEvent.requestPermission === 'function') {
          try {
            const perm = await DeviceOrientationEvent.requestPermission();
            if (perm === 'granted') {
              window.addEventListener('deviceorientation', handleOrientation, true);
            }
          } catch(permErr) {
            console.warn("Compass permission notice:", permErr);
          }
        }
        if (typeof DeviceMotionEvent !== 'undefined' && typeof DeviceMotionEvent.requestPermission === 'function') {
          try {
            await DeviceMotionEvent.requestPermission();
          } catch(motionErr) {}
        }

        // Pre-warm dedicated Microphone Stream for Push-to-Talk
        await ensureMicStream();

        isStreaming = true;
        if (startBtn) {
          startBtn.className = "start-btn active";
          startBtn.textContent = "🟢 AUDIO & SENSORS ACTIVE";
        }
        
        const heroMain = document.getElementById('statusHeroMain');
        const heroSub = document.getElementById('statusHeroSub');
        if (heroMain) heroMain.textContent = "PATHWAY CLEAR";
        if (heroSub) heroSub.textContent = "Earphone Audio & GPS Active (Phone in pocket)";
        
        playChime(660, 0.12);
        console.log("🎧 [ACCESSIBLE AUDIO & GPS ACTIVE]: Ready for pocket navigation!");
        // Initialize background hands-free wake word listener ("Navigate Me")
        startWakeWordDetector();
      } catch (err) {
        console.warn("Activation notice:", err.message);
      }
    }

    // --- Universal Microphone Recording Engine (MediaRecorder with AudioContext fallback) ---
    let isRecordingAudio = false;
    let audioRecordTimeout = null;
    let currentAudioCtx = null;
    let scriptProcessorNode = null;
    let micSourceNode = null;
    let audioChunks = [];
    let mediaRecorder = null;
    let mediaRecorderChunks = [];
    let mediaRecorderMime = '';

    function getSupportedMimeType() {
      if (!window.MediaRecorder) return '';
      const types = [
        'audio/webm;codecs=opus',
        'audio/webm',
        'audio/mp4',
        'audio/aac',
        'audio/ogg;codecs=opus',
        'audio/wav'
      ];
      for (const t of types) {
        if (MediaRecorder.isTypeSupported && MediaRecorder.isTypeSupported(t)) {
          return t;
        }
      }
      return '';
    }

    async function ensureMicStream() {
      if (micStream && micStream.active && micStream.getAudioTracks().length > 0 && micStream.getAudioTracks()[0].enabled) {
        return micStream;
      }
      try {
        micStream = await navigator.mediaDevices.getUserMedia({
          audio: { echoCancellation: true, noiseSuppression: true, autoGainControl: true }
        });
        console.log("🎤 Microphone stream initialized cleanly and pre-warmed!");
        return micStream;
      } catch (err) {
        console.warn("Microphone access request:", err.message);
        return null;
      }
    }

    // Downsamples Float32Array from input rate (e.g. 44.1k/48k) to 16,000 Hz
    function downsampleBuffer(buffer, inputSampleRate, outputSampleRate = 16000) {
      if (inputSampleRate === outputSampleRate) return buffer;
      const sampleRateRatio = inputSampleRate / outputSampleRate;
      const newLength = Math.round(buffer.length / sampleRateRatio);
      const result = new Float32Array(newLength);
      let offsetResult = 0;
      let offsetBuffer = 0;
      while (offsetResult < result.length) {
        const nextOffsetBuffer = Math.round((offsetResult + 1) * sampleRateRatio);
        let accum = 0, count = 0;
        for (let i = offsetBuffer; i < nextOffsetBuffer && i < buffer.length; i++) {
          accum += buffer[i];
          count++;
        }
        result[offsetResult] = count > 0 ? (accum / count) : 0;
        offsetResult++;
        offsetBuffer = nextOffsetBuffer;
      }
      return result;
    }

    // Encodes normalized float32 samples to standard 16-bit PCM WAV (mono)
    function encodeWAV(samples, sampleRate = 16000) {
      const buffer = new ArrayBuffer(44 + samples.length * 2);
      const view = new DataView(buffer);

      function writeString(v, offset, str) {
        for (let i = 0; i < str.length; i++) {
          v.setUint8(offset + i, str.charCodeAt(i));
        }
      }

      // RIFF chunk descriptor
      writeString(view, 0, 'RIFF');
      view.setUint32(4, 36 + samples.length * 2, true);
      writeString(view, 8, 'WAVE');
      
      // fmt sub-chunk
      writeString(view, 12, 'fmt ');
      view.setUint32(16, 16, true);
      view.setUint16(20, 1, true); // PCM
      view.setUint16(22, 1, true); // Mono
      view.setUint32(24, sampleRate, true);
      view.setUint32(28, sampleRate * 2, true); // Byte rate
      view.setUint16(32, 2, true); // Block align
      view.setUint16(34, 16, true); // 16-bit
      
      // data sub-chunk
      writeString(view, 36, 'data');
      view.setUint32(40, samples.length * 2, true);
      
      // Write 16-bit PCM samples with clipping protection
      let offset = 44;
      for (let i = 0; i < samples.length; i++, offset += 2) {
        const s = Math.max(-1.0, Math.min(1.0, samples[i]));
        view.setInt16(offset, s < 0 ? s * 0x8000 : s * 0x7FFF, true);
      }
      return buffer;
    }

    // Converts ArrayBuffer to Base64 in fast 32KB chunks
    function arrayBufferToBase64(buffer) {
      let binary = '';
      const bytes = new Uint8Array(buffer);
      const chunkSize = 0x8000;
      for (let i = 0; i < bytes.length; i += chunkSize) {
        binary += String.fromCharCode.apply(null, bytes.subarray(i, i + chunkSize));
      }
      return window.btoa(binary);
    }

    // Centralized Push-To-Talk State Machine
    let lastPttTapTime = 0;
    let isListening = false;

    function setPttState(state, text = '') {
      const btn = document.getElementById('pttBtn');
      if (!btn) return;
      btn.style.background = '';

      switch(state) {
        case 'standby':
          isListening = false;
          btn.className = 'ptt-btn';
          btn.textContent = '🎙️ SAY "NAVIGATE ME" OR TAP TO SPEAK';
          restartWakeWordDetector();
          break;
        case 'recording':
          btn.className = 'ptt-btn recording';
          btn.textContent = text || '🔴 LISTENING... TAP TO STOP';
          break;
        case 'processing':
          btn.className = 'ptt-btn processing';
          btn.textContent = text || '🧠 TRANSCRIBING WITH WHISPER AI...';
          break;
        case 'success':
          isListening = false;
          btn.className = 'ptt-btn success';
          btn.textContent = text || '🧭 ROUTING CONFIRMED';
          setTimeout(() => setPttState('standby'), 3200);
          break;
        case 'error':
          isListening = false;
          btn.className = 'ptt-btn error';
          btn.textContent = text || '⚠️ NO AUDIO HEARD. TAP TO RETRY';
          setTimeout(() => setPttState('standby'), 2800);
          break;
      }
    }

    function resetPttButton() {
      setPttState('standby');
    }

    // Stops MediaRecorder / ScriptProcessor recording, packages audio, sends to 5G Edge Server
    function stopAndSendAudio() {
      if (!isListening && !isRecordingAudio) return;
      isRecordingAudio = false;
      isListening = false;
      clearTimeout(audioRecordTimeout);
      playNext(); // Item 2: Resume any queued speech alerts that were held during recording
      setPttState('processing', '🧠 TRANSCRIBING WITH WHISPER AI...');

      // 1. MediaRecorder Path (Universal standard for iOS Safari & Android Chrome)
      if (mediaRecorder && mediaRecorder.state !== 'inactive') {
        try {
          mediaRecorder.onstop = () => {
            try {
              const mime = mediaRecorder.mimeType || mediaRecorderMime || 'audio/mp4';
              const blob = new Blob(mediaRecorderChunks, { type: mime });
              mediaRecorderChunks = [];

              if (blob.size === 0) {
                console.warn("⚠️ MediaRecorder produced 0 bytes.");
                setPttState('error', '⚠️ NO AUDIO HEARD. TAP TO RETRY');
                return;
              }

              let detectedFormat = "wav";
              if (mime.includes("mp4") || mime.includes("aac") || mime.includes("m4a")) {
                detectedFormat = "mp4";
              } else if (mime.includes("webm")) {
                detectedFormat = "webm";
              } else if (mime.includes("ogg")) {
                detectedFormat = "ogg";
              }

              const reader = new FileReader();
              reader.onloadend = () => {
                try {
                  const base64Data = reader.result.split(',')[1];
                  console.log(`🎙️ [5G AUDIO UPLINK]: ${blob.size} bytes (${mime}, format: ${detectedFormat}). Sending to Whisper AI...`);

                  const audioPayload = JSON.stringify({
                    type: "voice_audio",
                    format: detectedFormat,
                    mime_type: mime,
                    data: base64Data,
                    lat: lastLat,
                    lon: lastLon,
                    accuracy: lastAcc,
                    speed_kmh: lastSpd,
                    fix_time: lastFixTime,
                    fix_age: (lastFixTime > 0) ? Math.max(0.0, (Date.now() / 1000.0) - lastFixTime) : 0.0,
                    heading: currentHeading,
                    timestamp: Date.now() / 1000
                  });

                  if (ws && ws.readyState === WebSocket.OPEN) {
                    ws.send(audioPayload);
                  } else {
                    console.warn("⚠️ 5G link reconnecting. Queuing audio uplink...");
                    connectWebSocket();
                    setTimeout(() => {
                      if (ws && ws.readyState === WebSocket.OPEN) {
                        ws.send(audioPayload);
                        console.log("🚀 [QUEUED 5G AUDIO SENT SUCCESSFULLY]");
                      } else {
                        setPttState('error', '⚠️ 5G UPLINK RETRY FAILED');
                      }
                    }, 800);
                  }
                } catch (b64Err) {
                  console.error("Base64 conversion error:", b64Err);
                  setPttState('error', '⚠️ AUDIO PACKAGING FAILED');
                }
              };
              reader.readAsDataURL(blob);
            } catch (stopErr) {
              console.error("MediaRecorder processing error:", stopErr);
              setPttState('error', '⚠️ AUDIO PACKAGING FAILED');
            }
          };

          mediaRecorder.stop();
        } catch (mrStopErr) {
          console.warn("MediaRecorder stop error:", mrStopErr);
        }

        setTimeout(() => {
          const btn = document.getElementById('pttBtn');
          if (btn && btn.classList.contains('processing')) {
            setPttState('standby');
          }
        }, 7000);
        return;
      }

      // 2. Legacy ScriptProcessor / WAV Fallback (for older browsers without MediaRecorder)
      try {
        if (scriptProcessorNode) {
          scriptProcessorNode.disconnect();
          scriptProcessorNode = null;
        }
        if (micSourceNode) {
          micSourceNode.disconnect();
          micSourceNode = null;
        }

        const inputRate = (currentAudioCtx && currentAudioCtx.sampleRate) ? currentAudioCtx.sampleRate : 44100;
        if (currentAudioCtx && currentAudioCtx.state !== 'closed') {
          currentAudioCtx.close().catch(() => {});
        }
        currentAudioCtx = null;

        let totalLen = 0;
        for (let i = 0; i < audioChunks.length; i++) {
          totalLen += audioChunks[i].length;
        }

        if (totalLen === 0) {
          console.warn("No audio samples collected.");
          setPttState('error', '⚠️ NO AUDIO HEARD. TAP TO RETRY');
          return;
        }

        const mergedBuffer = new Float32Array(totalLen);
        let offset = 0;
        for (let i = 0; i < audioChunks.length; i++) {
          mergedBuffer.set(audioChunks[i], offset);
          offset += audioChunks[i].length;
        }

        const wavBuffer = encodeWAV(mergedBuffer, inputRate);
        const base64Wav = arrayBufferToBase64(wavBuffer);

        console.log(`🎙️ [5G AUDIO UPLINK (WAV)]: ${mergedBuffer.length} samples at ${inputRate}Hz (~${(base64Wav.length/1024).toFixed(1)} KB). Sending...`);

        if (ws && ws.readyState === WebSocket.OPEN) {
          ws.send(JSON.stringify({
            type: "voice_audio",
            format: "wav",
            sample_rate: inputRate,
            data: base64Wav,
            lat: lastLat,
            lon: lastLon,
            accuracy: lastAcc,
            speed_kmh: lastSpd,
            fix_time: lastFixTime,
            fix_age: (lastFixTime > 0) ? Math.max(0.0, (Date.now() / 1000.0) - lastFixTime) : 0.0,
            heading: currentHeading,
            timestamp: Date.now() / 1000
          }));
        }

        // Fallback reset if server doesn't respond
        setTimeout(() => {
          const btn = document.getElementById('pttBtn');
          if (btn && btn.classList.contains('processing')) {
            setPttState('standby');
          }
        }, 7000);

      } catch (err) {
        console.error("Audio packaging error:", err);
        setPttState('error', '⚠️ AUDIO PACKAGING FAILED');
      }
    }

    // Safety fallback prompt if browser blocks microphone
    function promptDestinationFallback() {
      isListening = false;
      isRecordingAudio = false;
      const fallback = prompt("Mic unavailable. Type your destination:", "New SAC");
      if (fallback && fallback.trim()) {
        const text = fallback.trim();
        setPttState('success', `🧭 ROUTING: ${text.toUpperCase()}`);
        // Bug 11: Server announces "Calculating walking route to X...", do not speak locally
        if (ws && ws.readyState === WebSocket.OPEN) {
          ws.send(JSON.stringify({
            type: "voice_command",
            command: text,
            text: text,
            is_exact: true,
            lat: lastLat,
            lon: lastLon,
            accuracy: lastAcc,
            speed_kmh: lastSpd,
            fix_time: lastFixTime,
            fix_age: (lastFixTime > 0) ? Math.max(0.0, (Date.now() / 1000.0) - lastFixTime) : 0.0,
            heading: currentHeading,
            timestamp: Date.now() / 1000
          }));
        }
      } else {
        setPttState('standby');
      }
    }

    // --- Hands-Free Conversational Wake-Word Trigger ("Navigate Me") ---
    let wakeWordRecognizer = null;
    let isWakeWordListening = false;
    let lastWakeWordTime = 0;

    function startWakeWordDetector() {
      const SpeechRecognition = window.SpeechRecognition || window.webkitSpeechRecognition;
      if (!SpeechRecognition) {
        console.log("ℹ️ Web SpeechRecognition not available in this browser. Push-to-Talk button remains active.");
        return;
      }
      if (wakeWordRecognizer) {
        try { wakeWordRecognizer.abort(); } catch(e) {}
        wakeWordRecognizer = null;
      }

      try {
        wakeWordRecognizer = new SpeechRecognition();
        wakeWordRecognizer.continuous = true;
        wakeWordRecognizer.interimResults = true;
        wakeWordRecognizer.maxAlternatives = 1;
        wakeWordRecognizer.lang = 'en-IN'; // Optimized for Indian English & IITG campus speech

        wakeWordRecognizer.onresult = (event) => {
          // Do not trigger wake word if already recording or listening
          if (isRecordingAudio || isListening) return;

          for (let i = event.resultIndex; i < event.results.length; i++) {
            const rawTranscript = event.results[i][0].transcript.toLowerCase().trim();
            console.log("👂 [AMBIENT AUDIO]:", rawTranscript);

            // Trigger match: "navigate me", "navigate to", or "navigator"
            if (rawTranscript.includes("navigate me") || 
               (rawTranscript.includes("navigate") && rawTranscript.includes("me"))) {
              const now = Date.now();
              if (now - lastWakeWordTime < 2500) return; // Debounce
              lastWakeWordTime = now;

              console.log("🎯 [WAKE-WORD ACTIVATED]: 'Navigate me' recognized!");
              pauseWakeWordDetector();

              // Conversational response through earphones
              speakAlert("Where do you want to go?");
              setPttState('recording', '🎙️ "WHERE DO YOU WANT TO GO?"');
              if (navigator.vibrate) { try { navigator.vibrate([100, 50, 100]); } catch(e) {} }

              // Open clean 4.5-second recording window after prompt finishes
              setTimeout(() => {
                startVoiceDestinationCapture(4500);
              }, 1300);
              break;
            }
          }
        };

        wakeWordRecognizer.onend = () => {
          // Automatically keep wake word listener alive in the background
          if (isStreaming && !isRecordingAudio && !isListening) {
            setTimeout(() => {
              if (isStreaming && !isRecordingAudio && !isListening && wakeWordRecognizer) {
                try { wakeWordRecognizer.start(); } catch(e) {}
              }
            }, 300);
          }
        };

        wakeWordRecognizer.onerror = (e) => {
          if (e.error !== 'no-speech' && e.error !== 'aborted') {
            console.log("ℹ️ Wake-word notice:", e.error);
          }
        };

        wakeWordRecognizer.start();
        isWakeWordListening = true;
        console.log("👂 [HANDS-FREE WAKE-WORD]: Listening for 'Navigate me' in earphones...");
        const pttBtn = document.getElementById('pttBtn');
        if (pttBtn && !pttBtn.classList.contains('recording')) {
          pttBtn.textContent = '🎙️ SAY "NAVIGATE ME" OR TAP TO SPEAK';
        }
      } catch (err) {
        console.warn("Wake-word initialization notice:", err.message);
      }
    }

    function pauseWakeWordDetector() {
      if (wakeWordRecognizer) {
        try { wakeWordRecognizer.abort(); } catch(e) {}
      }
    }

    function restartWakeWordDetector() {
      if (!isRecordingAudio && !isListening && isStreaming) {
        setTimeout(() => {
          startWakeWordDetector();
        }, 600);
      }
    }

    // Unified voice destination recording (Shared by Hands-Free Wake-Word & Manual Button)
    async function startVoiceDestinationCapture(timeoutMs = 5000) {
      if (isRecordingAudio || isListening) return;
      unlockAudio();
      pauseWakeWordDetector();

      isListening = true;
      isRecordingAudio = true;
      setPttState('recording', '🔴 LISTENING... TAP TO STOP (OR WAIT 5s)');
      if (navigator.vibrate) {
        try { navigator.vibrate(60); } catch(e) {}
      }
      try { playChime(880, 0.12); } catch(e) {}

      if (ws && ws.readyState === WebSocket.OPEN) {
        ws.send(JSON.stringify({ type: "ptt_start", timestamp: Date.now() / 1000 }));
      } else {
        console.warn("WebSocket not open during PTT tap, initiating reconnect...");
        connectWebSocket();
      }

      clearTimeout(audioRecordTimeout);
      audioRecordTimeout = setTimeout(() => {
        if (isRecordingAudio || isListening) {
          stopAndSendAudio();
        }
      }, timeoutMs);

      // 1. Ensure microphone stream is active
      const stream = await ensureMicStream();
      if (!stream) {
        clearTimeout(audioRecordTimeout);
        promptDestinationFallback();
        return;
      }

      if (!isListening) return;

      // 2. Prefer MediaRecorder (supported natively on iOS Safari 14.3+ and Android Chrome)
      if (window.MediaRecorder) {
        try {
          mediaRecorderChunks = [];
          const mime = getSupportedMimeType();
          mediaRecorderMime = mime;
          const options = mime ? { mimeType: mime } : {};
          mediaRecorder = new MediaRecorder(stream, options);

          mediaRecorder.ondataavailable = (event) => {
            if (event.data && event.data.size > 0) {
              mediaRecorderChunks.push(event.data);
            }
          };

          mediaRecorder.start(100); // Collect chunks every 100ms
          console.log(`🎙️ Destination recording started via MediaRecorder (${mediaRecorder.mimeType || mime || 'default'})`);
          return;
        } catch (mrErr) {
          console.warn("MediaRecorder init failed, falling back to Web Audio API:", mrErr);
        }
      }

      // 3. Fallback: Web Audio API ScriptProcessorNode
      try {
        const AudioContextClass = window.AudioContext || window.webkitAudioContext;
        currentAudioCtx = new AudioContextClass();
        if (currentAudioCtx.state === 'suspended') {
          await currentAudioCtx.resume();
        }

        micSourceNode = currentAudioCtx.createMediaStreamSource(stream);
        scriptProcessorNode = currentAudioCtx.createScriptProcessor(4096, 1, 1);
        audioChunks = [];

        scriptProcessorNode.onaudioprocess = (e) => {
          if (!isRecordingAudio) return;
          const channel = e.inputBuffer.getChannelData(0);
          audioChunks.push(new Float32Array(channel));
          const out = e.outputBuffer.getChannelData(0);
          out.fill(0);
        };

        micSourceNode.connect(scriptProcessorNode);
        scriptProcessorNode.connect(currentAudioCtx.destination);
        console.log("🎙️ Destination recording started via ScriptProcessor at", currentAudioCtx.sampleRate, "Hz");
      } catch (err) {
        console.error("Audio recording start error:", err);
        clearTimeout(audioRecordTimeout);
        promptDestinationFallback();
      }
    }

    // Push-to-Talk manual handler (Tap to speak / Tap to stop fallback)
    async function triggerPushToTalk(event) {
      if (event) event.preventDefault();
      unlockAudio();

      const now = Date.now();
      if (now - lastPttTapTime < 400) return;
      lastPttTapTime = now;

      // TOGGLE OFF: if currently recording or listening, stop and send immediately
      if (isRecordingAudio || isListening) {
        stopAndSendAudio();
        return;
      }

      // TOGGLE ON: manual recording start (5 seconds window)
      startVoiceDestinationCapture(5000);
    }

    // Instant Text Destination Dispatcher
    function sendTextDestination() {
      const input = document.getElementById('destTextInput');
      const goBtn = document.querySelector('.manual-dest-btn');
      if (!input) return;
      const text = input.value.trim();
      if (!text) return;

      unlockAudio();
      input.value = '';
      if (goBtn) {
        goBtn.textContent = '⏳';
        setTimeout(() => { if (goBtn) goBtn.textContent = 'GO ➡️'; }, 2000);
      }
      setPttState('success', `🧭 ROUTING: ${text.toUpperCase()}`);
      // Bug 11: Server announces "Calculating walking route to X...", do not speak locally
      if (ws && ws.readyState === WebSocket.OPEN) {
        ws.send(JSON.stringify({
          type: "voice_command",
          command: text,
          text: text,
          is_exact: true,
          lat: lastLat,
          lon: lastLon,
          accuracy: lastAcc,
          speed_kmh: lastSpd,
          fix_time: lastFixTime,
          fix_age: (lastFixTime > 0) ? Math.max(0.0, (Date.now() / 1000.0) - lastFixTime) : 0.0,
          heading: currentHeading,
          timestamp: Date.now() / 1000
        }));
      }
    }

    // --- Dual Camera Source Switcher (Arduino Wearable Camera <-> Phone Back Camera) ---
    // Toggle: false = Arduino Wearable Camera (Default), true = Phone Back Camera
    let USE_PHONE_CAMERA = false;
    let phoneCameraStream = null;
    let phoneCameraInterval = null;
    let phoneCameraVideoEl = null;
    let phoneCameraCanvas = null;

    async function startPhoneCameraStream() {
      if (phoneCameraStream) return;
      try {
        if (!phoneCameraVideoEl) {
          phoneCameraVideoEl = document.createElement('video');
          phoneCameraVideoEl.setAttribute('autoplay', '');
          phoneCameraVideoEl.setAttribute('playsinline', '');
          phoneCameraVideoEl.setAttribute('muted', '');
          phoneCameraVideoEl.style.display = 'none';
          document.body.appendChild(phoneCameraVideoEl);
        }

        if (!phoneCameraCanvas) {
          phoneCameraCanvas = document.createElement('canvas');
          phoneCameraCanvas.width = 480;
          phoneCameraCanvas.height = 320;
        }

        const stream = await navigator.mediaDevices.getUserMedia({
          video: {
            facingMode: { ideal: 'environment' }, // Dedicated back/rear camera for pedestrian path
            width: { ideal: 480 },
            height: { ideal: 320 }
          },
          audio: false
        });

        phoneCameraStream = stream;
        phoneCameraVideoEl.srcObject = stream;
        await phoneCameraVideoEl.play();

        const ctx = phoneCameraCanvas.getContext('2d');
        const sendFrame = () => {
          if (!USE_PHONE_CAMERA || !phoneCameraStream || !ws || ws.readyState !== WebSocket.OPEN) return;
          if (phoneCameraVideoEl.videoWidth > 0 && phoneCameraVideoEl.videoHeight > 0) {
            ctx.drawImage(phoneCameraVideoEl, 0, 0, 480, 320);
            phoneCameraCanvas.toBlob((blob) => {
              if (blob && ws && ws.readyState === WebSocket.OPEN && USE_PHONE_CAMERA) {
                ws.send(blob);
              }
            }, 'image/jpeg', 0.60);
          }
        };

        // Stream at smooth 10-12 FPS (~90ms interval) for low 5G latency
        phoneCameraInterval = setInterval(sendFrame, 90);
        console.log("📱 [PHONE CAMERA ACTIVE]: Streaming back camera frames to 5G Edge Server.");
        updateCameraUI(true);
      } catch (err) {
        console.warn("Phone camera initialization notice:", err.message);
        USE_PHONE_CAMERA = false;
        updateCameraUI(false);
      }
    }

    function stopPhoneCameraStream() {
      if (phoneCameraInterval) {
        clearInterval(phoneCameraInterval);
        phoneCameraInterval = null;
      }
      if (phoneCameraStream) {
        phoneCameraStream.getTracks().forEach(track => track.stop());
        phoneCameraStream = null;
      }
      if (phoneCameraVideoEl) {
        phoneCameraVideoEl.srcObject = null;
        phoneCameraVideoEl.remove();
        phoneCameraVideoEl = null;
      }
      phoneCameraCanvas = null;
      console.log("📷 [CAMERA SWITCHED]: Reverted to Arduino Wearable Camera.");
      updateCameraUI(false);
    }

    function toggleCameraSource(e) {
      if (e) {
        e.preventDefault();
        e.stopPropagation();
      }
      unlockAudio();
      USE_PHONE_CAMERA = !USE_PHONE_CAMERA;
      if (USE_PHONE_CAMERA) {
        startPhoneCameraStream();
      } else {
        stopPhoneCameraStream();
      }
    }

    function updateCameraUI(isPhone) {
      const text = document.getElementById('cameraToggleText');
      const btn = document.getElementById('cameraToggleBtn');
      if (text) {
        text.textContent = isPhone ? "📱 CAM: PHONE" : "📷 CAM: ARDUINO";
      }
      if (btn) {
        btn.classList.toggle('active', isPhone);
        if (isPhone) {
          btn.style.borderColor = "#00e676";
          btn.style.color = "#00e676";
        } else {
          btn.style.borderColor = "";
          btn.style.color = "";
        }
      }
      if (navigator.vibrate) { try { navigator.vibrate(40); } catch(e) {} }
    }

    // Problem 11: "Where am I?" query via long press or right-click context menu
    function queryWhereAmI() {
      if (ws && ws.readyState === WebSocket.OPEN) {
        ws.send(JSON.stringify({
          type: "nav_query",
          query: "where_am_i",
          lat: lastLat,
          lon: lastLon,
          fix_time: lastFixTime,
          timestamp: Date.now() / 1000
        }));
        if (navigator.vibrate) try { navigator.vibrate([80, 50, 80]); } catch(e) {}
      }
    }

    // Connect automatically when page loads & initialize audio/GPS
    window.addEventListener('DOMContentLoaded', () => {
      connectWebSocket();
      startGpsTracking();
      activateAudioAndGps();

      // Bug 6: Full-screen gestures for blind pedestrians (ignoring UI buttons/inputs)
      function isInteractiveTarget(e) {
        return !!(e.target && e.target.closest && e.target.closest('button, input, a, select, textarea, .gps-preset-btn, #micButton, #sendBtn'));
      }

      let pressTimer = null;
      let lastTap = 0;

      document.body.addEventListener('touchstart', (e) => {
        if (isInteractiveTarget(e)) return;
        pressTimer = setTimeout(queryWhereAmI, 750);
      }, { passive: true });

      document.body.addEventListener('touchend', (e) => {
        clearTimeout(pressTimer);
        if (isInteractiveTarget(e)) return;
        const currentTime = new Date().getTime();
        const tapGap = currentTime - lastTap;
        if (tapGap < 350 && tapGap > 40) {
          confirmDestination();
        }
        lastTap = currentTime;
      });

      document.body.addEventListener('touchcancel', () => clearTimeout(pressTimer));
      document.body.addEventListener('contextmenu', (e) => {
        if (!isInteractiveTarget(e)) {
          e.preventDefault();
          queryWhereAmI();
        }
      });
      document.body.addEventListener('dblclick', (e) => {
        if (!isInteractiveTarget(e)) {
          confirmDestination();
        }
      });
    });

    // Problem 11: Destination double-tap confirmation handler
    function confirmDestination() {
      if (ws && ws.readyState === WebSocket.OPEN) {
        console.log("👆👆 [DESTINATION CONFIRMED]: Double-tap sent to server!");
        ws.send(JSON.stringify({ type: "confirm_destination", ok: true }));
        if (navigator.vibrate) try { navigator.vibrate([100, 50, 100]); } catch(e) {}
      }
    }

    // If browser policy requires a user gesture for audio unlock, tap anywhere to unlock
    document.addEventListener('click', async () => {
      unlockAudio();
      ensureMicStream();
      if (!isStreaming) {
        activateAudioAndGps();
      }

      // Bug 4d: Request iOS orientation & motion permissions inside genuine user click gesture
      if (typeof DeviceOrientationEvent !== 'undefined' && typeof DeviceOrientationEvent.requestPermission === 'function') {
        try {
          const perm = await DeviceOrientationEvent.requestPermission();
          if (perm === 'granted') {
            window.addEventListener('deviceorientation', handleOrientation, true);
          }
        } catch(e) {}
      }
      if (typeof DeviceMotionEvent !== 'undefined' && typeof DeviceMotionEvent.requestPermission === 'function') {
        try {
          await DeviceMotionEvent.requestPermission();
        } catch(e) {}
      }
    }, { once: true });
