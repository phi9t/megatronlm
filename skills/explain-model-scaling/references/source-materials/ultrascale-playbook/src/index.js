// import { plotClusters } from './clusters'
import { init_memory_plot } from './memory'
import { loadFragments } from './fragmentLoader'
import { syncHFSpacesURLHash } from './syncHFSpacesURLHash'
// Dark mode is now handled manually via a CSS class on <html> and injected styles

document.addEventListener("DOMContentLoaded", () => {
    console.log("DOMContentLoaded");

    // Inject minimal styles for the theme toggle button
    const styleEl = document.createElement('style');
    styleEl.textContent = `
    .theme-toggle-btn{position:absolute;top:16px;left:16px;z-index:10000;display:inline-flex;align-items:center;justify-content:center;width:40px;height:40px;border-radius:999px;background:rgba(255,255,255,0.9);backdrop-filter:saturate(150%) blur(6px);cursor:pointer;border:1px solid transparent;outline:none;box-shadow:none;-webkit-appearance:none;appearance:none;-webkit-tap-highlight-color:transparent}
    .theme-toggle-btn:hover{border-color:transparent;box-shadow:none}
    .theme-toggle-btn:focus,.theme-toggle-btn:focus-visible{outline:none;border-color:transparent;box-shadow:none}
    .theme-toggle-btn img{width:22px;height:22px;transition:filter .15s ease}
    .theme-toggle-btn.dark img{filter: brightness(0) invert(1)}
    @media (prefers-color-scheme: dark){.theme-toggle-btn{background:rgba(30,30,30,0.85);border-color:transparent;box-shadow:none}}
    `;
    document.head.appendChild(styleEl);

    // Inject dark mode CSS (scoped via html.dark)
    const darkCSS = `
    html.dark{color-scheme:dark}
    html.dark body{background:#242525;color:#e5e7eb}
    html.dark a{color:#93c5fd}
    html.dark .figure-legend{color:#9ca3af}
    html.dark d-article, html.dark d-article p, html.dark d-article aside{color:white !important;}
    html.dark d-contents{background:#242525}
    html.dark d-contents nav a{color:#cbd5e1}
    html.dark d-contents nav a:hover{text-decoration:underline solid rgba(255,255,255,0.6)}
    html.dark .note-box{background:#111;border-left-color:#888}
    html.dark .note-box-title{color:#d1d5db}
    html.dark .note-box-content{color:#e5e7eb}
    html.dark .large-image-background{background:#242525}
    html.dark .boxed-image{background:#111;border-color:#262626;box-shadow:0 4px 6px rgba(0,0,0,.6)}
    html.dark #graph-all,html.dark #controls,html.dark .memory-block,html.dark .activation-memory,html.dark .gradient-memory{background:#111;border-color:#262626;box-shadow:0 4px 6px rgba(0,0,0,.6);color:#e5e7eb}
    html.dark label,html.dark .memory-title{color:#e5e7eb}
    html.dark .memory-value{color:#93c5fd}
    html.dark input,html.dark select,html.dark textarea{background:#0f0f0f;color:#e5e7eb;border:1px solid #333}
    html.dark input:hover,html.dark select:hover,html.dark textarea:hover{border-color:#60a5fa}
    html.dark input:focus,html.dark select:focus,html.dark textarea:focus{border-color:#3b82f6;box-shadow:0 0 0 2px rgba(59,130,246,0.35)}
    html.dark input[type=range]{background:#333}
    html.dark input[type=range]::-webkit-slider-thumb{background:#3b82f6}
    html.dark .plotly_caption{color:#9ca3af}
    html.dark .theme-toggle-btn{background:rgba(30,30,30,0.85);border-color:transparent}
    html.dark d-article img{background:white}
    html.dark summary {color:black !important;}
    html.dark .katex-container {color:white !important;}
    html.dark d-code {background: white!important;}
    html.dark .code-block div { background: white!important;}
    html.dark .code-block div p { color: black!important;}
    /* Table borders in dark mode */
    html.dark table{border-color:rgba(255,255,255,0.3)}
    html.dark th,html.dark td{border-color:rgba(255,255,255,0.3)}
    html.dark thead tr,html.dark tbody tr{border-color:rgba(255,255,255,0.3)}
    html.dark d-byline, html.dark d-article{border-top: 1px solid rgba(255, 255, 255, 0.5);}
    html.dark d-byline h3{color:white;}
    html.dark d-math *, html.dark span.katex{color:white !important;}
    html.dark d-appendix { color: white}
    html.dark h2 { border-bottom: 1px solid rgba(255, 255, 255, 0.25);}
    html.dark h1, html.dark h2, html.dark h3, html.dark h4, html.dark h5, html.dark h6 { color: white}
    html.dark .code-area { color: black;}
    html.dark .code-area a { color: black!important;}
    
    `;
    const darkStyleEl = document.createElement('style');
    darkStyleEl.id = 'darkmode-css';
    darkStyleEl.textContent = darkCSS;
    document.head.appendChild(darkStyleEl);

	// Inject equivalent dark CSS into all ShadowRoots using :host-context(.dark)
	// This ensures styles also apply inside web components with Shadow DOM
	const shadowDarkCSS = darkCSS.replace(/html\.dark/g, ':host-context(.dark)');

	const injectDarkStylesIntoRoot = (root) => {
		// Only target ShadowRoots here
		if (!root || !(root instanceof ShadowRoot)) return;
		if (root.querySelector('style#darkmode-css-shadow')) return;
		const style = document.createElement('style');
		style.id = 'darkmode-css-shadow';
		style.textContent = shadowDarkCSS;
		root.appendChild(style);
	};

	// Normalize inline SVGs: ensure viewBox and preserveAspectRatio for responsiveness
	const normalizeSvgElement = (svgEl) => {
		try {
			if (!svgEl || svgEl.hasAttribute('viewBox')) return;
			const widthAttr = svgEl.getAttribute('width');
			const heightAttr = svgEl.getAttribute('height');
			if (!widthAttr || !heightAttr) return;
			const width = parseFloat(widthAttr);
			const height = parseFloat(heightAttr);
			if (!Number.isFinite(width) || !Number.isFinite(height) || width <= 0 || height <= 0) return;
			svgEl.setAttribute('viewBox', `0 0 ${width} ${height}`);
			if (!svgEl.hasAttribute('preserveAspectRatio')) {
				svgEl.setAttribute('preserveAspectRatio', 'xMidYMid meet');
			}
		} catch (_) {
			// no-op
		}
	};

	const processRootForSVGs = (root) => {
		if (!root || typeof root.querySelectorAll !== 'function') return;
		const svgs = root.querySelectorAll('svg:not([viewBox])');
		svgs.forEach((svg) => normalizeSvgElement(svg));
	};

	const scanNodeForShadowRoots = (node) => {
		if (!node) return;
		if (node.shadowRoot) {
			injectDarkStylesIntoRoot(node.shadowRoot);
			processRootForSVGs(node.shadowRoot);
		}
		// Traverse children
		if (node.childNodes && node.childNodes.length) {
			node.childNodes.forEach((child) => {
				// Process SVGs in this subtree as well
				processRootForSVGs(child);
				scanNodeForShadowRoots(child);
			});
		}
	};

	// Intercept future shadow roots
	const originalAttachShadow = Element.prototype.attachShadow;
	Element.prototype.attachShadow = function(init) {
		const shadow = originalAttachShadow.call(this, init);
		try {
			injectDarkStylesIntoRoot(shadow);
			processRootForSVGs(shadow);
		} catch (e) {}
		return shadow;
	};

	// Initial sweep for any existing shadow roots
	scanNodeForShadowRoots(document.documentElement);
	// Initial pass for regular DOM SVGs
	processRootForSVGs(document);

	// Observe DOM mutations to catch dynamically added components
	const mo = new MutationObserver((mutations) => {
		for (const m of mutations) {
			m.addedNodes && m.addedNodes.forEach((n) => {
				scanNodeForShadowRoots(n);
				processRootForSVGs(n);
			});
		}
	});
	mo.observe(document.documentElement, { childList: true, subtree: true });

    // Create the toggle button
    const btn = document.createElement('button');
    btn.className = 'theme-toggle-btn';
    btn.setAttribute('type', 'button');
    btn.setAttribute('aria-label', 'Basculer le mode sombre');
    // Reuse icons declared in HTML and move them into the button
    const sunIcon = document.getElementById('sunIcon');
    const moonIcon = document.getElementById('moonIcon');

    if (sunIcon && moonIcon) {
        // Make sure they adopt button sizing
        sunIcon.style.display = 'none';
        sunIcon.style.width = '22px';
        sunIcon.style.height = '22px';
        moonIcon.style.display = 'none';
        moonIcon.style.width = '22px';
        moonIcon.style.height = '22px';
        btn.appendChild(sunIcon);
        btn.appendChild(moonIcon);
    }
    document.body.appendChild(btn);

    const setIcon = (enabled) => {
        // enabled = dark mode enabled -> show sun (to indicate turning off), hide moon
        sunIcon.style.display = enabled ? '' : 'none';
        moonIcon.style.display = enabled ? 'none' : '';
        btn.setAttribute('title', enabled ? 'Désactiver le mode sombre' : 'Activer le mode sombre');
        btn.setAttribute('aria-pressed', String(enabled));
        btn.classList.toggle('dark', enabled);
    };

    const setDark = (enabled) => {
        document.documentElement.classList.toggle('dark', enabled);
        setIcon(enabled);
    };

    const THEME_KEY = 'theme';
    let savedTheme = null;
    try {
        savedTheme = localStorage.getItem(THEME_KEY);
    } catch (e) {}

    const media = window.matchMedia && window.matchMedia('(prefers-color-scheme: dark)');
    const prefersDark = media ? media.matches : false;
    // Initialisation: priorité à la préférence sauvegardée, sinon préférence système
    if (savedTheme === 'dark') {
        setDark(true);
    } else if (savedTheme === 'light') {
        setDark(false);
    } else {
        setDark(prefersDark);
    }

    // Si l'utilisateur a déjà choisi manuellement, on ne suit plus la préférence système
    let manualOverride = savedTheme === 'dark' || savedTheme === 'light';

    // React to system preference changes dynamically (no persistence)
    if (media && typeof media.addEventListener === 'function') {
        media.addEventListener('change', (e) => {
            if (!manualOverride) {
                setDark(e.matches);
            }
        });
    } else if (media && typeof media.addListener === 'function') {
        // Fallback for older browsers
        media.addListener((e) => {
            if (!manualOverride) {
                setDark(e.matches);
            }
        });
    }

    // Toggle handler — for réduire les glitches, attendre le next frame avant d'ajuster l'icône
    btn.addEventListener('click', () => {
        manualOverride = true;
        const next = !document.documentElement.classList.contains('dark');
        setDark(next);
        try {
            localStorage.setItem(THEME_KEY, next ? 'dark' : 'light');
        } catch (e) {}
    });

    loadFragments();
    init_memory_plot();
    syncHFSpacesURLHash();
}, { once: true });
