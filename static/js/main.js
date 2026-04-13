/* ===== DENGROUS AI — Main JavaScript ===== */

// ===== NAVBAR TOGGLE =====
const navToggle = document.getElementById('navToggle');
const navLinks = document.getElementById('navLinks');
if (navToggle) {
  navToggle.addEventListener('click', () => {
    navLinks.classList.toggle('open');
    const spans = navToggle.querySelectorAll('span');
    const isOpen = navLinks.classList.contains('open');
    spans[0].style.transform = isOpen ? 'rotate(45deg) translate(5px,5px)' : '';
    spans[1].style.opacity = isOpen ? '0' : '1';
    spans[2].style.transform = isOpen ? 'rotate(-45deg) translate(5px,-5px)' : '';
  });
  document.addEventListener('click', (e) => {
    if (!navToggle.contains(e.target) && !navLinks.contains(e.target)) {
      navLinks.classList.remove('open');
    }
  });
}

// ===== SCROLL ANIMATIONS =====
const observerOpts = { threshold: 0.1, rootMargin: '0px 0px -40px 0px' };
const observer = new IntersectionObserver((entries) => {
  entries.forEach(entry => {
    if (entry.isIntersecting) {
      entry.target.style.opacity = '1';
      entry.target.style.transform = 'translateY(0)';
      observer.unobserve(entry.target);
    }
  });
}, observerOpts);

document.querySelectorAll('.pipeline-step, .class-card, .stat-card, .tech-card, .example-card, .res-card, .misty-card').forEach(el => {
  el.style.opacity = '0';
  el.style.transform = 'translateY(20px)';
  el.style.transition = 'opacity 0.5s ease, transform 0.5s ease';
  observer.observe(el);
});

// ===== ANIMATE BARS ON SCROLL =====
const barObserver = new IntersectionObserver((entries) => {
  entries.forEach(entry => {
    if (entry.isIntersecting) {
      entry.target.querySelectorAll('.lb-fill, .rc-fill').forEach(fill => {
        const w = fill.style.width;
        fill.style.width = '0';
        setTimeout(() => { fill.style.width = w; }, 100);
      });
      barObserver.unobserve(entry.target);
    }
  });
}, { threshold: 0.2 });

document.querySelectorAll('.misty-card, .res-card').forEach(el => barObserver.observe(el));

// ===== ANIMATE COUNTERS =====
function animateCounter(el, target, duration = 1200) {
  const isFloat = target % 1 !== 0;
  const decimals = isFloat ? (target.toString().split('.')[1] || '').length : 0;
  let start = 0;
  const increment = target / (duration / 16);
  const timer = setInterval(() => {
    start += increment;
    if (start >= target) {
      el.textContent = isFloat ? target.toFixed(decimals) : Math.floor(target).toLocaleString();
      clearInterval(timer);
    } else {
      el.textContent = isFloat ? start.toFixed(decimals) : Math.floor(start).toLocaleString();
    }
  }, 16);
}

const counterObserver = new IntersectionObserver((entries) => {
  entries.forEach(entry => {
    if (entry.isIntersecting) {
      const el = entry.target;
      const raw = el.textContent.replace(/,/g, '').replace('%', '').replace('ms', '').replace('K', '000');
      const val = parseFloat(raw);
      if (!isNaN(val)) animateCounter(el, val);
      counterObserver.unobserve(el);
    }
  });
}, { threshold: 0.5 });

document.querySelectorAll('.stat-value, .metric-value').forEach(el => counterObserver.observe(el));

// ===== CLASS CARD HOVER GLOW =====
document.querySelectorAll('.class-card').forEach(card => {
  card.addEventListener('mouseenter', function() {
    const danger = this.className.match(/danger-(\w+)/);
    if (danger) {
      const colors = { critical: '#ef4444', high: '#f97316', medium: '#eab308', low: '#22c55e' };
      this.style.boxShadow = `0 8px 32px ${colors[danger[1]]}33`;
    }
  });
  card.addEventListener('mouseleave', function() {
    this.style.boxShadow = '';
  });
});

// ===== TOOLTIP =====
function createTooltip(el, text) {
  el.style.position = 'relative';
  el.style.cursor = 'help';
  el.addEventListener('mouseenter', function(e) {
    const tip = document.createElement('div');
    tip.className = 'tooltip-popup';
    tip.textContent = text;
    tip.style.cssText = `
      position:fixed; background:#1c1c2e; border:1px solid #252538;
      color:#e2e8f0; padding:8px 12px; border-radius:8px; font-size:0.8rem;
      pointer-events:none; z-index:9999; white-space:nowrap;
      box-shadow:0 4px 16px rgba(0,0,0,0.4); max-width:280px;
      white-space:normal; line-height:1.5;
    `;
    document.body.appendChild(tip);
    const rect = el.getBoundingClientRect();
    tip.style.left = `${Math.min(rect.left, window.innerWidth - tip.offsetWidth - 10)}px`;
    tip.style.top = `${rect.top - tip.offsetHeight - 8}px`;
    el._tooltip = tip;
  });
  el.addEventListener('mouseleave', function() {
    if (el._tooltip) { el._tooltip.remove(); el._tooltip = null; }
  });
}

// Add tooltips to metrics
document.querySelectorAll('.metric-label').forEach(el => {
  const tips = {
    'mAP@0.5': 'Mean Average Precision at IoU ≥ 0.5. Asosiy aniqlik ko\'rsatkichi.',
    'Inference': 'YOLO modeli bir rasmni qayta ishlash vaqti (millisekundda).',
    'Sinflar': 'DENGROUS dataset sinflari: pichoq, qaychi, sanchqi, bolga...',
    'Tasvirlar': 'O\'rgatish uchun ishlatiladigan jami tasvir soni.',
  };
  const tip = tips[el.textContent.trim()];
  if (tip) createTooltip(el, tip);
});

// ===== SMOOTH LINK TRANSITIONS =====
document.querySelectorAll('a[href^="/"]').forEach(link => {
  link.addEventListener('click', function(e) {
    if (this.target || e.ctrlKey || e.metaKey || e.shiftKey) return;
    e.preventDefault();
    document.body.style.opacity = '0';
    document.body.style.transition = 'opacity 0.15s';
    setTimeout(() => { window.location.href = this.href; }, 150);
  });
});

document.body.style.opacity = '0';
document.body.style.transition = 'opacity 0.2s';
requestAnimationFrame(() => { document.body.style.opacity = '1'; });

// ===== KEYBOARD SHORTCUTS =====
document.addEventListener('keydown', (e) => {
  if (e.altKey) {
    const shortcuts = { '1': '/', '2': '/about/', '3': '/examples/', '4': '/results/', '5': '/demo/' };
    if (shortcuts[e.key]) { e.preventDefault(); window.location.href = shortcuts[e.key]; }
  }
});

// ===== NOTIFICATION SYSTEM =====
window.showNotif = function(message, type = 'info') {
  const colors = { success: '#22c55e', error: '#ef4444', info: '#6366f1', warning: '#eab308' };
  const icons = { success: '✅', error: '❌', info: 'ℹ️', warning: '⚠️' };
  const notif = document.createElement('div');
  notif.style.cssText = `
    position:fixed; bottom:24px; right:24px; z-index:9999;
    background:#1c1c2e; border:1px solid ${colors[type]};
    border-left:4px solid ${colors[type]};
    color:#e2e8f0; padding:16px 20px; border-radius:12px;
    font-size:0.9rem; font-family:'Inter',sans-serif;
    box-shadow:0 8px 32px rgba(0,0,0,0.5);
    display:flex; align-items:center; gap:10px;
    max-width:340px; animation:slideInNotif 0.3s ease;
  `;
  notif.innerHTML = `${icons[type]} ${message}`;
  const style = document.createElement('style');
  style.textContent = `@keyframes slideInNotif{from{opacity:0;transform:translateX(24px)}to{opacity:1;transform:translateX(0)}}`;
  document.head.appendChild(style);
  document.body.appendChild(notif);
  setTimeout(() => { notif.style.opacity='0'; notif.style.transition='opacity 0.3s'; setTimeout(()=>notif.remove(),300); }, 3000);
};

// ===== PER CLASS TABLE ANIMATION (results page) =====
document.querySelectorAll('#perClassTable tbody tr').forEach((row, i) => {
  row.style.animationDelay = `${i * 40}ms`;
});

// ===== DARK MODE ENHANCEMENT: Add glow to active nav =====
const activeLink = document.querySelector('.nav-link.active');
if (activeLink && !activeLink.classList.contains('nav-cta')) {
  activeLink.style.textShadow = '0 0 12px rgba(99,102,241,0.5)';
}

console.log('%c🛡️ DENGROUS AI', 'font-size:20px; font-weight:bold; color:#6366f1;');
console.log('%cBolalar xavfsizligi uchun AI tizimi', 'font-size:12px; color:#94a3b8;');
