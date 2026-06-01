import React from 'react';
import { createRoot } from 'react-dom/client';

import pages, { design, meta } from '../slides/hfrvla-training';

const CANVAS_WIDTH = 1920;
const CANVAS_HEIGHT = 1080;

const clamp = (value: number, min: number, max: number) => Math.max(min, Math.min(max, value));
const readSlideIndexFromLocation = () => {
  const params = new URLSearchParams(window.location.search);
  const token = params.get('slide') ?? window.location.hash.match(/\d+/)?.[0];
  if (!token) return 0;
  const parsed = Number.parseInt(token, 10);
  return Number.isFinite(parsed) ? clamp(parsed - 1, 0, pages.length - 1) : 0;
};

const useViewportScale = () => {
  const compute = () => Math.min(window.innerWidth / CANVAS_WIDTH, window.innerHeight / CANVAS_HEIGHT);
  const [scale, setScale] = React.useState(compute);

  React.useEffect(() => {
    const onResize = () => setScale(compute());
    window.addEventListener('resize', onResize);
    return () => window.removeEventListener('resize', onResize);
  }, []);

  return scale;
};

const GlobalStyle = () => (
  <style>{`
    html, body, #root {
      width: 100%;
      height: 100%;
      margin: 0;
      overflow: hidden;
      background: #040807;
      color: ${design.palette.text};
      font-family: ${design.fonts.body};
    }
    * { box-sizing: border-box; }
    button { font: inherit; }
    @media (prefers-reduced-motion: reduce) {
      *, *::before, *::after {
        animation-duration: 1ms !important;
        animation-iteration-count: 1 !important;
        transition-duration: 1ms !important;
      }
    }
  `}</style>
);

const App = () => {
  const [current, setCurrent] = React.useState(readSlideIndexFromLocation);
  const [touchStart, setTouchStart] = React.useState<number | null>(null);
  const scale = useViewportScale();
  const CurrentPage = pages[current] ?? pages[0];
  const go = React.useCallback((next: number) => {
    const clamped = clamp(next, 0, pages.length - 1);
    setCurrent(clamped);
    window.history.replaceState(null, '', `#slide=${clamped + 1}`);
  }, []);

  React.useEffect(() => {
    const onHashChange = () => setCurrent(readSlideIndexFromLocation());
    window.addEventListener('hashchange', onHashChange);
    return () => window.removeEventListener('hashchange', onHashChange);
  }, []);

  React.useEffect(() => {
    const onKeyDown = (event: KeyboardEvent) => {
      if (['ArrowRight', 'ArrowDown', 'PageDown', ' '].includes(event.key)) {
        event.preventDefault();
        go(current + 1);
      }
      if (['ArrowLeft', 'ArrowUp', 'PageUp'].includes(event.key)) {
        event.preventDefault();
        go(current - 1);
      }
      if (event.key === 'Home') go(0);
      if (event.key === 'End') go(pages.length - 1);
    };
    window.addEventListener('keydown', onKeyDown);
    return () => window.removeEventListener('keydown', onKeyDown);
  }, [current, go]);

  const canvasVars = {
    '--osd-bg': design.palette.bg,
    '--osd-text': design.palette.text,
    '--osd-accent': design.palette.accent,
    '--osd-font-display': design.fonts.display,
    '--osd-font-body': design.fonts.body,
    '--osd-size-hero': `${design.typeScale.hero}px`,
    '--osd-size-body': `${design.typeScale.body}px`,
    '--osd-radius': `${design.radius}px`,
  } as React.CSSProperties;

  return (
    <div
      style={{
        width: '100%',
        height: '100%',
        position: 'relative',
        background:
          'radial-gradient(circle at 82% 18%, rgba(50,211,153,.16), transparent 26%), linear-gradient(180deg, #06100e, #020504)',
      }}
      onTouchStart={(event) => setTouchStart(event.changedTouches[0].clientX)}
      onTouchEnd={(event) => {
        if (touchStart === null) return;
        const delta = touchStart - event.changedTouches[0].clientX;
        if (Math.abs(delta) > 48) go(current + (delta > 0 ? 1 : -1));
        setTouchStart(null);
      }}
    >
      <GlobalStyle />
      <div
        style={{
          position: 'absolute',
          left: '50%',
          top: '50%',
          width: CANVAS_WIDTH,
          height: CANVAS_HEIGHT,
          transform: `translate(-50%, -50%) scale(${scale})`,
          transformOrigin: 'center center',
          boxShadow: '0 40px 120px rgba(0,0,0,.45)',
          ...canvasVars,
        }}
      >
        <CurrentPage />
      </div>
      <div
        style={{
          position: 'fixed',
          left: 22,
          bottom: 18,
          display: 'flex',
          gap: 10,
          alignItems: 'center',
          color: '#a9c8bc',
          fontFamily: '"SFMono-Regular", "Cascadia Code", ui-monospace, monospace',
          fontSize: 13,
          letterSpacing: '0.08em',
          textTransform: 'uppercase',
          userSelect: 'none',
        }}
      >
        <span>{String(current + 1).padStart(2, '0')}</span>
        <span style={{ opacity: 0.45 }}>/</span>
        <span>{String(pages.length).padStart(2, '0')}</span>
        <span style={{ opacity: 0.45, marginLeft: 12 }}>{meta.title ?? 'HFRVLA Training'}</span>
      </div>
      <div
        style={{
          position: 'fixed',
          right: 22,
          top: '50%',
          transform: 'translateY(-50%)',
          display: 'flex',
          flexDirection: 'column',
          gap: 9,
        }}
        aria-label="Slide navigation"
      >
        {pages.map((_, index) => (
          <button
            key={index}
            type="button"
            aria-label={`Go to slide ${index + 1}`}
            onClick={() => go(index)}
            style={{
              width: 10,
              height: 10,
              borderRadius: 99,
              border: `1px solid ${index === current ? design.palette.accent : 'rgba(169,200,188,.5)'}`,
              background: index === current ? design.palette.accent : 'rgba(169,200,188,.18)',
              padding: 0,
              cursor: 'pointer',
            }}
          />
        ))}
      </div>
      <button
        type="button"
        aria-label="Previous slide"
        onClick={() => go(current - 1)}
        disabled={current === 0}
        style={{
          position: 'fixed',
          left: 18,
          top: '50%',
          transform: 'translateY(-50%)',
          width: 42,
          height: 58,
          border: '1px solid rgba(169,200,188,.26)',
          borderRadius: 8,
          background: 'rgba(4,8,7,.55)',
          color: '#a9c8bc',
          cursor: current === 0 ? 'default' : 'pointer',
          opacity: current === 0 ? 0.25 : 1,
        }}
      >
        ‹
      </button>
      <button
        type="button"
        aria-label="Next slide"
        onClick={() => go(current + 1)}
        disabled={current === pages.length - 1}
        style={{
          position: 'fixed',
          right: 18,
          top: '50%',
          transform: 'translateY(-50%)',
          width: 42,
          height: 58,
          border: '1px solid rgba(169,200,188,.26)',
          borderRadius: 8,
          background: 'rgba(4,8,7,.55)',
          color: '#a9c8bc',
          cursor: current === pages.length - 1 ? 'default' : 'pointer',
          opacity: current === pages.length - 1 ? 0.25 : 1,
        }}
      >
        ›
      </button>
    </div>
  );
};

createRoot(document.getElementById('root') as HTMLElement).render(<App />);
