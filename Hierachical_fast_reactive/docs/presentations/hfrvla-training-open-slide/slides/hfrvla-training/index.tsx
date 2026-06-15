import type { DesignSystem, Page, SlideMeta } from '@open-slide/core';

export const design: DesignSystem = {
  palette: {
    bg: '#f7f3ea',
    text: '#17211f',
    accent: '#0f9f7a',
  },
  fonts: {
    display: '"Avenir Next", "PingFang TC", "Microsoft JhengHei", system-ui, sans-serif',
    body: '"PingFang TC", "Microsoft JhengHei", "Noto Sans CJK TC", system-ui, sans-serif',
  },
  typeScale: {
    hero: 122,
    body: 34,
  },
  radius: 8,
};

const colors = {
  bg: 'var(--osd-bg)',
  text: 'var(--osd-text)',
  accent: 'var(--osd-accent)',
  muted: '#5d6d68',
  faint: '#d9d0c2',
  panel: '#fffaf1',
  panelAlt: '#eef8f4',
  ink: '#17211f',
  teal: '#0f9f7a',
  blue: '#2f6fb3',
  gold: '#b37a16',
  coral: '#d45b4f',
  violet: '#7357a8',
  line: 'rgba(23, 33, 31, 0.16)',
  lineStrong: 'rgba(23, 33, 31, 0.34)',
};

const font = {
  display: 'var(--osd-font-display)',
  body: 'var(--osd-font-body)',
  mono: '"SFMono-Regular", "Cascadia Code", "JetBrains Mono", ui-monospace, monospace',
};

const fill = {
  width: '100%',
  height: '100%',
  background: colors.bg,
  color: colors.text,
  fontFamily: font.body,
  position: 'relative' as const,
  overflow: 'hidden',
};

const styles = `
  @keyframes hfr-fade-up {
    from { opacity: 0; transform: translateY(18px); }
    to { opacity: 1; transform: translateY(0); }
  }
  @keyframes hfr-draw {
    from { stroke-dashoffset: 960; }
    to { stroke-dashoffset: 0; }
  }
  .fade-1, .fade-2, .fade-3, .fade-4 {
    opacity: 0;
    animation: hfr-fade-up 640ms cubic-bezier(.2,.7,.2,1) forwards;
  }
  .fade-2 { animation-delay: 110ms; }
  .fade-3 { animation-delay: 220ms; }
  .fade-4 { animation-delay: 330ms; }
  .draw-line {
    stroke-dasharray: 960;
    animation: hfr-draw 900ms cubic-bezier(.2,.7,.2,1) forwards;
  }
`;

const Styles = () => <style>{styles}</style>;

const SubtleGrid = () => (
  <div
    aria-hidden
    style={{
      position: 'absolute',
      inset: 0,
      backgroundImage:
        'linear-gradient(rgba(23,33,31,.045) 1px, transparent 1px), linear-gradient(90deg, rgba(23,33,31,.045) 1px, transparent 1px)',
      backgroundSize: '80px 80px',
      maskImage: 'linear-gradient(180deg, rgba(0,0,0,.75), rgba(0,0,0,.12) 74%, transparent)',
      WebkitMaskImage: 'linear-gradient(180deg, rgba(0,0,0,.75), rgba(0,0,0,.12) 74%, transparent)',
    }}
  />
);

const PageShell = ({
  children,
  label = 'HFRVLA experiment briefing',
}: {
  children: React.ReactNode;
  label?: string;
}) => (
  <div style={{ ...fill }}>
    <Styles />
    <SubtleGrid />
    <div
      style={{
        position: 'absolute',
        left: 120,
        top: 72,
        fontFamily: font.mono,
        fontSize: 22,
        fontWeight: 800,
        color: colors.gold,
        letterSpacing: '0.13em',
        textTransform: 'uppercase',
      }}
    >
      {label}
    </div>
    <div style={{ position: 'relative', height: '100%', padding: '128px 120px 96px' }}>{children}</div>
  </div>
);

const Heading = ({ children, maxWidth = 1320 }: { children: React.ReactNode; maxWidth?: number }) => (
  <h2
    className="fade-1"
    style={{
      margin: 0,
      maxWidth,
      fontFamily: font.display,
      fontSize: 74,
      lineHeight: 1.08,
      fontWeight: 860,
      letterSpacing: 0,
    }}
  >
    {children}
  </h2>
);

const Note = ({ children, width = 1380 }: { children: React.ReactNode; width?: number }) => (
  <p
    className="fade-2"
    style={{
      margin: '24px 0 0',
      maxWidth: width,
      fontSize: 31,
      lineHeight: 1.44,
      color: colors.muted,
    }}
  >
    {children}
  </p>
);

const Pill = ({ children, tone = colors.teal }: { children: React.ReactNode; tone?: string }) => (
  <div
    style={{
      display: 'inline-flex',
      alignItems: 'center',
      height: 48,
      padding: '0 18px',
      borderRadius: 24,
      border: `1px solid ${tone}`,
      color: tone,
      background: 'rgba(255, 250, 241, 0.72)',
      fontFamily: font.mono,
      fontSize: 20,
      fontWeight: 800,
      whiteSpace: 'nowrap',
    }}
  >
    {children}
  </div>
);

const Card = ({
  title,
  body,
  tone = colors.teal,
}: {
  title: string;
  body: string;
  tone?: string;
}) => (
  <div
    style={{
      minHeight: 150,
      borderRadius: 8,
      border: `1px solid ${colors.line}`,
      background: colors.panel,
      padding: '28px 30px',
      boxShadow: '0 18px 46px rgba(23,33,31,.08)',
      display: 'flex',
      flexDirection: 'column',
      justifyContent: 'space-between',
      gap: 14,
    }}
  >
    <div style={{ fontSize: 28, lineHeight: 1.14, fontWeight: 860, color: tone }}>{title}</div>
    <div style={{ fontSize: 25, lineHeight: 1.34, color: colors.muted }}>{body}</div>
  </div>
);

const StatCard = ({
  value,
  label,
  detail,
  tone = colors.teal,
}: {
  value: string;
  label: string;
  detail: string;
  tone?: string;
}) => (
  <div
    style={{
      borderRadius: 8,
      background: colors.panel,
      border: `1px solid ${colors.line}`,
      padding: '26px 30px',
      minHeight: 168,
      boxShadow: '0 18px 46px rgba(23,33,31,.08)',
    }}
  >
    <div style={{ fontFamily: font.display, fontSize: 54, lineHeight: 1, fontWeight: 900, color: tone }}>{value}</div>
    <div style={{ marginTop: 14, fontSize: 27, fontWeight: 820, color: colors.text }}>{label}</div>
    <div style={{ marginTop: 8, fontSize: 22, lineHeight: 1.32, color: colors.muted }}>{detail}</div>
  </div>
);

const Table = ({
  headers,
  rows,
  widths,
}: {
  headers: string[];
  rows: React.ReactNode[][];
  widths?: string[];
}) => (
  <div
    style={{
      borderRadius: 8,
      border: `1px solid ${colors.line}`,
      background: colors.panel,
      overflow: 'hidden',
      boxShadow: '0 18px 46px rgba(23,33,31,.08)',
    }}
  >
    <table style={{ width: '100%', borderCollapse: 'collapse', tableLayout: 'fixed' }}>
      <colgroup>
        {headers.map((header, index) => (
          <col key={header} style={widths?.[index] ? { width: widths[index] } : undefined} />
        ))}
      </colgroup>
      <thead>
        <tr style={{ background: '#efe5d5' }}>
          {headers.map((header) => (
            <th
              key={header}
              style={{
                padding: '12px 16px',
                textAlign: 'left',
                color: colors.ink,
                fontSize: 20,
                fontWeight: 860,
                borderBottom: `1px solid ${colors.lineStrong}`,
              }}
            >
              {header}
            </th>
          ))}
        </tr>
      </thead>
      <tbody>
        {rows.map((row, rowIndex) => (
          <tr key={`row-${rowIndex}`} style={{ borderTop: rowIndex === 0 ? 'none' : `1px solid ${colors.line}` }}>
            {row.map((cell, cellIndex) => (
              <td
                key={`cell-${rowIndex}-${cellIndex}`}
                style={{
                  padding: '10px 16px',
                  verticalAlign: 'middle',
                  fontSize: 19,
                  lineHeight: 1.18,
                  color: cellIndex === 0 ? colors.text : colors.muted,
                  fontWeight: cellIndex === 0 ? 760 : 520,
                }}
              >
                {cell}
              </td>
            ))}
          </tr>
        ))}
      </tbody>
    </table>
  </div>
);

const DeltaText = ({ value }: { value: number }) => (
  <span style={{ color: value >= 0 ? colors.teal : colors.coral, fontWeight: 900 }}>
    {value >= 0 ? '+' : ''}
    {value} pp
  </span>
);

const MethodDiagram = () => (
  <svg viewBox="0 0 1460 520" width="100%" height="520" role="img" aria-label="HFRVLA method diagram">
    <defs>
      <marker id="method-arrow-teal" markerWidth="12" markerHeight="12" refX="10" refY="6" orient="auto">
        <path d="M0,0 L12,6 L0,12 Z" fill={colors.teal} />
      </marker>
      <marker id="method-arrow-blue" markerWidth="12" markerHeight="12" refX="10" refY="6" orient="auto">
        <path d="M0,0 L12,6 L0,12 Z" fill={colors.blue} />
      </marker>
      <marker id="method-arrow-gold" markerWidth="12" markerHeight="12" refX="10" refY="6" orient="auto">
        <path d="M0,0 L12,6 L0,12 Z" fill={colors.gold} />
      </marker>
    </defs>

    <rect x="34" y="58" width="510" height="320" rx="18" fill="#e7f0fb" stroke={colors.blue} strokeWidth="4" />
    <text x="64" y="112" fill={colors.blue} fontSize="40" fontWeight="900">
      Frozen SmolVLA
    </text>
    <text x="64" y="158" fill={colors.muted} fontSize="26">
      slow planner: task + scene + robot state
    </text>
    <rect x="86" y="210" width="178" height="72" rx="10" fill="#ffffff" stroke={colors.blue} strokeWidth="2" />
    <text x="175" y="254" textAnchor="middle" fill={colors.ink} fontSize="24" fontWeight="850">
      base chunk
    </text>
    <rect x="314" y="210" width="178" height="72" rx="10" fill="#ffffff" stroke={colors.blue} strokeWidth="2" />
    <text x="403" y="254" textAnchor="middle" fill={colors.ink} fontSize="24" fontWeight="850">
      slow context
    </text>

    <rect x="34" y="408" width="510" height="72" rx="16" fill="#fff6df" stroke={colors.gold} strokeWidth="3" />
    <text x="64" y="454" fill={colors.gold} fontSize="27" fontWeight="900">
      SmolVLA weights stay frozen during HFRVLA training
    </text>

    <rect x="700" y="98" width="426" height="280" rx="18" fill="#e9f7f1" stroke={colors.teal} strokeWidth="4" />
    <text x="730" y="154" fill={colors.teal} fontSize="38" fontWeight="900">
      Fast wrist correction
    </text>
    <text x="730" y="198" fill={colors.muted} fontSize="25">
      small trainable correction head
    </text>
    <rect x="746" y="238" width="150" height="72" rx="10" fill="#ffffff" stroke={colors.teal} strokeWidth="2" />
    <text x="821" y="282" textAnchor="middle" fill={colors.ink} fontSize="23" fontWeight="850">
      wrist view
    </text>
    <rect x="926" y="238" width="150" height="72" rx="10" fill="#ffffff" stroke={colors.teal} strokeWidth="2" />
    <text x="1001" y="282" textAnchor="middle" fill={colors.ink} fontSize="23" fontWeight="850">
      robot state
    </text>

    <line x1="544" y1="246" x2="682" y2="246" stroke={colors.blue} strokeWidth="4" markerEnd="url(#method-arrow-blue)" />
    <line x1="544" y1="320" x2="682" y2="324" stroke={colors.blue} strokeWidth="3" markerEnd="url(#method-arrow-blue)" />

    <rect x="1190" y="150" width="210" height="86" rx="12" fill="#ffffff" stroke={colors.teal} strokeWidth="3" />
    <text x="1295" y="186" textAnchor="middle" fill={colors.teal} fontSize="30" fontWeight="900">
      δa
    </text>
    <text x="1295" y="217" textAnchor="middle" fill={colors.muted} fontSize="19">
      wrist correction
    </text>
    <line x1="1126" y1="238" x2="1174" y2="194" stroke={colors.teal} strokeWidth="4" markerEnd="url(#method-arrow-teal)" />

    <circle cx="1288" cy="338" r="42" fill="#fffaf1" stroke={colors.ink} strokeWidth="4" />
    <text x="1288" y="354" textAnchor="middle" fill={colors.ink} fontSize="52" fontWeight="900">
      +
    </text>
    <line x1="1295" y1="236" x2="1295" y2="286" stroke={colors.teal} strokeWidth="4" markerEnd="url(#method-arrow-teal)" />
    <path d="M544 382 C760 470 1070 440 1235 354" fill="none" stroke={colors.blue} strokeWidth="4" markerEnd="url(#method-arrow-blue)" />

    <rect x="1130" y="414" width="286" height="72" rx="12" fill="#fff6df" stroke={colors.gold} strokeWidth="3" />
    <text x="1273" y="458" textAnchor="middle" fill={colors.gold} fontSize="24" fontWeight="900">
      final robot action
    </text>
    <line x1="1328" y1="338" x2="1368" y2="414" stroke={colors.gold} strokeWidth="4" markerEnd="url(#method-arrow-gold)" />
  </svg>
);

type SuccessDatum = {
  setting: string;
  hfrvla: number;
  smolvla: number;
  hfrvlaText: string;
  smolvlaText: string;
  note: string;
};

const plan50Rows: SuccessDatum[] = [
  { setting: 'K=1', hfrvla: 75, smolvla: 79, hfrvlaText: '75/100', smolvlaText: '79/100', note: 'baseline stronger' },
  { setting: 'K=2', hfrvla: 79, smolvla: 72, hfrvlaText: '79/100', smolvlaText: '72/100', note: 'short gain' },
  { setting: 'K=4', hfrvla: 80, smolvla: 65, hfrvlaText: '80/100', smolvlaText: '65/100', note: 'best row' },
  { setting: 'K=8', hfrvla: 71, smolvla: 60, hfrvlaText: '71/100', smolvlaText: '60/100', note: 'deploy ref' },
  { setting: 'K=16', hfrvla: 75, smolvla: 57, hfrvlaText: '75/100', smolvlaText: '57/100', note: 'largest margin' },
  { setting: 'K=32', hfrvla: 59, smolvla: 50, hfrvlaText: '59/100', smolvlaText: '50/100', note: 'long weak' },
  { setting: 'K=50', hfrvla: 53, smolvla: 40, hfrvlaText: '53/100', smolvlaText: '40/100', note: 'long-chunk pain' },
];

const matchedChunkRows: SuccessDatum[] = [
  { setting: 'K=1', hfrvla: 73, smolvla: 77, hfrvlaText: '73/100', smolvlaText: '77/100', note: 'baseline stronger' },
  { setting: 'K=2', hfrvla: 71, smolvla: 77, hfrvlaText: '71/100', smolvlaText: '77/100', note: 'baseline stronger' },
  { setting: 'K=4', hfrvla: 73, smolvla: 65, hfrvlaText: '73/100', smolvlaText: '65/100', note: 'first gain' },
  { setting: 'K=8', hfrvla: 65, smolvla: 62, hfrvlaText: '65/100', smolvlaText: '62/100', note: 'small margin' },
  { setting: 'K=16', hfrvla: 60, smolvla: 54, hfrvlaText: '60/100', smolvlaText: '54/100', note: 'still useful' },
  { setting: 'K=32', hfrvla: 60, smolvla: 51, hfrvlaText: '60/100', smolvlaText: '51/100', note: 'largest margin' },
  { setting: 'K=50', hfrvla: 53, smolvla: 40, hfrvlaText: '53/100', smolvlaText: '40/100', note: 'long-chunk pain' },
];

const successTableRows = (rows: SuccessDatum[]) =>
  rows.map((row) => [
    row.setting,
    `${row.smolvlaText} = ${row.smolvla}%`,
    `${row.hfrvlaText} = ${row.hfrvla}%`,
    <DeltaText value={row.hfrvla - row.smolvla} />,
    row.note,
  ]);

const SuccessChart = ({
  title,
  data,
  yMin = 40,
  yMax = 90,
}: {
  title: string;
  data: SuccessDatum[];
  yMin?: number;
  yMax?: number;
}) => {
  const width = 860;
  const height = 426;
  const left = 76;
  const right = 30;
  const top = 58;
  const bottom = 76;
  const innerW = width - left - right;
  const innerH = height - top - bottom;
  const x = (index: number) => left + (innerW * index) / Math.max(1, data.length - 1);
  const y = (value: number) => top + innerH - ((value - yMin) / (yMax - yMin)) * innerH;
  const line = (key: 'hfrvla' | 'smolvla') => data.map((row, index) => `${x(index)},${y(row[key])}`).join(' ');

  return (
    <div style={{ background: colors.panel, border: `1px solid ${colors.line}`, borderRadius: 8, padding: '22px 24px' }}>
      <div style={{ fontSize: 25, fontWeight: 860, color: colors.ink }}>{title}</div>
      <svg width={width} height={height} viewBox={`0 0 ${width} ${height}`} role="img" aria-label={title}>
        {[40, 50, 60, 70, 80, 90].map((tick) => (
          <g key={tick}>
            <line x1={left} x2={width - right} y1={y(tick)} y2={y(tick)} stroke={colors.line} />
            <text x={left - 16} y={y(tick) + 7} textAnchor="end" fill={colors.muted} fontFamily={font.mono} fontSize="18">
              {tick}%
            </text>
          </g>
        ))}
        <line x1={left} x2={width - right} y1={height - bottom} y2={height - bottom} stroke={colors.lineStrong} />
        <line x1={left} x2={left} y1={top} y2={height - bottom} stroke={colors.lineStrong} />
        <polyline points={line('smolvla')} fill="none" stroke={colors.gold} strokeWidth="5" strokeLinejoin="round" strokeLinecap="round" />
        <polyline points={line('hfrvla')} fill="none" stroke={colors.teal} strokeWidth="6" strokeLinejoin="round" strokeLinecap="round" />
        {data.map((row, index) => (
          <g key={row.setting}>
            <circle cx={x(index)} cy={y(row.smolvla)} r="7" fill={colors.gold} />
            <circle cx={x(index)} cy={y(row.hfrvla)} r="8" fill={colors.teal} />
            <text x={x(index)} y={height - 36} textAnchor="middle" fill={colors.ink} fontFamily={font.mono} fontSize="20" fontWeight="800">
              {row.setting.replace('K=', '')}
            </text>
          </g>
        ))}
        <text x={left} y="32" fill={colors.teal} fontFamily={font.mono} fontSize="20" fontWeight="900">
          HFRVLA
        </text>
        <text x={left + 122} y="32" fill={colors.gold} fontFamily={font.mono} fontSize="20" fontWeight="900">
          SmolVLA
        </text>
        <text x={width - right} y={height - 10} textAnchor="end" fill={colors.muted} fontFamily={font.mono} fontSize="17">
          K = steps before refreshing execution
        </text>
      </svg>
    </div>
  );
};

type AlphaClipRow = {
  cap: string;
  values: { successes: number; percent: number; effective: string }[];
};

const alphaColumns = ['0.25', '0.50', '0.75', '1.00'];

const alphaClipRows: AlphaClipRow[] = [
  { cap: '0.05', values: [{ successes: 23, percent: 46, effective: '0.013' }, { successes: 27, percent: 54, effective: '0.025' }, { successes: 24, percent: 48, effective: '0.038' }, { successes: 25, percent: 50, effective: '0.050' }] },
  { cap: '0.10', values: [{ successes: 23, percent: 46, effective: '0.025' }, { successes: 23, percent: 46, effective: '0.050' }, { successes: 24, percent: 48, effective: '0.075' }, { successes: 28, percent: 56, effective: '0.100' }] },
  { cap: '0.15', values: [{ successes: 24, percent: 48, effective: '0.038' }, { successes: 22, percent: 44, effective: '0.075' }, { successes: 23, percent: 46, effective: '0.113' }, { successes: 24, percent: 48, effective: '0.150' }] },
  { cap: '0.18', values: [{ successes: 24, percent: 48, effective: '0.045' }, { successes: 23, percent: 46, effective: '0.090' }, { successes: 23, percent: 46, effective: '0.135' }, { successes: 20, percent: 40, effective: '0.180' }] },
  { cap: '0.20', values: [{ successes: 27, percent: 54, effective: '0.050' }, { successes: 28, percent: 56, effective: '0.100' }, { successes: 26, percent: 52, effective: '0.150' }, { successes: 23, percent: 46, effective: '0.200' }] },
  { cap: '0.22', values: [{ successes: 25, percent: 50, effective: '0.055' }, { successes: 26, percent: 52, effective: '0.110' }, { successes: 26, percent: 52, effective: '0.165' }, { successes: 20, percent: 40, effective: '0.220' }] },
  { cap: '0.25', values: [{ successes: 21, percent: 42, effective: '0.063' }, { successes: 25, percent: 50, effective: '0.125' }, { successes: 20, percent: 40, effective: '0.188' }, { successes: 21, percent: 42, effective: '0.250' }] },
  { cap: '0.30', values: [{ successes: 26, percent: 52, effective: '0.075' }, { successes: 22, percent: 44, effective: '0.150' }, { successes: 18, percent: 36, effective: '0.225' }, { successes: 19, percent: 38, effective: '0.300' }] },
  { cap: '∞', values: [{ successes: 22, percent: 44, effective: 'none' }, { successes: 8, percent: 16, effective: 'none' }, { successes: 2, percent: 4, effective: 'none' }, { successes: 1, percent: 2, effective: 'none' }] },
];

const heatTone = (percent: number) => {
  if (percent >= 56) return { fill: '#c9f1e5', text: colors.teal, stroke: colors.teal };
  if (percent >= 52) return { fill: '#fff0c9', text: colors.gold, stroke: colors.gold };
  if (percent <= 16) return { fill: '#f8d5d1', text: colors.coral, stroke: colors.coral };
  if (percent <= 40) return { fill: '#fce2df', text: colors.coral, stroke: colors.lineStrong };
  return { fill: '#fffaf1', text: colors.ink, stroke: colors.line };
};

const AlphaClipHeatmap = () => {
  const width = 930;
  const height = 510;
  const left = 94;
  const top = 82;
  const cellW = 196;
  const cellH = 38;
  const gap = 4;
  const gridW = alphaColumns.length * cellW + (alphaColumns.length - 1) * gap;
  const gridH = alphaClipRows.length * cellH + (alphaClipRows.length - 1) * gap;

  return (
    <div style={{ background: colors.panel, border: `1px solid ${colors.line}`, borderRadius: 8, padding: '18px 20px' }}>
      <svg width={width} height={height} viewBox={`0 0 ${width} ${height}`} role="img" aria-label="alpha delta calibration heatmap">
        <text x={left + gridW / 2} y="32" textAnchor="middle" fill={colors.teal} fontFamily={font.mono} fontSize="24" fontWeight="900">
          correction strength α
        </text>
        <text x="25" y={top + gridH / 2} textAnchor="middle" fill={colors.gold} fontFamily={font.mono} fontSize="22" fontWeight="900" transform={`rotate(-90 25 ${top + gridH / 2})`}>
          correction cap δ_max
        </text>
        <text x={left} y={height - 22} fill={colors.muted} fontFamily={font.mono} fontSize="15">
          cell text = successes / 50; αδ = effective cap after scaling
        </text>
        {alphaColumns.map((alpha, index) => {
          const x = left + index * (cellW + gap);
          return (
            <text key={alpha} x={x + cellW / 2} y="64" textAnchor="middle" fill={colors.teal} fontFamily={font.mono} fontSize="19" fontWeight="900">
              α={alpha}
            </text>
          );
        })}
        {alphaClipRows.map((row, rowIndex) => {
          const y = top + rowIndex * (cellH + gap);
          return (
            <g key={`row-${row.cap}`}>
              <text x={left - 18} y={y + cellH / 2 + 7} textAnchor="end" fill={row.cap === '∞' ? colors.coral : colors.gold} fontFamily={font.mono} fontSize="18" fontWeight="900">
                {row.cap}
              </text>
              {row.values.map((cell, colIndex) => {
                const x = left + colIndex * (cellW + gap);
                const tone = heatTone(cell.percent);
                const best = cell.percent === 56;
                return (
                  <g key={`${row.cap}-${alphaColumns[colIndex]}`}>
                    <rect x={x} y={y} width={cellW} height={cellH} rx="4" fill={tone.fill} stroke={best ? colors.teal : tone.stroke} strokeWidth={best ? 3 : 1} />
                    <text x={x + cellW / 2} y={y + 17} textAnchor="middle" fill={tone.text} fontFamily={font.mono} fontSize="16" fontWeight="900">
                      {cell.successes}/50 = {cell.percent}%
                    </text>
                    <text x={x + cellW / 2} y={y + 32} textAnchor="middle" fill={colors.muted} fontFamily={font.mono} fontSize="12">
                      {cell.effective === 'none' ? 'no cap' : `αδ ${cell.effective}`}
                    </text>
                  </g>
                );
              })}
            </g>
          );
        })}
      </svg>
    </div>
  );
};

type DelayDatum = {
  delay: number;
  hfrvla: number;
  disableFast: number;
  startIndex: number;
};

const delayRows: DelayDatum[] = [
  { delay: 0, hfrvla: 64, disableFast: 68, startIndex: 0 },
  { delay: 1, hfrvla: 68, disableFast: 64, startIndex: 1 },
  { delay: 2, hfrvla: 72, disableFast: 62, startIndex: 2 },
  { delay: 3, hfrvla: 68, disableFast: 61, startIndex: 3 },
  { delay: 4, hfrvla: 66, disableFast: 56, startIndex: 4 },
];

const DelayChart = () => {
  const width = 820;
  const height = 420;
  const left = 76;
  const right = 32;
  const top = 56;
  const bottom = 72;
  const yMin = 50;
  const yMax = 76;
  const innerW = width - left - right;
  const innerH = height - top - bottom;
  const x = (index: number) => left + (innerW * index) / Math.max(1, delayRows.length - 1);
  const y = (value: number) => top + innerH - ((value - yMin) / (yMax - yMin)) * innerH;
  const line = (key: 'hfrvla' | 'disableFast') => delayRows.map((row, index) => `${x(index)},${y(row[key])}`).join(' ');

  return (
    <div style={{ background: colors.panel, border: `1px solid ${colors.line}`, borderRadius: 8, padding: '22px 24px' }}>
      <div style={{ fontSize: 25, fontWeight: 860, color: colors.ink }}>planner-delay stress test</div>
      <svg width={width} height={height} viewBox={`0 0 ${width} ${height}`} role="img" aria-label="Planner delay success chart">
        {[50, 55, 60, 65, 70, 75].map((tick) => (
          <g key={tick}>
            <line x1={left} x2={width - right} y1={y(tick)} y2={y(tick)} stroke={colors.line} />
            <text x={left - 16} y={y(tick) + 7} textAnchor="end" fill={colors.muted} fontFamily={font.mono} fontSize="18">
              {tick}%
            </text>
          </g>
        ))}
        <line x1={left} x2={width - right} y1={height - bottom} y2={height - bottom} stroke={colors.lineStrong} />
        <line x1={left} x2={left} y1={top} y2={height - bottom} stroke={colors.lineStrong} />
        <polyline points={line('disableFast')} fill="none" stroke={colors.coral} strokeWidth="5" strokeLinejoin="round" strokeLinecap="round" />
        <polyline points={line('hfrvla')} fill="none" stroke={colors.teal} strokeWidth="6" strokeLinejoin="round" strokeLinecap="round" />
        {delayRows.map((row, index) => (
          <g key={row.delay}>
            <circle cx={x(index)} cy={y(row.disableFast)} r="7" fill={colors.coral} />
            <circle cx={x(index)} cy={y(row.hfrvla)} r="8" fill={colors.teal} />
            <text x={x(index)} y={height - 34} textAnchor="middle" fill={colors.ink} fontFamily={font.mono} fontSize="20" fontWeight="800">
              {row.delay}
            </text>
          </g>
        ))}
        <text x={left} y="32" fill={colors.teal} fontFamily={font.mono} fontSize="20" fontWeight="900">
          HFRVLA
        </text>
        <text x={left + 112} y="32" fill={colors.coral} fontFamily={font.mono} fontSize="20" fontWeight="900">
          fast path disabled
        </text>
        <text x={width - right} y={height - 10} textAnchor="end" fill={colors.muted} fontFamily={font.mono} fontSize="17">
          d = slow-planner delay in control steps
        </text>
      </svg>
    </div>
  );
};

const DelayTimeline = () => (
  <svg viewBox="0 0 1510 430" width="100%" height="430" role="img" aria-label="Planner delay timeline">
    <defs>
      <marker id="delay-arrow" markerWidth="12" markerHeight="12" refX="10" refY="6" orient="auto">
        <path d="M0,0 L12,6 L0,12 Z" fill={colors.teal} />
      </marker>
      <marker id="delay-arrow-coral" markerWidth="12" markerHeight="12" refX="10" refY="6" orient="auto">
        <path d="M0,0 L12,6 L0,12 Z" fill={colors.coral} />
      </marker>
    </defs>
    <rect x="36" y="36" width="1438" height="350" rx="16" fill={colors.panel} stroke={colors.line} />
    <line x1="132" y1="330" x2="1370" y2="330" stroke={colors.lineStrong} strokeWidth="3" markerEnd="url(#delay-arrow)" />
    <text x="130" y="370" fill={colors.ink} fontFamily={font.mono} fontSize="24" fontWeight="900">
      t
    </text>
    <text x="610" y="370" fill={colors.coral} fontFamily={font.mono} fontSize="24" fontWeight="900">
      t+d
    </text>
    <text x="1190" y="370" fill={colors.ink} fontFamily={font.mono} fontSize="24" fontWeight="900">
      later execution
    </text>
    <line x1="132" y1="72" x2="132" y2="338" stroke={colors.lineStrong} strokeWidth="2" />
    <line x1="610" y1="72" x2="610" y2="338" stroke={colors.coral} strokeWidth="3" strokeDasharray="9 8" />
    <rect x="168" y="92" width="402" height="76" rx="10" fill="#e7f0fb" stroke={colors.blue} strokeWidth="3" />
    <text x="369" y="122" textAnchor="middle" fill={colors.blue} fontSize="22" fontWeight="900">
      slow planner computes
    </text>
    <text x="369" y="150" textAnchor="middle" fill={colors.muted} fontSize="18">
      chunk is based on observation at t
    </text>
    <rect x="610" y="92" width="604" height="76" rx="10" fill="#e9f7f1" stroke={colors.teal} strokeWidth="3" />
    <text x="912" y="122" textAnchor="middle" fill={colors.teal} fontSize="22" fontWeight="900">
      robot executes stale base actions
    </text>
    <text x="912" y="150" textAnchor="middle" fill={colors.muted} fontSize="18">
      wrist correction is still computed from current feedback
    </text>
    <line x1="132" y1="212" x2="610" y2="212" stroke={colors.coral} strokeWidth="4" markerEnd="url(#delay-arrow-coral)" />
    <text x="371" y="198" textAnchor="middle" fill={colors.coral} fontSize="26" fontWeight="900">
      delay d
    </text>
    <rect x="690" y="230" width="392" height="62" rx="10" fill="#fff6df" stroke={colors.gold} strokeWidth="3" />
    <text x="886" y="270" textAnchor="middle" fill={colors.gold} fontSize="24" fontWeight="900">
      final = base + α · clipped(δa)
    </text>
  </svg>
);

const Cover: Page = () => (
  <div style={{ ...fill }}>
    <Styles />
    <SubtleGrid />
    <div style={{ position: 'absolute', left: 120, top: 112, display: 'flex', gap: 14 }}>
      <Pill tone={colors.teal}>HFRVLA</Pill>
      <Pill tone={colors.blue}>experiment record</Pill>
      <Pill tone={colors.gold}>paper briefing</Pill>
    </div>
    <div style={{ position: 'absolute', left: 120, top: 238, width: 1180 }}>
      <h1
        className="fade-1"
        style={{
          margin: 0,
          fontFamily: font.display,
          fontSize: 'var(--osd-size-hero)',
          lineHeight: 1.04,
          fontWeight: 900,
          letterSpacing: 0,
        }}
      >
        HFRVLA 實驗與論文資料簡報
      </h1>
      <p className="fade-2" style={{ margin: '36px 0 0', width: 1050, fontSize: 38, lineHeight: 1.34, color: colors.muted }}>
        這份文件記錄模型概念、已完成的實驗、目前能支持的主張，以及未來 paper 需要補齊的證據。
      </p>
    </div>
    <div className="fade-3" style={{ position: 'absolute', right: 120, bottom: 112, width: 520, display: 'grid', gap: 18 }}>
      <StatCard value="267" label="registry rows" detail="目前正式 eval master table 的資料列數" tone={colors.blue} />
      <StatCard value="14" label="briefing pages" detail="從舊的訓練 runbook 重整成人類可讀版本" tone={colors.teal} />
    </div>
  </div>
);

const Purpose: Page = () => (
  <PageShell label="purpose">
    <Heading maxWidth={1420}>這不是訓練操作手冊，而是實驗與寫作索引</Heading>
    <Note>
      操作命令留在 `docs/training.md`；這份 briefing 只保留人類需要理解與寫 paper 時會用到的資訊。
    </Note>
    <div className="fade-3" style={{ marginTop: 46, display: 'grid', gridTemplateColumns: '1fr 1fr 1fr', gap: 24 }}>
      <Card title="模型是什麼" body="Frozen SmolVLA 先給 base action，trainable wrist module 每步補局部 correction。" tone={colors.teal} />
      <Card title="做過哪些實驗" body="用 LIBERO-Spatial only 的統一表格記錄 execution interval、matched chunk、α/δ calibration、planner delay。" tone={colors.blue} />
      <Card title="paper 可用什麼" body="把可主張的結果、限制、缺口與資料來源分開，避免把臨時 run 寫成最終 claim。" tone={colors.gold} />
    </div>
    <div className="fade-4" style={{ marginTop: 38, display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 24 }}>
      <Card title="已移除" body="訓練啟動命令、cache build 指令、smoke protocol、自動重建細節、pitfalls、舊 Stage A/B/C gated-loss 細節。" tone={colors.coral} />
      <Card title="保留方式" body="舊 GRU/gate/contact path 只作 retired context，不再佔主簡報頁面，也不當作當前方法。" tone={colors.violet} />
    </div>
  </PageShell>
);

const ModelAtGlance: Page = () => (
  <PageShell label="model">
    <Heading>模型一句話：慢速規劃不動，快速 wrist correction 補上局部反應</Heading>
    <Note>
      HFRVLA 的重點不是重新訓練 SmolVLA，而是測試一個小型 wrist-camera correction module 能否改善 frozen action chunk。
    </Note>
    <div className="fade-3" style={{ marginTop: 28 }}>
      <MethodDiagram />
    </div>
  </PageShell>
);

const Formula: Page = () => (
  <PageShell label="merge rule">
    <Heading maxWidth={1500}>執行時只做一個人能讀懂的合併規則</Heading>
    <div className="fade-2" style={{ marginTop: 56, borderRadius: 8, background: colors.panel, border: `1px solid ${colors.line}`, padding: '48px 56px', width: 1460 }}>
      <div style={{ fontFamily: font.display, fontSize: 76, lineHeight: 1.1, fontWeight: 900, color: colors.ink }}>
        final action = base action + α · clipped(δa)
      </div>
      <div style={{ marginTop: 20, fontSize: 29, lineHeight: 1.4, color: colors.muted }}>
        其中 δa 是 wrist module 對當前 robot step 的小幅修正，α 是修正強度，δ_max 是 clipping 上限。
      </div>
    </div>
    <div className="fade-3" style={{ marginTop: 34, display: 'grid', gridTemplateColumns: '1fr 1fr 1fr', gap: 22 }}>
      <Card title="base action" body="Frozen SmolVLA 對目前 chunk 給出的原始動作。" tone={colors.blue} />
      <Card title="δa" body="由 wrist view 和 robot state 產生的局部修正，不是另一個完整 policy。" tone={colors.teal} />
      <Card title="α and δ_max" body="控制 residual 影響力；長 chunk 實驗顯示沒有上限會 over-correct。" tone={colors.gold} />
    </div>
  </PageShell>
);

const EvidenceBase: Page = () => (
  <PageShell label="evidence base">
    <Heading maxWidth={1480}>資料與 evidence base：paper 數字只從 registry 出來</Heading>
    <Note>
      模型訓練使用已驗證的 merged HFRVLA dataset；closed-loop evaluation 的長期紀錄由 eval registry 維護。
    </Note>
    <div className="fade-3" style={{ marginTop: 34, display: 'grid', gridTemplateColumns: '.85fr 1.15fr', gap: 26, alignItems: 'start' }}>
      <div style={{ display: 'grid', gap: 18 }}>
        <StatCard value="1,693" label="episodes" detail="verified merged local HFRVLA dataset" tone={colors.teal} />
        <StatCard value="273k" label="frames" detail="records SmolVLA context and wrist DINO features" tone={colors.blue} />
        <StatCard value="40" label="tasks" detail="LIBERO task coverage in the shared dataset" tone={colors.gold} />
      </div>
      <Table
        headers={['Paper term', 'What it means', 'Why it matters']}
        widths={['24%', '36%', '40%']}
        rows={[
          ['base action', 'SmolVLA action for the current step', 'baseline signal that HFRVLA corrects'],
          ['slow context', 'language / phase features from frozen SmolVLA', 'keeps the wrist module task-aware'],
          ['wrist patches', 'DINO features from the wrist camera', 'main per-step visual feedback path'],
          ['robot state', 'current proprioceptive state', 'anchors correction to the robot configuration'],
          ['expert action', 'LIBERO demonstration action', 'training target for residual correction'],
        ]}
      />
    </div>
  </PageShell>
);

const ExperimentInventory: Page = () => (
  <PageShell label="experiment map">
    <Heading maxWidth={1460}>已完成與待補的實驗分成四條主線</Heading>
    <Note>
      表格只記設計意義，不放 raw command。正式數字放在後續結果頁，完整 row-level provenance 留在 registry。
    </Note>
    <div className="fade-3" style={{ marginTop: 34 }}>
      <Table
        headers={['Track', 'Human question', 'Current status', 'Paper use']}
        widths={['23%', '37%', '18%', '22%']}
        rows={[
          ['Execution interval', '慢速 plan 固定 50 步時，每幾步刷新 execution 最合適？', 'completed, spatial-only', 'baseline comparison'],
          ['Matched chunk', 'planning、execution、replan 都用同一個 K 時，模型是否仍有優勢？', 'completed, spatial-only', 'chunk-length stress'],
          ['α / δ calibration', '修正強度和上限要多大才不會 over-correct？', 'completed, spatial-only', 'deployment default'],
          ['Planner delay', 'slow planner 有 d 步延遲時，wrist correction 是否能補償 stale chunk？', 'completed, spatial-only', 'reactivity evidence'],
          ['Hardware / diagnostics', '真實機器與 wrist attention 圖是否支持方法敘事？', 'planned', 'paper completeness'],
        ]}
      />
    </div>
  </PageShell>
);

const Plan50Result: Page = () => (
  <PageShell label="result: fixed plan">
    <Heading maxWidth={1540}>固定 slow plan=50：HFRVLA 在多數 K 提升 spatial success</Heading>
    <Note>
      LIBERO-Spatial only，100 episodes per setting。K 表示每幾個控制步刷新一次 execution。
    </Note>
    <div className="fade-3" style={{ marginTop: 24, display: 'grid', gridTemplateColumns: '900px 1fr', gap: 24, alignItems: 'start' }}>
      <SuccessChart title="fixed plan=50, execution interval swept" data={plan50Rows} />
      <Table
        headers={['Setting', 'SmolVLA', 'HFRVLA', 'Difference', 'Note']}
        widths={['15%', '23%', '23%', '17%', '22%']}
        rows={successTableRows(plan50Rows)}
      />
    </div>
  </PageShell>
);

const MatchedChunkResult: Page = () => (
  <PageShell label="result: matched chunk">
    <Heading maxWidth={1540}>Matched chunk：短 K 不贏，K=4 之後才出現優勢</Heading>
    <Note>
      LIBERO-Spatial only，100 episodes per setting；planning、execution、replan 都設成同一個 K。
    </Note>
    <div className="fade-3" style={{ marginTop: 24, display: 'grid', gridTemplateColumns: '900px 1fr', gap: 24, alignItems: 'start' }}>
      <SuccessChart title="planning = execution = replan = K" data={matchedChunkRows} />
      <Table
        headers={['Setting', 'SmolVLA', 'HFRVLA', 'Difference', 'Note']}
        widths={['15%', '23%', '23%', '17%', '22%']}
        rows={successTableRows(matchedChunkRows)}
      />
    </div>
  </PageShell>
);

const CalibrationResult: Page = () => (
  <PageShell label="result: α and δ">
    <Heading maxWidth={1560}>長 chunk calibration：有效上限 αδ≈0.10 是目前安全區</Heading>
    <Note>
      LIBERO-Spatial only，50 episodes per setting，planning=execution=replan=50。這頁用來決定後續 long-chunk / deployment 預設值。
    </Note>
    <div className="fade-3" style={{ marginTop: 22, display: 'grid', gridTemplateColumns: '1.14fr .86fr', gap: 22, alignItems: 'start' }}>
      <AlphaClipHeatmap />
      <div style={{ display: 'grid', gap: 16 }}>
        <Card title="Best observed rows" body="α=0.5, δ_max=0.2 與 α=1.0, δ_max=0.1 都是 28/50 = 56%。" tone={colors.teal} />
        <Card title="Practical default" body="目前偏好 α=0.5, δ_max=0.2；它保留較小 scale，同時達到有效上限 0.10。" tone={colors.gold} />
        <Card title="Important failure mode" body="沒有 residual cap 時，α=0.5/0.75/1.0 分別掉到 16%、4%、2%。" tone={colors.coral} />
      </div>
    </div>
  </PageShell>
);

const DelayDefinition: Page = () => (
  <PageShell label="planner delay">
    <Heading maxWidth={1540}>Planner delay 的問題：slow chunk 會變舊，但 wrist feedback 是即時的</Heading>
    <Note>
      Slow planner 在時間 t 看到 observation，chunk 到 t+d 才能開始執行；HFRVLA 測試 δa 能否補償這個 stale base action。
    </Note>
    <div className="fade-3" style={{ marginTop: 36 }}>
      <DelayTimeline />
    </div>
  </PageShell>
);

const DelayResult: Page = () => (
  <PageShell label="result: planner delay">
    <Heading maxWidth={1540}>Async delay：打開 wrist correction 後，d=1..4 沒有跟著 base-only 下滑</Heading>
    <Note>
      LIBERO-Spatial，100 episodes per delay，N=8，plan=50，execution=16。Disable-fast 是同一 wrapper 關掉 fast path 的對照。
    </Note>
    <div className="fade-3" style={{ marginTop: 24, display: 'grid', gridTemplateColumns: '860px 1fr', gap: 24, alignItems: 'start' }}>
      <DelayChart />
      <Table
        headers={['Delay d', 'Disable-fast', 'HFRVLA', 'Difference', 'Check']}
        widths={['17%', '23%', '22%', '18%', '20%']}
        rows={delayRows.map((row) => [
          `${row.delay}`,
          `${row.disableFast}/100 = ${row.disableFast}%`,
          `${row.hfrvla}/100 = ${row.hfrvla}%`,
          <DeltaText value={row.hfrvla - row.disableFast} />,
          `start index ${row.startIndex}`,
        ])}
      />
    </div>
  </PageShell>
);

const ClaimBoundary: Page = () => (
  <PageShell label="paper claims">
    <Heading maxWidth={1500}>目前可以說的事，和還不能說的事，要分開</Heading>
    <div className="fade-2" style={{ marginTop: 46, display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 24 }}>
      <div style={{ display: 'grid', gap: 18 }}>
      <Card title="Supported" body="在 LIBERO-Spatial 的多個 matched baseline 設定下，小型 fast wrist correction 可以提升 closed-loop success。" tone={colors.teal} />
        <Card title="Supported" body="Residual strength 必須校準；αδ≈0.10 是目前 long-chunk useful region。" tone={colors.teal} />
        <Card title="Supported" body="Planner-delay sweep 顯示 wrist correction 能抵抗 fast path disabled 時的 delay degradation。" tone={colors.teal} />
      </div>
      <div style={{ display: 'grid', gap: 18 }}>
        <Card title="Not yet" body="不能宣稱已完成 real-robot hardware validation；目前仍需要 tabletop rollout。" tone={colors.coral} />
        <Card title="Not yet" body="不能宣稱全面優於 A2C2；目前應寫成 closest related work 與 framing 差異。" tone={colors.coral} />
        <Card title="Retired" body="GRU、learned gate、contact auxiliary head 不是當前方法，只能放在歷史背景或 archive。" tone={colors.violet} />
      </div>
    </div>
  </PageShell>
);

const PaperNeeds: Page = () => (
  <PageShell label="future paper needs">
    <Heading maxWidth={1520}>下一步 paper 需要補齊的是 evidence，不是更多操作頁</Heading>
    <Note>
      長 paper 和碩士論文可以分開維護，但共用同一份 evidence base：registry、paper notes、figures、bibliography。
    </Note>
    <div className="fade-3" style={{ marginTop: 34, display: 'grid', gridTemplateColumns: '1fr 1fr 1fr', gap: 20 }}>
      <Card title="Final result table" body="從 registry 產生，不從 log 手抄；這份 briefing 的比較表統一採 LIBERO-Spatial only。" tone={colors.blue} />
      <Card title="Real robot rollout" body="2-3 個 tabletop tasks，matched task text/camera setup，記錄成功率與失敗類型。" tone={colors.gold} />
      <Card title="Timing instrumentation" body="記錄 fast latency、slow chunk latency、clip fraction、safety hits、applied ratio。" tone={colors.teal} />
      <Card title="Qualitative figures" body="wrist view、correction arrows、DINO patch heatmap；只作診斷，不單獨當 causal proof。" tone={colors.violet} />
      <Card title="Target ablations" body="residual target、time-age features、slow context removal 都是 reviewer 會問的補強項。" tone={colors.coral} />
      <Card title="Claim wording" body="把 method claim、system framing、hardware validation、limitations 明確分段。" tone={colors.blue} />
    </div>
  </PageShell>
);

const SourceOfTruth: Page = () => (
  <PageShell label="source of truth">
    <Heading maxWidth={1500}>維護規則：改內容從 source 改，引用數字從 registry 查</Heading>
    <Note>
      這頁是未來 agent 或人類接手時的入口。舊 `training_presentation.html` 只保留為相容 redirect。
    </Note>
    <div className="fade-3" style={{ marginTop: 34 }}>
      <Table
        headers={['Purpose', 'Current file', 'Rule']}
        widths={['27%', '40%', '33%']}
        rows={[
          ['Human briefing', 'docs/hfrvla_experiment_briefing.html', 'generated output; do not hand-edit'],
          ['Editable deck', 'docs/presentations/hfrvla-training-open-slide/slides/hfrvla-training/index.tsx', 'edit this React source'],
          ['Build command', 'python3 scripts/generate_training_presentation.py', 'writes new briefing and legacy redirect'],
          ['Eval manifest', 'experiments/eval_registry/sources.csv', 'append new runs here first'],
          ['Eval master table', 'experiments/eval_registry/eval_results_master.csv', 'regenerate; do not manually edit'],
          ['Operational training docs', 'docs/training.md', 'commands live here, not in the briefing'],
        ]}
      />
    </div>
  </PageShell>
);

export const meta: SlideMeta = {
  title: 'HFRVLA 實驗與論文資料簡報',
  createdAt: '2026-05-19T12:43:06.981Z',
};

export default [
  Cover,
  Purpose,
  ModelAtGlance,
  Formula,
  EvidenceBase,
  ExperimentInventory,
  Plan50Result,
  MatchedChunkResult,
  CalibrationResult,
  DelayDefinition,
  DelayResult,
  ClaimBoundary,
  PaperNeeds,
  SourceOfTruth,
] satisfies Page[];
