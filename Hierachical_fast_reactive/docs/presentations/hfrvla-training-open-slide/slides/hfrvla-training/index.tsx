import type { DesignSystem, Page, SlideMeta } from '@open-slide/core';

const systemReflexPreview = new URL(
  '../../assets/hfrvla-paper/system1-system2-wrist-reflex-preview.png',
  import.meta.url,
).href;
const wristFeedbackPreview = new URL(
  '../../assets/hfrvla-paper/wrist-feedback-rollout-preview.png',
  import.meta.url,
).href;
const dinoPatchPreview = new URL(
  '../../assets/hfrvla-paper/dino-patch-heatmap-preview.png',
  import.meta.url,
).href;
const asyncPlannerDelaySummaryPlot = new URL(
  '../../assets/hfrvla-paper/async-timestep-planner-delay-summary.png',
  import.meta.url,
).href;

export const design: DesignSystem = {
  palette: {
    bg: '#08110f',
    text: '#f4fbf7',
    accent: '#32d399',
  },
  fonts: {
    display: '"Avenir Next", "PingFang TC", "Microsoft JhengHei", system-ui, sans-serif',
    body: '"PingFang TC", "Microsoft JhengHei", "Noto Sans CJK TC", system-ui, sans-serif',
  },
  typeScale: {
    hero: 138,
    body: 36,
  },
  radius: 10,
};

const colors = {
  bg: 'var(--osd-bg)',
  text: 'var(--osd-text)',
  accent: 'var(--osd-accent)',
  muted: '#a9c8bc',
  dim: '#557368',
  panel: 'rgba(11, 31, 27, 0.82)',
  panelHi: 'rgba(21, 54, 46, 0.88)',
  line: 'rgba(120, 232, 190, 0.24)',
  lineStrong: 'rgba(120, 232, 190, 0.44)',
  gold: '#f7c66a',
  rose: '#ff7c9a',
  blue: '#77b7ff',
  violet: '#b59cff',
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
    from { opacity: 0; transform: translateY(20px); }
    to { opacity: 1; transform: translateY(0); }
  }
  @keyframes hfr-line {
    from { transform: scaleX(0); }
    to { transform: scaleX(1); }
  }
  @keyframes hfr-pulse {
    0%, 100% { opacity: 0.45; transform: scale(0.98); }
    50% { opacity: 0.95; transform: scale(1.02); }
  }
  @keyframes hfr-queue-pop {
    0%, 9% { opacity: 0.35; transform: translateY(0) scale(0.96); box-shadow: none; }
    12%, 22% { opacity: 1; transform: translateY(-14px) scale(1.08); box-shadow: 0 0 34px rgba(50,211,153,.42); }
    25%, 100% { opacity: 0.42; transform: translateY(0) scale(0.96); box-shadow: none; }
  }
  @keyframes hfr-queue-sweep {
    from { transform: translateX(0); }
    to { transform: translateX(710px); }
  }
  @keyframes hfr-correction-flash {
    0%, 100% { opacity: 0.38; }
    50% { opacity: 1; }
  }
  @keyframes hfr-live-tick {
    0%, 8% { transform: translateX(0); }
    13%, 21% { transform: translateX(126px); }
    26%, 34% { transform: translateX(252px); }
    39%, 47% { transform: translateX(378px); }
    52%, 60% { transform: translateX(504px); }
    65%, 73% { transform: translateX(630px); }
    78%, 86% { transform: translateX(756px); }
    91%, 100% { transform: translateX(882px); }
  }
  @keyframes hfr-live-cell {
    0%, 8% { opacity: 1; transform: translateY(-8px); }
    10%, 100% { opacity: 0.42; transform: translateY(0); }
  }
  @keyframes hfr-live-send {
    0%, 100% { box-shadow: 0 0 0 rgba(119,183,255,0); }
    50% { box-shadow: 0 0 42px rgba(119,183,255,.42); }
  }
  .fade-1, .fade-2, .fade-3, .fade-4, .fade-5 {
    opacity: 0;
    animation: hfr-fade-up 720ms cubic-bezier(.2,.7,.2,1) forwards;
  }
  .fade-2 { animation-delay: 120ms; }
  .fade-3 { animation-delay: 240ms; }
  .fade-4 { animation-delay: 360ms; }
  .fade-5 { animation-delay: 480ms; }
  .line-grow {
    transform-origin: left center;
    animation: hfr-line 900ms cubic-bezier(.2,.7,.2,1) forwards;
  }
  .pulse { animation: hfr-pulse 2.8s ease-in-out infinite; }
  .queue-pop {
    animation: hfr-queue-pop 3.6s ease-in-out infinite;
  }
  .queue-sweep {
    animation: hfr-queue-sweep 3.6s linear infinite;
  }
  .correction-flash {
    animation: hfr-correction-flash 1.2s ease-in-out infinite;
  }
  .live-tick {
    animation: hfr-live-tick 6.4s steps(1, end) infinite;
  }
  .live-cell {
    animation: hfr-live-cell 6.4s ease-in-out infinite;
  }
  .live-send {
    animation: hfr-live-send 1.1s ease-in-out infinite;
  }
`;

const Styles = () => <style>{styles}</style>;

const Grid = () => (
  <div
    style={{
      position: 'absolute',
      inset: 0,
      backgroundImage:
        'linear-gradient(rgba(120,232,190,.045) 1px, transparent 1px), linear-gradient(90deg, rgba(120,232,190,.045) 1px, transparent 1px)',
      backgroundSize: '72px 72px',
      maskImage: 'linear-gradient(180deg, rgba(0,0,0,.95), rgba(0,0,0,.42) 70%, rgba(0,0,0,.15))',
      WebkitMaskImage: 'linear-gradient(180deg, rgba(0,0,0,.95), rgba(0,0,0,.42) 70%, rgba(0,0,0,.15))',
    }}
  />
);

const Orbital = () => (
  <div aria-hidden style={{ position: 'absolute', right: 70, top: 80, width: 620, height: 620 }}>
    <div
      className="pulse"
      style={{
        position: 'absolute',
        inset: 54,
        border: `1px solid ${colors.lineStrong}`,
        borderRadius: '50%',
      }}
    />
    <div
      style={{
        position: 'absolute',
        inset: 0,
        border: `1px solid ${colors.line}`,
        borderRadius: '50%',
      }}
    />
    <div
      style={{
        position: 'absolute',
        left: 240,
        top: 240,
        width: 140,
        height: 140,
        borderRadius: '50%',
        background: colors.accent,
        boxShadow: '0 0 70px rgba(50,211,153,.38)',
      }}
    />
    <div
      style={{
        position: 'absolute',
        left: 80,
        top: 290,
        width: 112,
        height: 112,
        borderRadius: 28,
        border: `1px solid ${colors.gold}`,
        transform: 'rotate(-18deg)',
      }}
    />
  </div>
);

const PageShell = ({
  children,
  label = 'HFRVLA Training',
}: {
  children: React.ReactNode;
  label?: string;
}) => (
  <div style={{ ...fill }}>
    <Styles />
    <Grid />
    <div style={{ position: 'absolute', left: 120, top: 76, fontFamily: font.mono, fontSize: 24, color: colors.gold, letterSpacing: '0.18em', textTransform: 'uppercase' }}>
      {label}
    </div>
    <div style={{ position: 'relative', height: '100%', padding: '128px 120px 104px' }}>{children}</div>
  </div>
);

const Heading = ({ children, maxWidth = 1180 }: { children: React.ReactNode; maxWidth?: number }) => (
  <h2
    className="fade-1"
    style={{
      margin: 0,
      maxWidth,
      fontFamily: font.display,
      fontSize: 78,
      lineHeight: 1.06,
      fontWeight: 850,
      letterSpacing: 0,
    }}
  >
    {children}
  </h2>
);

const Note = ({ children, width = 960 }: { children: React.ReactNode; width?: number }) => (
  <p className="fade-2" style={{ margin: 0, maxWidth: width, fontSize: 34, lineHeight: 1.45, color: colors.muted }}>
    {children}
  </p>
);

const Card = ({
  title,
  body,
  tone = colors.accent,
}: {
  title: string;
  body: string;
  tone?: string;
}) => (
  <div
    style={{
      border: `1px solid ${colors.line}`,
      background: colors.panel,
      borderRadius: 10,
      padding: '34px 36px',
      minHeight: 198,
      display: 'flex',
      flexDirection: 'column',
      justifyContent: 'space-between',
      gap: 18,
    }}
  >
    <div style={{ fontSize: 30, fontWeight: 800, color: tone }}>{title}</div>
    <div style={{ fontSize: 28, lineHeight: 1.38, color: colors.muted }}>{body}</div>
  </div>
);

const MiniCard = ({
  title,
  body,
  tone = colors.accent,
}: {
  title: string;
  body: string;
  tone?: string;
}) => (
  <div
    style={{
      border: `1px solid ${colors.line}`,
      background: colors.panel,
      borderRadius: 10,
      padding: '24px 28px',
      minHeight: 132,
      display: 'flex',
      flexDirection: 'column',
      justifyContent: 'center',
      gap: 14,
    }}
  >
    <div style={{ fontSize: 27, fontWeight: 850, color: tone }}>{title}</div>
    <div style={{ fontSize: 24, lineHeight: 1.32, color: colors.muted }}>{body}</div>
  </div>
);

const Step = ({
  index,
  title,
  body,
}: {
  index: string;
  title: string;
  body: string;
}) => (
  <div style={{ display: 'grid', gridTemplateColumns: '86px 1fr', gap: 26, alignItems: 'start' }}>
    <div
      style={{
        width: 70,
        height: 70,
        borderRadius: 18,
        display: 'grid',
        placeItems: 'center',
        background: colors.panelHi,
        border: `1px solid ${colors.lineStrong}`,
        fontFamily: font.mono,
        fontSize: 26,
        color: colors.gold,
      }}
    >
      {index}
    </div>
    <div>
      <div style={{ fontSize: 33, fontWeight: 850, color: colors.text }}>{title}</div>
      <div style={{ marginTop: 8, fontSize: 27, lineHeight: 1.34, color: colors.muted }}>{body}</div>
    </div>
  </div>
);

const CodeBlock = ({
  children,
  size = 25,
  lineHeight = 1.32,
  padding = '30px 34px',
}: {
  children: string;
  size?: number;
  lineHeight?: number;
  padding?: string;
}) => (
  <pre
    style={{
      margin: 0,
      whiteSpace: 'pre-wrap',
      fontFamily: font.mono,
      fontSize: size,
      lineHeight,
      color: '#d7fff0',
      background: '#020807',
      border: `1px solid ${colors.lineStrong}`,
      borderRadius: 10,
      padding,
      boxShadow: '0 26px 70px rgba(0,0,0,.35)',
    }}
  >
    {children}
  </pre>
);

const Pill = ({ children, color = colors.accent }: { children: React.ReactNode; color?: string }) => (
  <div
    style={{
      display: 'inline-flex',
      alignItems: 'center',
      height: 54,
      padding: '0 22px',
      borderRadius: 999,
      border: `1px solid ${color}`,
      color,
      fontFamily: font.mono,
      fontSize: 22,
      background: 'rgba(0,0,0,.18)',
    }}
  >
    {children}
  </div>
);

const ArchBox = ({
  x,
  y,
  w,
  h,
  title,
  children,
  tone = colors.accent,
  dashed = false,
}: {
  x: number;
  y: number;
  w: number;
  h: number;
  title: string;
  children: React.ReactNode;
  tone?: string;
  dashed?: boolean;
}) => (
  <div
    style={{
      position: 'absolute',
      left: x,
      top: y,
      width: w,
      height: h,
      borderRadius: 10,
      border: `2px ${dashed ? 'dashed' : 'solid'} ${tone}`,
      background: 'rgba(5, 17, 15, .9)',
      boxShadow: '0 22px 70px rgba(0,0,0,.26)',
      padding: '18px 20px',
      display: 'flex',
      flexDirection: 'column',
      gap: 10,
      justifyContent: 'flex-start',
      overflow: 'hidden',
    }}
  >
    <div style={{ fontSize: 24, fontWeight: 900, color: tone, lineHeight: 1.08 }}>{title}</div>
    <div style={{ fontSize: 19, lineHeight: 1.28, color: colors.muted }}>{children}</div>
  </div>
);

const ArchChip = ({ children, tone = colors.lineStrong }: { children: React.ReactNode; tone?: string }) => (
  <div
    style={{
      display: 'inline-flex',
      alignItems: 'center',
      height: 30,
      padding: '0 9px',
      borderRadius: 7,
      border: `1px solid ${tone}`,
      color: colors.text,
      background: 'rgba(255,255,255,.035)',
      fontFamily: font.mono,
      fontSize: 15,
      margin: '0 6px 6px 0',
      whiteSpace: 'nowrap',
      maxWidth: '100%',
      overflow: 'hidden',
    }}
  >
    {children}
  </div>
);

const DiagramBox = ({
  title,
  subtitle,
  children,
  tone = colors.accent,
  style = {},
}: {
  title: string;
  subtitle?: string;
  children?: React.ReactNode;
  tone?: string;
  style?: React.CSSProperties;
}) => (
  <div
    style={{
      border: `2px solid ${tone}`,
      background: colors.panel,
      borderRadius: 10,
      padding: '24px 26px',
      minHeight: 146,
      boxShadow: '0 24px 70px rgba(0,0,0,.24)',
      ...style,
    }}
  >
    <div style={{ fontSize: 29, fontWeight: 900, color: tone, lineHeight: 1.08 }}>{title}</div>
    {subtitle ? (
      <div style={{ marginTop: 10, fontSize: 21, lineHeight: 1.28, color: colors.muted }}>{subtitle}</div>
    ) : null}
    {children ? <div style={{ marginTop: 18 }}>{children}</div> : null}
  </div>
);

const DiagramChip = ({
  label,
  detail,
  tone = colors.lineStrong,
}: {
  label: string;
  detail?: string;
  tone?: string;
}) => (
  <div
    style={{
      display: 'grid',
      gridTemplateColumns: detail ? '210px 1fr' : '1fr',
      gap: 12,
      alignItems: 'center',
      border: `1px solid ${tone}`,
      background: 'rgba(255,255,255,.035)',
      borderRadius: 8,
      padding: '10px 12px',
      minHeight: 44,
      fontFamily: font.mono,
      fontSize: 18,
      color: colors.text,
    }}
  >
    <span style={{ color: tone, fontWeight: 800 }}>{label}</span>
    {detail ? <span style={{ color: colors.muted, fontFamily: font.body }}>{detail}</span> : null}
  </div>
);

const FlowArrow = ({ label, tone = colors.accent }: { label?: string; tone?: string }) => (
  <div style={{ display: 'grid', placeItems: 'center', minWidth: 92 }}>
    <div style={{ fontFamily: font.mono, fontSize: 44, color: tone, lineHeight: 1 }}>→</div>
    {label ? <div style={{ marginTop: 8, fontFamily: font.mono, fontSize: 16, color: colors.muted }}>{label}</div> : null}
  </div>
);

const LossRow = ({
  name,
  target,
  role,
  tone = colors.rose,
}: {
  name: string;
  target: string;
  role: string;
  tone?: string;
}) => (
  <div
    style={{
      display: 'grid',
      gridTemplateColumns: '220px 530px 1fr',
      gap: 20,
      alignItems: 'center',
      border: `1px solid ${colors.line}`,
      background: colors.panel,
      borderRadius: 10,
      padding: '18px 22px',
      minHeight: 86,
    }}
  >
    <div style={{ fontFamily: font.mono, fontSize: 25, fontWeight: 900, color: tone }}>{name}</div>
    <div style={{ fontFamily: font.mono, fontSize: 22, color: colors.text, lineHeight: 1.28 }}>{target}</div>
    <div style={{ fontSize: 22, color: colors.muted, lineHeight: 1.32 }}>{role}</div>
  </div>
);

const ArchitectureArrows = () => (
  <svg
    viewBox="0 0 1680 720"
    style={{ position: 'absolute', inset: 0, width: '100%', height: '100%', overflow: 'visible', pointerEvents: 'none' }}
  >
    <defs>
      <marker id="arch-arrow" markerWidth="12" markerHeight="12" refX="10" refY="6" orient="auto">
        <path d="M0,0 L12,6 L0,12 Z" fill={colors.accent} />
      </marker>
      <marker id="arch-arrow-gold" markerWidth="12" markerHeight="12" refX="10" refY="6" orient="auto">
        <path d="M0,0 L12,6 L0,12 Z" fill={colors.gold} />
      </marker>
      <marker id="arch-arrow-blue" markerWidth="12" markerHeight="12" refX="10" refY="6" orient="auto">
        <path d="M0,0 L12,6 L0,12 Z" fill={colors.blue} />
      </marker>
      <marker id="arch-arrow-rose" markerWidth="12" markerHeight="12" refX="10" refY="6" orient="auto">
        <path d="M0,0 L12,6 L0,12 Z" fill={colors.rose} />
      </marker>
    </defs>
    <path d="M260 156 C294 156 296 145 330 145" stroke={colors.gold} strokeWidth="3" fill="none" markerEnd="url(#arch-arrow-gold)" />
    <path d="M260 236 C300 270 294 358 330 384" stroke={colors.blue} strokeWidth="3" fill="none" markerEnd="url(#arch-arrow-blue)" />
    <path d="M760 145 L830 145" stroke={colors.gold} strokeWidth="3" fill="none" markerEnd="url(#arch-arrow-gold)" />
    <path d="M980 220 C980 238 952 238 952 250" stroke={colors.gold} strokeWidth="3" fill="none" markerEnd="url(#arch-arrow-gold)" />
    <path d="M690 385 C728 385 724 424 760 424" stroke={colors.blue} strokeWidth="3" fill="none" markerEnd="url(#arch-arrow-blue)" />
    <path d="M1315 404 L1365 404" stroke={colors.accent} strokeWidth="3" fill="none" markerEnd="url(#arch-arrow)" />
    <path d="M1515 480 C1515 512 1494 518 1494 535" stroke={colors.accent} strokeWidth="3" fill="none" markerEnd="url(#arch-arrow)" />
    <path d="M760 190 C1085 190 1138 606 1335 606" stroke={colors.gold} strokeWidth="3" strokeDasharray="9 10" fill="none" markerEnd="url(#arch-arrow-gold)" />
    <path d="M1660 610 L1680 610" stroke={colors.accent} strokeWidth="3" fill="none" markerEnd="url(#arch-arrow)" />
    <text x="792" y="180" fill={colors.gold} fontFamily={font.mono} fontSize="19">a_base bypass</text>
    <text x="1018" y="242" fill={colors.gold} fontFamily={font.mono} fontSize="19">cached hooks</text>
    <text x="704" y="405" fill={colors.blue} fontFamily={font.mono} fontSize="19">wrist patches</text>
    <text x="1585" y="586" fill={colors.accent} fontFamily={font.mono} fontSize="21">a_final</text>
  </svg>
);

const Architecture: Page = () => (
  <PageShell label="Model architecture">
    <Heading>整體模型架構</Heading>
    <div
      className="fade-2"
      style={{
        position: 'relative',
        marginTop: 34,
        width: 1680,
        height: 720,
        borderRadius: 10,
        border: `1px solid ${colors.line}`,
        background:
          'linear-gradient(rgba(120,232,190,.03) 1px, transparent 1px), linear-gradient(90deg, rgba(120,232,190,.03) 1px, transparent 1px), rgba(0,0,0,.13)',
        backgroundSize: '54px 54px',
      }}
    >
      <ArchitectureArrows />
      <ArchBox x={20} y={80} w={240} h={195} title="Runtime inputs" tone={colors.violet}>
        <ArchChip tone={colors.violet}>task text</ArchChip>
        <ArchChip tone={colors.violet}>third-person image</ArchChip>
        <ArchChip tone={colors.violet}>wrist image2</ArchChip>
        <ArchChip tone={colors.violet}>state (8)</ArchChip>
      </ArchBox>
      <ArchBox x={330} y={40} w={430} h={205} title="Frozen SmolVLA slow planner" tone={colors.gold}>
        <ArchChip tone={colors.gold}>VLM text → z_goal (960)</ArchChip>
        <ArchChip tone={colors.gold}>LM expert → z_phase (480)</ArchChip>
        <ArchChip tone={colors.gold}>chunk action → a_base (7)</ArchChip>
        <ArchChip tone={colors.gold}>action queue</ArchChip>
      </ArchBox>
      <ArchBox x={830} y={50} w={300} h={170} title="Recorded extras" tone={colors.gold} dashed>
        <ArchChip tone={colors.gold}>a_base</ArchChip>
        <ArchChip tone={colors.gold}>z_goal</ArchChip>
        <ArchChip tone={colors.gold}>z_phase</ArchChip>
        <ArchChip tone={colors.gold}>k_idx_norm</ArchChip>
        <ArchChip tone={colors.gold}>offline cache</ArchChip>
      </ArchBox>
      <ArchBox x={330} y={310} w={360} h={150} title="DINOv3 wrist path" tone={colors.blue}>
        <ArchChip tone={colors.blue}>ViT-S/16 frozen</ArchChip>
        <ArchChip tone={colors.blue}>196 patches</ArchChip>
        <ArchChip tone={colors.blue}>384 dim</ArchChip>
        <ArchChip tone={colors.blue}>dino_patches</ArchChip>
      </ArchBox>
      <ArchBox x={760} y={250} w={555} h={360} title="Fast Wrist Residual core" tone={colors.accent}>
        <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 10 }}>
          <ArchChip>patch proj</ArchChip>
          <ArchChip>task query MLP</ArchChip>
          <ArchChip>cross-attn pool</ArchChip>
          <ArchChip>z_phase proj</ArchChip>
          <ArchChip>state fuser</ArchChip>
          <ArchChip tone={colors.gold}>prev delta</ArchChip>
        </div>
        <div style={{ marginTop: 4 }}>
          pooled wrist + state + a_base + k_idx + z_phase + previous correction。
        </div>
      </ArchBox>
      <ArchBox x={1365} y={270} w={295} h={210} title="Heads" tone={colors.accent}>
        <ArchChip>delta_a (7)</ArchChip>
        <ArchChip tone={colors.gold}>alpha scale</ArchChip>
        <ArchChip tone={colors.rose}>residual-only</ArchChip>
        <div>單一 residual head；輸出完整 7D correction。</div>
      </ArchBox>
      <ArchBox x={1335} y={535} w={325} h={155} title="Merge + safety" tone={colors.gold}>
        <div style={{ fontFamily: font.mono, fontSize: 19, color: colors.text, lineHeight: 1.35 }}>
          a_final = a_base + alpha * clip(delta_a)
        </div>
        <ArchChip tone={colors.gold}>delta_max = 0.2</ArchChip>
      </ArchBox>
    </div>
  </PageShell>
);

const TrainingBatchIO: Page = () => (
  <PageShell label="Training I/O">
    <Heading>training batch 先分清楚：哪些是輸入，哪些是學習目標</Heading>
    <Note width={1260}>
      LeRobot loader 讀的是已錄好的 HFRVLA dataset；training loop 不再 forward SmolVLA 或 DINO，只把 cache feature 餵進 fast module，並用 expert action 算 loss。
    </Note>
    <div className="fade-3" style={{ marginTop: 42, display: 'grid', gridTemplateColumns: '1fr 110px 1fr', gap: 28, alignItems: 'center', width: 1560 }}>
      <DiagramBox title="Fast module inputs" subtitle="policy.forward(batch) 會實際送進 FWR module" tone={colors.accent}>
        <div style={{ display: 'grid', gap: 10 }}>
          <DiagramChip label="observation.state" detail="proprio, shape (B,T,8)" tone={colors.violet} />
          <DiagramChip label="extra.a_base" detail="frozen SmolVLA action, shape (B,T,7)" tone={colors.gold} />
          <DiagramChip label="extra.k_idx_norm" detail="chunk position, shape (B,T,1)" tone={colors.gold} />
          <DiagramChip label="extra.z_goal" detail="task/text pool, shape (B,T,960)" tone={colors.blue} />
          <DiagramChip label="extra.z_phase" detail="expert hidden pool, shape (B,T,480)" tone={colors.blue} />
          <DiagramChip label="extra.dino_patches" detail="wrist patches, shape (B,T,196,384)" tone={colors.accent} />
        </div>
      </DiagramBox>
      <FlowArrow label="seq window" />
      <DiagramBox title="Learning targets" subtitle="FWR loss 對齊實際 deployment action" tone={colors.rose}>
        <div style={{ display: 'grid', gap: 10 }}>
          <DiagramChip label="action" detail="expert action a_expert, shape (B,T,7)" tone={colors.rose} />
          <DiagramChip label="prev_delta" detail="action[t-1] - a_base[t-1]" tone={colors.gold} />
          <DiagramChip label="target_delta" detail="raw residual term: action[t] - a_base[t]" tone={colors.gold} />
          <DiagramChip label="a_hat" detail="a_base + alpha * clip(delta_pred)" tone={colors.blue} />
          <DiagramChip label="loss" detail="delta + final + residual + clip terms" tone={colors.accent} />
        </div>
      </DiagramBox>
    </div>
  </PageShell>
);

const FastModuleOutputs: Page = () => (
  <PageShell label="Fast module outputs">
    <Heading>Fast Wrist Residual 只輸出一個 7D correction</Heading>
    <div className="fade-2" style={{ marginTop: 54, display: 'grid', gridTemplateColumns: '430px 90px 500px 90px 430px', gap: 20, alignItems: 'center', width: 1580 }}>
      <DiagramBox title="Inputs at time window" subtitle="state + cache feature + base action" tone={colors.blue}>
        <div style={{ display: 'grid', gap: 10 }}>
          <DiagramChip label="dino_patches" detail="visual wrist signal" tone={colors.blue} />
          <DiagramChip label="z_goal / z_phase" detail="slow planner context" tone={colors.gold} />
          <DiagramChip label="state + a_base + k" detail="robot and chunk context" tone={colors.violet} />
        </div>
      </DiagramBox>
      <FlowArrow />
      <DiagramBox title="FastWristResidualModule" subtitle="trainable: patch proj, cross-attn pool, fuser, delta head" tone={colors.accent} style={{ minHeight: 330 }}>
        <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 12 }}>
          <DiagramChip label="patch proj" tone={colors.accent} />
          <DiagramChip label="task query" tone={colors.accent} />
          <DiagramChip label="cross-attn" tone={colors.accent} />
          <DiagramChip label="state fuser" tone={colors.accent} />
          <DiagramChip label="prev delta" tone={colors.gold} />
          <DiagramChip label="delta head" tone={colors.accent} />
        </div>
      </DiagramBox>
      <FlowArrow />
      <DiagramBox title="Predicted residual" subtitle="single 7D correction head" tone={colors.rose}>
        <div style={{ display: 'grid', gap: 10 }}>
          <DiagramChip label="delta_a" detail="raw residual proposal, shape (B,7)" tone={colors.rose} />
          <DiagramChip label="clip" detail="deployment safety limit" tone={colors.gold} />
          <DiagramChip label="alpha" detail="simple scalar merge weight" tone={colors.blue} />
        </div>
      </DiagramBox>
    </div>
    <div className="fade-3" style={{ marginTop: 54, display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 26, width: 1540 }}>
      <DiagramBox title="Action merge" tone={colors.gold}>
        <div style={{ fontFamily: font.mono, fontSize: 36, lineHeight: 1.3, color: colors.text }}>
          a_final = a_base + alpha * clip(delta_a)
        </div>
      </DiagramBox>
      <DiagramBox title="Base action bypass" subtitle="a_base 來自 frozen SmolVLA queue；fast module 只輸出 residual correction。" tone={colors.gold} />
    </div>
  </PageShell>
);

const LeRobotHardwareDispatch: Page = () => (
  <PageShell label="Hardware dispatch">
    <Heading maxWidth={1580}>LeRobot 每 tick 只送一個命令</Heading>
    <Note width={1500}>
      async 路徑先把 action chunk 變成本地 queue；control loop 每個 tick 只 pop 一個點，再呼叫 robot.send_action()。
    </Note>
    <div className="fade-3" style={{ marginTop: 26, display: 'grid', gridTemplateColumns: '1.02fr .98fr', gap: 26, alignItems: 'start', width: 1640 }}>
      <div
        style={{
          border: `1px solid ${colors.lineStrong}`,
          background: colors.panel,
          borderRadius: 10,
          padding: '24px 28px',
          minHeight: 492,
          position: 'relative',
          overflow: 'hidden',
        }}
      >
        <div style={{ display: 'grid', gridTemplateColumns: '1fr 62px 1fr', gap: 14, alignItems: 'center' }}>
          <DiagramBox title="Policy server" subtitle="predict_action_chunk(obs)" tone={colors.gold} style={{ minHeight: 112, padding: '18px 20px', minWidth: 0 }}>
            <DiagramChip label="8 actions" detail="a[0] ... a[7]" tone={colors.gold} />
          </DiagramBox>
          <div style={{ display: 'grid', placeItems: 'center' }}>
            <div style={{ fontFamily: font.mono, fontSize: 38, color: colors.gold, lineHeight: 1 }}>→</div>
            <div style={{ marginTop: 6, fontFamily: font.mono, fontSize: 15, color: colors.muted }}>gRPC</div>
          </div>
          <DiagramBox title="Robot client" subtitle="local action queue" tone={colors.accent} style={{ minHeight: 112, padding: '18px 20px', minWidth: 0 }}>
            <DiagramChip label="TimedAction[]" detail="timestamped points" tone={colors.accent} />
          </DiagramBox>
        </div>

        <div style={{ marginTop: 32, position: 'relative', height: 146 }}>
          <div style={{ position: 'absolute', left: 22, right: 22, top: 64, height: 4, background: colors.lineStrong }} />
          <div
            className="queue-sweep"
            style={{
              position: 'absolute',
              left: 26,
              top: 50,
              width: 34,
              height: 34,
              borderRadius: '50%',
              background: colors.accent,
              boxShadow: '0 0 34px rgba(50,211,153,.46)',
            }}
          />
          <div style={{ display: 'grid', gridTemplateColumns: 'repeat(8, 1fr)', gap: 14, position: 'relative' }}>
            {Array.from({ length: 8 }, (_, i) => (
              <div key={i} style={{ display: 'grid', justifyItems: 'center', gap: 14 }}>
                <div
                  className="queue-pop"
                  style={{
                    animationDelay: `${i * 0.45}s`,
                    width: 66,
                    height: 66,
                    borderRadius: 10,
                    border: `2px solid ${i === 0 ? colors.gold : colors.accent}`,
                    background: i === 0 ? 'rgba(247,198,106,.16)' : 'rgba(50,211,153,.12)',
                    display: 'grid',
                    placeItems: 'center',
                    fontFamily: font.mono,
                    fontSize: 23,
                    fontWeight: 900,
                    color: i === 0 ? colors.gold : colors.text,
                  }}
                >
                  a{i}
                </div>
                <div style={{ fontFamily: font.mono, fontSize: 18, color: colors.muted }}>t+{i}dt</div>
              </div>
            ))}
          </div>
        </div>

        <div style={{ marginTop: 18, display: 'grid', gridTemplateColumns: '1fr 56px 1fr', gap: 12, alignItems: 'center' }}>
          <DiagramBox title="control_loop @ fps" subtitle="每 1/fps 秒 pop 一個 queued action" tone={colors.blue} style={{ minHeight: 108, padding: '18px 20px', minWidth: 0 }}>
            <DiagramChip label="a_i tensor" detail="1D action" tone={colors.blue} />
          </DiagramBox>
          <div style={{ display: 'grid', placeItems: 'center' }}>
            <div style={{ fontFamily: font.mono, fontSize: 34, color: colors.blue, lineHeight: 1 }}>→</div>
            <div style={{ marginTop: 6, fontFamily: font.mono, fontSize: 15, color: colors.muted }}>dict</div>
          </div>
          <DiagramBox title="send_action()" subtitle="送到 bus / SDK / robot driver" tone={colors.rose} style={{ minHeight: 108, padding: '18px 20px', minWidth: 0 }}>
            <DiagramChip label="single command" detail="不是整包 chunk" tone={colors.rose} />
          </DiagramBox>
        </div>
      </div>

      <div style={{ display: 'grid', gap: 12 }}>
        {[
          ['你的理解要修正一點', 'action chunk 是 server/client 的緩衝格式；硬體層通常看到的是每個 tick 的單一 action dict。', colors.gold],
          ['HFRVLA per-step correction', '正確語意是每個 a_base[i] 先變成 a_base[i] + alpha * clip(delta_i)，再進 queue 或 send_action。', colors.accent],
          ['controller 會不會吃不下', '問題不在 8 個點本身，而在 loop fps、bus latency、camera latency 是否能穩定執行每個 corrected point。', colors.rose],
          ['重要 caveat', 'LeRobot async server 呼叫 predict_action_chunk()；HFRVLA 現在的修正主路徑在 select_action()，直接走 async 可能只拿到 base chunk。', colors.blue],
        ].map(([title, body, tone]) => (
          <div
            key={title}
            style={{
              border: `1px solid ${colors.line}`,
              background: colors.panel,
              borderRadius: 10,
              padding: '17px 20px',
              minHeight: 84,
            }}
          >
            <div style={{ fontSize: 24, fontWeight: 900, color: tone as string }}>{title}</div>
            <div style={{ marginTop: 8, fontSize: 20, lineHeight: 1.28, color: colors.muted }}>{body}</div>
          </div>
        ))}
        <CodeBlock size={18} lineHeight={1.15} padding="18px 22px">{`async hardware path
  server: predict_action_chunk(obs) -> [a0..a7]
  server: attach timestamps t+i*dt
  client: merge into local queue
  loop:   pop one action per 1/fps
  robot:  send_action({joint_i: value, ...})`}</CodeBlock>
      </div>
    </div>
  </PageShell>
);

const LiveCorrectionTimeline: Page = () => {
  const ticks = Array.from({ length: 8 }, (_, i) => i);

  return (
    <PageShell label="Runtime timing">
      <Heading maxWidth={1600}>chunk 是慢計畫，correction 是每 tick 閉環</Heading>
      <Note width={1500}>
        第二種部署方式下，queue 裡保留 slow VLA 的 base plan；每個 control tick 讀最新 wrist/state，只修正當下即將送出的那一點。
      </Note>
      <div className="fade-3" style={{ marginTop: 28, display: 'grid', gridTemplateColumns: '1120px 1fr', gap: 28, width: 1640 }}>
        <div
          style={{
            border: `1px solid ${colors.lineStrong}`,
            background: colors.panel,
            borderRadius: 10,
            padding: '26px 30px',
            height: 565,
            position: 'relative',
            overflow: 'hidden',
          }}
        >
          <div style={{ display: 'grid', gap: 20 }}>
            <div>
              <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
                <div style={{ fontSize: 28, fontWeight: 900, color: colors.gold }}>Slow VLA chunk m</div>
                <div style={{ fontFamily: font.mono, fontSize: 18, color: colors.muted }}>replan after K ticks</div>
              </div>
              <div style={{ marginTop: 14, display: 'grid', gridTemplateColumns: '978px 1fr', gap: 18, alignItems: 'center' }}>
                <div style={{ display: 'grid', gridTemplateColumns: 'repeat(8, 96px)', gap: 30 }}>
                  {ticks.map((i) => (
                    <div
                      key={`base-${i}`}
                      style={{
                        height: 58,
                        borderRadius: 9,
                        border: `1px solid ${colors.gold}`,
                        background: 'rgba(247,198,106,.12)',
                        display: 'grid',
                        placeItems: 'center',
                        fontFamily: font.mono,
                        fontSize: 18,
                        color: colors.gold,
                      }}
                    >
                      a_base[{i}]
                    </div>
                  ))}
                </div>
                <div
                  style={{
                    height: 58,
                    borderRadius: 9,
                    border: `1px dashed ${colors.dim}`,
                    display: 'grid',
                    placeItems: 'center',
                    fontFamily: font.mono,
                    fontSize: 17,
                    color: colors.dim,
                  }}
                >
                  chunk m+1
                </div>
              </div>
            </div>

            <div>
              <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
                <div style={{ fontSize: 28, fontWeight: 900, color: colors.accent }}>Fast HFRVLA correction</div>
                <div style={{ fontFamily: font.mono, fontSize: 18, color: colors.muted }}>{'latest obs(t) -> delta(t)'}</div>
              </div>
              <div style={{ marginTop: 14, position: 'relative', height: 132 }}>
                <div style={{ position: 'absolute', left: 0, top: 58, width: 978, height: 4, background: colors.lineStrong }} />
                <div
                  className="live-tick"
                  style={{
                    position: 'absolute',
                    left: 30,
                    top: 42,
                    width: 38,
                    height: 38,
                    borderRadius: '50%',
                    background: colors.accent,
                    boxShadow: '0 0 36px rgba(50,211,153,.5)',
                    zIndex: 2,
                  }}
                />
                <div style={{ display: 'grid', gridTemplateColumns: 'repeat(8, 96px)', gap: 30, position: 'relative', zIndex: 1 }}>
                  {ticks.map((i) => (
                    <div
                      key={`delta-${i}`}
                      className="live-cell"
                      style={{
                        animationDelay: `${i * 0.8}s`,
                        height: 92,
                        borderRadius: 9,
                        border: `1px solid ${colors.accent}`,
                        background: 'rgba(50,211,153,.12)',
                        padding: '12px 8px',
                        display: 'grid',
                        alignContent: 'center',
                        gap: 6,
                        textAlign: 'center',
                      }}
                    >
                      <div style={{ fontFamily: font.mono, fontSize: 17, color: colors.text }}>obs{i}</div>
                      <div style={{ fontFamily: font.mono, fontSize: 19, fontWeight: 900, color: colors.accent }}>delta{i}</div>
                    </div>
                  ))}
                </div>
              </div>
            </div>

            <div>
              <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
                <div style={{ fontSize: 28, fontWeight: 900, color: colors.blue }}>Hardware command stream</div>
                <div style={{ fontFamily: font.mono, fontSize: 18, color: colors.muted }}>one send_action per tick</div>
              </div>
              <div style={{ marginTop: 14, display: 'grid', gridTemplateColumns: 'repeat(8, 96px)', gap: 30 }}>
                {ticks.map((i) => (
                  <div
                    key={`send-${i}`}
                    className="live-cell"
                    style={{
                      animationDelay: `${i * 0.8}s`,
                      height: 76,
                      borderRadius: 9,
                      border: `1px solid ${colors.blue}`,
                      background: 'rgba(119,183,255,.1)',
                      display: 'grid',
                      placeItems: 'center',
                      fontFamily: font.mono,
                      fontSize: 16,
                      lineHeight: 1.2,
                      color: colors.text,
                      textAlign: 'center',
                    }}
                  >
                    send
                    <br />
                    a_final[{i}]
                  </div>
                ))}
              </div>
            </div>
          </div>
        </div>

        <div style={{ display: 'grid', gap: 10 }}>
          {[
            ['Action chunk relation', '同一個 chunk 內的 a_base[0..7] 是 slow VLA 對未來 K 個 tick 的 nominal plan。', colors.gold],
            ['Correction relation', 'delta_i 用 obs_i、a_base_i、chunk index、previous delta 在當下重新估。', colors.accent],
            ['What controller sees', 'controller 只看到 a_final_i，不會同時收到 base/correction 兩條命令。', colors.blue],
            ['Latency budget', '如果 camera/bus 讓 tick 掉到 8 Hz，reactive correction 的有效頻率也只剩 8 Hz。', colors.rose],
          ].map(([title, body, tone]) => (
            <div
              key={title}
              style={{
                border: `1px solid ${colors.line}`,
                background: colors.panel,
                borderRadius: 10,
                padding: '14px 18px',
                minHeight: 82,
              }}
            >
              <div style={{ fontSize: 22, fontWeight: 900, color: tone as string }}>{title}</div>
              <div style={{ marginTop: 7, fontSize: 18, lineHeight: 1.24, color: colors.muted }}>{body}</div>
            </div>
          ))}
          <CodeBlock size={16} lineHeight={1.12} padding="14px 18px">{`for each tick i:
  obs_i = latest wrist/state
  a_i   = pop base action
  d_i   = fast(obs_i, a_i, i, prev_d)
  send_action(a_i + alpha * clip(d_i))`}</CodeBlock>
        </div>
      </div>
    </PageShell>
  );
};

const LearningObjective: Page = () => (
  <PageShell label="Learning objective">
    <Heading>FWR 目標要對齊實際送出的 action</Heading>
    <div className="fade-2" style={{ marginTop: 34, display: 'grid', gap: 12, width: 1640 }}>
      <LossRow
        name="L_delta"
        target="SmoothL1(delta_a, action[t] - a_base[t])"
        role="保留 raw residual supervision，讓 delta head 學完整 7D correction。"
        tone={colors.accent}
      />
      <LossRow
        name="L_final"
        target="SmoothL1(a_base + alpha * clip(delta_a), action[t])"
        role="訓練直接看 deployment 會送出的 clipped、alpha-scaled final action。"
        tone={colors.blue}
      />
      <LossRow
        name="L_residual"
        target="mean((alpha * clip(delta_a))^2)"
        role="用小權重限制 fast correction 能量，避免 residual 壓過 frozen base plan。"
        tone={colors.gold}
      />
      <LossRow
        name="L_clip"
        target="mean(relu(abs(delta_a) - delta_max))"
        role="預設權重可為 0，但仍保留 clip pressure 與 clip_fraction 監控。"
        tone={colors.rose}
      />
      <LossRow
        name="prev"
        target="prev_delta = action[t-1] - a_base[t-1]"
        role="前一幀 correction 是 conditioning；seq_len 仍只屬於 training-time window。"
        tone={colors.rose}
      />
    </div>
  </PageShell>
);

const Cover: Page = () => (
  <div style={{ ...fill }}>
    <Styles />
    <Grid />
    <Orbital />
    <div style={{ position: 'relative', height: '100%', padding: '150px 120px 118px', display: 'flex', flexDirection: 'column', justifyContent: 'space-between' }}>
      <div>
        <div className="fade-1" style={{ fontFamily: font.mono, fontSize: 26, color: colors.gold, letterSpacing: '0.2em', textTransform: 'uppercase' }}>
          Hierarchical Fast-Reactive VLA
        </div>
        <h1
          className="fade-2"
          style={{
            margin: '32px 0 0',
            width: 1050,
            fontFamily: font.display,
            fontSize: 'var(--osd-size-hero)',
            lineHeight: 0.98,
            fontWeight: 900,
            letterSpacing: 0,
          }}
        >
          HFRVLA 訓練流程
        </h1>
        <p className="fade-3" style={{ margin: '42px 0 0', width: 930, fontSize: 40, lineHeight: 1.42, color: colors.muted }}>
          frozen SmolVLA 慢規劃器，加上一個 wrist-camera fast residual module。
        </p>
      </div>
      <div className="fade-4" style={{ display: 'flex', gap: 18 }}>
        <Pill>open-slide build</Pill>
        <Pill color={colors.gold}>docs/training.md</Pill>
        <Pill color={colors.blue}>LeRobot-native</Pill>
      </div>
    </div>
  </div>
);

const Contract: Page = () => (
  <PageShell>
    <Heading>推論與訓練都圍繞同一個契約</Heading>
    <div className="fade-2" style={{ marginTop: 54, fontFamily: font.mono, fontSize: 56, color: colors.accent, background: colors.panel, border: `1px solid ${colors.lineStrong}`, borderRadius: 10, padding: '36px 42px', width: 1260 }}>
      a_final = a_base + alpha * clip(delta_a)
    </div>
    <div className="fade-3" style={{ marginTop: 54, display: 'grid', gridTemplateColumns: 'repeat(4, 1fr)', gap: 22, width: 1460 }}>
      <Card title="a_base" body="SmolVLA chunk action；慢系統已凍結。" tone={colors.gold} />
      <Card title="delta_a" body="FWR 學 raw residual，也用 final-action term 對齊 deployment。" />
      <Card title="alpha" body="訓練 config 是 1.0；正式 eval 常用 0.5 override。" tone={colors.blue} />
      <Card title="Safety" body="只 clip fast residual，不改寫 base action。" tone={colors.rose} />
    </div>
  </PageShell>
);

const Dataset: Page = () => (
  <PageShell>
    <Heading>訓練不是 raw LIBERO，而是 baked HFRVLA dataset</Heading>
    <Note>Recording 階段先跑 slow planner 與 wrist feature extractor，訓練 loop 只讀標準 LeRobotDataset v3。</Note>
    <div className="fade-3" style={{ marginTop: 52, display: 'grid', gridTemplateColumns: '1fr 1fr 1fr', gap: 24 }}>
      <Card title="Base action" body="observation.extra.a_base shape (7,)，作為 residual target 的基準。" tone={colors.gold} />
      <Card title="SmolVLA pools" body="z_goal 與 z_phase 已在 recording 時抽出，不在 train loop forward SmolVLA。" />
      <Card title="Wrist patches" body="DINOv3 dino_patches shape (196, 384)，提供 fast visual signal。" tone={colors.blue} />
    </div>
    <div className="fade-4" style={{ marginTop: 34, display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 24, width: 1190 }}>
      <Card title="Windowing" body="fast-cache 不存 seq_len；training loader 才用 policy.seq_len 建 window。FWR-v1 用 previous/current，FWR-v2 讀 full chunk。" tone={colors.violet} />
      <Card title="No extra normalization" body="observation.extra.* 刻意不放進 input_features normalization。" tone={colors.rose} />
    </div>
  </PageShell>
);

const Pipeline: Page = () => (
  <PageShell>
    <Heading>資料管線先建立契約，再進入長訓練</Heading>
    <div className="fade-2" style={{ marginTop: 60, display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 48, width: 1420 }}>
      <div style={{ display: 'grid', gap: 36 }}>
        <Step index="01" title="Record" body="從 HuggingFaceVLA/libero 產生 HFRVLA LeRobotDataset v3。" />
        <Step index="02" title="Merge" body="平行 shards 用 metadata/file stitch 合併，避免重新 encode。" />
        <Step index="03" title="Check" body="跑 training contract 與 alignment check，確認 zero-fast 近似 SmolVLA baseline。" />
      </div>
      <div style={{ background: colors.panel, border: `1px solid ${colors.line}`, borderRadius: 10, padding: '40px 44px' }}>
        <div style={{ fontSize: 32, fontWeight: 850, color: colors.gold }}>Verified dataset</div>
        <div style={{ marginTop: 28, fontFamily: font.mono, fontSize: 31, lineHeight: 1.42, color: colors.text }}>
          checkpoints/HFRVLA_libero_v1_merged_reindexed
        </div>
        <div className="line-grow" style={{ marginTop: 42, height: 2, background: colors.lineStrong }} />
        <p style={{ margin: '38px 0 0', fontSize: 30, lineHeight: 1.45, color: colors.muted }}>
          如果更換 slow planner，a_base、z_goal、z_phase 必須重錄，否則 residual target 會失真。
        </p>
      </div>
    </div>
  </PageShell>
);

const FastCache: Page = () => (
  <PageShell>
    <Heading>Fast-cache 只是一個本機加速層</Heading>
    <Note>Canonical source 仍是 LeRobotDataset v3；fast-cache 將大型欄位轉成 float16 memmap，減少 Parquet decode 開銷。</Note>
    <div className="fade-3" style={{ marginTop: 50, display: 'grid', gridTemplateColumns: '1.1fr .9fr', gap: 34, alignItems: 'stretch' }}>
      <CodeBlock size={25}>{`~/Robotic_infra/lerobot/.venv/bin/python scripts/build_hfrvla_fastcache.py \\
    --source-root checkpoints/HFRVLA_libero_v1_merged_reindexed \\
    --cache-root checkpoints/HFRVLA_libero_v1_fastcache_v3_plan50 \\
    --chunk-len 50`}</CodeBlock>
      <div style={{ display: 'grid', gap: 24 }}>
        <Card title="float16 cache" body="z_goal、z_phase、dino_patches 以 float16 儲存；進 Linear 前 cast 回 module dtype。" tone={colors.gold} />
        <Card title="frame-level cache" body="cache 只存 frame-aligned arrays；seq_len 是 training-time sampling choice。" tone={colors.rose} />
      </div>
    </div>
  </PageShell>
);

const TrainEntry: Page = () => (
  <PageShell>
    <Heading>正式訓練入口固定走 LeRobot wrapper</Heading>
    <div className="fade-2" style={{ marginTop: 52, display: 'grid', gridTemplateColumns: '1.16fr .84fr', gap: 34 }}>
      <CodeBlock size={23}>{`RUN_NAME=hfrvla_fwr_chunk_seq2 \\
WANDB_ENABLE=false \\
HFRVLA_TMP_ROOT=$HOME/tmp/hfrvla \\
HFRVLA_DATASET_BACKEND=fastcache \\
HFRVLA_FASTCACHE_ROOT=checkpoints/HFRVLA_libero_v1_fastcache_v3_plan50 \\
RESIDUAL_MERGE_MODE=fast_wrist_chunk \\
FAST_RESIDUAL_ALPHA=1.0 \\
SEQ_LEN=2 \\
BATCH_SIZE=256 \\
NUM_WORKERS=8 \\
scripts/run_hfrvla_training_foreground.sh`}</CodeBlock>
      <div style={{ display: 'grid', gap: 24 }}>
        <Card title="FWR-v2 chunk" body="stateless feed-forward correction：full base chunk cross-attn，學 current 7D delta。" tone={colors.gold} />
        <Card title="Expected log" body="[hfrvla-train] objective residual_mode=fast_wrist_chunk ..." />
        <Card title="Editable reinstall" body="policy source 修改後，先 uv pip install -e 再訓練。" tone={colors.rose} />
      </div>
    </div>
  </PageShell>
);

const SmokeProtocol: Page = () => (
  <PageShell>
    <Heading>每次改 code 後先跑小 smoke，再開長訓練</Heading>
    <Note>目標不是 500 step 就成功，而是先確認 training speed、VRAM、checkpoint packaging、eval I/O 都對齊。</Note>
    <div className="fade-3" style={{ marginTop: 42, display: 'grid', gridTemplateColumns: 'repeat(3, 1fr)', gap: 22 }}>
      <MiniCard title="Train smoke" body="500-1000 steps；若每 step 變成 10-20 秒要中止。" tone={colors.accent} />
      <MiniCard title="Package" body="把 fast checkpoint 和 frozen SmolVLA/DINO 組成 eval artifact。" tone={colors.blue} />
      <MiniCard title="Partial eval" body="先跑單 task / 1 episode，確認 eval_info 和 video 寫出。" tone={colors.gold} />
    </div>
    <div className="fade-4" style={{ marginTop: 32, width: 1320 }}>
      <CodeBlock size={22} lineHeight={1.24} padding="24px 30px">{`smoke checklist
  1. train short run and confirm loss is finite
  2. package checkpoint into LeRobot policy artifact
  3. run one LIBERO task / one episode with --policy.device=cuda
  4. confirm eval_info.json, log, and output_dir are written
  5. launch full spatial eval first; archive other suites separately if needed`}</CodeBlock>
    </div>
  </PageShell>
);

const DecisionLedger: Page = () => (
  <PageShell label="Decision ledger">
    <Heading maxWidth={1500}>目前要保留的研究決策</Heading>
    <Note width={1400}>
      這一頁只記 current mainline。若之後改方向，先更新這裡、memory、paper notes，再更新程式。
    </Note>
    <div className="fade-3" style={{ marginTop: 40, display: 'grid', gridTemplateColumns: 'repeat(3, 1fr)', gap: 20, width: 1600 }}>
      <MiniCard title="Slow planner" body="SmolVLA frozen；HFRVLA 不 fine-tune slow VLA，fast module 只修正 action chunk。" tone={colors.gold} />
      <MiniCard title="Fast signal" body="每步 visual correction 來自 wrist DINO patches；z_goal、z_phase、a_base 和 state 是 context。" tone={colors.blue} />
      <MiniCard title="Merge rule" body="a_final = a_base + alpha * clip(delta_a)；只限制 fast residual，不改寫 base action。" tone={colors.accent} />
      <MiniCard title="Output target" body="沿用完整 7D correction；raw residual term 加 final-action term 一起訓練。" tone={colors.rose} />
      <MiniCard title="Temporal input" body="保留 previous/current frame；seq_len 是 training-time window，不是 fast-cache schema。" tone={colors.violet} />
      <MiniCard title="Source of truth" body="正式數字以 eval registry 為準；HTML 是 active dashboard，不是 raw log archive。" tone={colors.gold} />
    </div>
  </PageShell>
);

const RetiredPath: Page = () => (
  <PageShell label="Retired path">
    <Heading maxWidth={1540}>已淘汰路線只保留一張備忘</Heading>
    <Note width={1400}>
      這張 slide 的目的只是避免未來 agent 把舊方向誤認成 current mainline；不保留舊 loss 表、實驗細節或歷史數字。
    </Note>
    <div className="fade-3" style={{ marginTop: 46, display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 26, width: 1480 }}>
      <Card title="Retired from HTML dashboard" body="GRU recurrent fast head、learned gate、contact auxiliary head、Stage B / preserve-style objective。" tone={colors.rose} />
      <Card title="Current replacement" body="FWR feed-forward residual head；v1 previous/current，v2 full-chunk context；simple alpha-scaled residual merge。" tone={colors.accent} />
      <Card title="Where history lives" body="Engineering archive can stay in docs/training.md and older notes; paper-facing HTML keeps only active decisions and planned experiments." tone={colors.gold} />
      <Card title="Reactivation rule" body="Only bring this path back into HTML if it becomes an active experiment again and receives a registry-backed eval plan." tone={colors.blue} />
    </div>
  </PageShell>
);

type SuccessDatum = {
  k: number;
  hfrvla: number;
  baseline: number;
  hfrvlaText: string;
  baselineText: string;
};

const actionStepComparison: SuccessDatum[] = [
  { k: 1, hfrvla: 75, baseline: 79, hfrvlaText: '75/100', baselineText: '79/100' },
  { k: 2, hfrvla: 79, baseline: 72, hfrvlaText: '79/100', baselineText: '72/100' },
  { k: 4, hfrvla: 80, baseline: 65, hfrvlaText: '80/100', baselineText: '65/100' },
  { k: 8, hfrvla: 71, baseline: 60, hfrvlaText: '71/100', baselineText: '60/100' },
  { k: 16, hfrvla: 75, baseline: 57, hfrvlaText: '75/100', baselineText: '57/100' },
  { k: 32, hfrvla: 59, baseline: 50, hfrvlaText: '59/100', baselineText: '50/100' },
  { k: 50, hfrvla: 53, baseline: 40, hfrvlaText: '53/100', baselineText: '40/100' },
];

const matchedChunkComparison: SuccessDatum[] = [
  { k: 1, hfrvla: 73, baseline: 77, hfrvlaText: '73/100', baselineText: '77/100' },
  { k: 2, hfrvla: 71, baseline: 77, hfrvlaText: '71/100', baselineText: '77/100' },
  { k: 4, hfrvla: 73, baseline: 65, hfrvlaText: '73/100', baselineText: '65/100' },
  { k: 8, hfrvla: 65, baseline: 62, hfrvlaText: '65/100', baselineText: '62/100' },
  { k: 16, hfrvla: 60, baseline: 54, hfrvlaText: '60/100', baselineText: '54/100' },
  { k: 32, hfrvla: 60, baseline: 51, hfrvlaText: '60/100', baselineText: '51/100' },
  { k: 50, hfrvla: 53, baseline: 40, hfrvlaText: '53/100', baselineText: '40/100' },
];

type AlphaClipRow = {
  delta: string;
  effectiveCaps: string[];
  values: { successes: number; percent: number }[];
};

const alphaClipColumns = ['0.25', '0.50', '0.75', '1.00'];

const alphaClipRows: AlphaClipRow[] = [
  { delta: '0.05', effectiveCaps: ['0.013', '0.025', '0.038', '0.050'], values: [{ successes: 23, percent: 46 }, { successes: 27, percent: 54 }, { successes: 24, percent: 48 }, { successes: 25, percent: 50 }] },
  { delta: '0.10', effectiveCaps: ['0.025', '0.050', '0.075', '0.100'], values: [{ successes: 23, percent: 46 }, { successes: 23, percent: 46 }, { successes: 24, percent: 48 }, { successes: 28, percent: 56 }] },
  { delta: '0.15', effectiveCaps: ['0.038', '0.075', '0.113', '0.150'], values: [{ successes: 24, percent: 48 }, { successes: 22, percent: 44 }, { successes: 23, percent: 46 }, { successes: 24, percent: 48 }] },
  { delta: '0.18', effectiveCaps: ['0.045', '0.090', '0.135', '0.180'], values: [{ successes: 24, percent: 48 }, { successes: 23, percent: 46 }, { successes: 23, percent: 46 }, { successes: 20, percent: 40 }] },
  { delta: '0.20', effectiveCaps: ['0.050', '0.100', '0.150', '0.200'], values: [{ successes: 27, percent: 54 }, { successes: 28, percent: 56 }, { successes: 26, percent: 52 }, { successes: 23, percent: 46 }] },
  { delta: '0.22', effectiveCaps: ['0.055', '0.110', '0.165', '0.220'], values: [{ successes: 25, percent: 50 }, { successes: 26, percent: 52 }, { successes: 26, percent: 52 }, { successes: 20, percent: 40 }] },
  { delta: '0.25', effectiveCaps: ['0.063', '0.125', '0.188', '0.250'], values: [{ successes: 21, percent: 42 }, { successes: 25, percent: 50 }, { successes: 20, percent: 40 }, { successes: 21, percent: 42 }] },
  { delta: '0.30', effectiveCaps: ['0.075', '0.150', '0.225', '0.300'], values: [{ successes: 26, percent: 52 }, { successes: 22, percent: 44 }, { successes: 18, percent: 36 }, { successes: 19, percent: 38 }] },
  { delta: '999', effectiveCaps: ['none', 'none', 'none', 'none'], values: [{ successes: 22, percent: 44 }, { successes: 8, percent: 16 }, { successes: 2, percent: 4 }, { successes: 1, percent: 2 }] },
];

const signedPts = (value: number) => `${value >= 0 ? '+' : ''}${value} pts`;

const alphaClipHeatTone = (percent: number) => {
  if (percent >= 56) return { fill: 'rgba(50, 211, 153, 0.43)', text: colors.accent, stroke: colors.accent };
  if (percent >= 52) return { fill: 'rgba(247, 198, 106, 0.31)', text: colors.gold, stroke: colors.gold };
  if (percent <= 16) return { fill: 'rgba(255, 124, 154, 0.40)', text: colors.rose, stroke: colors.rose };
  if (percent <= 40) return { fill: 'rgba(255, 124, 154, 0.23)', text: colors.rose, stroke: colors.lineStrong };
  return { fill: 'rgba(120, 232, 190, 0.08)', text: colors.text, stroke: colors.line };
};

const AlphaClipHeatmap = () => {
  const width = 920;
  const height = 510;
  const left = 90;
  const top = 88;
  const cellW = 198;
  const cellH = 38;
  const gap = 4;
  const gridW = alphaClipColumns.length * cellW + (alphaClipColumns.length - 1) * gap;
  const gridH = alphaClipRows.length * cellH + (alphaClipRows.length - 1) * gap;

  return (
    <div style={{ background: colors.panel, border: `1px solid ${colors.line}`, borderRadius: 10, padding: '18px 20px' }}>
      <svg width={width} height={height} viewBox={`0 0 ${width} ${height}`} role="img" aria-label="alpha clip heatmap">
        <text x={left + gridW / 2} y={32} textAnchor="middle" fill={colors.gold} fontFamily={font.mono} fontSize="24" fontWeight="900">
          alpha
        </text>
        <text x={26} y={top + gridH / 2} textAnchor="middle" fill={colors.gold} fontFamily={font.mono} fontSize="22" fontWeight="900" transform={`rotate(-90 26 ${top + gridH / 2})`}>
          clip / delta_max
        </text>
        <text x={left} y={height - 22} fill={colors.muted} fontFamily={font.mono} fontSize="15">
          color = success rate; text = successes / 50; cap = alpha * clip
        </text>
        {alphaClipColumns.map((alpha, index) => {
          const x = left + index * (cellW + gap);
          return (
            <text key={alpha} x={x + cellW / 2} y={64} textAnchor="middle" fill={colors.gold} fontFamily={font.mono} fontSize="19" fontWeight="900">
              {alpha}
            </text>
          );
        })}
        {alphaClipRows.map((row, rowIndex) => {
          const y = top + rowIndex * (cellH + gap);
          return (
            <g key={`row-${row.delta}`}>
              <text x={left - 18} y={y + cellH / 2 + 7} textAnchor="end" fill={row.delta === '999' ? colors.rose : colors.gold} fontFamily={font.mono} fontSize="18" fontWeight="900">
                {row.delta}
              </text>
              {row.values.map((cell, colIndex) => {
                const x = left + colIndex * (cellW + gap);
                const tone = alphaClipHeatTone(cell.percent);
                const isBest = cell.percent === 56;
                return (
                  <g key={`${row.delta}-${alphaClipColumns[colIndex]}`}>
                    <rect
                      x={x}
                      y={y}
                      width={cellW}
                      height={cellH}
                      rx="4"
                      fill={tone.fill}
                      stroke={isBest ? colors.accent : tone.stroke}
                      strokeWidth={isBest ? 3 : 1}
                    />
                    <text x={x + cellW / 2} y={y + 17} textAnchor="middle" fill={tone.text} fontFamily={font.mono} fontSize="16" fontWeight="900">
                      {cell.successes}/50 = {cell.percent}%
                    </text>
                    <text x={x + cellW / 2} y={y + 32} textAnchor="middle" fill={colors.muted} fontFamily={font.mono} fontSize="12">
                      cap {row.effectiveCaps[colIndex]}
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

const SuccessLineChart = ({
  title,
  data,
}: {
  title: string;
  data: SuccessDatum[];
}) => {
  const width = 900;
  const height = 390;
  const left = 78;
  const right = 34;
  const top = 46;
  const bottom = 72;
  const yMin = 40;
  const yMax = 90;
  const innerW = width - left - right;
  const innerH = height - top - bottom;
  const x = (index: number) => left + (innerW * index) / Math.max(1, data.length - 1);
  const y = (value: number) => top + innerH - ((value - yMin) / (yMax - yMin)) * innerH;
  const points = (key: 'hfrvla' | 'baseline') =>
    data.map((row, index) => `${x(index)},${y(row[key])}`).join(' ');

  return (
    <div style={{ background: colors.panel, border: `1px solid ${colors.line}`, borderRadius: 10, padding: '22px 24px' }}>
      <div style={{ fontSize: 25, fontWeight: 850, color: colors.gold, marginBottom: 8 }}>{title}</div>
      <svg width={width} height={height} viewBox={`0 0 ${width} ${height}`} role="img" aria-label={title}>
        {[40, 50, 60, 70, 80, 90].map((tick) => (
          <g key={tick}>
            <line x1={left} x2={width - right} y1={y(tick)} y2={y(tick)} stroke={colors.line} />
            <text x={left - 16} y={y(tick) + 7} textAnchor="end" fill={colors.muted} fontFamily={font.mono} fontSize="20">
              {tick}%
            </text>
          </g>
        ))}
        <line x1={left} x2={width - right} y1={height - bottom} y2={height - bottom} stroke={colors.lineStrong} />
        <line x1={left} x2={left} y1={top} y2={height - bottom} stroke={colors.lineStrong} />
        <polyline points={points('baseline')} fill="none" stroke={colors.gold} strokeWidth="5" strokeLinejoin="round" strokeLinecap="round" />
        <polyline points={points('hfrvla')} fill="none" stroke={colors.accent} strokeWidth="6" strokeLinejoin="round" strokeLinecap="round" />
        {data.map((row, index) => (
          <g key={`pt-${row.k}`}>
            <circle cx={x(index)} cy={y(row.baseline)} r="8" fill={colors.gold} />
            <circle cx={x(index)} cy={y(row.hfrvla)} r="9" fill={colors.accent} />
            <text x={x(index)} y={height - 34} textAnchor="middle" fill={colors.text} fontFamily={font.mono} fontSize="22">
              {row.k}
            </text>
          </g>
        ))}
        <text x={left} y={30} fill={colors.accent} fontFamily={font.mono} fontSize="21">
          HFRVLA
        </text>
        <text x={left + 120} y={30} fill={colors.gold} fontFamily={font.mono} fontSize="21">
          SmolVLA baseline
        </text>
      </svg>
    </div>
  );
};

const SuccessComparisonTable = ({
  rows,
  firstColumn,
}: {
  rows: SuccessDatum[];
  firstColumn: string;
}) => (
  <div style={{ background: colors.panel, border: `1px solid ${colors.line}`, borderRadius: 10, overflow: 'hidden' }}>
    <table style={{ width: '100%', borderCollapse: 'collapse', fontFamily: font.mono, fontSize: 21 }}>
      <thead>
        <tr style={{ background: 'rgba(120, 232, 190, 0.10)', color: colors.gold }}>
          <th style={{ textAlign: 'left', padding: '16px 18px' }}>{firstColumn}</th>
          <th style={{ textAlign: 'right', padding: '16px 18px' }}>HFRVLA</th>
          <th style={{ textAlign: 'right', padding: '16px 18px' }}>SmolVLA</th>
          <th style={{ textAlign: 'right', padding: '16px 18px' }}>delta</th>
        </tr>
      </thead>
      <tbody>
        {rows.map((row) => {
          const delta = row.hfrvla - row.baseline;
          return (
            <tr key={row.k} style={{ borderTop: `1px solid ${colors.line}` }}>
              <td style={{ padding: '15px 18px', color: colors.text }}>{row.k}</td>
              <td style={{ padding: '15px 18px', textAlign: 'right', color: colors.accent }}>{row.hfrvlaText} = {row.hfrvla.toFixed(1)}%</td>
              <td style={{ padding: '15px 18px', textAlign: 'right', color: colors.gold }}>{row.baselineText} = {row.baseline.toFixed(1)}%</td>
              <td style={{ padding: '15px 18px', textAlign: 'right', color: delta >= 0 ? colors.accent : colors.rose }}>{signedPts(delta)}</td>
            </tr>
          );
        })}
      </tbody>
    </table>
  </div>
);

const EvidenceSnapshot: Page = () => (
  <PageShell label="Evidence snapshot">
    <Heading maxWidth={1580}>穩定 evidence：保留 baseline 比較，不提前寫成 paper claim</Heading>
    <Note width={1500}>
      這頁只摘要 registry-backed LIBERO-Spatial rows；完整成功率比較在後面兩張折線圖與表格。Paper 正文先保留方法大架構，等全部實驗完成後再寫細節主張。
    </Note>
    <div className="fade-3" style={{ marginTop: 34, display: 'grid', gridTemplateColumns: '1.12fr .88fr', gap: 24, alignItems: 'start' }}>
      <CodeBlock size={17} lineHeight={1.18} padding="24px 28px">{`seed=42, LIBERO-Spatial, 100 episodes, batch=3
sweep          policy                    plan exec   success
plan50 exec    HFRVLA gen alpha=.5 d=.2  50   4      80/100
plan50 exec    SmolVLA                   50   4      65/100
plan50 exec    HFRVLA gen alpha=.5 d=.2  50   16     75/100
plan50 exec    SmolVLA                   50   16     57/100
matched chunk  HFRVLA gen alpha=.5 d=.2  8    8      65/100
matched chunk  SmolVLA                   8    8      62/100
matched chunk  HFRVLA gen alpha=.5 d=.2  50   50     53/100
matched chunk  SmolVLA                   50   50     40/100`}</CodeBlock>
      <div style={{ display: 'grid', gap: 18 }}>
        <Card title="Dashboard read" body="目前只說明比較趨勢，不把任何 success gap 寫成最終 paper claim。" tone={colors.accent} />
        <Card title="Main risk" body="plan=exec=50 時兩邊都掉很多；這仍是 calibration / target-deployment mismatch 的診斷重點。" tone={colors.rose} />
        <Card title="Source of truth" body="正式表格從 experiments/eval_registry/eval_results_master.csv 產生，不從 sandbox log 手抄。" tone={colors.gold} />
      </div>
    </div>
  </PageShell>
);

const ActionStepBaselineComparison: Page = () => (
  <PageShell label="Success comparison">
    <Heading maxWidth={1580}>固定 plan=50：success rate vs SmolVLA baseline</Heading>
    <Note width={1500}>
      LIBERO-Spatial，seed=42，100 episodes/setting，eval batch size=3。X 軸是 execution/replan interval K；slow planner 固定產生 50-step chunk。
    </Note>
    <div className="fade-3" style={{ marginTop: 26, display: 'grid', gridTemplateColumns: '920px 1fr', gap: 24, alignItems: 'start' }}>
      <SuccessLineChart title="Action-step sweep: plan=50, exec/replan=K" data={actionStepComparison} />
      <SuccessComparisonTable rows={actionStepComparison} firstColumn="exec K" />
    </div>
  </PageShell>
);

const MatchedChunkBaselineComparison: Page = () => (
  <PageShell label="Success comparison">
    <Heading maxWidth={1580}>Matched chunk：success rate vs SmolVLA baseline</Heading>
    <Note width={1500}>
      LIBERO-Spatial，seed=42，100 episodes/setting，eval batch size=3。X 軸是 matched K，其中 planning=execution=replan=K。
    </Note>
    <div className="fade-3" style={{ marginTop: 26, display: 'grid', gridTemplateColumns: '920px 1fr', gap: 24, alignItems: 'start' }}>
      <SuccessLineChart title="Matched chunk sweep: plan=exec=replan=K" data={matchedChunkComparison} />
      <SuccessComparisonTable rows={matchedChunkComparison} firstColumn="matched K" />
    </div>
  </PageShell>
);

const ActiveN50Calibration: Page = () => (
  <PageShell label="n=50 calibration result">
    <Heading maxWidth={1580}>n_action_steps=50：alpha/clip sweep 結果</Heading>
    <Note width={1500}>
      LIBERO-Spatial，seed=42，50 episodes/setting，planning=execution=replan=50，eval batch size=3。Baseline reference：matched SmolVLA K=50 是 40/100 = 40% in the 10x10 spatial run。
    </Note>
    <div className="fade-3" style={{ marginTop: 24, display: 'grid', gridTemplateColumns: '1.15fr .85fr', gap: 22, alignItems: 'start' }}>
      <div>
        <AlphaClipHeatmap />
        <div style={{ marginTop: 14, fontFamily: font.mono, fontSize: 15, color: colors.muted }}>
          Source: experiments/eval_registry/eval_results_master.csv; raw CSV:
          outputs/eval_hfrvla_fwr_chunk_generated_seq2_b1024p3_50k_n50_alpha_clip_spatial_50eps_b3/results.csv
        </div>
      </div>
      <div style={{ display: 'grid', gap: 15 }}>
        <MiniCard title="Best settings" body="最高是 28/50 = 56%：alpha=.5 clip=.2，以及 alpha=1.0 clip=.1；兩者 effective cap 都是 0.1。" tone={colors.accent} />
        <MiniCard title="Useful conclusion" body="最佳 56% 比 SmolVLA K=50 baseline 高 16 pts；但長 chunk 不是越大 residual 越好，較大 cap 會 over-correct。" tone={colors.gold} />
        <MiniCard title="No-limit result" body="delta=999 在 alpha=.5/.75/1.0 時掉到 16%、4%、2%；clip 是必要 deployment constraint。" tone={colors.rose} />
      </div>
    </div>
  </PageShell>
);

const AsyncTimestepPlannerDelayProtocol: Page = () => (
  <PageShell label="Async-timestep protocol">
    <Heading maxWidth={1580}>新的 async_timestep：server 只跑 slow planner，client 每步做 wrist residual</Heading>
    <Note width={1500}>
      Phase 1 用 deterministic control timestep 模擬 latency；不修改 LeRobot source。Slow planner observe 的時間、chunk ready 的時間、以及從 chunk 第幾個 index 開始執行都寫入 debug stats。
    </Note>
    <div className="fade-3" style={{ marginTop: 24, display: 'grid', gridTemplateColumns: '1.08fr .92fr', gap: 24, alignItems: 'start' }}>
      <div style={{ background: colors.panel, border: `1px solid ${colors.line}`, borderRadius: 10, padding: '22px 26px' }}>
        <CodeBlock size={20} lineHeight={1.18} padding="22px 24px">{`async_timestep semantics
  request step t:
    observe o_t
    slow planner starts A_t = SmolVLA(o_t)

  ready step t+d:
    A_t arrives
    active queue is replaced immediately
    executor starts from A_t[d], not A_t[0]

  every control step:
    current wrist/state -> delta_a
    a_final = a_base + alpha * clip(delta_a)

main phase-1 grid
  N = async_request_interval_steps = 8
  d = planner_delay_steps = 0,1,2,3,4
  constraint: N + d <= execution = 16`}</CodeBlock>
      </div>
      <div style={{ display: 'grid', gap: 14 }}>
        <MiniCard title="Controlled point" body="新的 chunk 是在 ready step t+d 取代 queue；開始計算的 observation 是 request step t 的 o_t。" tone={colors.gold} />
        <MiniCard title="Controlled index" body="delay=d 時第一個執行的 base action 是 A_t[d]；k_idx_norm 使用 d,d+1,... 的 absolute chunk index。" tone={colors.accent} />
        <MiniCard title="Split-system meaning" body="server 只代表 frozen slow planner；client 保留 HFRVLA fast residual 和 safety merge，不走 LeRobot RTC。" tone={colors.blue} />
        <MiniCard title="Debug fields" body="async_request_count、async_activation_count、chunk_start_index、dropped_old_queue_steps、event trace 會進 CSV/registry。" tone={colors.violet} />
      </div>
    </div>
  </PageShell>
);

const AsyncTimestepPlannerDelayResult: Page = () => (
  <PageShell label="Async-timestep result">
    <Heading maxWidth={1580}>Async-timestep result：wrist residual 阻止 disable-fast 的 delay drop</Heading>
    <Note width={1500}>
      LIBERO-Spatial，100 episodes/row，N=8，plan=50，exec=16，delay=0..4。這是正式 planner-delay protocol。
    </Note>
    <div className="fade-3" style={{ marginTop: 22, display: 'grid', gridTemplateColumns: '1.05fr .95fr', gap: 24, alignItems: 'start' }}>
      <div style={{ background: '#ffffff', border: `1px solid ${colors.lineStrong}`, borderRadius: 10, padding: '18px 20px' }}>
        <img
          src={asyncPlannerDelaySummaryPlot}
          alt="Async-timestep planner-delay summary for HFRVLA and disable-fast"
          style={{ width: '100%', height: 565, objectFit: 'contain', display: 'block' }}
        />
      </div>
      <div style={{ display: 'grid', gap: 13 }}>
        <MiniCard title="HFRVLA" body="Success = 64、68、72、68、66%。相對 d=0 是 +0、+4、+8、+4、+2 pp；沒有出現 0..4 delay degradation。" tone={colors.accent} />
        <MiniCard title="Disable-fast" body="Success = 68、64、62、61、56%。相對 d=0 到 d=4 是 -12 pp，顯示 slow chunk latency 對 base-only execution 有明顯傷害。" tone={colors.rose} />
        <MiniCard title="Margin" body="HFRVLA - disable-fast = -4、+4、+10、+7、+10 pp；d=1..4 平均 margin 是 +7.75 pp。" tone={colors.gold} />
        <MiniCard title="Timing check" body="async_chunk_start_index_mean 精準等於 d=0..4；代表實際執行確實從 A_t[d] 開始。" tone={colors.blue} />
        <CodeBlock size={13} lineHeight={1.08} padding="13px 15px">{`source artifacts
outputs/async_timestep_planner_delay_eval_sweep/results.csv
outputs/async_timestep_planner_delay_eval_sweep/analysis.md
paper/notes/async_timestep_planner_delay_eval_results.md`}</CodeBlock>
      </div>
    </div>
  </PageShell>
);

const EvalRegistry: Page = () => (
  <PageShell label="Experiment registry">
    <Heading maxWidth={1520}>所有正式 eval 要進 master CSV</Heading>
    <Note width={1380}>
      `outputs/` 是 raw runs；長期論文比較以 repo-tracked registry 為準。新增實驗時只加 manifest row，再重建 master。
    </Note>
    <div className="fade-3" style={{ marginTop: 42, display: 'grid', gridTemplateColumns: '.95fr 1.05fr', gap: 28, alignItems: 'start' }}>
      <CodeBlock size={21} lineHeight={1.24} padding="24px 28px">{`experiments/eval_registry/
  sources.csv
    source_csv + sweep_id + policy
    metadata_profile + short tags / notes

  eval_results_master.csv
    compact long/tidy table
    267 data rows, 52 columns
    dashboard uses only stable evidence

scripts/build_eval_results_master.py
  rebuilds master from manifest
  expands sweep/profile metadata
  keeps paper-facing CSV compact`}</CodeBlock>
      <div style={{ display: 'grid', gap: 18 }}>
        <MiniCard title="Current sources" body="18 manifest rows；master check 目前為 267 data rows，只保留 async planner-delay sweep。" tone={colors.accent} />
        <MiniCard title="Do not hand-edit" body="master CSV 是 build artifact；未來不同 training params 都透過 sources.csv 記錄。" tone={colors.rose} />
        <MiniCard title="Paper use" body="HTML 不再保留舊 raw result pages；paper table 從 master CSV filter/groupby。" tone={colors.gold} />
      </div>
    </div>
  </PageShell>
);

const ExperimentMatrix: Page = () => (
  <PageShell label="Experiment matrix">
    <Heading maxWidth={1560}>下一輪未決實驗矩陣</Heading>
    <Note width={1460}>
      Matrix 只記實驗設計與狀態，不放 decision rule。跑完後把正式結果納入 registry，再回來更新 dashboard。
    </Note>
    <div className="fade-3" style={{ marginTop: 32, width: 1640 }}>
      <CodeBlock size={14} lineHeight={1.1} padding="22px 24px">{`experiment_id                  variable                 values / plan                         fixed_controls                                      status
n50_alpha_clip_spatial_50eps   alpha x delta_max        best 56% at effective cap=.1        plan=exec=replan=50; spatial 50eps; batch=3        completed
async_timestep_delay_phase1    async planner delay      N=8; d=0..4; plan=50 exec=16        hfrvla vs disable-fast; spatial 100eps           completed
target_alignment_ablation      residual target          raw residual vs final-action vs age   same FWR architecture and deploy merge            planned
time_age_features              timing features          k_norm, sin/cos, age_norm, latency    same checkpoint/data pipeline                    planned
runtime_instrumentation        deployment logs          fast_applied_ratio, hook misses       latency, delta_norm, clip_fraction, safety hits    planned
real_robot_tabletop_validation  hardware rollout         2-3 tasks; 10-20 trials/method/task  matched task text/cameras where possible          planned
dino_wrist_visualization       visual diagnosis         patch heatmap + occlusion sensitivity diagnostic only; not a causal claim by itself    planned
latent_context_ablation        slow context             keep/remove z_goal,z_phase            wrist/state/a_base fixed                         planned
tool_frame_residual            target frame             raw 7D vs tool-frame residual          same deploy merge                                planned`}</CodeBlock>
    </div>
  </PageShell>
);

const CompactDelayCard = ({
  title,
  body,
  tone,
}: {
  title: string;
  body: string;
  tone: string;
}) => (
  <div
    style={{
      border: `1px solid ${colors.line}`,
      background: colors.panel,
      borderRadius: 10,
      padding: '13px 18px 15px',
      display: 'grid',
      gap: 7,
    }}
  >
    <div style={{ fontSize: 21, fontWeight: 850, color: tone }}>{title}</div>
    <div style={{ fontSize: 18, lineHeight: 1.24, color: colors.muted }}>{body}</div>
  </div>
);

const DelayTimingSchematic: Page = () => (
  <PageShell label="Delay timing">
    <Heading maxWidth={1560}>Planner delay 要畫成 observe / compute / execute 的時間切分</Heading>
    <Note width={1500}>
      Delay 定義為 slow VLA chunk generation latency；wrist residual 不延遲，仍在每個 control step 使用 current wrist feedback。
    </Note>
    <div className="fade-3" style={{ marginTop: 28, display: 'grid', gridTemplateColumns: '1.18fr .82fr', gap: 24, alignItems: 'start' }}>
      <div style={{ background: colors.panel, border: `1px solid ${colors.line}`, borderRadius: 10, padding: '20px 24px 24px' }}>
        <svg viewBox="0 0 1040 560" width="100%" height="520" role="img" aria-label="HFRVLA planner delay timing schematic">
          <defs>
            <marker id="arrow-green" markerWidth="10" markerHeight="10" refX="8" refY="5" orient="auto">
              <path d="M0,0 L10,5 L0,10 Z" fill={colors.accent} />
            </marker>
            <marker id="arrow-blue" markerWidth="10" markerHeight="10" refX="8" refY="5" orient="auto">
              <path d="M0,0 L10,5 L0,10 Z" fill={colors.blue} />
            </marker>
            <marker id="arrow-gold" markerWidth="10" markerHeight="10" refX="8" refY="5" orient="auto">
              <path d="M0,0 L10,5 L0,10 Z" fill={colors.gold} />
            </marker>
          </defs>

          {[170, 450, 890].map((x, i) => (
            <g key={x}>
              <line x1={x} y1={58} x2={x} y2={500} stroke={i === 1 ? colors.gold : colors.lineStrong} strokeWidth={i === 1 ? 2 : 1.4} strokeDasharray={i === 0 ? '0' : '8 8'} />
            </g>
          ))}
          <line x1={130} y1={500} x2={970} y2={500} stroke={colors.lineStrong} strokeWidth={2} markerEnd="url(#arrow-green)" />
          <text x={170} y={532} textAnchor="middle" fill={colors.text} fontFamily={font.mono} fontSize={22}>t</text>
          <text x={450} y={532} textAnchor="middle" fill={colors.gold} fontFamily={font.mono} fontSize={22}>t+d</text>
          <text x={890} y={532} textAnchor="middle" fill={colors.text} fontFamily={font.mono} fontSize={22}>t+d+K</text>

          <text x={28} y={112} fill={colors.gold} fontWeight={850} fontSize={23}>Slow VLA</text>
          <text x={28} y={140} fill={colors.muted} fontSize={17}>System-2 planner</text>
          <text x={28} y={262} fill={colors.blue} fontWeight={850} fontSize={23}>Robot</text>
          <text x={28} y={290} fill={colors.muted} fontSize={17}>execution queue</text>
          <text x={28} y={414} fill={colors.accent} fontWeight={850} fontSize={23}>Wrist reflex</text>
          <text x={28} y={442} fill={colors.muted} fontSize={17}>System-1 correction</text>

          <rect x={138} y={88} width={116} height={54} rx={9} fill="rgba(247,198,106,.16)" stroke={colors.gold} strokeWidth={2} />
          <text x={196} y={110} textAnchor="middle" fill={colors.gold} fontFamily={font.mono} fontSize={15}>sample</text>
          <text x={196} y={130} textAnchor="middle" fill={colors.text} fontFamily={font.mono} fontSize={17}>o_t</text>

          <rect x={258} y={88} width={282} height={54} rx={9} fill="rgba(119,183,255,.14)" stroke={colors.blue} strokeWidth={2} />
          <text x={399} y={110} textAnchor="middle" fill={colors.blue} fontFamily={font.mono} fontSize={15}>slow compute window</text>
          <text x={399} y={130} textAnchor="middle" fill={colors.text} fontFamily={font.mono} fontSize={17}>generate chunk A_t</text>
          <line x1={254} y1={115} x2={258} y2={115} stroke={colors.gold} strokeWidth={2} markerEnd="url(#arrow-gold)" />

          <polygon points="450,74 494,115 450,156 406,115" fill="rgba(50,211,153,.16)" stroke={colors.accent} strokeWidth={2} />
          <text x={450} y={111} textAnchor="middle" fill={colors.accent} fontFamily={font.mono} fontSize={15}>A_t</text>
          <text x={450} y={132} textAnchor="middle" fill={colors.text} fontFamily={font.mono} fontSize={14}>ready</text>

          <rect x={150} y={230} width={274} height={58} rx={9} fill="rgba(255,124,154,.12)" stroke={colors.rose} strokeWidth={2} />
          <text x={287} y={254} textAnchor="middle" fill={colors.rose} fontFamily={font.mono} fontSize={16}>previous queue</text>
          <text x={287} y={278} textAnchor="middle" fill={colors.text} fontFamily={font.mono} fontSize={16}>or hold-last fallback</text>
          <rect x={450} y={230} width={420} height={58} rx={9} fill="rgba(50,211,153,.13)" stroke={colors.accent} strokeWidth={2} />
          {['a^0', 'a^1', 'a^2', '...', 'a^{K-1}'].map((label, idx) => (
            <g key={label}>
              <rect x={470 + idx * 76} y={244} width={58} height={30} rx={6} fill="rgba(8,17,15,.58)" stroke={colors.lineStrong} />
              <text x={499 + idx * 76} y={264} textAnchor="middle" fill={colors.text} fontFamily={font.mono} fontSize={15}>{label}</text>
            </g>
          ))}
          <text x={660} y={314} textAnchor="middle" fill={colors.muted} fontFamily={font.mono} fontSize={16}>execute chunk predicted from o_t</text>

          <line x1={170} y1={346} x2={450} y2={346} stroke={colors.rose} strokeWidth={3} markerEnd="url(#arrow-blue)" />
          <text x={310} y={336} textAnchor="middle" fill={colors.rose} fontFamily={font.mono} fontSize={17}>prediction-execution offset d</text>

          {Array.from({ length: 8 }).map((_, idx) => {
            const x = 154 + idx * 108;
            return (
              <g key={idx}>
                <circle cx={x} cy={410} r={12} fill="rgba(50,211,153,.18)" stroke={colors.accent} strokeWidth={2} />
                <text x={x} y={448} textAnchor="middle" fill={colors.muted} fontFamily={font.mono} fontSize={13}>w{idx === 0 ? '_t' : `_${idx}`}</text>
                <line x1={x} y1={394} x2={x} y2={292} stroke={colors.accent} strokeWidth={1.5} strokeDasharray="5 7" markerEnd="url(#arrow-green)" />
              </g>
            );
          })}
          <rect x={360} y={388} width={304} height={48} rx={9} fill="rgba(50,211,153,.12)" stroke={colors.accent} strokeWidth={2} />
          <text x={512} y={418} textAnchor="middle" fill={colors.text} fontFamily={font.mono} fontSize={18}>a_final = a_base + alpha clip(delta a)</text>
        </svg>
      </div>
      <div style={{ display: 'grid', gap: 9 }}>
        <CompactDelayCard title="Delay semantics" body="planner_delay_steps=d：slow VLA 在 t 取樣 observation，chunk 到 t+d 才進 queue。" tone={colors.gold} />
        <CompactDelayCard title="Main sweep" body="plan=50, exec/replan=16；delay={0,1,2,4,8}；Spatial 10 eps/task。" tone={colors.accent} />
        <CompactDelayCard title="Fallback" body="主實驗 hold-last-action，另記 fallback_steps；zero-action 只做 ablation。" tone={colors.rose} />
        <CompactDelayCard title="Arms" body="SmolVLA、HFRVLA、HFRVLA disable-fast，用第三組隔離 wrist residual。" tone={colors.blue} />
        <CodeBlock size={13} lineHeight={1.08} padding="13px 16px">{`metrics
success_rate
planner_delay_steps
fallback_steps_total / mean
slow_replan_count
fast_latency_ms_mean
fast_applied_ratio
delta_norm_mean
delta_clip_fraction_mean
k_mean`}</CodeBlock>
      </div>
    </div>
  </PageShell>
);

const DelayTimingProblemDefinition: Page = () => (
  <PageShell label="Delay timing">
    <Heading maxWidth={1580}>Inference delay：A_t 在 t 生成，t+d 才執行</Heading>
    <Note width={1500}>
      slow VLA 用舊 observation 產生 action chunk；HFRVLA 測試 wrist feedback 是否能修正這個 stale chunk。
    </Note>
    <div className="fade-3" style={{ marginTop: 22, width: 1640 }}>
      <div style={{ background: colors.panel, border: `1px solid ${colors.line}`, borderRadius: 10, padding: '20px 28px 24px' }}>
        <svg viewBox="0 0 1600 560" width="100%" height="500" role="img" aria-label="HFRVLA planner delay problem definition">
          <defs>
            <marker id="delay-arrow-green" markerWidth="10" markerHeight="10" refX="8" refY="5" orient="auto">
              <path d="M0,0 L10,5 L0,10 Z" fill={colors.accent} />
            </marker>
            <marker id="delay-arrow-gold" markerWidth="10" markerHeight="10" refX="8" refY="5" orient="auto">
              <path d="M0,0 L10,5 L0,10 Z" fill={colors.gold} />
            </marker>
            <marker id="delay-arrow-rose" markerWidth="10" markerHeight="10" refX="8" refY="5" orient="auto">
              <path d="M0,0 L10,5 L0,10 Z" fill={colors.rose} />
            </marker>
          </defs>

          {[260, 760, 1320].map((x, i) => (
            <g key={x}>
              <line x1={x} y1={54} x2={x} y2={502} stroke={i === 1 ? colors.gold : colors.lineStrong} strokeWidth={i === 1 ? 2.4 : 1.4} strokeDasharray={i === 0 ? '0' : '8 8'} />
            </g>
          ))}
          <line x1={206} y1={502} x2={1460} y2={502} stroke={colors.lineStrong} strokeWidth={2} markerEnd="url(#delay-arrow-green)" />
          <text x={260} y={538} textAnchor="middle" fill={colors.text} fontFamily={font.mono} fontSize={23}>t</text>
          <text x={760} y={538} textAnchor="middle" fill={colors.gold} fontFamily={font.mono} fontSize={23}>t+d</text>
          <text x={1320} y={538} textAnchor="middle" fill={colors.text} fontFamily={font.mono} fontSize={23}>t+d+K</text>

          <text x={34} y={92} fill={colors.gold} fontWeight={850} fontSize={26}>Observation</text>
          <text x={34} y={122} fill={colors.muted} fontSize={18}>what planner sees</text>
          <text x={34} y={214} fill={colors.blue} fontWeight={850} fontSize={26}>Slow VLA</text>
          <text x={34} y={244} fill={colors.muted} fontSize={18}>System-2 planner</text>
          <text x={34} y={346} fill={colors.rose} fontWeight={850} fontSize={26}>Robot</text>
          <text x={34} y={376} fill={colors.muted} fontSize={18}>what executes</text>
          <text x={34} y={452} fill={colors.accent} fontWeight={850} fontSize={26}>Wrist reflex</text>
          <text x={34} y={482} fill={colors.muted} fontSize={18}>current feedback</text>

          <rect x={220} y={62} width={240} height={74} rx={9} fill="rgba(247,198,106,.14)" stroke={colors.gold} strokeWidth={2} />
          <text x={340} y={92} textAnchor="middle" fill={colors.gold} fontFamily={font.mono} fontSize={18}>observe</text>
          <text x={340} y={118} textAnchor="middle" fill={colors.text} fontFamily={font.mono} fontSize={22}>o_t</text>

          <rect x={742} y={62} width={260} height={74} rx={9} fill="rgba(255,124,154,.11)" stroke={colors.rose} strokeWidth={2} />
          <text x={872} y={92} textAnchor="middle" fill={colors.rose} fontFamily={font.mono} fontSize={18}>world has moved</text>
          <text x={872} y={118} textAnchor="middle" fill={colors.text} fontFamily={font.mono} fontSize={22}>state at t+d</text>
          <line x1={460} y1={99} x2={742} y2={99} stroke={colors.rose} strokeWidth={2.5} strokeDasharray="8 8" markerEnd="url(#delay-arrow-rose)" />

          <rect x={260} y={184} width={500} height={70} rx={9} fill="rgba(119,183,255,.14)" stroke={colors.blue} strokeWidth={2} />
          <text x={510} y={213} textAnchor="middle" fill={colors.blue} fontFamily={font.mono} fontSize={18}>slow compute window</text>
          <text x={510} y={238} textAnchor="middle" fill={colors.text} fontFamily={font.mono} fontSize={21}>generate A_t from o_t</text>
          <line x1={760} y1={219} x2={862} y2={219} stroke={colors.gold} strokeWidth={2.4} markerEnd="url(#delay-arrow-gold)" />
          <rect x={882} y={184} width={156} height={70} rx={9} fill="rgba(50,211,153,.14)" stroke={colors.accent} strokeWidth={2} />
          <text x={960} y={213} textAnchor="middle" fill={colors.accent} fontFamily={font.mono} fontSize={18}>A_t ready</text>
          <text x={960} y={238} textAnchor="middle" fill={colors.text} fontFamily={font.mono} fontSize={18}>enqueue</text>

          <rect x={260} y={318} width={500} height={76} rx={9} fill="rgba(255,124,154,.12)" stroke={colors.rose} strokeWidth={2} />
          <text x={510} y={348} textAnchor="middle" fill={colors.rose} fontFamily={font.mono} fontSize={19}>previous queue / hold-last</text>
          <text x={510} y={374} textAnchor="middle" fill={colors.text} fontFamily={font.mono} fontSize={18}>robot cannot use A_t yet</text>
          <rect x={760} y={318} width={560} height={76} rx={9} fill="rgba(50,211,153,.13)" stroke={colors.accent} strokeWidth={2.2} />
          {['a^0', 'a^1', 'a^2', '...', 'a^{K-1}'].map((label, idx) => (
            <g key={label}>
              <rect x={804 + idx * 96} y={338} width={72} height={34} rx={6} fill="rgba(8,17,15,.58)" stroke={colors.lineStrong} />
              <text x={840 + idx * 96} y={360} textAnchor="middle" fill={colors.text} fontFamily={font.mono} fontSize={17}>{label}</text>
            </g>
          ))}
          <text x={1040} y={302} textAnchor="middle" fill={colors.rose} fontWeight={850} fontSize={25}>A_t is stale when it starts executing</text>
          <line x1={340} y1={154} x2={760} y2={154} stroke={colors.rose} strokeWidth={3} strokeDasharray="10 9" markerEnd="url(#delay-arrow-rose)" />
          <text x={550} y={178} textAnchor="middle" fill={colors.rose} fontFamily={font.mono} fontSize={20}>staleness = d control steps</text>

          {Array.from({ length: 7 }).map((_, idx) => {
            const x = 776 + idx * 78;
            return (
              <g key={idx}>
                <circle cx={x} cy={418} r={11} fill="rgba(50,211,153,.18)" stroke={colors.accent} strokeWidth={2} />
                <line x1={x} y1={406} x2={x} y2={394} stroke={colors.accent} strokeWidth={1.6} strokeDasharray="5 7" markerEnd="url(#delay-arrow-green)" />
              </g>
            );
          })}
          <rect x={260} y={430} width={430} height={56} rx={9} fill="rgba(50,211,153,.12)" stroke={colors.accent} strokeWidth={2} />
          <text x={475} y={464} textAnchor="middle" fill={colors.text} fontFamily={font.mono} fontSize={20}>current wrist {'->'} delta a every step</text>
          <line x1={690} y1={458} x2={768} y2={458} stroke={colors.accent} strokeWidth={2.5} markerEnd="url(#delay-arrow-green)" />
          <rect x={790} y={430} width={500} height={56} rx={9} fill="rgba(50,211,153,.12)" stroke={colors.accent} strokeWidth={2} />
          <text x={1040} y={464} textAnchor="middle" fill={colors.text} fontFamily={font.mono} fontSize={20}>a_final = a_base + alpha clip(delta a)</text>
        </svg>
      </div>
    </div>
  </PageShell>
);

const FigureConceptPreviews: Page = () => (
  <PageShell label="Figure concepts">
    <Heading maxWidth={1560}>可審核的概念圖：只當 placeholder，不當實驗截圖</Heading>
    <Note width={1460}>
      這三張由 imagegen 產生，用來先檢查 paper visual direction。正式稿建議重畫成 vector method diagram，或替換成真實 robot rollout / wrist-frame visualization。
    </Note>
    <div className="fade-3" style={{ marginTop: 30, display: 'grid', gridTemplateColumns: '1fr 1fr 1fr', gap: 18 }}>
      {[
        {
          title: 'System-1 wrist reflex',
          src: systemReflexPreview,
          body: 'Slow frozen planner + local wrist residual；適合當 architecture schematic 的構圖參考。',
          tone: colors.accent,
        },
        {
          title: 'Wrist feedback rollout',
          src: wristFeedbackPreview,
          body: '展示 wrist inset 與 correction arrow；適合 real-robot validation slide 的構圖參考。',
          tone: colors.gold,
        },
        {
          title: 'DINO patch heatmap',
          src: dinoPatchPreview,
          body: '只作 diagnostic visualization；若要寫 causal claim，需要 occlusion/sensitivity ablation。',
          tone: colors.blue,
        },
      ].map((item) => (
        <div key={item.title} style={{ background: colors.panel, border: `1px solid ${colors.line}`, borderRadius: 10, overflow: 'hidden' }}>
          <div style={{ height: 262, background: '#101817' }}>
            <img src={item.src} alt={item.title} style={{ width: '100%', height: '100%', objectFit: 'cover', display: 'block' }} />
          </div>
          <div style={{ padding: '18px 20px 22px' }}>
            <div style={{ fontSize: 23, fontWeight: 850, color: item.tone }}>{item.title}</div>
            <p style={{ margin: '10px 0 0', fontSize: 19, lineHeight: 1.35, color: colors.muted }}>{item.body}</p>
          </div>
        </div>
      ))}
    </div>
    <div className="fade-4" style={{ marginTop: 26, width: 1500 }}>
      <CodeBlock size={18} lineHeight={1.2} padding="22px 28px">{`asset paths
docs/assets/hfrvla-paper/system1-system2-wrist-reflex-preview.png
docs/assets/hfrvla-paper/wrist-feedback-rollout-preview.png
docs/assets/hfrvla-paper/dino-patch-heatmap-preview.png`}</CodeBlock>
    </div>
  </PageShell>
);

const PaperWritingAnchors: Page = () => (
  <PageShell label="Paper anchors">
    <Heading maxWidth={1560}>之後寫論文時要抓住的資訊</Heading>
    <div className="fade-2" style={{ marginTop: 48, display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 22, width: 1580 }}>
      <Card title="Core claim" body="一個小型 wrist-feedback residual module 可以在 frozen SmolVLA action chunks 上做局部反射式 correction，不需要重訓 slow planner。" tone={colors.accent} />
      <Card title="System 1 / System 2" body="把 slow SmolVLA 寫成 System-2-like planner，把 wrist residual 寫成局部 System-1-like reflex；避免宣稱 full dual-system VLA。" tone={colors.gold} />
      <Card title="Comparison framing" body="A2C2 是 closest related work，不寫直接勝負比較；real robot rollout 是額外 hardware validation，不是核心 novelty。" tone={colors.blue} />
      <Card title="Baseline discipline" body="SmolVLA baseline 必須用相同 planning/execution override 比較；default n=1 只能作上限參考，不是 matched baseline。" tone={colors.blue} />
      <Card title="Evidence source" body="數字引用 `eval_results_master.csv`；raw outputs 和 eval logs 只作追溯，不作手抄表格來源。" tone={colors.rose} />
      <Card title="Incomplete experiments" body="Inference delay、chunk-age target、runtime instrumentation、real robot rollout 與 DINO wrist visualization 是 paper 完整度的主要缺口。" tone={colors.rose} />
    </div>
    <div className="fade-3" style={{ marginTop: 34, width: 1500 }}>
      <CodeBlock size={22} lineHeight={1.25} padding="24px 30px">{`write-up checklist
1. State the two chunk axes explicitly:
   planning_chunk_size vs execution_chunk_size / replan_interval_steps.
2. Frame wrist feedback as a local reflex, not as a full wrist-only policy.
3. Add inference-delay stress before strong reactivity claims.
4. Add real-robot rollouts before claiming hardware validation.
5. Record each new run in experiments/eval_registry/sources.csv.`}</CodeBlock>
    </div>
  </PageShell>
);

const Knobs: Page = () => (
  <PageShell>
    <Heading>調參先看 throughput 與 sample budget</Heading>
    <div className="fade-2" style={{ marginTop: 56, display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 28, width: 1340 }}>
      <Card title="SEQ_LEN" body="training-time window；FWR-v1 用 seq_len=2 的 previous/current pair；FWR-v2 另讀 full chunk。" tone={colors.violet} />
      <Card title="Batch size" body="不要只看每 step 秒數；要看每秒處理多少 samples。" tone={colors.blue} />
    </div>
    <div className="fade-3" style={{ marginTop: 50, fontFamily: font.mono, fontSize: 52, color: colors.gold, borderLeft: `10px solid ${colors.accent}`, background: colors.panel, padding: '34px 44px', borderRadius: 10, width: 1220 }}>
      samples/sec ~= BATCH_SIZE / (data_s + updt_s)
    </div>
    <p className="fade-4" style={{ margin: '48px 0 0', width: 1180, fontSize: 34, lineHeight: 1.45, color: colors.muted }}>
      Batch 翻倍若要同等 sample budget，STEPS 與 curriculum boundaries 也要大約減半。
    </p>
  </PageShell>
);

const Automation: Page = () => (
  <PageShell label="Codex hook + open-slide">
    <Heading>hook 現在負責跑 open-slide build，而不是手寫 HTML</Heading>
    <div className="fade-2" style={{ marginTop: 58, display: 'grid', gridTemplateColumns: '1fr 1fr 1fr', gap: 24 }}>
      <Card title="Source" body="React deck 位於 docs/presentations/hfrvla-training-open-slide/slides/hfrvla-training。" tone={colors.blue} />
      <Card title="Build" body="generator 呼叫 npm run build -- --out-dir，再讀取 open-slide 產物。" tone={colors.accent} />
      <Card title="Bundle" body="CSS / JS / font assets 內嵌成 docs/training_presentation.html 單檔。" tone={colors.gold} />
    </div>
    <div className="fade-3" style={{ marginTop: 52, display: 'grid', gridTemplateColumns: '.95fr 1.05fr', gap: 36, alignItems: 'stretch' }}>
      <CodeBlock size={25}>{`.agents/plugins/plugins/hfrvla-training-docs-hook/hooks.json
  PostToolUse -> scripts/sync_training_presentation.sh
  changed docs/training.md or open-slide source
  -> scripts/generate_training_presentation.py`}</CodeBlock>
      <div style={{ background: colors.panel, border: `1px solid ${colors.line}`, borderRadius: 10, padding: '36px 40px' }}>
        <div style={{ fontSize: 31, fontWeight: 850, color: colors.gold }}>Why this matters</div>
        <p style={{ margin: '22px 0 0', fontSize: 30, lineHeight: 1.45, color: colors.muted }}>
          之後改 deck 只要改 open-slide React page；hook 會重建 static HTML，讓簡報來源維持在同一套 open-slide workflow。
        </p>
      </div>
    </div>
  </PageShell>
);

const Pitfalls: Page = () => (
  <PageShell>
    <Heading>近期踩過的坑已寫進流程</Heading>
    <div className="fade-2" style={{ marginTop: 58, display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 28, width: 1390 }}>
      <Card title="Plugin reinstall" body="policy plugin source 改完後必須重新 editable install。" tone={colors.rose} />
      <Card title="Sandbox CUDA" body="default sandbox 看不到 CUDA；GPU smoke/eval 要在外層執行。" tone={colors.gold} />
      <Card title="Eval device" body="若 package 時 CUDA 不可見，formal eval 要加 --policy.device=cuda。" tone={colors.blue} />
      <Card title="dtype bridge" body="float16 cache 餵進 float32 Linear 前要 cast 到 module dtype。" />
    </div>
  </PageShell>
);

const Checklist: Page = () => (
  <PageShell>
    <Heading>操作檢查表</Heading>
    <div className="fade-2" style={{ marginTop: 62, display: 'grid', gap: 32, width: 1280 }}>
      <Step index="01" title="確認 slow planner contract" body="merged dataset 必須對應同一個 SmolVLA checkpoint。" />
      <Step index="02" title="重建或重用 fast-cache" body="fast-cache 不帶 seq_len；確認 task_index、dtype 與 training-time window 設定。" />
      <Step index="03" title="先 smoke，再正式 run" body="500-1000 step training + package + partial eval 通過後才開長訓練。" />
      <Step index="04" title="用數據決策" body="用 data_s、updt_s、grad_norm、delta_norm、clip_fraction 和 closed-loop eval 判斷下一步。" />
    </div>
  </PageShell>
);

export const meta: SlideMeta = {
  title: 'HFRVLA 訓練流程',
  createdAt: '2026-05-19T12:43:06.981Z',
};

export default [
  Cover,
  Contract,
  Architecture,
  TrainingBatchIO,
  FastModuleOutputs,
  LeRobotHardwareDispatch,
  LiveCorrectionTimeline,
  LearningObjective,
  Dataset,
  Pipeline,
  FastCache,
  TrainEntry,
  SmokeProtocol,
  DecisionLedger,
  RetiredPath,
  EvidenceSnapshot,
  ActionStepBaselineComparison,
  MatchedChunkBaselineComparison,
  ActiveN50Calibration,
  ExperimentMatrix,
  AsyncTimestepPlannerDelayProtocol,
  AsyncTimestepPlannerDelayResult,
  DelayTimingProblemDefinition,
  EvalRegistry,
  FigureConceptPreviews,
  PaperWritingAnchors,
  Knobs,
  Automation,
  Pitfalls,
  Checklist,
] satisfies Page[];
