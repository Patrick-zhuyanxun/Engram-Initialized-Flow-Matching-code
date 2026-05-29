import type { DesignSystem, Page, SlideMeta } from '@open-slide/core';

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
      <ArchBox x={760} y={250} w={555} h={360} title="A2C2-Wrist trainable core" tone={colors.accent}>
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
      <DiagramBox title="Fast module inputs" subtitle="policy.forward(batch) 會實際送進 A2C2-Wrist module" tone={colors.accent}>
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
      <DiagramBox title="Learning targets" subtitle="A2C2 只用 current step raw residual MSE" tone={colors.rose}>
        <div style={{ display: 'grid', gap: 10 }}>
          <DiagramChip label="action" detail="expert action a_expert, shape (B,T,7)" tone={colors.rose} />
          <DiagramChip label="prev_delta" detail="action[t-1] - a_base[t-1]" tone={colors.gold} />
          <DiagramChip label="target_delta" detail="action[t] - a_base[t]" tone={colors.gold} />
          <DiagramChip label="loss" detail="MSE(delta_pred, target_delta)" tone={colors.accent} />
        </div>
      </DiagramBox>
    </div>
  </PageShell>
);

const FastModuleOutputs: Page = () => (
  <PageShell label="Fast module outputs">
    <Heading>A2C2-Wrist 只輸出一個 7D correction</Heading>
    <div className="fade-2" style={{ marginTop: 54, display: 'grid', gridTemplateColumns: '430px 90px 500px 90px 430px', gap: 20, alignItems: 'center', width: 1580 }}>
      <DiagramBox title="Inputs at time window" subtitle="state + cache feature + base action" tone={colors.blue}>
        <div style={{ display: 'grid', gap: 10 }}>
          <DiagramChip label="dino_patches" detail="visual wrist signal" tone={colors.blue} />
          <DiagramChip label="z_goal / z_phase" detail="slow planner context" tone={colors.gold} />
          <DiagramChip label="state + a_base + k" detail="robot and chunk context" tone={colors.violet} />
        </div>
      </DiagramBox>
      <FlowArrow />
      <DiagramBox title="A2C2WristCorrectionModule" subtitle="trainable: patch proj, cross-attn pool, fuser, delta head" tone={colors.accent} style={{ minHeight: 330 }}>
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

const LearningObjective: Page = () => (
  <PageShell label="Learning objective">
    <Heading>目前先用最小可行的 residual 目標</Heading>
    <div className="fade-2" style={{ marginTop: 34, display: 'grid', gap: 12, width: 1640 }}>
      <LossRow
        name="L_delta"
        target="delta_a ≈ action[t] - a_base[t]"
        role="A2C2-Wrist 目前唯一訓練 loss：學 current step raw 7D correction。"
        tone={colors.accent}
      />
      <LossRow
        name="prev"
        target="prev_delta = action[t-1] - a_base[t-1]"
        role="保留前一幀 correction conditioning，但不在 fast-cache 建立時固定 seq_len。"
        tone={colors.blue}
      />
      <LossRow
        name="deploy"
        target="a_final = a_base + alpha * clip(delta_a)"
        role="推論時只做 scalar alpha + residual clipping，保留 base action bypass。"
        tone={colors.gold}
      />
      <LossRow
        name="monitor"
        target="delta_norm / target_delta_norm / clip_fraction"
        role="先用 smoke run 確認輸入輸出與 loss 行為，再進長訓練和 closed-loop eval。"
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
      <Card title="delta_a" body="A2C2-Wrist 學 raw current-step residual。" />
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
      <Card title="Windowing" body="fast-cache 不存 seq_len；training loader 才用 policy.seq_len 建 window。A2C2 首跑用 previous/current 兩幀。" tone={colors.violet} />
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
    --cache-root checkpoints/HFRVLA_libero_v1_fastcache_v2`}</CodeBlock>
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
      <CodeBlock size={23}>{`RUN_NAME=hfrvla_a2c2_wrist_seq2 \\
WANDB_ENABLE=false \\
HFRVLA_TMP_ROOT=$HOME/tmp/hfrvla \\
HFRVLA_DATASET_BACKEND=fastcache \\
HFRVLA_FASTCACHE_ROOT=checkpoints/HFRVLA_libero_v1_fastcache_v2 \\
RESIDUAL_MERGE_MODE=a2c2 \\
A2C2_ALPHA=1.0 \\
SEQ_LEN=2 \\
BATCH_SIZE=256 \\
NUM_WORKERS=8 \\
scripts/run_hfrvla_training_foreground.sh`}</CodeBlock>
      <div style={{ display: 'grid', gap: 24 }}>
        <Card title="A2C2-Wrist" body="stateless feed-forward correction：單一 residual head，學完整 7D delta。" tone={colors.gold} />
        <Card title="Expected log" body="[hfrvla-train] objective residual_mode=a2c2 ..." />
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
  5. only then launch full spatial + object eval`}</CodeBlock>
    </div>
  </PageShell>
);

const SpatialObjectComparison: Page = () => (
  <PageShell>
    <Heading maxWidth={1540}>n_action_steps=8 後 baseline 比較變公平</Heading>
    <Note>同樣 seed=42、每 suite 10 tasks x 5 episodes。兩邊都覆蓋成 n_action_steps=8；HFRVLA 使用 alpha=0.5。</Note>
    <div className="fade-3" style={{ marginTop: 34, display: 'grid', gridTemplateColumns: '.82fr 1.18fr', gap: 28, alignItems: 'start' }}>
      <div style={{ display: 'grid', gap: 16 }}>
        <MiniCard title="HFRVLA n=8" body="spatial 34/50、object 47/50，合計 81/100 = 81.0%。" tone={colors.accent} />
        <MiniCard title="SmolVLA n=8" body="spatial 32/50、object 45/50，合計 77/100 = 77.0%。" tone={colors.gold} />
        <MiniCard title="Interpretation" body="相同 8-step chunk 下，HFRVLA +4/100；這頁只比較 matched 8-step baseline。" tone={colors.rose} />
      </div>
      <CodeBlock size={20} lineHeight={1.22} padding="24px 30px">{`matched n_action_steps=8
policy                         spatial       object        combined
HFRVLA 30k alpha=0.5           34/50 68.0%   47/50 94.0%   81/100 81.0%
SmolVLA                        32/50 64.0%   45/50 90.0%   77/100 77.0%

interpretation
same 8-step chunk baseline removes the main mismatch.
HFRVLA correction is +2/50 spatial and +2/50 object.`}</CodeBlock>
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
      <MiniCard title="Output target" body="沿用完整 7D correction，直接學 action[t] - a_base[t]。" tone={colors.rose} />
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
      <Card title="Current replacement" body="A2C2-Wrist feed-forward residual head；previous/current training window；simple alpha-scaled residual merge。" tone={colors.accent} />
      <Card title="Where history lives" body="Engineering archive can stay in docs/training.md and older notes; paper-facing HTML keeps only active decisions and planned experiments." tone={colors.gold} />
      <Card title="Reactivation rule" body="Only bring this path back into HTML if it becomes an active experiment again and receives a registry-backed eval plan." tone={colors.blue} />
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
    148 rows, 37 columns
    spatial + object + combined rows

scripts/build_eval_results_master.py
  rebuilds master from manifest
  expands sweep/profile metadata
  keeps paper-facing CSV compact`}</CodeBlock>
      <div style={{ display: 'grid', gap: 18 }}>
        <MiniCard title="Current sources" body="action_steps: 30；chunk_size: 36；alpha_clip: 36；no_clip controls: 22；lr/wd: 24 rows。" tone={colors.accent} />
        <MiniCard title="Do not hand-edit" body="master CSV 是 build artifact；未來不同 training params 都透過 sources.csv 記錄。" tone={colors.rose} />
        <MiniCard title="Paper use" body="表格與圖都從 master CSV filter/groupby，避免把 sandbox log 或手抄數字當 ground truth。" tone={colors.gold} />
      </div>
    </div>
  </PageShell>
);

const ActionStepSweepResults: Page = () => (
  <PageShell label="Action-step sweep">
    <Heading maxWidth={1560}>Action-step sweep：固定 plan=50，只改 execution/replan</Heading>
    <Note width={1460}>
      這組測試回答：當 slow planner 仍規劃 50-step chunk 時，實際執行幾步才重新規劃比較好？
    </Note>
    <div className="fade-3" style={{ marginTop: 34, display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 24, alignItems: 'start' }}>
      <CodeBlock size={17} lineHeight={1.18} padding="22px 24px">{`HFRVLA 30k, alpha=0.5
plan exec spatial       object        combined
50   2    41/50 82.0%   44/50 88.0%   85/100 85.0%
50   4    38/50 76.0%   45/50 90.0%   83/100 83.0%
50   8    34/50 68.0%   47/50 94.0%   81/100 81.0%
50   16   34/50 68.0%   40/50 80.0%   74/100 74.0%
50   32   26/50 52.0%   45/50 90.0%   71/100 71.0%`}</CodeBlock>
      <CodeBlock size={17} lineHeight={1.18} padding="22px 24px">{`SmolVLA baseline
plan exec spatial       object        combined
50   2    36/50 72.0%   46/50 92.0%   82/100 82.0%
50   4    31/50 62.0%   42/50 84.0%   73/100 73.0%
50   8    32/50 64.0%   45/50 90.0%   77/100 77.0%
50   16   34/50 68.0%   44/50 88.0%   78/100 78.0%
50   32   31/50 62.0%   42/50 84.0%   73/100 73.0%`}</CodeBlock>
    </div>
  </PageShell>
);

const ChunkSizeSweepResults: Page = () => (
  <PageShell label="Chunk-size sweep">
    <Heading maxWidth={1560}>Chunk-size sweep：plan=exec=K 一起改</Heading>
    <Note width={1460}>
      這組測試回答：如果模型只規劃 K 步、也執行 K 步才重新規劃，成功率如何變化？
    </Note>
    <div className="fade-3" style={{ marginTop: 34, display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 24, alignItems: 'start' }}>
      <CodeBlock size={16} lineHeight={1.16} padding="22px 24px">{`HFRVLA 30k, alpha=0.5
K    spatial       object        combined
2    38/50 76.0%   44/50 88.0%   82/100 82.0%
4    31/50 62.0%   47/50 94.0%   78/100 78.0%
8    36/50 72.0%   47/50 94.0%   83/100 83.0%
16   32/50 64.0%   48/50 96.0%   80/100 80.0%
32   27/50 54.0%   44/50 88.0%   71/100 71.0%
50   22/50 44.0%   24/50 48.0%   46/100 46.0%`}</CodeBlock>
      <CodeBlock size={16} lineHeight={1.16} padding="22px 24px">{`SmolVLA baseline
K    spatial       object        combined
2    33/50 66.0%   43/50 86.0%   76/100 76.0%
4    34/50 68.0%   45/50 90.0%   79/100 79.0%
8    29/50 58.0%   45/50 90.0%   74/100 74.0%
16   30/50 60.0%   44/50 88.0%   74/100 74.0%
32   26/50 52.0%   41/50 82.0%   67/100 67.0%
50   20/50 40.0%   23/50 46.0%   43/100 43.0%`}</CodeBlock>
    </div>
  </PageShell>
);

const AlphaClipSweepResults: Page = () => (
  <PageShell label="Alpha/clip sweep">
    <Heading maxWidth={1560}>Alpha/clip sweep：n_action_steps=8 時最佳是 alpha=0.75, clip=0.2</Heading>
    <Note width={1500}>
      固定 SmolVLA plan=50、execution/replan=8、seed=42；delta_max 同步設定 safety joint velocity limit，所以 effective cap = alpha x delta_max。
    </Note>
    <div className="fade-3" style={{ marginTop: 34, display: 'grid', gridTemplateColumns: '1.12fr .88fr', gap: 24, alignItems: 'start' }}>
      <CodeBlock size={15} lineHeight={1.13} padding="22px 24px">{`HFRVLA 30k, plan=50, exec=8
alpha clip cap    spatial       object        combined
0.75  .2  .150   41/50 82.0%   47/50 94.0%   88/100 88.0%
1.00  .2  .200   38/50 76.0%   44/50 88.0%   82/100 82.0%
0.25  .3  .075   37/50 74.0%   44/50 88.0%   81/100 81.0%
0.75  .4  .300   37/50 74.0%   44/50 88.0%   81/100 81.0%
0.50  .2  .100   34/50 68.0%   47/50 94.0%   81/100 81.0%
0.50  .4  .200   35/50 70.0%   45/50 90.0%   80/100 80.0%
0.75  .3  .225   38/50 76.0%   42/50 84.0%   80/100 80.0%
0.25  .4  .100   32/50 64.0%   44/50 88.0%   76/100 76.0%
0.50  .3  .150   33/50 66.0%   42/50 84.0%   75/100 75.0%
1.00  .3  .300   34/50 68.0%   40/50 80.0%   74/100 74.0%
0.25  .2  .050   25/50 50.0%   44/50 88.0%   69/100 69.0%
1.00  .4  .400   36/50 72.0%   29/50 58.0%   65/100 65.0%`}</CodeBlock>
      <div style={{ display: 'grid', gap: 18 }}>
        <Card title="Decision" body="目前 n=8 部署建議使用 alpha=0.75、delta_max=0.2；combined 88/100，是完整 sweep 最佳。" tone={colors.accent} />
        <Card title="Why not larger clip" body="同 alpha 下 clip 從 .2 提到 .3/.4 會降低 combined；alpha=1.0 在 .3/.4 也明顯退化。" tone={colors.rose} />
        <Card title="Tie rule" body="主排序 combined，其次 spatial，再選較小 effective cap；這避免只靠 object suite 撐高總分。" tone={colors.gold} />
      </div>
    </div>
  </PageShell>
);

const LrWdSweepResults: Page = () => (
  <PageShell label="LR/WD sweep">
    <Heading maxWidth={1560}>LR/WD sweep：比較標準固定為 SmolVLA plan=50, exec=8 baseline</Heading>
    <Note width={1500}>
      固定 eval alpha=0.75、delta_max=0.2、planning=50、execution/replan=8、seed=42；主要比較基準是同 protocol 的 SmolVLA baseline，而不是不同 training run 的內部 reference。
    </Note>
    <div className="fade-3" style={{ marginTop: 34, display: 'grid', gridTemplateColumns: '1.08fr .92fr', gap: 24, alignItems: 'start' }}>
      <CodeBlock size={15} lineHeight={1.13} padding="22px 24px">{`HFRVLA LR/WD, b512 50k, alpha=.75, clip=.2
lr     wd      spatial       object        combined   vs baseline
3e-4   0       36/50 72.0%   49/50 98.0%   85/100 85.0%   +8
3e-4   1e-5    36/50 72.0%   49/50 98.0%   85/100 85.0%   +8
3e-4   1e-4    36/50 72.0%   48/50 96.0%   84/100 84.0%   +7
5e-4   0       34/50 68.0%   49/50 98.0%   83/100 83.0%   +6
5e-4   1e-5    34/50 68.0%   49/50 98.0%   83/100 83.0%   +6
5e-4   1e-4    33/50 66.0%   49/50 98.0%   82/100 82.0%   +5
3e-4   3e-4    37/50 74.0%   44/50 88.0%   81/100 81.0%   +4
5e-4   3e-4    22/50 44.0%   48/50 96.0%   70/100 70.0%   -7`}</CodeBlock>
      <div style={{ display: 'grid', gap: 18 }}>
        <Card title="Matched baseline" body="SmolVLA plan=50、exec/replan=8：spatial 32/50、object 45/50、combined 77/100。" tone={colors.gold} />
        <Card title="Decision" body="預設改為 lr=3e-4、wd=1e-5；它和 wd=0 並列 85/100，但保留極小 regularization，且比 matched baseline 高 +8/100。" tone={colors.accent} />
        <Card title="Caution" body="wd=3e-4 不穩：lr=3e-4 時 object 掉到 88%，lr=5e-4 時 spatial 掉到 44%。" tone={colors.rose} />
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
      <CodeBlock size={15} lineHeight={1.13} padding="22px 24px">{`experiment_id                  variable                 values / plan                         fixed_controls                                      status
action_step8_alpha_clip_sweep  alpha x delta_max        alpha=.25,.5,.75,1; delta=.2,.3,.4  n=8; plan=50; seed=42; both suites                complete: best alpha=.75, delta=.2
hfrvla_lrwd_alpha075_clip02    lr x weight_decay        lr={3e-4,5e-4}; wd={0,1e-5,1e-4,3e-4} plan=50; exec=8; alpha=.75; clip=.2       complete: default lr=3e-4, wd=1e-5
seq_len_temporal_wrist         seq_len                  2,4,8                               HFRVLA; dataset=v1; alpha=0.5; spatial+object     pending
latent_context_ablation        A2C2_USE_LATENT_CONTEXT  true,false                          seq_len=2; steps=30k; alpha=0.5; spatial+object   pending
action_context_ablation        prev_delta/action ctx    keep,remove                         seq_len=2; steps=30k; alpha=0.5; spatial+object   pending
alpha_training_alignment       train/deploy alpha       train target aligned to 0.5 vs 1.0   seq_len=2; steps=30k; spatial+object              pending
planning_execution_grid        plan x exec              plan={4,8,16}; exec={2,4,8}          best checkpoint; seed=42; 5ep/task; both suites   pending`}</CodeBlock>
    </div>
  </PageShell>
);

const PaperWritingAnchors: Page = () => (
  <PageShell label="Paper anchors">
    <Heading maxWidth={1560}>之後寫論文時要抓住的資訊</Heading>
    <div className="fade-2" style={{ marginTop: 48, display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 22, width: 1580 }}>
      <Card title="Core claim" body="一個小型 wrist-camera residual module 可以在 frozen SmolVLA action chunks 上做 fast correction，不需要重訓 slow planner。" tone={colors.accent} />
      <Card title="Comparison framing" body="跟 A2C2 類方法對照時，強調 wrist visual correction 的簡化；但不要說整個 policy 只看 wrist，因為仍使用 slow planner context。" tone={colors.gold} />
      <Card title="Baseline discipline" body="SmolVLA baseline 必須用相同 planning/execution override 比較；default n=1 只能作上限參考，不是 matched baseline。" tone={colors.blue} />
      <Card title="Evidence source" body="數字引用 `eval_results_master.csv`；raw outputs 和 eval logs 只作追溯，不作手抄表格來源。" tone={colors.rose} />
    </div>
    <div className="fade-3" style={{ marginTop: 34, width: 1500 }}>
      <CodeBlock size={22} lineHeight={1.25} padding="24px 30px">{`write-up checklist
1. State the two chunk axes explicitly:
   planning_chunk_size vs execution_chunk_size / replan_interval_steps.
2. Report spatial and object together when claiming overall LIBERO improvement.
3. Record each new training run in experiments/eval_registry/sources.csv.
4. Keep docs/training_presentation.html, paper/notes, and memory synchronized.`}</CodeBlock>
    </div>
  </PageShell>
);

const Knobs: Page = () => (
  <PageShell>
    <Heading>調參先看 throughput 與 sample budget</Heading>
    <div className="fade-2" style={{ marginTop: 56, display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 28, width: 1340 }}>
      <Card title="SEQ_LEN" body="training-time window；A2C2 path 用 seq_len=2 的 previous/current pair。" tone={colors.violet} />
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
  LearningObjective,
  Dataset,
  Pipeline,
  FastCache,
  TrainEntry,
  SmokeProtocol,
  DecisionLedger,
  RetiredPath,
  SpatialObjectComparison,
  ActionStepSweepResults,
  ChunkSizeSweepResults,
  AlphaClipSweepResults,
  LrWdSweepResults,
  ExperimentMatrix,
  EvalRegistry,
  PaperWritingAnchors,
  Knobs,
  Automation,
  Pitfalls,
  Checklist,
] satisfies Page[];
