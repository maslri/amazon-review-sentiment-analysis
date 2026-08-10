const models = [
  {
    name: "RoBERTa + LoRA",
    eyebrow: "Parameter efficient",
    micro: "0.6748",
    macro: "0.6751",
    trainable: "3.26%",
    detail: "Rank-16 adapters preserve the pretrained backbone while learning rating-specific language.",
    tone: "cyan",
  },
  {
    name: "DeBERTa-v3 + LoRA",
    eyebrow: "Resumed after timeout",
    micro: "0.6757",
    macro: "0.6738",
    trainable: "2.22%",
    detail: "Disentangled attention, a 12-hour interruption, and an exact checkpoint continuation through epoch four.",
    tone: "violet",
  },
  {
    name: "RoBERTa Full + Ordinal",
    eyebrow: "Best validation score",
    micro: "0.6948",
    macro: "0.6960",
    trainable: "100%",
    detail: "Full-backbone adaptation combines categorical accuracy with the natural order of one-to-five-star labels.",
    tone: "gold",
    best: true,
  },
];

const pipeline = [
  ["01", "Inspect", "Quality, imbalance, length and metadata coverage"],
  ["02", "Clean", "Remove HTML and URLs; preserve negation and tone"],
  ["03", "Enrich", "Verified status, helpful votes, summary and review"],
  ["04", "Balance", "50,000 examples from each rating class"],
  ["05", "Fine-tune", "LoRA and full-backbone Transformer experiments"],
  ["06", "Audit", "Reproduce validation before hidden-test inference"],
];

const notebooks = [
  ["01", "EDA", "Dataset shape, imbalance and text quality"],
  ["02", "Preprocessing", "Local and Kaggle-ready dataset builders"],
  ["03", "RoBERTa full", "Hybrid ordinal fine-tuning"],
  ["04", "DeBERTa", "Initial LoRA training session"],
  ["05", "LoRA inference", "Validation audit and submission"],
  ["06", "Resume", "Timeout-safe training continuation"],
  ["07", "Full inference", "Best-model hidden-test prediction"],
  ["08", "DeBERTa inference", "Third-model submission workflow"],
];

const classScores = [
  ["1 star", 75.32],
  ["2 stars", 61.09],
  ["3 stars", 62.03],
  ["4 stars", 67.45],
  ["5 stars", 82.11],
];

export default function Home() {
  return (
    <main>
      <header className="site-nav">
        <a className="brand" href="#top" aria-label="Amazon Review Intelligence home">
          <span className="brand-mark">AR</span>
          <span>Review Intelligence</span>
        </a>
        <nav aria-label="Primary navigation">
          <a href="#pipeline">Pipeline</a>
          <a href="#models">Models</a>
          <a href="#results">Results</a>
          <a href="#notebooks">Notebooks</a>
        </nav>
        <a className="nav-action" href="https://github.com/maslri/amazon-review-sentiment-analysis" target="_blank" rel="noreferrer">
          GitHub repository
        </a>
      </header>

      <section className="hero" id="top">
        <div className="hero-grid" aria-hidden="true" />
        <div className="hero-copy">
          <p className="kicker"><span /> 838,944 reviews · five exact ratings · one hidden test</p>
          <h1>Can a model hear the difference between <em>three stars</em> and <em>four?</em></h1>
          <p className="hero-lede">
            An end-to-end NLP study that turns long, messy Amazon electronics reviews into precise one-to-five-star predictions—then tests three ways to adapt modern Transformer models.
          </p>
          <div className="hero-actions">
            <a className="button primary" href="#results">Explore the results</a>
            <a className="button secondary" href="https://www.kaggle.com/datasets/maslri/balanced-50000" target="_blank" rel="noreferrer">Open the dataset</a>
          </div>
          <div className="hero-facts" aria-label="Project summary">
            <div><strong>250K</strong><span>balanced labelled rows</span></div>
            <div><strong>256</strong><span>tokens per sequence</span></div>
            <div><strong>20K</strong><span>untouched test rows</span></div>
          </div>
        </div>
        <div className="hero-visual" aria-label="Best model validation micro F1 is 0.6948">
          <div className="orbit orbit-one" />
          <div className="orbit orbit-two" />
          <div className="score-core">
            <span>Best micro-F1</span>
            <strong>0.6948</strong>
            <small>RoBERTa full + ordinal</small>
          </div>
          <div className="rating-node n1">1</div><div className="rating-node n2">2</div><div className="rating-node n3">3</div><div className="rating-node n4">4</div><div className="rating-node n5">5</div>
        </div>
      </section>

      <section className="signal-strip" aria-label="Project principles">
        <span>Micro-F1 first</span><i />
        <span>Macro-F1 always visible</span><i />
        <span>No test leakage</span><i />
        <span>Reproducible checkpoints</span><i />
        <span>Original row order preserved</span>
      </section>

      <section className="section data-story">
        <div className="section-heading split-heading">
          <div>
            <p className="kicker"><span /> The raw signal</p>
            <h2>A dataset where “accurate” can still mean biased.</h2>
          </div>
          <p>Five-star reviews dominate the original distribution. The experiment creates an equal learning surface—50,000 reviews per rating—so improvements in minority classes cannot hide behind the majority.</p>
        </div>
        <div className="data-layout">
          <div className="distribution-panel">
            <div className="panel-label"><span>Original training distribution</span><strong>838,944 rows</strong></div>
            {[
              ["1", "82,950", 18], ["2", "56,756", 12], ["3", "81,239", 18], ["4", "156,514", 34], ["5", "461,485", 100],
            ].map(([label, count, width]) => (
              <div className="dist-row" key={label as string}>
                <span>{label}★</span><div><i style={{ width: `${width}%` }} /></div><strong>{count}</strong>
              </div>
            ))}
          </div>
          <div className="input-panel">
            <p className="panel-label">One structured model input</p>
            <div className="input-code">
              <span className="field">Verified</span><b>:</b> yes <i>|</i><br />
              <span className="field">Helpful votes</span><b>:</b> 10_to_49 <i>|</i><br />
              <span className="field">Summary</span><b>:</b> Almost perfect <i>|</i><br />
              <span className="field">Review</span><b>:</b> Great display, but the battery...
            </div>
            <p className="input-note">Natural language stays natural: punctuation, numbers, emoji, stopwords and negation remain intact.</p>
          </div>
        </div>
      </section>

      <section className="section pipeline-section" id="pipeline">
        <div className="section-heading">
          <p className="kicker"><span /> The experiment</p>
          <h2>Every decision is traceable.</h2>
          <p>From the first inspection to the final CSV, the workflow separates training evidence, validation decisions and hidden-test inference.</p>
        </div>
        <div className="pipeline-grid">
          {pipeline.map(([number, title, text]) => (
            <article className="pipeline-step" key={number}>
              <span>{number}</span><h3>{title}</h3><p>{text}</p>
            </article>
          ))}
        </div>
      </section>

      <section className="section models-section" id="models">
        <div className="section-heading split-heading">
          <div><p className="kicker"><span /> Three adaptations</p><h2>Same data. Different ways to learn.</h2></div>
          <p>LoRA asks how far a small trainable adapter can go. Full fine-tuning asks what happens when the whole backbone is allowed to move.</p>
        </div>
        <div className="model-grid">
          {models.map((model) => (
            <article className={`model-card ${model.tone} ${model.best ? "best" : ""}`} key={model.name}>
              <div className="model-top"><span>{model.eyebrow}</span>{model.best && <b>Winner</b>}</div>
              <h3>{model.name}</h3>
              <p>{model.detail}</p>
              <div className="model-schema" aria-hidden="true"><i /><i /><i /><span>→</span><b /></div>
              <div className="model-metrics">
                <div><span>Micro-F1</span><strong>{model.micro}</strong></div>
                <div><span>Macro-F1</span><strong>{model.macro}</strong></div>
                <div><span>Trainable</span><strong>{model.trainable}</strong></div>
              </div>
            </article>
          ))}
        </div>
      </section>

      <section className="section results-section" id="results">
        <div className="section-heading">
          <p className="kicker"><span /> What won</p>
          <h2>Capacity delivered the clearest gain.</h2>
          <p>Full RoBERTa improves balanced validation micro-F1 by two points over the LoRA baseline, with gains across every rating class.</p>
        </div>
        <div className="results-layout">
          <div className="comparison-chart" aria-label="Validation micro F1 comparison">
            <div className="chart-scale"><span>0.65</span><span>0.67</span><span>0.69</span><span>0.70</span></div>
            {[
              ["RoBERTa LoRA", "0.6748", 49], ["DeBERTa-v3 LoRA", "0.6757", 51], ["RoBERTa Full + Ordinal", "0.6948", 90],
            ].map(([name, score, width], index) => (
              <div className={`chart-row ${index === 2 ? "winner" : ""}`} key={name as string}>
                <span>{name}</span><div><i style={{ width: `${width}%` }} /></div><strong>{score}</strong>
              </div>
            ))}
            <p className="chart-caption">Balanced validation · 10,000 reviews · 2,000 per class</p>
          </div>
          <div className="class-chart">
            <div className="panel-label"><span>Best-model F1 by rating</span><strong>Extremes are clearer</strong></div>
            {classScores.map(([label, score]) => (
              <div className="class-row" key={label as string}>
                <span>{label}</span><div><i style={{ width: `${score}%` }} /></div><strong>{Number(score).toFixed(1)}%</strong>
              </div>
            ))}
          </div>
        </div>
        <aside className="result-insight">
          <span>The hard truth</span>
          <p>The models understand sentiment direction better than sentiment intensity. Ratings 2, 3 and 4 overlap in language—and often in human intent.</p>
        </aside>
      </section>

      <section className="section resilience-section">
        <div className="resilience-copy">
          <p className="kicker"><span /> Engineering the run</p>
          <h2>Twelve hours elapsed. The experiment did not restart.</h2>
          <p>DeBERTa exceeded Kaggle’s session limit inside epoch three. A writable copy of the complete training state restored weights, optimizer progress and the learning-rate schedule in a fresh session.</p>
          <div className="resume-stats"><div><strong>500</strong><span>batches per backup</span></div><div><strong>4</strong><span>epochs completed</span></div><div><strong>0.6757</strong><span>final micro-F1</span></div></div>
        </div>
        <div className="resume-track" aria-label="DeBERTa training resume timeline">
          <div className="track-line" />
          <div className="track-event"><i /><span>Initial run</span><strong>Epoch 1–2</strong></div>
          <div className="track-event timeout"><i /><span>Kaggle timeout</span><strong>Epoch 3 · step 6,499</strong></div>
          <div className="track-event"><i /><span>State restored</span><strong>Optimizer + schedule</strong></div>
          <div className="track-event complete"><i /><span>Completed</span><strong>Epoch 4</strong></div>
        </div>
      </section>

      <section className="section notebooks-section" id="notebooks">
        <div className="section-heading split-heading">
          <div><p className="kicker"><span /> Reproduce it</p><h2>Eight notebooks. One continuous story.</h2></div>
          <p>Clean templates stay separate from executed evidence, so the repository is readable without losing the outputs that support each claim.</p>
        </div>
        <div className="notebook-grid">
          {notebooks.map(([number, title, text]) => (
            <article key={number}><span>{number}</span><div><h3>{title}</h3><p>{text}</p></div></article>
          ))}
        </div>
      </section>

      <section className="closing">
        <p className="kicker"><span /> The final contract</p>
        <h2>20,000 rows in. One trustworthy column out.</h2>
        <div className="submission"><span>q2_submission.csv</span><i>→</i><strong>predicted</strong><small>integer · 1 to 5 · no index</small></div>
        <p>Every inference notebook reproduces its known validation score before it is allowed to touch the test set.</p>
        <div className="hero-actions centered">
          <a className="button primary" href="https://github.com/maslri/amazon-review-sentiment-analysis" target="_blank" rel="noreferrer">Explore the source</a>
          <a className="button secondary" href="https://www.kaggle.com/models/yasin86/roberta-keras-full-ordinal" target="_blank" rel="noreferrer">Use the best model</a>
        </div>
      </section>

      <footer><span>Amazon Review Rating Prediction</span><span>TensorFlow · KerasHub · Kaggle</span><span>MIT License</span></footer>
    </main>
  );
}
