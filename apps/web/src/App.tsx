import { FormEvent, useEffect, useState } from "react";

type VoiceProfile = {
  voice_id: string;
  source: string;
  preset: string;
  language_priors: Record<string, number>;
};

type LanguageInfo = {
  code: string;
  name: string;
  production: boolean;
};

type Tab = "enroll" | "design" | "synthesize";

const apiBase = "";

async function fetchJson<T>(path: string, init?: RequestInit): Promise<T> {
  const response = await fetch(`${apiBase}${path}`, init);
  if (!response.ok) {
    throw new Error(await response.text());
  }
  return response.json();
}

export default function App() {
  const [tab, setTab] = useState<Tab>("enroll");
  const [voices, setVoices] = useState<VoiceProfile[]>([]);
  const [languages, setLanguages] = useState<LanguageInfo[]>([]);
  const [message, setMessage] = useState<string>("");
  const [audioUrl, setAudioUrl] = useState<string>("");
  const [streamProgress, setStreamProgress] = useState<number>(0);

  const [enrollVoiceId, setEnrollVoiceId] = useState("demo-voice");
  const [enrollFile, setEnrollFile] = useState<File | null>(null);

  const [preset, setPreset] = useState("neutral");
  const [designVoiceId, setDesignVoiceId] = useState("designed-voice");
  const [sourceVoiceId, setSourceVoiceId] = useState("");
  const [pitch, setPitch] = useState(1);
  const [pace, setPace] = useState(1);
  const [energy, setEnergy] = useState(1);
  const [brightness, setBrightness] = useState(1);

  const [text, setText] = useState("Hello from Aoede.");
  const [languageCode, setLanguageCode] = useState("en");
  const [selectedVoiceId, setSelectedVoiceId] = useState("");

  useEffect(() => {
    void refresh();
  }, []);

  async function refresh() {
    const [voicesResponse, languagesResponse] = await Promise.all([
      fetchJson<{ voices: VoiceProfile[] }>("/api/v1/voices"),
      fetchJson<{ production: LanguageInfo[]; experimental: LanguageInfo[] }>("/api/v1/languages"),
    ]);
    setVoices(voicesResponse.voices);
    setLanguages([...languagesResponse.production, ...languagesResponse.experimental]);
    if (!selectedVoiceId && voicesResponse.voices[0]) {
      setSelectedVoiceId(voicesResponse.voices[0].voice_id);
    }
  }

  async function onEnroll(event: FormEvent<HTMLFormElement>) {
    event.preventDefault();
    if (!enrollFile) {
      setMessage("Select a WAV file first.");
      return;
    }
    const form = new FormData();
    form.set("voice_id", enrollVoiceId);
    form.set("audio", enrollFile);
    const response = await fetch(`${apiBase}/api/v1/voices/enroll`, { method: "POST", body: form });
    if (!response.ok) {
      setMessage(await response.text());
      return;
    }
    const payload = await response.json();
    setMessage(`Enrolled ${payload.voice_id}`);
    await refresh();
  }

  async function onDesign(event: FormEvent<HTMLFormElement>) {
    event.preventDefault();
    const payload = {
      preset,
      voice_id: designVoiceId,
      source_voice_id: sourceVoiceId || null,
      style_controls: { pitch, pace, energy, brightness },
    };
    const response = await fetchJson<VoiceProfile>("/api/v1/voices/design", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload),
    });
    setMessage(`Designed ${response.voice_id}`);
    setSelectedVoiceId(response.voice_id);
    await refresh();
  }

  async function onSynthesize(event: FormEvent<HTMLFormElement>) {
    event.preventDefault();
    setStreamProgress(0);
    setMessage("Starting synthesis stream...");
    const protocol = window.location.protocol === "https:" ? "wss" : "ws";
    const ws = new WebSocket(`${protocol}://${window.location.host}/api/v1/synthesis/stream`);
    const chunks: Uint8Array[] = [];

    ws.onopen = () => {
      ws.send(
        JSON.stringify({
          text,
          language_code: languageCode,
          voice_id: selectedVoiceId,
          style_controls: { pitch, pace, energy, brightness },
          stream: true,
        }),
      );
    };

    ws.onmessage = async (rawEvent) => {
      const eventData = JSON.parse(rawEvent.data);
      setStreamProgress(eventData.progress ?? 0);
      if (eventData.type === "audio_chunk") {
        const binary = Uint8Array.from(atob(eventData.payload.chunk_b64), (char) => char.charCodeAt(0));
        chunks.push(binary);
      }
      if (eventData.type === "done") {
        const totalLength = chunks.reduce((sum, chunk) => sum + chunk.length, 0);
        const merged = new Uint8Array(totalLength);
        let offset = 0;
        for (const chunk of chunks) {
          merged.set(chunk, offset);
          offset += chunk.length;
        }
        const blob = new Blob([merged], { type: "audio/wav" });
        setAudioUrl(URL.createObjectURL(blob));
        setMessage(`Finished synthesis in ${eventData.payload.duration_s}s`);
      }
      if (eventData.type === "error") {
        setMessage(eventData.payload.message ?? "Synthesis failed");
      }
    };
  }

  const productionLanguages = languages.filter((language) => language.production);

  return (
    <div className="shell">
      <aside className="hero">
        <p className="eyebrow">Aoede v1</p>
        <h1>Multilingual voice cloning with a product-ready control surface.</h1>
        <p className="lede">
          Enroll a voice, design a new one, then synthesize speech across the launch
          language set with streaming feedback.
        </p>
        <div className="status">
          <span>{voices.length} voices saved</span>
          <span>{productionLanguages.length} production languages</span>
        </div>
      </aside>

      <main className="panel">
        <nav className="tabs">
          {(["enroll", "design", "synthesize"] as Tab[]).map((item) => (
            <button
              key={item}
              type="button"
              className={tab === item ? "active" : ""}
              onClick={() => setTab(item)}
            >
              {item}
            </button>
          ))}
        </nav>

        {tab === "enroll" && (
          <form className="stack" onSubmit={onEnroll}>
            <label>
              Voice ID
              <input value={enrollVoiceId} onChange={(event) => setEnrollVoiceId(event.target.value)} />
            </label>
            <label>
              Enrollment WAV
              <input type="file" accept="audio/wav" onChange={(event) => setEnrollFile(event.target.files?.[0] ?? null)} />
            </label>
            <button type="submit">Enroll Voice</button>
          </form>
        )}

        {tab === "design" && (
          <form className="stack" onSubmit={onDesign}>
            <label>
              Designed Voice ID
              <input value={designVoiceId} onChange={(event) => setDesignVoiceId(event.target.value)} />
            </label>
            <label>
              Source Voice
              <select value={sourceVoiceId} onChange={(event) => setSourceVoiceId(event.target.value)}>
                <option value="">Preset only</option>
                {voices.map((voice) => (
                  <option key={voice.voice_id} value={voice.voice_id}>
                    {voice.voice_id}
                  </option>
                ))}
              </select>
            </label>
            <label>
              Preset
              <select value={preset} onChange={(event) => setPreset(event.target.value)}>
                <option value="neutral">Neutral</option>
                <option value="velvet">Velvet</option>
                <option value="bright">Bright</option>
                <option value="deep">Deep</option>
              </select>
            </label>
            <Slider label="Pitch" value={pitch} onChange={setPitch} />
            <Slider label="Pace" value={pace} onChange={setPace} />
            <Slider label="Energy" value={energy} onChange={setEnergy} />
            <Slider label="Brightness" value={brightness} onChange={setBrightness} />
            <button type="submit">Save Designed Voice</button>
          </form>
        )}

        {tab === "synthesize" && (
          <form className="stack" onSubmit={onSynthesize}>
            <label>
              Voice
              <select value={selectedVoiceId} onChange={(event) => setSelectedVoiceId(event.target.value)}>
                <option value="">Select a voice</option>
                {voices.map((voice) => (
                  <option key={voice.voice_id} value={voice.voice_id}>
                    {voice.voice_id}
                  </option>
                ))}
              </select>
            </label>
            <label>
              Language
              <select value={languageCode} onChange={(event) => setLanguageCode(event.target.value)}>
                {languages.map((language) => (
                  <option key={language.code} value={language.code}>
                    {language.name}
                  </option>
                ))}
              </select>
            </label>
            <label>
              Text
              <textarea value={text} onChange={(event) => setText(event.target.value)} rows={5} />
            </label>
            <Slider label="Pitch" value={pitch} onChange={setPitch} />
            <Slider label="Pace" value={pace} onChange={setPace} />
            <Slider label="Energy" value={energy} onChange={setEnergy} />
            <Slider label="Brightness" value={brightness} onChange={setBrightness} />
            <button type="submit">Stream Synthesis</button>
            <div className="meter">
              <div className="meter-fill" style={{ width: `${Math.round(streamProgress * 100)}%` }} />
            </div>
            {audioUrl && <audio controls src={audioUrl} />}
          </form>
        )}

        <section className="foot">
          <p>{message}</p>
          <button type="button" onClick={() => void refresh()}>
            Refresh Data
          </button>
        </section>
      </main>
    </div>
  );
}

function Slider({
  label,
  value,
  onChange,
}: {
  label: string;
  value: number;
  onChange: (value: number) => void;
}) {
  return (
    <label>
      {label}: {value.toFixed(2)}
      <input
        type="range"
        min="0.5"
        max="1.5"
        step="0.01"
        value={value}
        onChange={(event) => onChange(Number(event.target.value))}
      />
    </label>
  );
}
