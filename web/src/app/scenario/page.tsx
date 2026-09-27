"use client";

import { useCallback, useEffect, useState } from "react";
import { BookOpen, BrainCircuit, Clock3, FileText, Loader2, Plus, Sparkles, Workflow } from "lucide-react";

const API = "http://localhost:8008";
const stages = ["research", "analyze", "timeline", "script", "visual-plan"] as const;

type Scenario = {
  id: string;
  premise: string;
  title: string;
  status: string;
  current_stage?: string | null;
  job_id?: string | null;
  error?: string | null;
  research_data?: { summary?: string; items?: ResearchItem[]; sources?: Record<string, Source> } | null;
  analysis_data?: { hypothesis?: string; branches?: Branch[] } | null;
  timeline_data?: { entries?: TimelineEntry[] } | null;
  script_data?: { title?: string; narration?: string; sections?: { heading?: string; text?: string }[] } | null;
  visual_plan_data?: { scenes?: { id?: string; narration?: string; visual_prompt?: string }[] } | null;
};

type ResearchItem = { kind: string; claim: string; confidence: number; uncertainty?: string[]; source_ids?: string[] };
type Source = { title: string; url?: string; publisher?: string };
type Branch = { id: string; label: string; nodes?: { level: string; claim: string; confidence: number; uncertainty?: string[] }[] };
type TimelineEntry = { phase: string; title: string; consequence: string; confidence: number; uncertainty?: string[] };

export default function ScenarioPage() {
  const [scenarios, setScenarios] = useState<Scenario[]>([]);
  const [selected, setSelected] = useState<Scenario | null>(null);
  const [premise, setPremise] = useState("");
  const [title, setTitle] = useState("");
  const [busy, setBusy] = useState(false);
  const [message, setMessage] = useState("");

  const loadScenarios = useCallback(async () => {
    const response = await fetch(`${API}/api/scenarios`);
    if (!response.ok) throw new Error("Could not load scenarios");
    const data = await response.json();
    setScenarios(data);
    if (selected) {
      const refreshed = data.find((item: Scenario) => item.id === selected.id);
      if (refreshed) setSelected(refreshed);
    }
  }, [selected]);

  useEffect(() => {
    let active = true;
    fetch(`${API}/api/scenarios`)
      .then(response => {
        if (!response.ok) throw new Error("Could not load scenarios");
        return response.json();
      })
      .then(data => {
        if (active) setScenarios(data);
      })
      .catch(error => {
        if (active) setMessage(error.message);
      });
    return () => {
      active = false;
    };
  }, [loadScenarios]);

  useEffect(() => {
    if (!selected?.job_id || !selected.status.endsWith("_queued") && !selected.status.endsWith("_running")) return;
    const socket = new WebSocket(`ws://localhost:8008/api/generate/ws/${selected.job_id}`);
    socket.onmessage = event => {
      const messageData = JSON.parse(event.data);
      const state = messageData.data || messageData;
      if (messageData.type === "log") setMessage(messageData.data || messageData.message || "Working...");
      if (state.status === "completed" || state.status === "failed") {
        loadScenarios().catch(error => setMessage(error.message));
      }
    };
    return () => socket.close();
  }, [loadScenarios, selected?.job_id, selected?.status]);

  const createScenario = async (event: React.FormEvent) => {
    event.preventDefault();
    setBusy(true);
    setMessage("");
    try {
      const response = await fetch(`${API}/api/scenarios`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ premise, title: title || undefined }),
      });
      if (!response.ok) throw new Error((await response.json()).detail || "Could not create scenario");
      const scenario = await response.json();
      setPremise("");
      setTitle("");
      setSelected(scenario);
      await loadScenarios();
    } catch (error) {
      setMessage(error instanceof Error ? error.message : "Could not create scenario");
    } finally {
      setBusy(false);
    }
  };

  const runStage = async (stage: string) => {
    if (!selected) return;
    setBusy(true);
    setMessage(`Starting ${stage}...`);
    try {
      const response = await fetch(`${API}/api/scenarios/${selected.id}/${stage}`, { method: "POST" });
      if (!response.ok) throw new Error((await response.json()).detail || `Could not start ${stage}`);
      await loadScenarios();
    } catch (error) {
      setMessage(error instanceof Error ? error.message : "Could not start stage");
    } finally {
      setBusy(false);
    }
  };

  const canRun = (stage: string) => {
    if (!selected) return false;
    const requirements: Record<string, string[]> = {
      research: ["created", "failed"],
      analyze: ["research_complete", "failed"],
      timeline: ["analysis_complete", "failed"],
      script: ["timeline_complete", "failed"],
      "visual-plan": ["script_complete", "failed"],
    };
    return requirements[stage].includes(selected.status);
  };

  return (
    <div className="max-w-[1500px] mx-auto space-y-6 animate-in fade-in duration-500 pb-16">
      <header className="mb-8">
        <h1 className="text-3xl font-bold text-white tracking-tight flex items-center gap-3"><Sparkles className="h-8 w-8 text-[#66fcf1]" /> Hypothetical Scenario Engine</h1>
        <p className="text-zinc-400 mt-1">Change one thing about reality. Then follow what happens.</p>
      </header>

      <div className="grid grid-cols-1 xl:grid-cols-12 gap-6">
        <aside className="xl:col-span-3 space-y-6">
          <form onSubmit={createScenario} className="glass-card p-6 space-y-4">
            <h2 className="text-xs font-bold text-white uppercase tracking-widest border-b border-white/10 pb-3 flex items-center gap-2"><Plus className="h-4 w-4 text-[#66fcf1]" /> New scenario</h2>
            <input className="modern-input" value={title} onChange={event => setTitle(event.target.value)} placeholder="Optional title" />
            <textarea className="modern-input min-h-32 resize-y" required minLength={5} value={premise} onChange={event => setPremise(event.target.value)} placeholder="What if Earth stopped spinning?" />
            <button className="btn-primary" disabled={busy}><Plus className="h-4 w-4" /> Create scenario</button>
          </form>
          <div className="glass-card p-4 space-y-2">
            <h2 className="text-xs font-bold text-zinc-400 uppercase tracking-widest px-2 pb-2">Saved scenarios</h2>
            {scenarios.length === 0 ? <p className="text-sm text-zinc-600 px-2 py-4">No scenarios yet.</p> : scenarios.map(scenario => (
              <button key={scenario.id} onClick={() => setSelected(scenario)} className={`w-full text-left p-3 rounded-xl border transition-all ${selected?.id === scenario.id ? "bg-[#66fcf1]/10 border-[#66fcf1]/50" : "border-white/5 hover:bg-white/5"}`}>
                <span className="block text-sm font-bold text-white truncate">{scenario.title}</span>
                <span className="block text-xs text-zinc-500 mt-1 truncate">{scenario.status}</span>
              </button>
            ))}
          </div>
        </aside>

        <main className="xl:col-span-9 space-y-6">
          {!selected ? <div className="glass-card min-h-[500px] flex items-center justify-center text-zinc-500"><div className="text-center space-y-3"><BrainCircuit className="h-12 w-12 mx-auto opacity-30" /><p>Create or select a scenario to begin.</p></div></div> : <>
            <section className="glass-card p-6 space-y-4">
              <div className="flex flex-wrap justify-between gap-4"><div><p className="text-xs uppercase tracking-widest text-[#66fcf1] font-bold">Premise</p><h2 className="text-2xl font-bold text-white mt-2">{selected.title}</h2></div><span className="h-fit px-3 py-1 rounded-full border border-white/10 text-xs text-zinc-300">{selected.status}</span></div>
              <p className="text-zinc-300 leading-relaxed">{selected.premise}</p>
              {message && <p className="text-sm text-[#66fcf1]">{message}</p>}
              {selected.error && <p className="text-sm text-rose-400">{selected.error}</p>}
              <div className="flex flex-wrap gap-2 pt-2">{stages.map(stage => <button key={stage} onClick={() => runStage(stage)} disabled={!canRun(stage) || busy} className="px-3 py-2 rounded-lg border border-white/10 text-xs font-bold text-zinc-300 enabled:hover:border-[#66fcf1]/60 enabled:hover:text-white disabled:opacity-35 disabled:cursor-not-allowed">{busy && selected.current_stage === stage ? <Loader2 className="inline h-3 w-3 animate-spin mr-1" /> : null}{stage.replace("-", " ")}</button>)}</div>
            </section>

            <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
              <Panel icon={<BookOpen className="h-4 w-4" />} title="Structured research">
                <p className="text-sm text-zinc-400 mb-4">{selected.research_data?.summary || "Research has not been generated."}</p>
                <div className="space-y-2">{selected.research_data?.items?.map((item, index) => <div key={index} className="p-3 bg-black/30 rounded-lg border border-white/5"><div className="flex justify-between gap-3"><span className="text-[10px] font-bold text-[#66fcf1]">{item.kind}</span><span className="text-[10px] text-zinc-500">{Math.round(item.confidence * 100)}% confidence</span></div><p className="text-sm text-zinc-200 mt-1">{item.claim}</p>{item.uncertainty?.length ? <p className="text-xs text-amber-300/80 mt-2">Uncertainty: {item.uncertainty.join(" ")}</p> : null}</div>)}</div>
              </Panel>
              <Panel icon={<Workflow className="h-4 w-4" />} title="Causal analysis">
                <p className="text-sm text-zinc-300 mb-4">{selected.analysis_data?.hypothesis || "Analysis has not been generated."}</p>
                <div className="space-y-4">{selected.analysis_data?.branches?.map(branch => <div key={branch.id} className="border-l border-[#66fcf1]/40 pl-3"><p className="text-sm font-bold text-white">{branch.label}</p>{branch.nodes?.map(node => <div key={node.claim} className="mt-2"><span className="text-[10px] font-bold text-[#66fcf1]">{node.level}</span><p className="text-sm text-zinc-300">{node.claim}</p></div>)}</div>)}</div>
              </Panel>
              <Panel icon={<Clock3 className="h-4 w-4" />} title="Timeline">
                <div className="space-y-3">{selected.timeline_data?.entries?.map(entry => <div key={entry.title} className="p-3 rounded-lg bg-black/30 border border-white/5"><span className="text-[10px] uppercase tracking-wider text-[#66fcf1] font-bold">{entry.phase.replace("_", " ")}</span><h3 className="text-sm font-bold text-white mt-1">{entry.title}</h3><p className="text-sm text-zinc-300 mt-1">{entry.consequence}</p><p className="text-xs text-zinc-500 mt-2">Confidence {Math.round(entry.confidence * 100)}%</p></div>)}</div>
              </Panel>
              <Panel icon={<FileText className="h-4 w-4" />} title="Script and visual plan">
                <h3 className="text-sm font-bold text-white">{selected.script_data?.title || "Script not generated"}</h3><p className="text-sm text-zinc-300 whitespace-pre-wrap mt-3 max-h-60 overflow-y-auto">{selected.script_data?.narration}</p>
                <div className="mt-4 space-y-2">{selected.visual_plan_data?.scenes?.map(scene => <div key={scene.id} className="text-xs border-t border-white/10 pt-2"><span className="text-[#66fcf1]">Scene {scene.id}</span><p className="text-zinc-300">{scene.visual_prompt}</p></div>)}</div>
              </Panel>
            </div>
          </>}
        </main>
      </div>
    </div>
  );
}

function Panel({ icon, title, children }: { icon: React.ReactNode; title: string; children: React.ReactNode }) {
  return <section className="glass-card p-5 min-h-56"><h2 className="text-xs font-bold text-white uppercase tracking-widest border-b border-white/10 pb-3 mb-4 flex items-center gap-2">{icon}{title}</h2>{children}</section>;
}
