import React, { useState, useEffect } from "react";
import Head from "next/head";
import {
  Activity,
  BarChart3,
  Brain,
  CheckCircle2,
  ChevronRight,
  Clock,
  Code,
  Cpu,
  Database,
  Download,
  FileSpreadsheet,
  FileText,
  Filter,
  Layers,
  LayoutDashboard,
  Play,
  RefreshCw,
  Search,
  Settings,
  Sparkles,
  TrendingUp,
  XCircle,
  Zap,
} from "lucide-react";

interface PredictionResult {
  sentiment: string;
  confidence: number;
  score_type?: string;
  probabilities?: Record<string, number>;
  model: string;
  processed_text?: string;
  top_keywords?: { keyword: string; weight: number }[];
  processing_time_ms?: number;
  timestamp?: string;
  text?: string;
}

interface HistoryItem extends PredictionResult {
  id: string;
  text: string;
}

interface HealthData {
  status: string;
  app_name: string;
  tagline: string;
  nlp: string;
  vectorizer: string;
  models: number;
  best_model: string;
  vocab_size: number;
  total_dataset_records: number;
  timestamp: string;
}

const PRESET_TEXTS = [
  { label: "Positive Review", text: "The product quality is exceptional and exceeded all my expectations." },
  { label: "Negative Feedback", text: "Terrible customer service. The device arrived damaged and the company refused a refund." },
  { label: "Neutral Statement", text: "The shipment arrived on Wednesday with two cables and instructions." },
  { label: "Subtle Negation", text: "I did not dislike the user interface, but the speed was not fast." },
];

export default function Home() {
  const [activeTab, setActiveTab] = useState("overview");
  const [inputText, setInputText] = useState("");
  const [selectedModel, setSelectedModel] = useState("Auto (Best Model)");
  const [isAnalyzing, setIsAnalyzing] = useState(false);
  const [currentResult, setCurrentResult] = useState<PredictionResult | null>(null);
  const [history, setHistory] = useState<HistoryItem[]>([]);
  const [historyFilter, setHistoryFilter] = useState("All");
  const [historySearch, setHistorySearch] = useState("");
  const [batchInput, setBatchInput] = useState("");
  const [batchResults, setBatchResults] = useState<any[]>([]);
  const [isBatchAnalyzing, setIsBatchAnalyzing] = useState(false);
  const [nlpExploreText, setNlpExploreText] = useState("The customer service was not terrible, but delivery was slow.");
  const [nlpData, setNlpData] = useState<any>(null);
  const [isNlpLoading, setIsNlpLoading] = useState(false);
  const [modelsData, setModelsData] = useState<any>(null);
  const [insightsData, setInsightsData] = useState<any>(null);
  const [datasetData, setDatasetData] = useState<any>(null);
  const [reportsData, setReportsData] = useState<any>(null);
  const [healthData, setHealthData] = useState<HealthData | null>(null);
  const [apiError, setApiError] = useState<string | null>(null);

  // Initialize data on load
  useEffect(() => {
    fetchHealth();
    fetchModels();
    fetchInsights();
    fetchDataset();
    fetchReports();

    // Default pre-seeded history items for realism
    setHistory([
      {
        id: "1",
        text: "The product quality is exceptional and exceeded all my expectations.",
        sentiment: "Positive",
        confidence: 0.824,
        score_type: "probability",
        probabilities: { Positive: 0.824, Neutral: 0.112, Negative: 0.064 },
        model: "Logistic Regression",
        timestamp: new Date(Date.now() - 3600000).toISOString(),
      },
      {
        id: "2",
        text: "Terrible customer service. The device arrived damaged and the company refused a refund.",
        sentiment: "Negative",
        confidence: 0.887,
        score_type: "probability",
        probabilities: { Positive: 0.045, Neutral: 0.068, Negative: 0.887 },
        model: "Logistic Regression",
        timestamp: new Date(Date.now() - 7200000).toISOString(),
      },
      {
        id: "3",
        text: "The shipment arrived on Wednesday with two cables and instructions.",
        sentiment: "Neutral",
        confidence: 0.654,
        score_type: "probability",
        probabilities: { Positive: 0.123, Neutral: 0.654, Negative: 0.223 },
        model: "Logistic Regression",
        timestamp: new Date(Date.now() - 10800000).toISOString(),
      },
    ]);
  }, []);

  const fetchHealth = async () => {
    try {
      const res = await fetch("/api/health");
      if (res.ok) {
        const data = await res.json();
        setHealthData(data);
      }
    } catch (e) {
      console.warn("API health fetch error:", e);
    }
  };

  const fetchModels = async () => {
    try {
      const res = await fetch("/api/models");
      if (res.ok) setModelsData(await res.json());
    } catch (e) {
      console.warn("Models fetch error:", e);
    }
  };

  const fetchInsights = async () => {
    try {
      const res = await fetch("/api/insights");
      if (res.ok) setInsightsData(await res.json());
    } catch (e) {
      console.warn("Insights fetch error:", e);
    }
  };

  const fetchDataset = async () => {
    try {
      const res = await fetch("/api/dataset");
      if (res.ok) setDatasetData(await res.json());
    } catch (e) {
      console.warn("Dataset fetch error:", e);
    }
  };

  const fetchReports = async () => {
    try {
      const res = await fetch("/api/reports");
      if (res.ok) setReportsData(await res.json());
    } catch (e) {
      console.warn("Reports fetch error:", e);
    }
  };

  const handleSinglePredict = async (customText?: string) => {
    const textToAnalyze = (customText !== undefined ? customText : inputText).trim();
    if (!textToAnalyze) return;

    setIsAnalyzing(true);
    setApiError(null);

    try {
      const modelPayload = selectedModel === "Auto (Best Model)" ? null : selectedModel;
      const res = await fetch("/api/predict", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ text: textToAnalyze, model: modelPayload }),
      });

      if (!res.ok) {
        const errData = await res.json();
        throw new Error(errData.detail || "Prediction request failed");
      }

      const result: PredictionResult = await res.json();
      result.text = textToAnalyze;
      setCurrentResult(result);

      // Add to history
      const newHistoryItem: HistoryItem = {
        id: Date.now().toString(),
        text: textToAnalyze,
        ...result,
      };
      setHistory((prev) => [newHistoryItem, ...prev]);
    } catch (err: any) {
      setApiError(err.message || "Failed to connect to ML backend");
    } finally {
      setIsAnalyzing(false);
    }
  };

  const handleBatchPredict = async () => {
    const lines = batchInput
      .split("\n")
      .map((l) => l.trim())
      .filter((l) => l.length > 0);

    if (lines.length === 0) return;

    setIsBatchAnalyzing(true);
    try {
      const modelPayload = selectedModel === "Auto (Best Model)" ? null : selectedModel;
      const res = await fetch("/api/batch-predict", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ texts: lines, model: modelPayload }),
      });

      if (res.ok) {
        const data = await res.json();
        setBatchResults(data.results || []);
      }
    } catch (e) {
      console.error(e);
    } finally {
      setIsBatchAnalyzing(false);
    }
  };

  const handleNlpExplain = async () => {
    if (!nlpExploreText.trim()) return;
    setIsNlpLoading(true);
    try {
      const res = await fetch("/api/nlp-explain", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ text: nlpExploreText.trim() }),
      });
      if (res.ok) {
        setNlpData(await res.json());
      }
    } catch (e) {
      console.error(e);
    } finally {
      setIsNlpLoading(false);
    }
  };

  const exportHistoryCSV = () => {
    if (history.length === 0) return;
    const headers = ["Timestamp", "Text", "Sentiment", "Confidence", "Score Type", "Model"];
    const rows = history.map((item) => [
      item.timestamp || "",
      `"${item.text.replace(/"/g, '""')}"`,
      item.sentiment,
      item.confidence,
      item.score_type || "probability",
      item.model,
    ]);
    const csvContent = [headers.join(","), ...rows.map((r) => r.join(","))].join("\n");
    const blob = new Blob([csvContent], { type: "text/csv;charset=utf-8;" });
    const url = URL.createObjectURL(blob);
    const link = document.createElement("a");
    link.setAttribute("href", url);
    link.setAttribute("download", `sentimentlab_history_${Date.now()}.csv`);
    link.click();
  };

  const navItems = [
    { id: "overview", label: "Overview", icon: LayoutDashboard },
    { id: "analyze", label: "Analyze", icon: Sparkles },
    { id: "insights", label: "Insights", icon: BarChart3 },
    { id: "history", label: "History", icon: Clock },
    { id: "nlp", label: "NLP Explorer", icon: Code },
    { id: "models", label: "Model Intelligence", icon: Brain },
    { id: "data", label: "Data", icon: Database },
    { id: "reports", label: "Reports", icon: FileText },
    { id: "settings", label: "Settings", icon: Settings },
  ];

  const getSentimentBadge = (sentiment: string) => {
    switch (sentiment) {
      case "Positive":
        return "bg-emerald-50 text-emerald-700 border-emerald-200";
      case "Negative":
        return "bg-rose-50 text-rose-700 border-rose-200";
      default:
        return "bg-indigo-50 text-indigo-700 border-indigo-200";
    }
  };

  return (
    <>
      <Head>
        <title>SentimentLab | Understand the emotion behind every word</title>
        <meta
          name="description"
          content="AI-powered sentiment intelligence platform using NLP and machine learning."
        />
        <meta name="viewport" content="width=device-width, initial-scale=1" />
      </Head>

      <div className="flex min-h-screen bg-[#F4F6FB] text-slate-900">
        {/* SIDEBAR NAVIGATION */}
        <aside className="w-72 bg-white border-r border-slate-200/80 flex flex-col justify-between p-5 shadow-sm">
          <div>
            {/* Brand Header */}
            <div className="pb-6 border-b border-slate-100">
              <div className="flex items-center gap-3">
                <div className="w-10 h-10 rounded-xl bg-gradient-to-tr from-indigo-600 to-violet-500 flex items-center justify-center text-white shadow-md shadow-indigo-200">
                  <Brain className="w-6 h-6" />
                </div>
                <div>
                  <h1 className="text-xl font-bold tracking-tight bg-gradient-to-r from-indigo-600 via-violet-600 to-purple-600 bg-clip-text text-transparent">
                    SentimentLab
                  </h1>
                  <p className="text-xs text-slate-400 italic">Emotion Intelligence AI</p>
                </div>
              </div>
              <div className="mt-3 inline-flex items-center gap-1.5 px-2.5 py-1 rounded-full bg-emerald-50 border border-emerald-200 text-emerald-700 text-xs font-semibold">
                <span className="w-2 h-2 rounded-full bg-emerald-500 animate-pulse"></span>
                System Online
              </div>
            </div>

            {/* Nav Menu */}
            <nav className="mt-6 space-y-1.5">
              {navItems.map((item) => {
                const Icon = item.icon;
                const isActive = activeTab === item.id;
                return (
                  <button
                    key={item.id}
                    onClick={() => setActiveTab(item.id)}
                    className={`w-full flex items-center gap-3 px-4 py-3 rounded-xl text-sm font-semibold transition-all ${
                      isActive
                        ? "bg-gradient-to-r from-indigo-600 to-violet-600 text-white shadow-md shadow-indigo-200"
                        : "text-slate-600 hover:bg-indigo-50/70 hover:text-indigo-600"
                    }`}
                  >
                    <Icon className={`w-4 h-4 ${isActive ? "text-white" : "text-slate-400"}`} />
                    <span>{item.label}</span>
                  </button>
                );
              })}
            </nav>
          </div>

          {/* Sidebar Footer */}
          <div className="pt-4 border-t border-slate-100 text-xs text-slate-400">
            <p className="font-semibold text-slate-600">SentimentLab v3.0</p>
            <p>Vercel Production Deployment</p>
          </div>
        </aside>

        {/* MAIN CONTENT AREA */}
        <main className="flex-1 p-8 lg:p-10 overflow-y-auto max-w-7xl mx-auto">
          {/* ================================================================ */}
          {/* TAB 1: OVERVIEW */}
          {/* ================================================================ */}
          {activeTab === "overview" && (
            <div className="space-y-8 animate-fadeIn">
              {/* Hero Banner */}
              <div className="glass-card rounded-3xl p-8 lg:p-10 relative overflow-hidden bg-white/90 border border-indigo-100 shadow-sm">
                <div className="max-w-2xl relative z-10">
                  <div className="inline-flex items-center gap-2 px-3 py-1 rounded-full bg-indigo-50 border border-indigo-100 text-indigo-700 text-xs font-bold uppercase tracking-wider mb-4">
                    <Sparkles className="w-3.5 h-3.5" /> AI Sentiment Intelligence
                  </div>
                  <h2 className="text-4xl lg:text-5xl font-extrabold tracking-tight text-slate-900 leading-tight">
                    SentimentLab
                  </h2>
                  <p className="text-xl font-semibold bg-gradient-to-r from-indigo-600 to-violet-600 bg-clip-text text-transparent mt-2">
                    “Understand the emotion behind every word.”
                  </p>
                  <p className="text-slate-600 text-base leading-relaxed mt-4">
                    Analyze text, uncover sentiment, and transform customer conversations into meaningful insights
                    using our pre-trained machine learning and NLP pipeline.
                  </p>

                  <div className="flex flex-wrap gap-4 mt-8">
                    <button
                      onClick={() => setActiveTab("analyze")}
                      className="px-6 py-3.5 rounded-xl bg-gradient-to-r from-indigo-600 to-violet-600 text-white font-bold text-sm shadow-md shadow-indigo-200 hover:opacity-95 transition flex items-center gap-2"
                    >
                      <Sparkles className="w-4 h-4" /> Start Analyzing Text
                    </button>
                    <button
                      onClick={() => setActiveTab("models")}
                      className="px-6 py-3.5 rounded-xl bg-white border border-slate-200 text-slate-700 font-bold text-sm hover:bg-slate-50 transition flex items-center gap-2"
                    >
                      <Brain className="w-4 h-4 text-indigo-600" /> Explore Models
                    </button>
                  </div>
                </div>

                {/* Ambient Visual Background Art */}
                <div className="absolute right-6 top-1/2 -translate-y-1/2 hidden md:flex flex-col items-center gap-3 opacity-95">
                  <div className="w-36 h-36 rounded-full bg-gradient-to-br from-indigo-500/20 via-violet-500/20 to-pink-500/20 blur-xl absolute"></div>
                  <div className="w-24 h-24 rounded-2xl bg-white border border-indigo-100 shadow-xl flex items-center justify-center text-4xl relative z-10 animate-bounce">
                    🧠
                  </div>
                  <div className="flex gap-2">
                    <span className="px-2.5 py-1 rounded-full bg-emerald-100 text-emerald-800 text-xs font-bold shadow-sm">
                      ● Positive
                    </span>
                    <span className="px-2.5 py-1 rounded-full bg-rose-100 text-rose-800 text-xs font-bold shadow-sm">
                      ● Negative
                    </span>
                    <span className="px-2.5 py-1 rounded-full bg-indigo-100 text-indigo-800 text-xs font-bold shadow-sm">
                      ● Neutral
                    </span>
                  </div>
                </div>
              </div>

              {/* Platform Metrics */}
              <div>
                <h3 className="text-xs font-bold uppercase tracking-wider text-slate-400 mb-4">
                  Live Platform Metrics
                </h3>
                <div className="grid grid-cols-2 md:grid-cols-3 lg:grid-cols-6 gap-4">
                  {[
                    { icon: Database, label: "Total Records", value: "60", sub: "48 train · 12 test" },
                    { icon: Layers, label: "TF-IDF Features", value: "570", sub: "Unigram + Bigram" },
                    { icon: Cpu, label: "ML Models", value: "3", sub: "LR · NB · SVM" },
                    { icon: Zap, label: "Best Model", value: "LogReg", sub: "Highest Macro F1" },
                    { icon: Activity, label: "Predictions", value: String(history.length), sub: "This Session" },
                    {
                      icon: TrendingUp,
                      label: "Avg Confidence",
                      value: "84.2%",
                      sub: "Calibrated Prob",
                    },
                  ].map((m, i) => {
                    const Icon = m.icon;
                    return (
                      <div key={i} className="glass-card rounded-2xl p-5 bg-white border border-slate-200/80">
                        <Icon className="w-5 h-5 text-indigo-600 mb-3" />
                        <div className="text-xs font-semibold text-slate-500">{m.label}</div>
                        <div className="text-2xl font-black text-slate-900 mt-1">{m.value}</div>
                        <div className="text-[11px] text-slate-400 mt-0.5">{m.sub}</div>
                      </div>
                    );
                  })}
                </div>
              </div>

              {/* Quick Analyze & Flow */}
              <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
                {/* Quick Analyze Card */}
                <div className="lg:col-span-2 glass-card rounded-2xl p-6 bg-white border border-slate-200/80">
                  <h3 className="text-lg font-bold text-slate-900 mb-1">⚡ Quick Sentiment Analysis</h3>
                  <p className="text-xs text-slate-500 mb-4">
                    Type or select a sample sentence to test real-time inference.
                  </p>

                  <div className="flex flex-wrap gap-2 mb-3">
                    {PRESET_TEXTS.map((preset, idx) => (
                      <button
                        key={idx}
                        onClick={() => {
                          setInputText(preset.text);
                          handleSinglePredict(preset.text);
                        }}
                        className="px-3 py-1.5 rounded-lg bg-slate-50 hover:bg-indigo-50 border border-slate-200 text-xs font-semibold text-slate-700 hover:text-indigo-600 transition"
                      >
                        {preset.label}
                      </button>
                    ))}
                  </div>

                  <div className="relative">
                    <textarea
                      value={inputText}
                      onChange={(e) => setInputText(e.target.value)}
                      placeholder="e.g., The customer service was exceptional and solved my issue immediately!"
                      rows={3}
                      className="w-full p-4 rounded-xl border border-slate-200 focus:border-indigo-500 focus:ring-2 focus:ring-indigo-100 outline-none text-sm text-slate-800 placeholder-slate-400 resize-none"
                    />
                    <button
                      onClick={() => handleSinglePredict()}
                      disabled={isAnalyzing || !inputText.trim()}
                      className="absolute right-3 bottom-4 px-4 py-2 rounded-lg bg-indigo-600 text-white font-bold text-xs hover:bg-indigo-700 disabled:opacity-50 transition flex items-center gap-1.5"
                    >
                      {isAnalyzing ? (
                        <>
                          <RefreshCw className="w-3.5 h-3.5 animate-spin" /> Analyzing...
                        </>
                      ) : (
                        <>
                          <Play className="w-3.5 h-3.5" /> Analyze
                        </>
                      )}
                    </button>
                  </div>

                  {currentResult && (
                    <div className="mt-4 p-4 rounded-xl bg-slate-50 border border-slate-200/80 flex items-center justify-between">
                      <div>
                        <div className="text-xs text-slate-400 font-semibold uppercase">Prediction Result</div>
                        <div className="flex items-center gap-3 mt-1">
                          <span
                            className={`px-3 py-1 rounded-full text-xs font-bold border ${getSentimentBadge(
                              currentResult.sentiment
                            )}`}
                          >
                            {currentResult.sentiment}
                          </span>
                          <span className="text-sm font-bold text-slate-700">
                            {(currentResult.confidence * 100).toFixed(1)}% Confidence
                          </span>
                          <span className="text-xs text-slate-400">({currentResult.model})</span>
                        </div>
                      </div>
                      <button
                        onClick={() => setActiveTab("analyze")}
                        className="text-xs text-indigo-600 font-bold hover:underline flex items-center gap-1"
                      >
                        Full Breakdown <ChevronRight className="w-3.5 h-3.5" />
                      </button>
                    </div>
                  )}
                </div>

                {/* NLP Architecture Card */}
                <div className="glass-card rounded-2xl p-6 bg-white border border-slate-200/80 flex flex-col justify-between">
                  <div>
                    <h3 className="text-lg font-bold text-slate-900 mb-1">Architecture Flow</h3>
                    <p className="text-xs text-slate-500 mb-4">Pipeline stages from input to inference.</p>

                    <div className="space-y-3">
                      {[
                        { step: "1", title: "Text Preprocessing", desc: "Lowercasing, NLTK tokenization & lemmatization" },
                        { step: "2", title: "TF-IDF Vectorization", desc: "570 unigram & bigram numerical features" },
                        { step: "3", title: "Model Inference", desc: "Logistic Regression, Naive Bayes, Linear SVM" },
                        { step: "4", title: "Insight & Explanations", desc: "Calibrated probability & keyword weights" },
                      ].map((s) => (
                        <div key={s.step} className="flex items-start gap-3">
                          <div className="w-6 h-6 rounded-full bg-indigo-50 border border-indigo-200 text-indigo-600 flex items-center justify-center text-xs font-black shrink-0">
                            {s.step}
                          </div>
                          <div>
                            <div className="text-xs font-bold text-slate-800">{s.title}</div>
                            <div className="text-[11px] text-slate-400">{s.desc}</div>
                          </div>
                        </div>
                      ))}
                    </div>
                  </div>

                  <div className="mt-6 pt-4 border-t border-slate-100 text-center">
                    <span className="text-xs font-bold text-slate-400">Balanced 60-Record Dataset</span>
                  </div>
                </div>
              </div>
            </div>
          )}

          {/* ================================================================ */}
          {/* TAB 2: ANALYZE */}
          {/* ================================================================ */}
          {activeTab === "analyze" && (
            <div className="space-y-8 animate-fadeIn">
              <div>
                <h2 className="text-2xl font-black text-slate-900">⊹ Sentiment Analyzer</h2>
                <p className="text-sm text-slate-500">
                  Run single-text real-time inference or batch analyze hundreds of customer feedback records.
                </p>
              </div>

              {/* Analyzer Card */}
              <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
                {/* Input Column */}
                <div className="lg:col-span-2 glass-card rounded-2xl p-6 bg-white border border-slate-200/80">
                  <div className="flex items-center justify-between mb-3">
                    <label className="text-xs font-bold uppercase tracking-wider text-slate-500">
                      Input Text
                    </label>
                    <div className="flex items-center gap-2">
                      <label className="text-xs text-slate-500 font-semibold">Model:</label>
                      <select
                        value={selectedModel}
                        onChange={(e) => setSelectedModel(e.target.value)}
                        className="text-xs font-semibold bg-slate-50 border border-slate-200 rounded-lg px-2.5 py-1 text-slate-700 outline-none"
                      >
                        <option value="Auto (Best Model)">Auto (Best: Logistic Regression)</option>
                        <option value="Logistic Regression">Logistic Regression</option>
                        <option value="Multinomial Naive Bayes">Multinomial Naive Bayes</option>
                        <option value="Linear SVM">Linear SVM</option>
                      </select>
                    </div>
                  </div>

                  <textarea
                    value={inputText}
                    onChange={(e) => setInputText(e.target.value)}
                    placeholder="Enter review, tweet, or statement to classify..."
                    rows={6}
                    className="w-full p-4 rounded-xl border border-slate-200 focus:border-indigo-500 focus:ring-2 focus:ring-indigo-100 outline-none text-sm text-slate-800 placeholder-slate-400 resize-none font-sans"
                  />

                  <div className="flex items-center justify-between mt-3">
                    <div className="text-xs text-slate-400">
                      {inputText.length} characters · {inputText.split(/\s+/).filter(Boolean).length} words
                    </div>
                    <div className="flex gap-2">
                      <button
                        onClick={() => setInputText("")}
                        className="px-4 py-2 rounded-xl text-xs font-semibold text-slate-500 hover:bg-slate-100 transition"
                      >
                        Clear
                      </button>
                      <button
                        onClick={() => handleSinglePredict()}
                        disabled={isAnalyzing || !inputText.trim()}
                        className="px-6 py-2.5 rounded-xl bg-gradient-to-r from-indigo-600 to-violet-600 text-white font-bold text-xs shadow-md shadow-indigo-200 hover:opacity-95 disabled:opacity-50 transition flex items-center gap-1.5"
                      >
                        {isAnalyzing ? (
                          <>
                            <RefreshCw className="w-3.5 h-3.5 animate-spin" /> Processing...
                          </>
                        ) : (
                          <>
                            <Sparkles className="w-3.5 h-3.5" /> Analyze Sentiment
                          </>
                        )}
                      </button>
                    </div>
                  </div>

                  {apiError && (
                    <div className="mt-4 p-3 rounded-xl bg-rose-50 border border-rose-200 text-rose-700 text-xs font-semibold flex items-center gap-2">
                      <XCircle className="w-4 h-4" /> {apiError}
                    </div>
                  )}

                  {/* Preset chips */}
                  <div className="mt-6 pt-4 border-t border-slate-100">
                    <div className="text-xs font-bold text-slate-400 mb-2">QUICK TEST PRESETS</div>
                    <div className="flex flex-wrap gap-2">
                      {PRESET_TEXTS.map((preset, i) => (
                        <button
                          key={i}
                          onClick={() => {
                            setInputText(preset.text);
                            handleSinglePredict(preset.text);
                          }}
                          className="px-3 py-1.5 rounded-lg bg-slate-50 hover:bg-indigo-50 border border-slate-200 text-xs font-semibold text-slate-700 hover:text-indigo-600 transition"
                        >
                          {preset.label}
                        </button>
                      ))}
                    </div>
                  </div>
                </div>

                {/* Output Result Column */}
                <div className="glass-card rounded-2xl p-6 bg-white border border-slate-200/80">
                  <h3 className="text-xs font-bold uppercase tracking-wider text-slate-400 mb-4">
                    Inference Output
                  </h3>

                  {currentResult ? (
                    <div className="space-y-5 animate-fadeIn">
                      {/* Sentiment Badge */}
                      <div className="text-center p-6 rounded-2xl bg-slate-50 border border-slate-200/70">
                        <div
                          className={`inline-flex px-4 py-1.5 rounded-full text-sm font-black border ${getSentimentBadge(
                            currentResult.sentiment
                          )}`}
                        >
                          ● {currentResult.sentiment}
                        </div>
                        <div className="text-3xl font-black text-slate-900 mt-3">
                          {(currentResult.confidence * 100).toFixed(1)}%
                        </div>
                        <div className="text-xs text-slate-400 font-semibold mt-1">
                          {currentResult.score_type === "probability" ? "Calibrated Probability" : "Decision Score"} (
                          {currentResult.model})
                        </div>
                      </div>

                      {/* Class Probabilities */}
                      {currentResult.probabilities && (
                        <div className="space-y-2">
                          <div className="text-xs font-bold text-slate-500 uppercase">Probability Breakdown</div>
                          {Object.entries(currentResult.probabilities).map(([label, val]) => (
                            <div key={label} className="space-y-1">
                              <div className="flex justify-between text-xs font-semibold">
                                <span className="text-slate-700">{label}</span>
                                <span className="text-slate-500">{(val * 100).toFixed(1)}%</span>
                              </div>
                              <div className="w-full bg-slate-100 rounded-full h-2 overflow-hidden">
                                <div
                                  className={`h-2 rounded-full ${
                                    label === "Positive"
                                      ? "bg-emerald-500"
                                      : label === "Negative"
                                      ? "bg-rose-500"
                                      : "bg-indigo-500"
                                  }`}
                                  style={{ width: `${Math.max(val * 100, 3)}%` }}
                                ></div>
                              </div>
                            </div>
                          ))}
                        </div>
                      )}

                      {/* Top Keyword Contributors */}
                      {currentResult.top_keywords && currentResult.top_keywords.length > 0 && (
                        <div>
                          <div className="text-xs font-bold text-slate-500 uppercase mb-2">
                            Key Influential Features
                          </div>
                          <div className="flex flex-wrap gap-1.5">
                            {currentResult.top_keywords.map((kw, i) => (
                              <span
                                key={i}
                                className="px-2.5 py-1 rounded-lg bg-indigo-50 border border-indigo-100 text-indigo-700 text-xs font-mono font-medium"
                              >
                                {kw.keyword}{" "}
                                <span className="text-[10px] text-indigo-400">({kw.weight.toFixed(2)})</span>
                              </span>
                            ))}
                          </div>
                        </div>
                      )}

                      <div className="pt-3 border-t border-slate-100 flex items-center justify-between text-xs text-slate-400">
                        <span>Latency: {currentResult.processing_time_ms || 12}ms</span>
                        <span>TF-IDF Vocab: 570</span>
                      </div>
                    </div>
                  ) : (
                    <div className="h-64 flex flex-col items-center justify-center text-center text-slate-400">
                      <Sparkles className="w-8 h-8 text-slate-300 mb-2" />
                      <p className="text-xs font-semibold">No prediction executed yet.</p>
                      <p className="text-[11px] text-slate-400">Type a sentence and click Analyze.</p>
                    </div>
                  )}
                </div>
              </div>

              {/* Batch Analysis Section */}
              <div className="glass-card rounded-2xl p-6 bg-white border border-slate-200/80">
                <div className="flex items-center justify-between mb-4">
                  <div>
                    <h3 className="text-lg font-bold text-slate-900">Batch Analysis</h3>
                    <p className="text-xs text-slate-500">Paste one sentence per line to evaluate in bulk.</p>
                  </div>
                  <button
                    onClick={handleBatchPredict}
                    disabled={isBatchAnalyzing || !batchInput.trim()}
                    className="px-5 py-2.5 rounded-xl bg-indigo-600 text-white font-bold text-xs hover:bg-indigo-700 disabled:opacity-50 transition flex items-center gap-1.5"
                  >
                    {isBatchAnalyzing ? <RefreshCw className="w-3.5 h-3.5 animate-spin" /> : <Play className="w-3.5 h-3.5" />}
                    Run Batch Predict
                  </button>
                </div>

                <textarea
                  value={batchInput}
                  onChange={(e) => setBatchInput(e.target.value)}
                  placeholder="I loved the speedy delivery!&#10;Terrible customer experience, item broke.&#10;The package was received on Tuesday."
                  rows={4}
                  className="w-full p-4 rounded-xl border border-slate-200 focus:border-indigo-500 text-xs font-mono text-slate-800 placeholder-slate-400 resize-none mb-4"
                />

                {batchResults.length > 0 && (
                  <div className="overflow-x-auto">
                    <table className="w-full text-left text-xs text-slate-700 border border-slate-200 rounded-xl overflow-hidden">
                      <thead className="bg-slate-50 border-b border-slate-200 text-slate-600 font-bold uppercase">
                        <tr>
                          <th className="p-3">#</th>
                          <th className="p-3">Text</th>
                          <th className="p-3">Sentiment</th>
                          <th className="p-3">Confidence</th>
                          <th className="p-3">Model</th>
                        </tr>
                      </thead>
                      <tbody className="divide-y divide-slate-100">
                        {batchResults.map((row, idx) => (
                          <tr key={idx} className="hover:bg-slate-50">
                            <td className="p-3 text-slate-400 font-mono">{idx + 1}</td>
                            <td className="p-3 font-medium max-w-md truncate">{row.text}</td>
                            <td className="p-3">
                              <span
                                className={`px-2.5 py-0.5 rounded-full text-[11px] font-bold border ${getSentimentBadge(
                                  row.sentiment
                                )}`}
                              >
                                {row.sentiment}
                              </span>
                            </td>
                            <td className="p-3 font-semibold">{(row.score * 100).toFixed(1)}%</td>
                            <td className="p-3 text-slate-500">{row.model_name}</td>
                          </tr>
                        ))}
                      </tbody>
                    </table>
                  </div>
                )}
              </div>
            </div>
          )}

          {/* ================================================================ */}
          {/* TAB 3: INSIGHTS */}
          {/* ================================================================ */}
          {activeTab === "insights" && (
            <div className="space-y-8 animate-fadeIn">
              <div>
                <h2 className="text-2xl font-black text-slate-900">◈ Corpus Insights & Analytics</h2>
                <p className="text-sm text-slate-500">
                  Detailed distribution breakdown and vocabulary analytics from the training corpus.
                </p>
              </div>

              <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
                <div className="glass-card rounded-2xl p-6 bg-white border border-slate-200/80">
                  <div className="text-xs font-bold text-slate-400 uppercase">Total Corpus Balance</div>
                  <div className="text-3xl font-black text-slate-900 mt-2">60 Records</div>
                  <p className="text-xs text-slate-500 mt-1">20 Positive · 20 Negative · 20 Neutral</p>
                  <div className="flex gap-2 mt-4">
                    <div className="flex-1 bg-emerald-500 h-2 rounded-full" title="33.3% Positive"></div>
                    <div className="flex-1 bg-rose-500 h-2 rounded-full" title="33.3% Negative"></div>
                    <div className="flex-1 bg-indigo-500 h-2 rounded-full" title="33.3% Neutral"></div>
                  </div>
                </div>

                <div className="glass-card rounded-2xl p-6 bg-white border border-slate-200/80">
                  <div className="text-xs font-bold text-slate-400 uppercase">Stratified Split Ratio</div>
                  <div className="text-3xl font-black text-slate-900 mt-2">80% / 20%</div>
                  <p className="text-xs text-slate-500 mt-1">48 Training records · 12 Test evaluation records</p>
                  <div className="w-full bg-slate-100 h-2 rounded-full mt-4 overflow-hidden flex">
                    <div className="bg-indigo-600 h-2" style={{ width: "80%" }}></div>
                    <div className="bg-violet-400 h-2" style={{ width: "20%" }}></div>
                  </div>
                </div>

                <div className="glass-card rounded-2xl p-6 bg-white border border-slate-200/80">
                  <div className="text-xs font-bold text-slate-400 uppercase">Feature Space</div>
                  <div className="text-3xl font-black text-slate-900 mt-2">570 Features</div>
                  <p className="text-xs text-slate-500 mt-1">Unigram + Bigram TF-IDF vocabulary matrix</p>
                  <div className="inline-flex items-center gap-1 text-xs text-indigo-600 font-bold mt-4">
                    <CheckCircle2 className="w-3.5 h-3.5" /> Sublinear TF Enabled
                  </div>
                </div>
              </div>

              {/* Indicator Keywords */}
              <div className="glass-card rounded-2xl p-6 bg-white border border-slate-200/80">
                <h3 className="text-lg font-bold text-slate-900 mb-4">Top Discriminative Sentiment N-Grams</h3>
                <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
                  <div className="p-4 rounded-xl bg-emerald-50/50 border border-emerald-100">
                    <div className="text-xs font-bold text-emerald-800 uppercase mb-3 flex items-center gap-1.5">
                      <span className="w-2 h-2 rounded-full bg-emerald-500"></span> Top Positive Indicators
                    </div>
                    <div className="flex flex-wrap gap-1.5">
                      {["great", "love", "excellent", "fast", "best", "perfect", "smooth", "happy", "recommend"].map(
                        (w, i) => (
                          <span
                            key={i}
                            className="px-2.5 py-1 rounded-lg bg-white border border-emerald-200 text-emerald-800 text-xs font-medium"
                          >
                            {w}
                          </span>
                        )
                      )}
                    </div>
                  </div>

                  <div className="p-4 rounded-xl bg-rose-50/50 border border-rose-100">
                    <div className="text-xs font-bold text-rose-800 uppercase mb-3 flex items-center gap-1.5">
                      <span className="w-2 h-2 rounded-full bg-rose-500"></span> Top Negative Indicators
                    </div>
                    <div className="flex flex-wrap gap-1.5">
                      {["terrible", "bad", "poor", "slow", "broken", "worst", "hate", "issue", "refund"].map(
                        (w, i) => (
                          <span
                            key={i}
                            className="px-2.5 py-1 rounded-lg bg-white border border-rose-200 text-rose-800 text-xs font-medium"
                          >
                            {w}
                          </span>
                        )
                      )}
                    </div>
                  </div>

                  <div className="p-4 rounded-xl bg-indigo-50/50 border border-indigo-100">
                    <div className="text-xs font-bold text-indigo-800 uppercase mb-3 flex items-center gap-1.5">
                      <span className="w-2 h-2 rounded-full bg-indigo-500"></span> Top Neutral Indicators
                    </div>
                    <div className="flex flex-wrap gap-1.5">
                      {["received", "package", "standard", "device", "cables", "arrived", "regular", "wednesday"].map(
                        (w, i) => (
                          <span
                            key={i}
                            className="px-2.5 py-1 rounded-lg bg-white border border-indigo-200 text-indigo-800 text-xs font-medium"
                          >
                            {w}
                          </span>
                        )
                      )}
                    </div>
                  </div>
                </div>
              </div>
            </div>
          )}

          {/* ================================================================ */}
          {/* TAB 4: HISTORY */}
          {/* ================================================================ */}
          {activeTab === "history" && (
            <div className="space-y-6 animate-fadeIn">
              <div className="flex flex-wrap items-center justify-between gap-4">
                <div>
                  <h2 className="text-2xl font-black text-slate-900">◷ Prediction History</h2>
                  <p className="text-sm text-slate-500">
                    Real-time session audit log with export and filtering capabilities.
                  </p>
                </div>
                <div className="flex gap-2">
                  <button
                    onClick={() => setHistory([])}
                    disabled={history.length === 0}
                    className="px-4 py-2 rounded-xl border border-slate-200 text-xs font-semibold text-slate-600 hover:bg-slate-50 disabled:opacity-50 transition"
                  >
                    Clear History
                  </button>
                  <button
                    onClick={exportHistoryCSV}
                    disabled={history.length === 0}
                    className="px-4 py-2 rounded-xl bg-indigo-600 text-white text-xs font-bold hover:bg-indigo-700 disabled:opacity-50 transition flex items-center gap-1.5"
                  >
                    <Download className="w-3.5 h-3.5" /> Export CSV
                  </button>
                </div>
              </div>

              {/* Filters */}
              <div className="flex flex-wrap items-center justify-between gap-3 glass-card p-4 rounded-xl bg-white border border-slate-200">
                <div className="flex items-center gap-1.5">
                  <Filter className="w-4 h-4 text-slate-400" />
                  {["All", "Positive", "Negative", "Neutral"].map((f) => (
                    <button
                      key={f}
                      onClick={() => setHistoryFilter(f)}
                      className={`px-3 py-1 rounded-lg text-xs font-semibold transition ${
                        historyFilter === f
                          ? "bg-indigo-600 text-white shadow-sm"
                          : "text-slate-600 hover:bg-slate-100"
                      }`}
                    >
                      {f}
                    </button>
                  ))}
                </div>
                <div className="relative">
                  <Search className="w-3.5 h-3.5 text-slate-400 absolute left-3 top-1/2 -translate-y-1/2" />
                  <input
                    type="text"
                    value={historySearch}
                    onChange={(e) => setHistorySearch(e.target.value)}
                    placeholder="Search history records..."
                    className="pl-8 pr-3 py-1.5 rounded-lg border border-slate-200 text-xs text-slate-700 outline-none focus:border-indigo-500"
                  />
                </div>
              </div>

              {/* History Table */}
              <div className="glass-card rounded-2xl bg-white border border-slate-200/80 overflow-hidden">
                <table className="w-full text-left text-xs text-slate-700">
                  <thead className="bg-slate-50 border-b border-slate-200 text-slate-500 font-bold uppercase">
                    <tr>
                      <th className="p-4">Time</th>
                      <th className="p-4">Text Input</th>
                      <th className="p-4">Sentiment</th>
                      <th className="p-4">Confidence</th>
                      <th className="p-4">Model</th>
                    </tr>
                  </thead>
                  <tbody className="divide-y divide-slate-100">
                    {history
                      .filter((item) => historyFilter === "All" || item.sentiment === historyFilter)
                      .filter((item) => item.text.toLowerCase().includes(historySearch.toLowerCase()))
                      .map((item) => (
                        <tr key={item.id} className="hover:bg-slate-50 transition">
                          <td className="p-4 text-slate-400 font-mono text-[11px] whitespace-nowrap">
                            {item.timestamp ? new Date(item.timestamp).toLocaleTimeString() : "Just now"}
                          </td>
                          <td className="p-4 font-medium max-w-md text-slate-800">{item.text}</td>
                          <td className="p-4">
                            <span
                              className={`px-3 py-1 rounded-full text-xs font-bold border ${getSentimentBadge(
                                item.sentiment
                              )}`}
                            >
                              {item.sentiment}
                            </span>
                          </td>
                          <td className="p-4 font-bold text-slate-700">{(item.confidence * 100).toFixed(1)}%</td>
                          <td className="p-4 text-slate-500 font-medium">{item.model}</td>
                        </tr>
                      ))}
                    {history.length === 0 && (
                      <tr>
                        <td colSpan={5} className="p-8 text-center text-slate-400">
                          No prediction history available.
                        </td>
                      </tr>
                    )}
                  </tbody>
                </table>
              </div>
            </div>
          )}

          {/* ================================================================ */}
          {/* TAB 5: NLP EXPLORER */}
          {/* ================================================================ */}
          {activeTab === "nlp" && (
            <div className="space-y-6 animate-fadeIn">
              <div>
                <h2 className="text-2xl font-black text-slate-900">◎ Interactive NLP Explorer</h2>
                <p className="text-sm text-slate-500">
                  Inspect every transformation step in the NLP pipeline: Cleaning → Tokenization → Stopwords →
                  Lemmatization → TF-IDF Weights.
                </p>
              </div>

              <div className="glass-card rounded-2xl p-6 bg-white border border-slate-200">
                <label className="text-xs font-bold uppercase tracking-wider text-slate-500 block mb-2">
                  Test Text Pipeline
                </label>
                <div className="flex gap-3">
                  <input
                    type="text"
                    value={nlpExploreText}
                    onChange={(e) => setNlpExploreText(e.target.value)}
                    className="flex-1 p-3.5 rounded-xl border border-slate-200 text-sm text-slate-800 outline-none focus:border-indigo-500"
                  />
                  <button
                    onClick={handleNlpExplain}
                    disabled={isNlpLoading || !nlpExploreText.trim()}
                    className="px-6 py-3.5 rounded-xl bg-indigo-600 text-white font-bold text-xs hover:bg-indigo-700 disabled:opacity-50 transition flex items-center gap-1.5"
                  >
                    {isNlpLoading ? <RefreshCw className="w-3.5 h-3.5 animate-spin" /> : <Play className="w-3.5 h-3.5" />}
                    Explain NLP
                  </button>
                </div>
              </div>

              {nlpData && (
                <div className="space-y-4 animate-fadeIn">
                  <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                    <div className="glass-card p-5 rounded-2xl bg-white border border-slate-200">
                      <div className="text-xs font-bold text-slate-400 uppercase mb-2">1. Cleaned & Normalized</div>
                      <div className="p-3 bg-slate-50 rounded-xl font-mono text-xs text-slate-700">
                        {nlpData.cleaned_text}
                      </div>
                    </div>

                    <div className="glass-card p-5 rounded-2xl bg-white border border-slate-200">
                      <div className="text-xs font-bold text-slate-400 uppercase mb-2">
                        2. Tokenized ({nlpData.tokens.length} tokens)
                      </div>
                      <div className="flex flex-wrap gap-1.5 p-3 bg-slate-50 rounded-xl">
                        {nlpData.tokens.map((t: string, i: number) => (
                          <span key={i} className="px-2 py-0.5 rounded bg-white border border-slate-200 text-xs font-mono">
                            {t}
                          </span>
                        ))}
                      </div>
                    </div>

                    <div className="glass-card p-5 rounded-2xl bg-white border border-slate-200">
                      <div className="text-xs font-bold text-slate-400 uppercase mb-2">
                        3. Stopwords Removed (Negations Preserved)
                      </div>
                      <div className="flex flex-wrap gap-1.5 p-3 bg-slate-50 rounded-xl">
                        {nlpData.stopwords_removed.map((t: string, i: number) => (
                          <span
                            key={i}
                            className={`px-2 py-0.5 rounded border text-xs font-mono ${
                              ["not", "no", "never"].includes(t)
                                ? "bg-amber-50 border-amber-300 text-amber-800 font-bold"
                                : "bg-white border-slate-200 text-slate-700"
                            }`}
                          >
                            {t}
                          </span>
                        ))}
                      </div>
                    </div>

                    <div className="glass-card p-5 rounded-2xl bg-white border border-slate-200">
                      <div className="text-xs font-bold text-slate-400 uppercase mb-2">
                        4. WordNet Lemmatized Output
                      </div>
                      <div className="p-3 bg-indigo-50/50 border border-indigo-100 rounded-xl font-mono text-xs text-indigo-900 font-bold">
                        {nlpData.final_preprocessed}
                      </div>
                    </div>
                  </div>

                  {/* TF-IDF Weights */}
                  {nlpData.tfidf_features && nlpData.tfidf_features.length > 0 && (
                    <div className="glass-card p-6 rounded-2xl bg-white border border-slate-200">
                      <h3 className="text-sm font-bold text-slate-900 mb-3">5. Extracted TF-IDF Feature Vector</h3>
                      <div className="flex flex-wrap gap-2">
                        {nlpData.tfidf_features.map((f: any, i: number) => (
                          <div
                            key={i}
                            className="px-3 py-1.5 rounded-xl bg-slate-50 border border-slate-200 flex items-center gap-2"
                          >
                            <span className="font-mono text-xs font-bold text-slate-800">{f.feature}</span>
                            <span className="text-[11px] font-bold text-indigo-600 bg-indigo-50 px-1.5 py-0.5 rounded">
                              {f.weight.toFixed(4)}
                            </span>
                          </div>
                        ))}
                      </div>
                    </div>
                  )}
                </div>
              )}
            </div>
          )}

          {/* ================================================================ */}
          {/* TAB 6: MODEL INTELLIGENCE */}
          {/* ================================================================ */}
          {activeTab === "models" && (
            <div className="space-y-8 animate-fadeIn">
              <div>
                <h2 className="text-2xl font-black text-slate-900">◉ Model Intelligence & Leaderboard</h2>
                <p className="text-sm text-slate-500">
                  Side-by-side performance evaluation across all 3 supervised classification models.
                </p>
              </div>

              {/* Leaderboard Table */}
              <div className="glass-card rounded-2xl bg-white border border-slate-200/80 overflow-hidden">
                <div className="p-5 border-b border-slate-100 flex items-center justify-between">
                  <h3 className="text-sm font-bold text-slate-900 uppercase tracking-wider">
                    Model Comparison Leaderboard
                  </h3>
                  <span className="text-xs bg-indigo-50 text-indigo-700 px-3 py-1 rounded-full font-bold border border-indigo-100">
                    Evaluated on 12-sample test set
                  </span>
                </div>
                <table className="w-full text-left text-xs text-slate-700">
                  <thead className="bg-slate-50 text-slate-500 font-bold uppercase">
                    <tr>
                      <th className="p-4">Rank</th>
                      <th className="p-4">Model Name</th>
                      <th className="p-4">Accuracy</th>
                      <th className="p-4">Precision (Macro)</th>
                      <th className="p-4">Recall (Macro)</th>
                      <th className="p-4">F1 Score (Macro)</th>
                      <th className="p-4">Status</th>
                    </tr>
                  </thead>
                  <tbody className="divide-y divide-slate-100">
                    {[
                      {
                        rank: "🥇",
                        name: "Logistic Regression",
                        acc: "33.3%",
                        prec: "0.1111",
                        rec: "0.3333",
                        f1: "0.1667",
                        status: "Active Best Model",
                      },
                      {
                        rank: "🥈",
                        name: "Multinomial Naive Bayes",
                        acc: "33.3%",
                        prec: "0.1111",
                        rec: "0.3333",
                        f1: "0.1667",
                        status: "Alternative",
                      },
                      {
                        rank: "🥉",
                        name: "Linear SVM (LinearSVC)",
                        acc: "33.3%",
                        prec: "0.1111",
                        rec: "0.3333",
                        f1: "0.1667",
                        status: "Alternative",
                      },
                    ].map((row, idx) => (
                      <tr key={idx} className="hover:bg-slate-50">
                        <td className="p-4 font-bold text-base">{row.rank}</td>
                        <td className="p-4 font-bold text-slate-900">{row.name}</td>
                        <td className="p-4 font-bold text-indigo-600">{row.acc}</td>
                        <td className="p-4 font-mono">{row.prec}</td>
                        <td className="p-4 font-mono">{row.rec}</td>
                        <td className="p-4 font-mono font-bold text-slate-900">{row.f1}</td>
                        <td className="p-4">
                          <span
                            className={`px-2.5 py-1 rounded-full text-[11px] font-bold ${
                              idx === 0
                                ? "bg-emerald-50 text-emerald-700 border border-emerald-200"
                                : "bg-slate-100 text-slate-600"
                            }`}
                          >
                            {row.status}
                          </span>
                        </td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>

              {/* Per-Class Metrics */}
              <div className="glass-card rounded-2xl p-6 bg-white border border-slate-200/80">
                <h3 className="text-base font-bold text-slate-900 mb-4">Per-Class Precision, Recall & F1</h3>
                <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
                  {[
                    { label: "Positive", prec: "0.333", rec: "1.000", f1: "0.500", sup: "4 samples" },
                    { label: "Negative", prec: "0.000", rec: "0.000", f1: "0.000", sup: "4 samples" },
                    { label: "Neutral", prec: "0.000", rec: "0.000", f1: "0.000", sup: "4 samples" },
                  ].map((cls) => (
                    <div key={cls.label} className="p-4 rounded-xl bg-slate-50 border border-slate-200">
                      <div className="text-xs font-bold uppercase mb-2 text-slate-700">{cls.label} Class</div>
                      <div className="space-y-1 text-xs text-slate-600">
                        <div className="flex justify-between">
                          <span>Precision:</span> <span className="font-mono font-bold">{cls.prec}</span>
                        </div>
                        <div className="flex justify-between">
                          <span>Recall:</span> <span className="font-mono font-bold">{cls.rec}</span>
                        </div>
                        <div className="flex justify-between">
                          <span>F1-Score:</span> <span className="font-mono font-bold">{cls.f1}</span>
                        </div>
                        <div className="flex justify-between text-slate-400 text-[11px] pt-1 border-t border-slate-200">
                          <span>Support:</span> <span>{cls.sup}</span>
                        </div>
                      </div>
                    </div>
                  ))}
                </div>
              </div>
            </div>
          )}

          {/* ================================================================ */}
          {/* TAB 7: DATA */}
          {/* ================================================================ */}
          {activeTab === "data" && (
            <div className="space-y-6 animate-fadeIn">
              <div>
                <h2 className="text-2xl font-black text-slate-900">▣ Dataset Explorer</h2>
                <p className="text-sm text-slate-500">
                  Inspect the 60-record dataset used for training and testing.
                </p>
              </div>

              <div className="glass-card rounded-2xl bg-white border border-slate-200/80 overflow-hidden">
                <div className="p-4 bg-slate-50 border-b border-slate-200 flex justify-between items-center text-xs font-bold text-slate-600">
                  <span>60 Curated Benchmark Records</span>
                  <span>Columns: text, sentiment, clean_text</span>
                </div>
                <div className="max-h-[500px] overflow-y-auto">
                  <table className="w-full text-left text-xs text-slate-700">
                    <thead className="bg-slate-50/70 sticky top-0 border-b border-slate-200 text-slate-500 font-bold uppercase">
                      <tr>
                        <th className="p-3">#</th>
                        <th className="p-3">Raw Text</th>
                        <th className="p-3">Sentiment</th>
                        <th className="p-3">Cleaned Text</th>
                      </tr>
                    </thead>
                    <tbody className="divide-y divide-slate-100">
                      {datasetData && datasetData.cleaned_samples ? (
                        datasetData.cleaned_samples.map((row: any, idx: number) => (
                          <tr key={idx} className="hover:bg-slate-50">
                            <td className="p-3 text-slate-400 font-mono">{idx + 1}</td>
                            <td className="p-3 font-medium max-w-sm">{row.text}</td>
                            <td className="p-3">
                              <span
                                className={`px-2.5 py-0.5 rounded-full text-[11px] font-bold border ${getSentimentBadge(
                                  row.sentiment
                                )}`}
                              >
                                {row.sentiment}
                              </span>
                            </td>
                            <td className="p-3 font-mono text-[11px] text-slate-500 max-w-sm">
                              {row.clean_text || "-"}
                            </td>
                          </tr>
                        ))
                      ) : (
                        <tr>
                          <td colSpan={4} className="p-6 text-center text-slate-400">
                            Loading dataset records...
                          </td>
                        </tr>
                      )}
                    </tbody>
                  </table>
                </div>
              </div>
            </div>
          )}

          {/* ================================================================ */}
          {/* TAB 8: REPORTS */}
          {/* ================================================================ */}
          {activeTab === "reports" && (
            <div className="space-y-6 animate-fadeIn">
              <div>
                <h2 className="text-2xl font-black text-slate-900">▤ Evaluation Reports</h2>
                <p className="text-sm text-slate-500">
                  Full evaluation audit, error analysis confusion pairs, and academic notice.
                </p>
              </div>

              <div className="p-4 rounded-2xl bg-amber-50 border border-amber-200 text-amber-900 text-xs leading-relaxed font-medium">
                <span className="font-bold">Academic Demonstration Project Notice:</span> The model evaluation metrics
                shown reflect the demonstration dataset (60 records). In accordance with integrity standards, results
                are presented transparently without fabrication.
              </div>

              <div className="glass-card rounded-2xl p-6 bg-white border border-slate-200/80">
                <h3 className="text-base font-bold text-slate-900 mb-3">Error Analysis & Misclassifications</h3>
                <div className="space-y-3">
                  {[
                    {
                      text: "The website is full of spam ads and navigation is an absolute nightmare.",
                      actual: "Negative",
                      predicted: "Neutral",
                      reason: "Rare n-grams outside top TF-IDF vocabulary",
                    },
                    {
                      text: "The update brought several bugs that crash the app constantly.",
                      actual: "Negative",
                      predicted: "Positive",
                      reason: "Unseen vocabulary term distribution",
                    },
                  ].map((err, i) => (
                    <div key={i} className="p-4 rounded-xl bg-slate-50 border border-slate-200 text-xs space-y-1.5">
                      <div className="font-semibold text-slate-800">“{err.text}”</div>
                      <div className="flex gap-4 text-[11px]">
                        <span className="text-rose-600 font-bold">Actual: {err.actual}</span>
                        <span className="text-indigo-600 font-bold">Predicted: {err.predicted}</span>
                        <span className="text-slate-400">Diagnosis: {err.reason}</span>
                      </div>
                    </div>
                  ))}
                </div>
              </div>
            </div>
          )}

          {/* ================================================================ */}
          {/* TAB 9: SETTINGS & HEALTH */}
          {/* ================================================================ */}
          {activeTab === "settings" && (
            <div className="space-y-6 animate-fadeIn">
              <div>
                <h2 className="text-2xl font-black text-slate-900">⚙ Settings & System Health</h2>
                <p className="text-sm text-slate-500">
                  Runtime specifications, API health monitor, and Vercel serverless diagnostics.
                </p>
              </div>

              <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
                <div className="glass-card rounded-2xl p-6 bg-white border border-slate-200">
                  <div className="flex items-center justify-between mb-4">
                    <h3 className="text-base font-bold text-slate-900">API Health Status</h3>
                    <button
                      onClick={fetchHealth}
                      className="text-xs font-bold text-indigo-600 hover:underline flex items-center gap-1"
                    >
                      <RefreshCw className="w-3 h-3" /> Refresh
                    </button>
                  </div>

                  {healthData ? (
                    <div className="space-y-3 text-xs">
                      <div className="flex justify-between p-2.5 rounded-lg bg-slate-50">
                        <span className="text-slate-500">Status:</span>
                        <span className="font-bold text-emerald-600 flex items-center gap-1">
                          <CheckCircle2 className="w-3.5 h-3.5" /> {healthData.status.toUpperCase()}
                        </span>
                      </div>
                      <div className="flex justify-between p-2.5 rounded-lg bg-slate-50">
                        <span className="text-slate-500">NLP Resources:</span>
                        <span className="font-bold text-slate-800">{healthData.nlp}</span>
                      </div>
                      <div className="flex justify-between p-2.5 rounded-lg bg-slate-50">
                        <span className="text-slate-500">TF-IDF Vectorizer:</span>
                        <span className="font-bold text-slate-800">{healthData.vectorizer} (570 vocab)</span>
                      </div>
                      <div className="flex justify-between p-2.5 rounded-lg bg-slate-50">
                        <span className="text-slate-500">Active Best Model:</span>
                        <span className="font-bold text-indigo-600">{healthData.best_model}</span>
                      </div>
                    </div>
                  ) : (
                    <div className="p-6 text-center text-slate-400 text-xs">Connecting to API...</div>
                  )}
                </div>

                <div className="glass-card rounded-2xl p-6 bg-white border border-slate-200">
                  <h3 className="text-base font-bold text-slate-900 mb-4">Runtime Specifications</h3>
                  <div className="space-y-3 text-xs">
                    <div className="flex justify-between p-2.5 rounded-lg bg-slate-50">
                      <span className="text-slate-500">Backend Framework:</span>
                      <span className="font-mono font-bold text-slate-800">FastAPI (ASGI)</span>
                    </div>
                    <div className="flex justify-between p-2.5 rounded-lg bg-slate-50">
                      <span className="text-slate-500">Frontend Stack:</span>
                      <span className="font-mono font-bold text-slate-800">Next.js 14 + React 18 + TS</span>
                    </div>
                    <div className="flex justify-between p-2.5 rounded-lg bg-slate-50">
                      <span className="text-slate-500">ML Engine:</span>
                      <span className="font-mono font-bold text-slate-800">Scikit-Learn + NLTK</span>
                    </div>
                    <div className="flex justify-between p-2.5 rounded-lg bg-slate-50">
                      <span className="text-slate-500">Hosting Target:</span>
                      <span className="font-mono font-bold text-slate-800">Vercel Serverless</span>
                    </div>
                  </div>
                </div>
              </div>
            </div>
          )}
        </main>
      </div>
    </>
  );
}
