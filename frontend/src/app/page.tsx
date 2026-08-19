'use client';

import React, { useState, useEffect } from 'react';
import { TacticalPitch } from '@/components/TacticalPitch';
import { PredictionCard } from '@/components/PredictionCard';
import { BestPassCard } from '@/components/BestPassCard';
import { DefensiveDangerCard } from '@/components/DefensiveDangerCard';
import { WhatIfSimulatorCard } from '@/components/WhatIfSimulatorCard';
import { ExplainabilityCard } from '@/components/ExplainabilityCard';
import { VideoUploadSection } from '@/components/VideoUploadSection';
import {
  FrameTacticalState,
  ActionPrediction,
  PassRecommendation,
  DefensiveCollapseIndex,
  WhatIfResponse,
  ExplainablePrediction,
  GoalExplanation,
  PlayerState,
} from '@/types/football';
import { Play, Pause, RotateCcw, Activity, Shield, Cpu, Video, Layout } from 'lucide-react';

const API_BASE = 'http://127.0.0.1:8000/api';

export default function FootballMindDashboard() {
  const [activeTab, setActiveTab] = useState<'interactive' | 'video_upload'>('interactive');
  const [frameIndex, setFrameIndex] = useState<number>(30);
  const [isPlaying, setIsPlaying] = useState<boolean>(false);
  const [selectedPlayerId, setSelectedPlayerId] = useState<number>(7);
  const [isWhatIfMode, setIsWhatIfMode] = useState<boolean>(false);


  const [frameState, setFrameState] = useState<FrameTacticalState | null>(null);
  const [prediction, setPrediction] = useState<ActionPrediction | null>(null);
  const [passRecs, setPassRecs] = useState<PassRecommendation[]>([]);
  const [defensiveIndex, setDefensiveIndex] = useState<DefensiveCollapseIndex | null>(null);
  const [simulationRes, setSimulationRes] = useState<WhatIfResponse | null>(null);
  const [explainability, setExplainability] = useState<ExplainablePrediction | null>(null);
  const [goalExplanation, setGoalExplanation] = useState<GoalExplanation | null>(null);

  const [apiConnected, setApiConnected] = useState<boolean>(true);

  // Fetch match data for frameIndex
  useEffect(() => {
    async function fetchData() {
      try {
        const [trackRes, predRes, recRes, tactRes, expRes] = await Promise.all([
          fetch(`${API_BASE}/tracking/demo_match_01?frame_index=${frameIndex}`),
          fetch(`${API_BASE}/prediction/demo_match_01?player_id=${selectedPlayerId}&frame_index=${frameIndex}`),
          fetch(`${API_BASE}/recommendations/demo_match_01?player_id=${selectedPlayerId}&frame_index=${frameIndex}`),
          fetch(`${API_BASE}/tactical/demo_match_01?frame_index=${frameIndex}`),
          fetch(`${API_BASE}/explanation/demo_match_01?player_id=${selectedPlayerId}&frame_index=${frameIndex}`),
        ]);

        if (trackRes.ok) {
          setFrameState(await trackRes.json());
          setApiConnected(true);
        }
        if (predRes.ok) {
          const data = await predRes.json();
          setPrediction(data.action_prediction);
        }
        if (recRes.ok) {
          const data = await recRes.json();
          setPassRecs(data.recommendations || []);
        }
        if (tactRes.ok) {
          const data = await tactRes.json();
          setDefensiveIndex(data.defensive_collapse_index);
        }
        if (expRes.ok) {
          const data = await expRes.json();
          setExplainability(data.prediction_explanation);
          setGoalExplanation(data.goal_explanation);
        }
      } catch (err) {
        console.warn('API fetch warning:', err);
        setApiConnected(false);
      }
    }
    fetchData();
  }, [frameIndex, selectedPlayerId]);


  // Handle Play/Pause animation loop
  useEffect(() => {
    let interval: NodeJS.Timeout;
    if (isPlaying) {
      interval = setInterval(() => {
        setFrameIndex((prev) => (prev >= 115 ? 0 : prev + 1));
      }, 200);
    }
    return () => clearInterval(interval);
  }, [isPlaying]);

  // Handle What-If Position Change
  const handlePositionChange = async (id: number, newX: number, newY: number) => {
    if (!frameState) return;

    // Local UI State update
    const updatedPlayers = frameState.players.map((p) =>
      p.id === id ? { ...p, x: newX, y: newY } : p
    );
    setFrameState({ ...frameState, players: updatedPlayers });

    // Call Backend What-If Simulation API
    try {
      const res = await fetch(`${API_BASE}/simulation`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          match_id: 'demo_match_01',
          frame_index: frameIndex,
          modified_player_id: id,
          new_x: newX,
          new_y: newY,
        }),
      });
      if (res.ok) {
        setSimulationRes(await res.json());
      }
    } catch (e) {
      console.warn('Simulation call failed:', e);
    }
  };

  const handleRunPresetScenario = () => {
    handlePositionChange(3, 22.0, 26.0);
  };

  return (
    <div className="min-h-screen bg-slate-950 text-slate-100 p-4 md:p-6 font-sans">
      {/* Header */}
      <header className="mb-6 flex flex-col md:flex-row md:items-center justify-between border-b border-slate-800 pb-4 gap-4">
        <div>
          <div className="flex items-center space-x-3">
            <Cpu className="w-8 h-8 text-emerald-400" />
            <h1 className="text-2xl md:text-3xl font-black tracking-tight text-white uppercase flex items-center gap-3">
              FootballMind <span className="text-emerald-400">AI</span>
              <span className={`text-[10px] normal-case font-mono px-2 py-0.5 rounded-full border ${apiConnected ? 'bg-emerald-950/80 text-emerald-400 border-emerald-800' : 'bg-amber-950/80 text-amber-400 border-amber-800'}`}>
                {apiConnected ? '● FastApi Live' : '○ Standalone / Connecting'}
              </span>
            </h1>
          </div>
          <p className="text-xs text-slate-400 mt-1">
            Multimodal Deep Learning Framework for Predictive, Tactical and Explainable Football Intelligence
          </p>
          <div className="flex items-center space-x-2 bg-slate-900/80 p-1.5 rounded-xl border border-slate-800 mt-2">
            <button
              onClick={() => setActiveTab('interactive')}
              className={`flex items-center space-x-2 px-3 py-1.5 rounded-lg text-xs font-bold transition-all ${
                activeTab === 'interactive'
                  ? 'bg-emerald-500 text-slate-950 shadow-md'
                  : 'text-slate-400 hover:text-white hover:bg-slate-800'
              }`}
            >
              <Layout className="w-4 h-4" />
              <span>Interactive Tactical View</span>
            </button>

            <button
              onClick={() => setActiveTab('video_upload')}
              className={`flex items-center space-x-2 px-3 py-1.5 rounded-lg text-xs font-bold transition-all ${
                activeTab === 'video_upload'
                  ? 'bg-emerald-500 text-slate-950 shadow-md'
                  : 'text-slate-400 hover:text-white hover:bg-slate-800'
              }`}
            >
              <Video className="w-4 h-4" />
              <span>Video Upload & Analysis</span>
            </button>
          </div>
        </div>

        {activeTab === 'interactive' && (
          <div className="flex items-center space-x-3 bg-slate-900/80 p-2 rounded-xl border border-slate-800">
            <button
              onClick={() => setIsPlaying(!isPlaying)}
              className="p-2 rounded-lg bg-emerald-500 hover:bg-emerald-400 text-slate-950 transition-all font-bold"
            >
              {isPlaying ? <Pause className="w-5 h-5" /> : <Play className="w-5 h-5 fill-current" />}
            </button>
            <button
              onClick={() => setFrameIndex(0)}
              className="p-2 rounded-lg bg-slate-800 hover:bg-slate-700 text-slate-300 transition-all"
            >
              <RotateCcw className="w-5 h-5" />
            </button>

            <div className="flex flex-col w-44">
              <div className="flex justify-between text-[11px] text-slate-400 mb-1">
                <span>Frame: {frameIndex}</span>
                <span>Time: {(frameIndex * 0.067).toFixed(1)}s</span>
              </div>
              <input
                type="range"
                min="0"
                max="115"
                value={frameIndex}
                onChange={(e) => setFrameIndex(Number(e.target.value))}
                className="accent-emerald-400 h-1.5 bg-slate-800 rounded-lg cursor-pointer"
              />
            </div>
          </div>
        )}
      </header>

      {/* Main Grid */}
      {activeTab === 'video_upload' ? (
        <VideoUploadSection apiBase={API_BASE} />
      ) : (
        <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
          {/* Left 2 Columns: Pitch & Simulation */}
          <div className="lg:col-span-2 space-y-6">
            {/* Tactical Pitch Component */}
            {frameState ? (
              <TacticalPitch
                players={frameState.players}
                ball={frameState.ball}
                passRecommendations={passRecs}
                selectedPlayerId={selectedPlayerId}
                onSelectPlayer={(id) => setSelectedPlayerId(id)}
                onPositionChange={handlePositionChange}
                isWhatIfMode={isWhatIfMode}
              />
            ) : (
              <div className="w-full aspect-[1000/640] bg-slate-900 rounded-xl flex items-center justify-center border border-slate-800 text-slate-500">
                Loading Tactical Match Frame...
              </div>
            )}

            {/* What-If Simulator */}
            <WhatIfSimulatorCard
              simulation={simulationRes}
              selectedPlayerId={selectedPlayerId}
              onRunPresetScenario={handleRunPresetScenario}
              isWhatIfMode={isWhatIfMode}
              setIsWhatIfMode={setIsWhatIfMode}
            />

            {/* Explainability Card */}
            <ExplainabilityCard
              explainability={explainability}
              goalExplanation={goalExplanation}
            />
          </div>

          {/* Right 1 Column: Intelligence Cards */}
          <div className="space-y-6">
            {/* Next Action Intention */}
            <PredictionCard prediction={prediction} playerId={selectedPlayerId} />

            {/* Best Pass Recommender */}
            <BestPassCard recommendations={passRecs} />

            {/* Defensive Collapse Index */}
            <DefensiveDangerCard defensiveIndex={defensiveIndex} />
          </div>
        </div>
      )}
    </div>
  );
}

