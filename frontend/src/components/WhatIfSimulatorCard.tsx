'use client';

import React from 'react';
import { WhatIfResponse } from '@/types/football';
import { Sliders, ArrowRight } from 'lucide-react';

interface WhatIfSimulatorCardProps {
  simulation?: WhatIfResponse | null;
  selectedPlayerId: number;
  onRunPresetScenario: () => void;
  isWhatIfMode: boolean;
  setIsWhatIfMode: (val: boolean) => void;
}

export const WhatIfSimulatorCard: React.FC<WhatIfSimulatorCardProps> = ({
  simulation,
  selectedPlayerId,
  onRunPresetScenario,
  isWhatIfMode,
  setIsWhatIfMode,
}) => {
  return (
    <div className="bg-slate-900/90 border border-slate-800 rounded-xl p-4 shadow-xl backdrop-blur-md">
      <div className="flex items-center justify-between mb-3 border-b border-slate-800 pb-2">
        <div className="flex items-center space-x-2">
          <Sliders className="w-5 h-5 text-amber-400" />
          <h3 className="text-sm font-semibold text-slate-200 uppercase tracking-wider">What-If Tactical Simulator</h3>
        </div>
        <button
          onClick={() => setIsWhatIfMode(!isWhatIfMode)}
          className={`text-xs font-semibold px-3 py-1 rounded transition-colors border ${
            isWhatIfMode
              ? 'bg-amber-500 text-slate-950 border-amber-400 font-bold shadow-lg shadow-amber-500/30'
              : 'bg-slate-800 text-slate-300 border-slate-700 hover:bg-slate-700'
          }`}
        >
          {isWhatIfMode ? 'Exit What-If Mode' : 'Enter Drag Edit Mode'}
        </button>
      </div>

      <div className="mb-3 text-xs text-slate-400">
        Click or drag player positions on the tactical pitch to test counterfactual scenarios and observe real-time defensive collapse & xG impact.
      </div>

      <div className="mb-3">
        <button
          onClick={onRunPresetScenario}
          className="w-full bg-slate-800 hover:bg-slate-700 text-amber-300 text-xs font-semibold py-2 px-3 rounded-lg border border-slate-700 flex items-center justify-center space-x-2 transition-all"
        >
          <span>Run Preset Scenario: Left-Back Inward Shift (3.2m)</span>
          <ArrowRight className="w-3.5 h-3.5" />
        </button>
      </div>

      {simulation && (
        <div className="bg-slate-950/80 p-3 rounded-lg border border-slate-800 space-y-2">
          <div className="flex justify-between items-center text-xs">
            <span className="text-slate-400">Baseline Danger:</span>
            <span className="font-mono text-slate-200">{(simulation.baseline_danger * 100).toFixed(0)}%</span>
          </div>
          <div className="flex justify-between items-center text-xs">
            <span className="text-slate-400">Scenario Danger:</span>
            <span className="font-mono text-amber-400 font-bold">{(simulation.scenario_danger * 100).toFixed(0)}%</span>
          </div>
          <div className="flex justify-between items-center text-xs border-t border-slate-800 pt-1">
            <span className="text-slate-300 font-semibold">Danger Delta Impact:</span>
            <span
              className={`font-mono font-bold ${
                simulation.danger_delta <= 0 ? 'text-emerald-400' : 'text-red-400'
              }`}
            >
              {simulation.danger_delta <= 0 ? '' : '+'}{(simulation.danger_delta * 100).toFixed(1)} pp
            </span>
          </div>

          <div className="text-[11px] text-slate-400 italic pt-1 border-t border-slate-800/60">
            {simulation.summary}
          </div>
        </div>
      )}
    </div>
  );
};
