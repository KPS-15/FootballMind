'use client';

import React from 'react';
import { ActionPrediction } from '@/types/football';
import { Target, Zap } from 'lucide-react';

interface PredictionCardProps {
  prediction?: ActionPrediction | null;
  playerId: number;
}

export const PredictionCard: React.FC<PredictionCardProps> = ({ prediction, playerId }) => {
  if (!prediction) return null;

  return (
    <div className="bg-slate-900/90 border border-slate-800 rounded-xl p-4 shadow-xl backdrop-blur-md">
      <div className="flex items-center justify-between mb-3 border-b border-slate-800 pb-2">
        <div className="flex items-center space-x-2">
          <Target className="w-5 h-5 text-sky-400" />
          <h3 className="text-sm font-semibold text-slate-200 uppercase tracking-wider">Next Action Intention</h3>
        </div>
        <span className="text-xs px-2 py-0.5 rounded bg-sky-950 text-sky-400 font-mono border border-sky-800">
          Player #{playerId}
        </span>
      </div>

      <div className="mb-4">
        <div className="flex items-center justify-between mb-1.5">
          <span className="text-2xl font-black text-white tracking-wide">{prediction.action}</span>
          <span className="text-lg font-bold text-emerald-400 font-mono">
            {(prediction.confidence * 100).toFixed(0)}%
          </span>
        </div>

        {/* Confidence Bar */}
        <div className="w-full bg-slate-800 h-2.5 rounded-full overflow-hidden mb-2">
          <div
            className="bg-gradient-to-r from-emerald-500 to-sky-400 h-full rounded-full transition-all duration-500"
            style={{ width: `${prediction.confidence * 100}%` }}
          />
        </div>

        {/* Model Badge & Calibration Status */}
        <div className="flex items-center justify-between text-[11px]">
          <span className="text-slate-400 font-medium">
            {prediction.model_type || 'FootballTemporalLSTM Neural Model'}
          </span>
          <span className="text-emerald-400 font-mono bg-emerald-950/60 px-1.5 py-0.5 rounded border border-emerald-800/60">
            {prediction.calibration_status || 'Calibrated'}
          </span>
        </div>
      </div>


      {/* Alternatives */}
      <div className="space-y-1.5">
        <span className="text-xs text-slate-400 uppercase font-semibold">Alternative Choices</span>
        {prediction.alternatives.map((alt, idx) => (
          <div key={idx} className="flex justify-between items-center text-xs py-1 px-2 rounded bg-slate-800/50">
            <span className="text-slate-300 font-medium">{alt.action}</span>
            <span className="text-slate-400 font-mono">{(alt.confidence * 100).toFixed(0)}%</span>
          </div>
        ))}
      </div>
    </div>
  );
};
