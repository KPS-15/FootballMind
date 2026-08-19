'use client';

import React from 'react';
import { PassRecommendation } from '@/types/football';
import { Compass, CheckCircle2 } from 'lucide-react';

interface BestPassCardProps {
  recommendations: PassRecommendation[];
}

export const BestPassCard: React.FC<BestPassCardProps> = ({ recommendations }) => {
  return (
    <div className="bg-slate-900/90 border border-slate-800 rounded-xl p-4 shadow-xl backdrop-blur-md">
      <div className="flex items-center justify-between mb-3 border-b border-slate-800 pb-2">
        <div className="flex items-center space-x-2">
          <Compass className="w-5 h-5 text-emerald-400" />
          <h3 className="text-sm font-semibold text-slate-200 uppercase tracking-wider">Best Pass Recommendations</h3>
        </div>
      </div>

      <div className="space-y-2.5">
        {recommendations.slice(0, 3).map((pass, idx) => (
          <div
            key={idx}
            className={`p-3 rounded-lg border transition-all ${
              idx === 0
                ? 'bg-emerald-950/40 border-emerald-500/50 shadow-lg shadow-emerald-950/50'
                : 'bg-slate-800/40 border-slate-700/50'
            }`}
          >
            <div className="flex items-center justify-between mb-1.5">
              <div className="flex items-center space-x-2">
                {idx === 0 && <CheckCircle2 className="w-4 h-4 text-emerald-400" />}
                <span className="font-bold text-white text-sm">Target Receiver #{pass.receiver_id}</span>
              </div>
              <span className="font-mono text-xs font-semibold px-2 py-0.5 rounded bg-emerald-950 text-emerald-400 border border-emerald-800">
                Score: {pass.score}
              </span>
            </div>

            <div className="grid grid-cols-3 gap-2 text-[11px] text-slate-300">
              <div className="bg-slate-900/60 p-1.5 rounded">
                <span className="text-slate-400 block">Success</span>
                <span className="font-mono font-semibold text-emerald-400">{(pass.success_probability * 100).toFixed(0)}%</span>
              </div>
              <div className="bg-slate-900/60 p-1.5 rounded">
                <span className="text-slate-400 block">Advantage</span>
                <span className="font-mono font-semibold text-sky-400">+{pass.attacking_advantage.toFixed(2)}</span>
              </div>
              <div className="bg-slate-900/60 p-1.5 rounded">
                <span className="text-slate-400 block">Space</span>
                <span className="font-mono font-semibold text-amber-400">{pass.space_created.toFixed(1)}m</span>
              </div>
            </div>
          </div>
        ))}
      </div>
    </div>
  );
};
