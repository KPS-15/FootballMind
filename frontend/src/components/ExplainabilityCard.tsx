'use client';

import React from 'react';
import { ExplainablePrediction, GoalExplanation } from '@/types/football';
import { BrainCircuit, Info } from 'lucide-react';

interface ExplainabilityCardProps {
  explainability?: ExplainablePrediction | null;
  goalExplanation?: GoalExplanation | null;
}

export const ExplainabilityCard: React.FC<ExplainabilityCardProps> = ({
  explainability,
  goalExplanation,
}) => {
  return (
    <div className="bg-slate-900/90 border border-slate-800 rounded-xl p-4 shadow-xl backdrop-blur-md">
      <div className="flex items-center justify-between mb-3 border-b border-slate-800 pb-2">
        <div className="flex items-center space-x-2">
          <BrainCircuit className="w-5 h-5 text-indigo-400" />
          <h3 className="text-sm font-semibold text-slate-200 uppercase tracking-wider">Explainable AI & Goal Analysis</h3>
        </div>
      </div>

      {explainability && (
        <div className="mb-4">
          <h4 className="text-xs font-semibold text-indigo-300 uppercase tracking-wider mb-2 flex items-center space-x-1">
            <Info className="w-3.5 h-3.5" />
            <span>Prediction Rationale (SHAP Attributions)</span>
          </h4>
          <p className="text-xs text-slate-300 bg-slate-950/60 p-2.5 rounded border border-slate-800 mb-3 italic">
            "{explainability.narrative_reason}"
          </p>

          <div className="space-y-1.5">
            {explainability.top_features.map((feat, idx) => (
              <div key={idx} className="flex justify-between items-center text-xs bg-slate-800/40 p-1.5 rounded">
                <span className="text-slate-300 font-medium">{feat.feature_name}</span>
                <span
                  className={`font-mono font-semibold ${
                    feat.contribution >= 0 ? 'text-emerald-400' : 'text-red-400'
                  }`}
                >
                  {feat.contribution >= 0 ? '+' : ''}{feat.contribution.toFixed(2)}
                </span>
              </div>
            ))}
          </div>
        </div>
      )}

      {goalExplanation && (
        <div className="border-t border-slate-800 pt-3">
          <h4 className="text-xs font-semibold text-red-400 uppercase tracking-wider mb-1.5">Goal Cause Reconstruction</h4>
          <div className="bg-red-950/30 border border-red-900/50 p-2.5 rounded text-xs text-slate-200 space-y-1">
            <p className="font-semibold text-red-300">Primary Cause: {goalExplanation.primary_cause}</p>
            <p className="text-slate-400 text-[11px]">Recommended Counteraction: {goalExplanation.alternative_counteraction}</p>
          </div>
        </div>
      )}
    </div>
  );
};
