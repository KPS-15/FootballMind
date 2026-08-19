'use client';

import React from 'react';
import { DefensiveCollapseIndex } from '@/types/football';
import { ShieldAlert } from 'lucide-react';

interface DefensiveDangerCardProps {
  defensiveIndex?: DefensiveCollapseIndex | null;
}

export const DefensiveDangerCard: React.FC<DefensiveDangerCardProps> = ({ defensiveIndex }) => {
  if (!defensiveIndex) return null;

  const dangerPercent = Math.round(defensiveIndex.overall_danger * 100);

  return (
    <div className="bg-slate-900/90 border border-slate-800 rounded-xl p-4 shadow-xl backdrop-blur-md">
      <div className="flex items-center justify-between mb-3 border-b border-slate-800 pb-2">
        <div className="flex items-center space-x-2">
          <ShieldAlert className="w-5 h-5 text-red-400" />
          <h3 className="text-sm font-semibold text-slate-200 uppercase tracking-wider">Defensive Collapse Danger</h3>
        </div>
        <span className={`text-xs font-bold px-2.5 py-0.5 rounded border font-mono ${
          dangerPercent > 60 ? 'bg-red-950 text-red-400 border-red-800' : 'bg-amber-950 text-amber-400 border-amber-800'
        }`}>
          {dangerPercent}% DANGER
        </span>
      </div>

      {/* Main Meter */}
      <div className="mb-4">
        <div className="w-full bg-slate-800 h-3 rounded-full overflow-hidden p-0.5 border border-slate-700">
          <div
            className={`h-full rounded-full transition-all duration-500 ${
              dangerPercent > 60 ? 'bg-gradient-to-r from-amber-500 to-red-500' : 'bg-gradient-to-r from-emerald-500 to-amber-400'
            }`}
            style={{ width: `${dangerPercent}%` }}
          />
        </div>
      </div>

      {/* Breakdown Factors */}
      <div className="space-y-2 text-xs">
        <div>
          <div className="flex justify-between text-slate-300 mb-1">
            <span>CB-LB Gap Risk</span>
            <span className="font-mono text-red-400 font-bold">{(defensiveIndex.cb_lb_gap_risk * 100).toFixed(0)}%</span>
          </div>
          <div className="w-full bg-slate-800 h-1.5 rounded-full overflow-hidden">
            <div className="bg-red-500 h-full" style={{ width: `${defensiveIndex.cb_lb_gap_risk * 100}%` }} />
          </div>
        </div>

        <div>
          <div className="flex justify-between text-slate-300 mb-1">
            <span>Passing Lane Exposure</span>
            <span className="font-mono text-amber-400 font-bold">{(defensiveIndex.passing_lane_exposure * 100).toFixed(0)}%</span>
          </div>
          <div className="w-full bg-slate-800 h-1.5 rounded-full overflow-hidden">
            <div className="bg-amber-400 h-full" style={{ width: `${defensiveIndex.passing_lane_exposure * 100}%` }} />
          </div>
        </div>

        <div>
          <div className="flex justify-between text-slate-300 mb-1">
            <span>Unmarked Attacker Risk</span>
            <span className="font-mono text-orange-400 font-bold">{(defensiveIndex.unmarked_attacker_risk * 100).toFixed(0)}%</span>
          </div>
          <div className="w-full bg-slate-800 h-1.5 rounded-full overflow-hidden">
            <div className="bg-orange-400 h-full" style={{ width: `${defensiveIndex.unmarked_attacker_risk * 100}%` }} />
          </div>
        </div>
      </div>
    </div>
  );
};
