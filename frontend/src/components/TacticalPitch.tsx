'use client';

import React, { useState } from 'react';
import { PlayerState, BallState, PassRecommendation } from '@/types/football';

interface TacticalPitchProps {
  players: PlayerState[];
  ball: BallState;
  passRecommendations?: PassRecommendation[];
  selectedPlayerId?: number | null;
  onSelectPlayer?: (id: number) => void;
  onPositionChange?: (id: number, newX: number, newY: number) => void;
  isWhatIfMode?: boolean;
}

export const TacticalPitch: React.FC<TacticalPitchProps> = ({
  players,
  ball,
  passRecommendations = [],
  selectedPlayerId,
  onSelectPlayer,
  onPositionChange,
  isWhatIfMode = false,
}) => {
  const pitchWidth = 105;
  const pitchHeight = 68;

  const toSvgX = (x: number) => (x / pitchWidth) * 1000;
  const toSvgY = (y: number) => (y / pitchHeight) * 640;

  const [draggingId, setDraggingId] = useState<number | null>(null);

  const handleMouseDown = (id: number) => {
    if (isWhatIfMode) {
      setDraggingId(id);
      if (onSelectPlayer) onSelectPlayer(id);
    } else if (onSelectPlayer) {
      onSelectPlayer(id);
    }
  };

  const handleMouseMove = (e: React.MouseEvent<SVGSVGElement>) => {
    if (!draggingId || !isWhatIfMode || !onPositionChange) return;
    const rect = e.currentTarget.getBoundingClientRect();
    const clickX = e.clientX - rect.left;
    const clickY = e.clientY - rect.top;

    const normX = Math.max(1, Math.min(104, (clickX / rect.width) * pitchWidth));
    const normY = Math.max(1, Math.min(67, (clickY / rect.height) * pitchHeight));

    onPositionChange(draggingId, Number(normX.toFixed(1)), Number(normY.toFixed(1)));
  };

  const handleMouseUp = () => {
    setDraggingId(null);
  };

  return (
    <div className="relative w-full aspect-[1000/640] bg-emerald-950 rounded-xl overflow-hidden shadow-2xl border border-emerald-800/40">
      <svg
        viewBox="0 0 1000 640"
        className="w-full h-full select-none cursor-crosshair"
        onMouseMove={handleMouseMove}
        onMouseUp={handleMouseUp}
        onMouseLeave={handleMouseUp}
      >
        {/* Pitch Turf & Stripes */}
        <rect width="1000" height="640" fill="#042f1a" />
        {[...Array(10)].map((_, i) => (
          <rect
            key={i}
            x={i * 100}
            y="0"
            width="50"
            height="640"
            fill="#063c22"
            opacity="0.5"
          />
        ))}

        {/* Pitch Boundary Lines */}
        <rect x="30" y="30" width="940" height="580" fill="none" stroke="#22c55e" strokeWidth="3" opacity="0.6" />
        
        {/* Halfway Line & Center Circle */}
        <line x1="500" y1="30" x2="500" y2="610" stroke="#22c55e" strokeWidth="3" opacity="0.6" />
        <circle cx="500" cy="320" r="90" fill="none" stroke="#22c55e" strokeWidth="3" opacity="0.6" />
        <circle cx="500" cy="320" r="4" fill="#22c55e" opacity="0.8" />

        {/* Penalty Areas */}
        {/* Left Box (Home GK) */}
        <rect x="30" y="140" width="165" height="360" fill="none" stroke="#22c55e" strokeWidth="3" opacity="0.6" />
        <rect x="30" y="230" width="55" height="180" fill="none" stroke="#22c55e" strokeWidth="3" opacity="0.6" />
        {/* Right Box (Away GK) */}
        <rect x="805" y="140" width="165" height="360" fill="none" stroke="#22c55e" strokeWidth="3" opacity="0.6" />
        <rect x="915" y="230" width="55" height="180" fill="none" stroke="#22c55e" strokeWidth="3" opacity="0.6" />

        {/* Goal Frames */}
        <rect x="15" y="280" width="15" height="80" fill="none" stroke="#ffffff" strokeWidth="4" />
        <rect x="970" y="280" width="15" height="80" fill="none" stroke="#ffffff" strokeWidth="4" />

        {/* Pass Recommendation Vectors */}
        {passRecommendations.map((pass, idx) => {
          const sx = toSvgX(pass.start_pos[0]);
          const sy = toSvgY(pass.start_pos[1]);
          const ex = toSvgX(pass.end_pos[0]);
          const ey = toSvgY(pass.end_pos[1]);

          return (
            <g key={idx}>
              <line
                x1={sx}
                y1={sy}
                x2={ex}
                y2={ey}
                stroke={idx === 0 ? '#38bdf8' : '#34d399'}
                strokeWidth={idx === 0 ? '4' : '2'}
                strokeDasharray="8 4"
                className="animate-pulse"
              />
              <circle cx={ex} cy={ey} r="8" fill="none" stroke={idx === 0 ? '#38bdf8' : '#34d399'} strokeWidth="2" />
            </g>
          );
        })}

        {/* Players */}
        {players.map((p) => {
          const cx = toSvgX(p.x);
          const cy = toSvgY(p.y);
          const isSelected = selectedPlayerId === p.id;
          const isHome = p.team === 'home';

          return (
            <g
              key={p.id}
              className="cursor-pointer transition-all duration-100"
              onMouseDown={() => handleMouseDown(p.id)}
            >
              {/* Highlight selection */}
              {isSelected && (
                <circle cx={cx} cy={cy} r="22" fill="none" stroke="#f59e0b" strokeWidth="3" className="animate-ping opacity-75" />
              )}

              {/* Player Node */}
              <circle
                cx={cx}
                cy={cy}
                r="14"
                fill={isHome ? '#ef4444' : '#3b82f6'}
                stroke={isSelected ? '#f59e0b' : '#ffffff'}
                strokeWidth="2.5"
                className="drop-shadow-lg"
              />

              {/* Player Jersey ID */}
              <text
                x={cx}
                y={cy + 4}
                textAnchor="middle"
                fontSize="11"
                fontWeight="bold"
                fill="#ffffff"
              >
                {p.id}
              </text>
            </g>
          );
        })}

        {/* Ball Node */}
        {ball && (
          <g>
            <circle cx={toSvgX(ball.x)} cy={toSvgY(ball.y)} r="12" fill="#eab308" opacity="0.3" className="animate-pulse" />
            <circle cx={toSvgX(ball.x)} cy={toSvgY(ball.y)} r="7" fill="#ffffff" stroke="#15803d" strokeWidth="2" />
          </g>
        )}
      </svg>

      {/* Mode Overlay Legend */}
      <div className="absolute top-3 left-3 bg-slate-900/80 backdrop-blur-md px-3 py-1.5 rounded-lg border border-slate-700/50 text-xs text-slate-300 flex items-center space-x-3">
        <span className="flex items-center space-x-1"><span className="w-2.5 h-2.5 rounded-full bg-red-500 inline-block"></span> <span>Home Team</span></span>
        <span className="flex items-center space-x-1"><span className="w-2.5 h-2.5 rounded-full bg-blue-500 inline-block"></span> <span>Away Team</span></span>
        <span className="flex items-center space-x-1"><span className="w-2.5 h-2.5 rounded-full bg-amber-400 inline-block"></span> <span>Ball</span></span>
        {isWhatIfMode && <span className="text-amber-400 font-bold ml-2">[WHAT-IF EDIT MODE ACTIVE]</span>}
      </div>
    </div>
  );
};
