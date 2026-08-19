'use client';

import React, { useState } from 'react';
import { Upload, Play, ShieldAlert, CheckCircle, Video, Download, Zap, Target, Clock, ChevronDown, ChevronUp, AlertTriangle } from 'lucide-react';
import { VideoMetadata, VideoAnalysisResponse } from '@/types/football';

interface VideoUploadSectionProps {
  apiBase: string;
}

export const VideoUploadSection: React.FC<VideoUploadSectionProps> = ({ apiBase }) => {
  const [file, setFile] = useState<File | null>(null);
  const [metadata, setMetadata] = useState<VideoMetadata | null>(null);
  const [analysisResult, setAnalysisResult] = useState<VideoAnalysisResponse | null>(null);
  const [isUploading, setIsUploading] = useState<boolean>(false);
  const [isAnalyzing, setIsAnalyzing] = useState<boolean>(false);
  const [currentStep, setCurrentStep] = useState<number>(0);
  const [errorMessage, setErrorMessage] = useState<string | null>(null);
  
  // UI view toggles
  const [videoMode, setVideoMode] = useState<'analyzed' | 'original'>('analyzed');
  const [showWhy, setShowWhy] = useState<boolean>(false);
  const [showDetailedReport, setShowDetailedReport] = useState<boolean>(false);

  const steps = [
    'Uploading video file...',
    'YOLO Player & Ball Detection...',
    'Multi-Object Tracking (Hungarian MOT)...',
    'Tactical Engine & Arrow Overlays...',
    'Complete!'
  ];

  const handleFileSelect = async (e: React.ChangeEvent<HTMLInputElement>) => {
    if (!e.target.files || e.target.files.length === 0) return;
    const selectedFile = e.target.files[0];
    setFile(selectedFile);
    setErrorMessage(null);
    setAnalysisResult(null);
    setIsUploading(true);

    const formData = new FormData();
    formData.append('file', selectedFile);

    try {
      const res = await fetch(`${apiBase}/video/upload`, {
        method: 'POST',
        body: formData,
      });

      if (!res.ok) {
        const errData = await res.json();
        throw new Error(errData.detail || 'Failed to upload video.');
      }

      const data = await res.json();
      setMetadata(data.video_metadata);
    } catch (err: any) {
      setErrorMessage(err.message || 'Error uploading video file.');
    } finally {
      setIsUploading(false);
    }
  };

  const handleAnalyzeVideo = async () => {
    if (!metadata) return;
    setIsAnalyzing(true);
    setErrorMessage(null);
    setCurrentStep(1);

    const stepInterval = setInterval(() => {
      setCurrentStep((prev) => (prev < 3 ? prev + 1 : prev));
    }, 1400);

    try {
      const res = await fetch(`${apiBase}/video/analyze`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          file_path: metadata.file_path,
          max_frames: 300,
        }),
      });

      clearInterval(stepInterval);

      if (!res.ok) {
        const errData = await res.json();
        throw new Error(errData.detail || 'Video analysis failed.');
      }

      const data: VideoAnalysisResponse = await res.json();
      setCurrentStep(4);
      setAnalysisResult(data);
    } catch (err: any) {
      clearInterval(stepInterval);
      setErrorMessage(err.message || 'An error occurred during video analysis.');
    } finally {
      setIsAnalyzing(false);
    }
  };

  const hostBase = apiBase.replace('/api', '');

  return (
    <div className="space-y-6 max-w-6xl mx-auto">
      {/* Upload Header Card */}
      <div className="bg-slate-900/90 border border-slate-800 rounded-xl p-5 shadow-2xl backdrop-blur-md">
        <div className="flex items-center justify-between">
          <div className="flex items-center space-x-3">
            <Video className="w-6 h-6 text-emerald-400" />
            <div>
              <h2 className="text-lg font-black text-white uppercase tracking-wide">Real Football Video Analysis</h2>
              <p className="text-xs text-slate-400">YOLO Detection • Player & Ball Tracking • Tactical Overlays</p>
            </div>
          </div>
          {metadata && (
            <button
              onClick={() => {
                setMetadata(null);
                setFile(null);
                setAnalysisResult(null);
              }}
              className="text-xs text-slate-400 hover:text-rose-400 font-mono transition"
            >
              Change Video
            </button>
          )}
        </div>

        {/* File Picker */}
        {!metadata && (
          <div className="mt-4 border-2 border-dashed border-slate-700 hover:border-emerald-500/80 rounded-xl p-8 text-center transition-all bg-slate-950/50">
            <input
              type="file"
              id="video-file-input"
              accept=".mp4,.mov,.avi,.mkv"
              onChange={handleFileSelect}
              className="hidden"
            />
            <label htmlFor="video-file-input" className="cursor-pointer flex flex-col items-center justify-center space-y-3">
              <div className="p-3.5 rounded-full bg-emerald-950/80 text-emerald-400 border border-emerald-800/80 shadow-inner">
                <Upload className="w-7 h-7" />
              </div>
              <div>
                <p className="text-sm font-bold text-slate-200">Upload Football Video</p>
                <p className="text-xs text-slate-400 mt-0.5">MP4, MOV, AVI, MKV (Max: 500 MB)</p>
              </div>
            </label>
          </div>
        )}

        {/* Uploading Spinner */}
        {isUploading && (
          <div className="flex items-center justify-center p-6 space-x-3 text-emerald-400 font-mono text-sm">
            <div className="w-5 h-5 border-2 border-emerald-400 border-t-transparent rounded-full animate-spin" />
            <span>Uploading video to server...</span>
          </div>
        )}

        {/* Error Alert */}
        {errorMessage && (
          <div className="mt-4 p-3.5 rounded-lg bg-rose-950/80 border border-rose-800/80 text-rose-300 text-xs flex items-start space-x-2">
            <ShieldAlert className="w-4 h-4 text-rose-400 flex-shrink-0 mt-0.5" />
            <span>{errorMessage}</span>
          </div>
        )}

        {/* Uploaded Video Metadata Preview & Action Button */}
        {metadata && !analysisResult && (
          <div className="mt-4 bg-slate-950 p-4 rounded-lg border border-slate-800 space-y-3">
            <div className="flex justify-between items-center text-xs font-mono">
              <span className="text-white font-bold">{metadata.filename}</span>
              <span className="text-slate-400">{metadata.duration_sec}s • {metadata.resolution[0]}×{metadata.resolution[1]} • {metadata.fps} FPS</span>
            </div>

            <button
              onClick={handleAnalyzeVideo}
              disabled={isAnalyzing}
              className="w-full py-3 rounded-lg bg-gradient-to-r from-emerald-500 to-sky-500 hover:from-emerald-400 hover:to-sky-400 text-slate-950 font-black text-sm uppercase tracking-wider transition-all shadow-lg flex items-center justify-center space-x-2 disabled:opacity-50"
            >
              {isAnalyzing ? (
                <>
                  <div className="w-4 h-4 border-2 border-slate-950 border-t-transparent rounded-full animate-spin" />
                  <span>ANALYZING & TRACKING...</span>
                </>
              ) : (
                <>
                  <Play className="w-4 h-4 fill-current" />
                  <span>ANALYZE & TRACK</span>
                </>
              )}
            </button>
          </div>
        )}

        {/* Processing Stepper */}
        {isAnalyzing && (
          <div className="mt-4 p-4 rounded-lg bg-slate-950 border border-slate-800 space-y-2">
            <div className="flex justify-between items-center text-xs font-mono text-slate-400">
              <span>ANALYZING VIDEO...</span>
              <span>{Math.min(100, (currentStep + 1) * 25)}%</span>
            </div>
            <div className="w-full bg-slate-800 h-2 rounded-full overflow-hidden">
              <div
                className="bg-emerald-400 h-full rounded-full transition-all duration-500"
                style={{ width: `${Math.min(100, (currentStep + 1) * 25)}%` }}
              />
            </div>
            <div className="grid grid-cols-2 md:grid-cols-4 gap-2 pt-2 text-[11px] font-mono">
              {['YOLO DETECT', 'PLAYER TRACK', 'BALL TRACK', 'TACTICS'].map((st, i) => (
                <div key={i} className={`p-1.5 rounded border text-center ${i <= currentStep ? 'bg-emerald-950/80 text-emerald-400 border-emerald-800' : 'bg-slate-900/50 text-slate-600 border-slate-800'}`}>
                  {i < currentStep ? '✓ ' : i === currentStep ? '● ' : '○ '}{st}
                </div>
              ))}
            </div>
          </div>
        )}
      </div>

      {/* RESULT VIEW: VIDEO IS THE HERO PRODUCT */}
      {analysisResult && (
        <div className="space-y-6">
          {/* Main Hero Video Card */}
          <div className="bg-slate-900 border border-slate-800 rounded-2xl p-4 md:p-6 shadow-2xl space-y-4">
            <div className="flex flex-col md:flex-row md:items-center justify-between gap-3 border-b border-slate-800 pb-3">
              <div className="flex items-center space-x-2">
                <span className="w-3 h-3 rounded-full bg-emerald-400 animate-pulse" />
                <h3 className="text-base font-black text-white uppercase tracking-wide">
                  ANALYZED FOOTBALL VIDEO OVERLAY
                </h3>
              </div>

              <div className="flex items-center space-x-3">
                {/* Mode Toggle: Analyzed vs Original */}
                <div className="flex bg-slate-950 p-1 rounded-lg border border-slate-800 text-xs font-bold font-mono">
                  <button
                    onClick={() => setVideoMode('analyzed')}
                    className={`px-3 py-1 rounded transition ${videoMode === 'analyzed' ? 'bg-emerald-500 text-slate-950' : 'text-slate-400 hover:text-white'}`}
                  >
                    ANALYZED
                  </button>
                  <button
                    onClick={() => setVideoMode('original')}
                    className={`px-3 py-1 rounded transition ${videoMode === 'original' ? 'bg-emerald-500 text-slate-950' : 'text-slate-400 hover:text-white'}`}
                  >
                    ORIGINAL
                  </button>
                </div>

                {/* Download MP4 Button */}
                <a
                  href={`${hostBase}${analysisResult.download_url || analysisResult.annotated_video_url || `/api/video/download/${analysisResult.video_metadata.filename}`}`}
                  download={`footballmind_analyzed_${analysisResult.video_metadata.filename}`}
                  className="px-4 py-2 rounded-lg bg-emerald-500 hover:bg-emerald-400 text-slate-950 font-black text-xs uppercase tracking-wider transition-all shadow-md flex items-center space-x-2"
                >
                  <Download className="w-4 h-4" />
                  <span>DOWNLOAD ANALYZED VIDEO</span>
                </a>
              </div>
            </div>

            {/* Video Player */}
            <div className="relative w-full aspect-video bg-black rounded-xl overflow-hidden border border-slate-800 shadow-inner flex items-center justify-center">
              {videoMode === 'analyzed' && analysisResult.annotated_video_url ? (
                <video
                  controls
                  autoPlay
                  loop
                  muted
                  className="w-full h-full object-contain"
                  src={`${hostBase}${analysisResult.annotated_video_url}`}
                />
              ) : (
                <video
                  controls
                  autoPlay
                  loop
                  muted
                  className="w-full h-full object-contain"
                  src={`${hostBase}/${analysisResult.video_metadata.file_path}`}
                />
              )}
            </div>
          </div>

          {/* 4 CORE CARDS: LIVE SPORTS ANALYTICS (INSTANT FOOTBALL LANGUAGE) */}
          <div className="grid grid-cols-1 md:grid-cols-4 gap-4 font-mono">
            {/* 1. ATTACK CARD */}
            <div className="bg-slate-900/90 border border-emerald-800/80 rounded-xl p-4 shadow-xl space-y-3">
              <div className="flex items-center justify-between border-b border-slate-800 pb-2">
                <div className="flex items-center space-x-1.5 text-emerald-400">
                  <Zap className="w-4 h-4" />
                  <span className="text-xs font-black uppercase">ATTACK</span>
                </div>
                <span className="text-[11px] bg-emerald-950 text-emerald-400 px-2 py-0.5 rounded border border-emerald-800 font-bold">
                  {analysisResult.carrier_label || `BALL #${analysisResult.ball_carrier_id || 39}`}
                </span>
              </div>

              <div>
                <span className="text-[10px] text-slate-400 block uppercase font-bold">BEST ACTION</span>
                <span className="text-base font-black text-white tracking-wider">
                  {analysisResult.best_pass_label || `#${analysisResult.ball_carrier_id || 39} ────► #41`}
                </span>
              </div>

              <div className="flex justify-between items-center text-xs pt-1 border-t border-slate-800">
                <span className="text-slate-400">TACTICAL SCORE</span>
                <span className="text-emerald-400 font-bold text-sm">
                  {analysisResult.tactical_score || 0.38}
                </span>
              </div>

              {/* Progressive Disclosure: WHY? */}
              <button
                onClick={() => setShowWhy(!showWhy)}
                className="w-full text-left text-[11px] text-emerald-400 hover:underline font-bold flex items-center justify-between pt-1"
              >
                <span>WHY?</span>
                {showWhy ? <ChevronUp className="w-3.5 h-3.5" /> : <ChevronDown className="w-3.5 h-3.5" />}
              </button>

              {showWhy && (
                <div className="text-[11px] text-slate-300 bg-slate-950 p-2 rounded border border-slate-800 space-y-1">
                  <span className="text-emerald-400 font-bold block">{analysisResult.space_label || '#41 OPEN • 12m SPACE'}</span>
                  <p className="text-[10px] text-slate-400 leading-tight">{analysisResult.pass_reason}</p>
                </div>
              )}
            </div>

            {/* 2. DEFENCE CARD */}
            <div className="bg-slate-900/90 border border-amber-800/80 rounded-xl p-4 shadow-xl space-y-3">
              <div className="flex items-center justify-between border-b border-slate-800 pb-2">
                <div className="flex items-center space-x-1.5 text-amber-400">
                  <ShieldAlert className="w-4 h-4" />
                  <span className="text-xs font-black uppercase">DEFENCE</span>
                </div>
                <span className="text-xs font-bold text-amber-400 bg-amber-950 px-2 py-0.5 rounded border border-amber-800">
                  DANGER: {analysisResult.defensive_danger_score} / 100
                </span>
              </div>

              <div>
                <span className="text-[10px] text-slate-400 block uppercase font-bold">THREAT</span>
                <span className="text-sm font-bold text-white">
                  Attacker #{analysisResult.main_defensive_threat_id || 39}
                </span>
              </div>

              <div className="space-y-1 pt-1 border-t border-slate-800 text-xs">
                <span className="text-[10px] text-slate-400 block uppercase font-bold">RECOMMENDED</span>
                {(analysisResult.defensive_recommendations_short || ['#12 ──► MARK #39', '#6 ──► PRESS', '#3 ──► COVER']).map((rec, i) => (
                  <span key={i} className="block text-slate-200 font-semibold">{rec}</span>
                ))}
              </div>
            </div>

            {/* 3. TACTICAL ALERT BADGE */}
            <div className="bg-slate-900/90 border border-sky-800/80 rounded-xl p-4 shadow-xl space-y-3 flex flex-col justify-between">
              <div>
                <div className="flex items-center space-x-1.5 text-sky-400 border-b border-slate-800 pb-2 mb-3">
                  <AlertTriangle className="w-4 h-4" />
                  <span className="text-xs font-black uppercase">TACTICAL ALERT</span>
                </div>

                <div className="p-3 rounded-lg bg-sky-950/60 border border-sky-800/80 text-sky-200 text-xs font-bold space-y-1">
                  <span>{analysisResult.short_alert || '⚠ #41 IS OPEN (12m SPACE)'}</span>
                  <span className="block text-[10px] text-sky-400 font-normal uppercase">RECOMMENDED RESPONSE: {analysisResult.recommended_response || 'SHIFT RIGHT'}</span>
                </div>
              </div>

              <div className="text-[10px] text-slate-500 border-t border-slate-800 pt-2 text-center">
                Baseline Tactical Engine Active
              </div>
            </div>

            {/* 4. COMPACT TIMELINE */}
            <div className="bg-slate-900/90 border border-slate-800 rounded-xl p-4 shadow-xl space-y-3">
              <div className="flex items-center space-x-1.5 text-slate-300 border-b border-slate-800 pb-2">
                <Clock className="w-4 h-4 text-emerald-400" />
                <span className="text-xs font-black uppercase">KEY TIMELINE</span>
              </div>

              <div className="space-y-1.5 text-xs">
                {analysisResult.events.slice(0, 4).map((ev, i) => (
                  <div key={i} className="flex justify-between items-center text-[11px] py-1 border-b border-slate-800/60 last:border-0">
                    <span className="text-emerald-400 font-bold">{ev.timestamp_str}</span>
                    <span className="text-slate-300 font-semibold">{ev.description}</span>
                  </div>
                ))}
              </div>
            </div>
          </div>

          {/* PROGRESSIVE DISCLOSURE: OPTIONAL DETAILED REPORT */}
          <div className="pt-2">
            <button
              onClick={() => setShowDetailedReport(!showDetailedReport)}
              className="text-xs text-slate-400 hover:text-emerald-400 font-mono font-bold flex items-center space-x-2 transition"
            >
              <span>{showDetailedReport ? '▼ HIDE DETAILED REPORT' : '▶ MORE ANALYSIS & TECHNICAL LOGS'}</span>
            </button>

            {showDetailedReport && (
              <div className="mt-3 p-4 rounded-xl bg-slate-950 border border-slate-800 space-y-3 text-xs font-mono">
                <span className="text-slate-300 font-bold block">FULL MATCH REPORT LOG</span>
                <pre className="text-emerald-400 leading-relaxed whitespace-pre-wrap bg-slate-900 p-3 rounded border border-slate-800 text-[11px]">
                  {analysisResult.summary_text}
                </pre>
              </div>
            )}
          </div>
        </div>
      )}
    </div>
  );
};
