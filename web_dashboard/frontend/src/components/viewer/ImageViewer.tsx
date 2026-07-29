/**
 * Image viewer component for annotated frames and heatmaps
 */
import { useCallback, useEffect, useState } from 'react';
import { ChevronLeft, ChevronRight, X, AlertTriangle, ImageOff } from 'lucide-react';

interface ImageViewerProps {
  runId: string;
  frameNames: string[];
  currentIndex: number;
  mode: 'annotated' | 'heatmap';
  isEvent?: boolean;
  onClose: () => void;
  onNavigate: (index: number) => void;
}

/** frame_001.jpg -> frame_001_heatmap.jpg (matches the pipeline's naming) */
const heatmapName = (frame: string) => frame.replace(/(\.\w+)$/, '_heatmap$1');

export const ImageViewer = ({
  runId,
  frameNames,
  currentIndex,
  mode: initialMode,
  isEvent,
  onClose,
  onNavigate,
}: ImageViewerProps) => {
  const [loading, setLoading] = useState(true);
  const [failed, setFailed] = useState(false);
  const [mode, setMode] = useState<'annotated' | 'heatmap'>(initialMode);

  const currentFrame = frameNames[currentIndex];

  // Reset load state whenever the displayed image changes, regardless of
  // whether navigation came from inside or from the parent
  useEffect(() => {
    setLoading(true);
    setFailed(false);
  }, [currentIndex, mode]);

  const goPrev = useCallback(() => {
    if (currentIndex > 0) onNavigate(currentIndex - 1);
  }, [currentIndex, onNavigate]);

  const goNext = useCallback(() => {
    if (currentIndex < frameNames.length - 1) onNavigate(currentIndex + 1);
  }, [currentIndex, frameNames.length, onNavigate]);

  useEffect(() => {
    const handleKeyDown = (e: KeyboardEvent) => {
      if (e.key === 'Escape') onClose();
      if (e.key === 'ArrowLeft') {
        e.preventDefault();
        goPrev();
      }
      if (e.key === 'ArrowRight') {
        e.preventDefault();
        goNext();
      }
    };
    window.addEventListener('keydown', handleKeyDown);
    return () => window.removeEventListener('keydown', handleKeyDown);
  }, [onClose, goPrev, goNext]);

  if (!currentFrame) return null;

  const imageUrl =
    mode === 'annotated'
      ? `/api/runs/${runId}/artifacts/annotated/${encodeURIComponent(currentFrame)}`
      : `/api/runs/${runId}/artifacts/heatmaps/${encodeURIComponent(heatmapName(currentFrame))}`;

  return (
    <div className="fixed inset-0 z-50 bg-black/90 flex items-center justify-center">
      {/* Header */}
      <div className="absolute top-0 left-0 right-0 p-4 flex items-center justify-between bg-gradient-to-b from-black/50 to-transparent">
        <div className="flex items-center gap-3">
          <span className="text-white font-medium">
            {currentIndex + 1} / {frameNames.length}
          </span>
          <span className="text-white/80">{currentFrame}</span>
          {isEvent && (
            <span className="flex items-center gap-1 text-red-400 text-sm">
              <AlertTriangle className="h-4 w-4" />
              Event Frame
            </span>
          )}
        </div>
        <div className="flex items-center gap-2">
          <div className="flex rounded overflow-hidden border border-white/30 text-sm">
            <button
              onClick={() => setMode('annotated')}
              className={`px-3 py-1 ${mode === 'annotated' ? 'bg-white/20 text-white' : 'text-white/60 hover:text-white'}`}
            >
              Annotated
            </button>
            <button
              onClick={() => setMode('heatmap')}
              className={`px-3 py-1 ${mode === 'heatmap' ? 'bg-white/20 text-white' : 'text-white/60 hover:text-white'}`}
            >
              Heatmap
            </button>
          </div>
          <button onClick={onClose} className="p-2 text-white hover:bg-white/10 rounded">
            <X className="h-6 w-6" />
          </button>
        </div>
      </div>

      {/* Navigation */}
      {currentIndex > 0 && (
        <button
          onClick={goPrev}
          className="absolute left-4 p-3 text-white hover:bg-white/10 rounded-full"
        >
          <ChevronLeft className="h-8 w-8" />
        </button>
      )}
      {currentIndex < frameNames.length - 1 && (
        <button
          onClick={goNext}
          className="absolute right-4 p-3 text-white hover:bg-white/10 rounded-full"
        >
          <ChevronRight className="h-8 w-8" />
        </button>
      )}

      {/* Image */}
      <div className="max-w-[90vw] max-h-[80vh] relative">
        {loading && !failed && (
          <div className="absolute inset-0 flex items-center justify-center">
            <div className="animate-spin h-8 w-8 border-2 border-white border-t-transparent rounded-full" />
          </div>
        )}
        {failed ? (
          <div className="flex flex-col items-center gap-3 text-white/70 p-12">
            <ImageOff className="h-10 w-10" />
            <span>
              {mode === 'heatmap' ? 'Heatmap' : 'Annotated frame'} not available
            </span>
          </div>
        ) : (
          <img
            src={imageUrl}
            alt={currentFrame}
            className="max-w-full max-h-[80vh] object-contain"
            onLoad={() => setLoading(false)}
            onError={() => {
              setLoading(false);
              setFailed(true);
            }}
          />
        )}
      </div>

      {/* Footer */}
      <div className="absolute bottom-0 left-0 right-0 p-4 bg-gradient-to-t from-black/50 to-transparent text-center">
        <span className="text-white/60 text-sm">
          Use arrow keys to navigate, ESC to close
        </span>
      </div>
    </div>
  );
};
