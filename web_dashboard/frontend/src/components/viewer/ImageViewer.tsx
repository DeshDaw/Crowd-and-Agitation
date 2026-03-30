/**
 * Image viewer component for annotated frames and heatmaps
 */
import { useEffect, useState } from 'react';
import { ChevronLeft, ChevronRight, X, AlertTriangle } from 'lucide-react';
import { API_BASE_URL } from '../../api/runs';

interface ImageViewerProps {
  runId: string;
  frameNames: string[];
  currentIndex: number;
  mode: 'annotated' | 'heatmap';
  isEvent?: boolean;
  onClose: () => void;
  onNavigate: (index: number) => void;
}

export const ImageViewer = ({
  runId,
  frameNames,
  currentIndex,
  mode,
  isEvent,
  onClose,
  onNavigate,
}: ImageViewerProps) => {
  const [loading, setLoading] = useState(true);

  const currentFrame = frameNames[currentIndex];
  const imageUrl = mode === 'annotated'
    ? `${API_BASE_URL}/api/runs/${runId}/artifacts/annotated/${currentFrame}`
    : `${API_BASE_URL}/api/runs/${runId}/artifacts/heatmaps/${currentFrame.replace(/\.\w+$/, '')}_heatmap$&`;

  const goPrev = () => {
    if (currentIndex > 0) {
      setLoading(true);
      onNavigate(currentIndex - 1);
    }
  };

  const goNext = () => {
    if (currentIndex < frameNames.length - 1) {
      setLoading(true);
      onNavigate(currentIndex + 1);
    }
  };

  const handleKeyDown = (e: KeyboardEvent) => {
    if (e.key === 'Escape') onClose();
    if (e.key === 'ArrowLeft') goPrev();
    if (e.key === 'ArrowRight') goNext();
  };

  // Add keyboard listener
  useEffect(() => {
    window.addEventListener('keydown', handleKeyDown);
    return () => window.removeEventListener('keydown', handleKeyDown);
  });

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
        <button onClick={onClose} className="p-2 text-white hover:bg-white/10 rounded">
          <X className="h-6 w-6" />
        </button>
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
        {loading && (
          <div className="absolute inset-0 flex items-center justify-center">
            <div className="animate-spin h-8 w-8 border-2 border-white border-t-transparent rounded-full" />
          </div>
        )}
        <img
          src={imageUrl}
          alt={currentFrame}
          className="max-w-full max-h-[80vh] object-contain"
          onLoad={() => setLoading(false)}
        />
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
