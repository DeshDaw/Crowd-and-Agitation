/**
 * Ground-plane calibration editor: click the 4 corners of a real-world
 * rectangle (known size in metres) on the first input frame. Enables
 * persons/m², Fruin LOS and metric speeds for the run.
 */
import { useEffect, useRef, useState } from 'react';
import { Ruler, RotateCcw, Check } from 'lucide-react';
import { Card, CardHeader, CardTitle, CardContent } from '../ui/Card';
import { Button } from '../ui/Button';
import { Input } from '../ui/Input';
import { runsApi } from '../../api/runs';

interface CalibrationEditorProps {
  runId: string;
}

const POINT_LABELS = ['top-left', 'top-right', 'bottom-right', 'bottom-left'];

export const CalibrationEditor = ({ runId }: CalibrationEditorProps) => {
  // Points stored in natural (full-resolution) image pixels
  const [points, setPoints] = useState<[number, number][]>([]);
  const [widthM, setWidthM] = useState(5);
  const [heightM, setHeightM] = useState(5);
  const [saved, setSaved] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [previewFailed, setPreviewFailed] = useState(false);
  const imgRef = useRef<HTMLImageElement | null>(null);

  useEffect(() => {
    // Restore an already-saved calibration when revisiting the step
    runsApi
      .getCalibration(runId)
      .then((c) => {
        setPoints(c.image_points.map((p) => [p[0], p[1]] as [number, number]));
        setWidthM(c.width_m);
        setHeightM(c.height_m);
        setSaved(true);
      })
      .catch((err: any) => {
        // 404 = no calibration yet (normal); anything else is worth surfacing
        if (err.response?.status !== 404) {
          setError('Could not load existing calibration');
        }
      });
  }, [runId]);

  const handleClick = (e: React.MouseEvent<HTMLImageElement>) => {
    if (points.length >= 4 || !imgRef.current) return;
    const img = imgRef.current;
    // Ignore clicks before the image has decoded — naturalWidth is 0 and
    // every point would collapse to (0,0)
    if (!img.naturalWidth || !img.naturalHeight) return;
    const rect = img.getBoundingClientRect();
    if (!rect.width || !rect.height) return;
    const scaleX = img.naturalWidth / rect.width;
    const scaleY = img.naturalHeight / rect.height;
    const x = (e.clientX - rect.left) * scaleX;
    const y = (e.clientY - rect.top) * scaleY;
    setPoints((p) => [...p, [x, y]]);
    setSaved(false);
  };

  const reset = () => {
    setPoints([]);
    setSaved(false);
    setError(null);
  };

  const save = async () => {
    if (points.length !== 4 || !imgRef.current) return;
    try {
      setError(null);
      const img = imgRef.current;
      await runsApi.saveCalibration(runId, {
        image_points: points.map((p) => [p[0], p[1]]),
        width_m: widthM,
        height_m: heightM,
        image_size: [img.naturalWidth, img.naturalHeight],
      });
      setSaved(true);
    } catch (err: any) {
      setError(err.response?.data?.detail || 'Failed to save calibration');
    }
  };

  // Display coordinates for overlay dots (natural px -> rendered px)
  const toDisplay = (p: [number, number]): [number, number] => {
    const img = imgRef.current;
    if (!img || !img.naturalWidth || !img.naturalHeight) return [0, 0];
    const rect = img.getBoundingClientRect();
    return [
      (p[0] / img.naturalWidth) * rect.width,
      (p[1] / img.naturalHeight) * rect.height,
    ];
  };

  if (previewFailed) return null; // no previewable input — hide the card

  return (
    <Card>
      <CardHeader>
        <div className="flex items-center justify-between">
          <div className="flex items-center gap-2">
            <Ruler className="h-5 w-5 text-primary-600" />
            <CardTitle>Ground Calibration (optional)</CardTitle>
          </div>
          {saved && (
            <span className="flex items-center gap-1 text-sm text-green-600">
              <Check className="h-4 w-4" /> saved
            </span>
          )}
        </div>
        <p className="text-sm text-slate-500 mt-1">
          Click the 4 corners of a rectangle on the ground with known real
          dimensions (tile grid, court markings, taped square) in the order:
          top-left, top-right, bottom-right, bottom-left. Enables persons/m²,
          Fruin Level of Service, and metric speeds.
        </p>
      </CardHeader>
      <CardContent className="space-y-4">
        <div className="relative inline-block max-w-full">
          <img
            ref={imgRef}
            src={`/api/runs/${runId}/input-preview`}
            alt="calibration preview"
            className={`max-w-full rounded border ${points.length < 4 ? 'cursor-crosshair' : ''}`}
            onClick={handleClick}
            onError={() => setPreviewFailed(true)}
            draggable={false}
          />
          {points.map((p, i) => {
            const [dx, dy] = toDisplay(p);
            return (
              <div
                key={i}
                className="absolute -translate-x-1/2 -translate-y-1/2 pointer-events-none"
                style={{ left: dx, top: dy }}
              >
                <div className="w-3 h-3 bg-red-500 rounded-full ring-2 ring-white" />
                <span className="absolute left-3 top-0 text-xs font-medium text-red-600 bg-white/80 px-1 rounded whitespace-nowrap">
                  {POINT_LABELS[i]}
                </span>
              </div>
            );
          })}
        </div>

        <div className="text-sm text-slate-600">
          {points.length < 4
            ? `Click point ${points.length + 1} of 4: ${POINT_LABELS[points.length]}`
            : 'All 4 points placed — set the real dimensions and save.'}
        </div>

        <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
          <Input
            label="Rectangle width (m)"
            type="number"
            min="0.1"
            step="0.1"
            value={widthM}
            onChange={(e) => {
              const v = parseFloat(e.target.value);
              if (!Number.isNaN(v)) { setWidthM(v); setSaved(false); }
            }}
            helperText="Top-left → top-right distance"
          />
          <Input
            label="Rectangle height (m)"
            type="number"
            min="0.1"
            step="0.1"
            value={heightM}
            onChange={(e) => {
              const v = parseFloat(e.target.value);
              if (!Number.isNaN(v)) { setHeightM(v); setSaved(false); }
            }}
            helperText="Top-left → bottom-left distance"
          />
        </div>

        {error && (
          <div className="p-3 bg-red-50 text-red-700 rounded text-sm">{error}</div>
        )}

        <div className="flex gap-3">
          <Button variant="secondary" onClick={reset} disabled={points.length === 0}>
            <RotateCcw className="h-4 w-4 mr-2" />
            Reset points
          </Button>
          <Button onClick={save} disabled={points.length !== 4 || saved}>
            {saved ? 'Saved' : 'Save calibration'}
          </Button>
        </div>
      </CardContent>
    </Card>
  );
};
