/**
 * Run configuration form
 */
import { Input, Select, Checkbox } from '../ui/Input';
import { Card, CardHeader, CardTitle, CardContent } from '../ui/Card';
import type { RunConfig } from '../../types/api';

interface ConfigFormProps {
  config: Partial<RunConfig>;
  onChange: (config: Partial<RunConfig>) => void;
  cudaAvailable: boolean;
}

export const ConfigForm = ({ config, onChange, cudaAvailable }: ConfigFormProps) => {
  const handleChange = (key: keyof RunConfig, value: any) => {
    onChange({ ...config, [key]: value });
  };

  return (
    <Card>
      <CardHeader>
        <CardTitle>Processing Configuration</CardTitle>
      </CardHeader>
      <CardContent className="space-y-6">
        {/* Device Selection */}
        <Select
          label="Device"
          value={config.device || 'cpu'}
          onChange={(e) => handleChange('device', e.target.value)}
          options={[
            { value: 'cpu', label: 'CPU' },
            { value: 'cuda', label: `CUDA ${cudaAvailable ? '(available)' : '(not available)'}`,
          },
          ]}
          disabled={!cudaAvailable}
        />

        {/* Inference Settings */}
        <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
          <Input
            label="Confidence Threshold"
            type="number"
            min="0"
            max="1"
            step="0.05"
            value={config.confidence_threshold ?? 0.5}
            onChange={(e) => handleChange('confidence_threshold', parseFloat(e.target.value))}
            helperText="Detection confidence threshold (0-1)"
          />
          <Input
            label="Pose Confidence Threshold"
            type="number"
            min="0"
            max="1"
            step="0.05"
            value={config.pose_confidence_threshold ?? 0.5}
            onChange={(e) => handleChange('pose_confidence_threshold', parseFloat(e.target.value))}
            helperText="Keypoint detection threshold"
          />
        </div>

        <Input
          label="Max Inference Width"
          type="number"
          min="320"
          max="2048"
          step="32"
          value={config.max_inference_width ?? 960}
          onChange={(e) => handleChange('max_inference_width', parseInt(e.target.value))}
          helperText="Resize width for inference (smaller = faster)"
        />

        {/* Tracking Settings */}
        <div className="border-t pt-4">
          <h4 className="text-sm font-medium text-slate-900 mb-3">Tracking</h4>
          <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
            <Input
              label="IoU Threshold"
              type="number"
              min="0"
              max="1"
              step="0.05"
              value={config.tracker_iou_threshold ?? 0.3}
              onChange={(e) => handleChange('tracker_iou_threshold', parseFloat(e.target.value))}
              helperText="Intersection-over-Union threshold for matching"
            />
            <Input
              label="Max Lost Frames"
              type="number"
              min="1"
              max="100"
              value={config.tracker_max_lost ?? 5}
              onChange={(e) => handleChange('tracker_max_lost', parseInt(e.target.value))}
              helperText="Frames before dropping a track"
            />
          </div>
        </div>

        {/* Density Settings */}
        <div className="border-t pt-4">
          <h4 className="text-sm font-medium text-slate-900 mb-3">Density Classification</h4>
          <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
            <Input
              label="Low Density Sigma"
              type="number"
              min="0.1"
              max="5"
              step="0.1"
              value={config.density_low_sigma ?? 0.5}
              onChange={(e) => handleChange('density_low_sigma', parseFloat(e.target.value))}
            />
            <Input
              label="High Density Sigma"
              type="number"
              min="0.1"
              max="10"
              step="0.1"
              value={config.density_high_sigma ?? 1.5}
              onChange={(e) => handleChange('density_high_sigma', parseFloat(e.target.value))}
            />
          </div>
        </div>

        {/* Agitation Settings */}
        <div className="border-t pt-4">
          <h4 className="text-sm font-medium text-slate-900 mb-3">Agitation Detection</h4>
          <Input
            label="Agitation Threshold Sigma"
            type="number"
            min="0"
            max="5"
            step="0.1"
            value={config.agitation_threshold_sigma ?? 2.0}
            onChange={(e) => handleChange('agitation_threshold_sigma', parseFloat(e.target.value))}
            helperText="Standard deviations above mean for event threshold"
          />
        </div>

        {/* Video Settings */}
        <div className="border-t pt-4">
          <h4 className="text-sm font-medium text-slate-900 mb-3">Video Extraction</h4>
          <Input
            label="Extract FPS (optional)"
            type="number"
            min="1"
            max="60"
            value={config.video_extract_fps || ''}
            onChange={(e) => handleChange('video_extract_fps', e.target.value ? parseFloat(e.target.value) : undefined)}
            helperText="Leave empty to use video's native FPS"
          />
        </div>

        {/* Output Toggles */}
        <div className="border-t pt-4">
          <h4 className="text-sm font-medium text-slate-900 mb-3">Output Options</h4>
          <div className="grid grid-cols-2 gap-4">
            <Checkbox
              label="Save annotated frames"
              checked={config.save_annotated ?? true}
              onChange={(e) => handleChange('save_annotated', e.target.checked)}
            />
            <Checkbox
              label="Save heatmaps"
              checked={config.save_heatmaps ?? true}
              onChange={(e) => handleChange('save_heatmaps', e.target.checked)}
            />
            <Checkbox
              label="Generate plots"
              checked={config.generate_plots ?? true}
              onChange={(e) => handleChange('generate_plots', e.target.checked)}
            />
            <Checkbox
              label="Save database"
              checked={config.save_database ?? true}
              onChange={(e) => handleChange('save_database', e.target.checked)}
            />
          </div>
        </div>
      </CardContent>
    </Card>
  );
};
