/**
 * Drag-and-drop file uploader component
 */
import { useCallback, useState } from 'react';
import { useDropzone } from 'react-dropzone';
import { Upload, X, FileImage, Film, AlertCircle } from 'lucide-react';
import { Button } from '../ui/Button';
import { ProgressBar } from '../ui/ProgressBar';

interface FileUploaderProps {
  onFilesSelected: (files: File[], video?: File) => void;
  uploadProgress: number;
  maxFiles?: number;
  maxSizeMB?: number;
}

export const FileUploader = ({
  onFilesSelected,
  uploadProgress,
  maxFiles = 1000,
  maxSizeMB = 100,
}: FileUploaderProps) => {
  const [selectedFiles, setSelectedFiles] = useState<File[]>([]);
  const [selectedVideo, setSelectedVideo] = useState<File | null>(null);
  const [error, setError] = useState<string | null>(null);

  const onDrop = useCallback(
    (acceptedFiles: File[], fileRejections: any[]) => {
      setError(null);

      if (fileRejections.length > 0) {
        const rejection = fileRejections[0];
        if (rejection.errors[0].code === 'file-too-large') {
          setError(`File too large. Maximum size is ${maxSizeMB}MB.`);
        } else if (rejection.errors[0].code === 'file-invalid-type') {
          setError('Invalid file type. Supported: images (jpg, png, webp) and videos (mp4, avi, mov).');
        } else {
          setError(rejection.errors[0].message);
        }
        return;
      }

      // Separate images and video
      const images: File[] = [];
      let video: File | null = null;

      for (const file of acceptedFiles) {
        // Robust video check: some browsers/OS combinations report empty file.type
        const isVideo =
          file.type.startsWith('video/') ||
          /\.(mp4|avi|mov|mkv)$/i.test(file.name);

        if (isVideo) {
          if (video) {
            setError('Only one video file can be uploaded at a time.');
            return;
          }
          video = file;
        } else {
          images.push(file);
        }
      }

      // Check if mixing video and images
      if (video && images.length > 0) {
        setError('Please upload either multiple images OR a single video, not both.');
        return;
      }

      // Check file count
      if (images.length > maxFiles) {
        setError(`Maximum ${maxFiles} images allowed.`);
        return;
      }

      if (video) {
        setSelectedVideo(video);
        setSelectedFiles([]);
      } else {
        setSelectedFiles(images);
        setSelectedVideo(null);
      }
    },
    [maxFiles, maxSizeMB]
  );

  const { getRootProps, getInputProps, isDragActive } = useDropzone({
    onDrop,
    accept: {
      'image/*': ['.jpg', '.jpeg', '.png', '.webp'],
      'video/mp4': ['.mp4'],
      'video/x-msvideo': ['.avi'],
      'video/quicktime': ['.mov'],
      'video/x-matroska': ['.mkv'],
    },
    maxSize: maxSizeMB * 1024 * 1024,
  });

  const handleRemoveFile = (index: number) => {
    setSelectedFiles((prev) => prev.filter((_, i) => i !== index));
  };

  const handleRemoveVideo = () => {
    setSelectedVideo(null);
  };

  const handleUpload = () => {
    if (selectedVideo) {
      onFilesSelected([], selectedVideo);
    } else if (selectedFiles.length > 0) {
      onFilesSelected(selectedFiles);
    }
  };

  const totalSelected = selectedFiles.length + (selectedVideo ? 1 : 0);
  const hasFiles = totalSelected > 0;

  return (
    <div className="space-y-4">
      {/* Dropzone */}
      <div
        {...getRootProps()}
        className={`
          border-2 border-dashed rounded-lg p-8 text-center cursor-pointer
          transition-colors duration-200
          ${isDragActive
            ? 'border-primary-500 bg-primary-50'
            : 'border-slate-300 hover:border-slate-400'
          }
        `}
      >
        <input {...getInputProps()} />
        <div className="flex flex-col items-center">
          <Upload className="h-12 w-12 text-slate-400 mb-4" />
          <p className="text-lg font-medium text-slate-900">
            {isDragActive ? 'Drop files here' : 'Drag & drop files here'}
          </p>
          <p className="text-sm text-slate-500 mt-1">
            or click to select files
          </p>
          <p className="text-xs text-slate-400 mt-4">
            Images: JPG, PNG, WebP (max {maxFiles} files, {maxSizeMB}MB each)
            <br />
            Video: MP4, AVI, MOV (1 file max, {maxSizeMB}MB)
          </p>
        </div>
      </div>

      {/* Error */}
      {error && (
        <div className="flex items-center gap-2 p-3 bg-red-50 text-red-700 rounded-md">
          <AlertCircle className="h-5 w-5 flex-shrink-0" />
          <span className="text-sm">{error}</span>
        </div>
      )}

      {/* Selected Files List */}
      {hasFiles && uploadProgress === 0 && (
        <div className="border rounded-lg p-4 bg-slate-50">
          <h4 className="text-sm font-medium text-slate-900 mb-3">
            Selected ({totalSelected} {selectedVideo ? 'video' : 'images'})
          </h4>
          <div className="space-y-2 max-h-48 overflow-y-auto">
            {selectedVideo && (
              <div className="flex items-center justify-between p-2 bg-white rounded border">
                <div className="flex items-center gap-2">
                  <Film className="h-5 w-5 text-primary-600" />
                  <span className="text-sm truncate max-w-md">{selectedVideo.name}</span>
                  <span className="text-xs text-slate-500">
                    ({(selectedVideo.size / 1024 / 1024).toFixed(2)} MB)
                  </span>
                </div>
                <button
                  onClick={handleRemoveVideo}
                  className="p-1 hover:bg-slate-100 rounded"
                >
                  <X className="h-4 w-4 text-slate-500" />
                </button>
              </div>
            )}
            {selectedFiles.map((file, index) => (
              <div
                key={index}
                className="flex items-center justify-between p-2 bg-white rounded border"
              >
                <div className="flex items-center gap-2">
                  <FileImage className="h-5 w-5 text-primary-600" />
                  <span className="text-sm truncate max-w-md">{file.name}</span>
                  <span className="text-xs text-slate-500">
                    ({(file.size / 1024 / 1024).toFixed(2)} MB)
                  </span>
                </div>
                <button
                  onClick={() => handleRemoveFile(index)}
                  className="p-1 hover:bg-slate-100 rounded"
                >
                  <X className="h-4 w-4 text-slate-500" />
                </button>
              </div>
            ))}
          </div>

          <div className="mt-4 flex justify-end">
            <Button onClick={handleUpload} size="lg">
              Upload Files
            </Button>
          </div>
        </div>
      )}

      {/* Upload Progress */}
      {uploadProgress > 0 && uploadProgress < 100 && (
        <div className="border rounded-lg p-4">
          <p className="text-sm font-medium text-slate-900 mb-2">
            Uploading...
          </p>
          <ProgressBar progress={uploadProgress} />
        </div>
      )}
    </div>
  );
};
