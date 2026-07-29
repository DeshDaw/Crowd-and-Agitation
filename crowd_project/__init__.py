"""
Crowd Intelligence System — Detectron2 multi-model batch pipeline.

Making this a real package guarantees `crowd_project.config` is a single
module object; the pre-package layout allowed `import config` and
`import crowd_project.config` to create two independent copies, which made
runtime configuration overrides silently invisible to the pipeline.
"""
