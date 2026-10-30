import logging
log=logging.getLogger("rife_onnx")
def emit(metrics,pairs,wall):
 for m in metrics.values(): log.info("device=%s active=%s pairs=%d cuts=%d generated=%d worker=%.3fs errors=%s",m.device.name,m.active_provider,m.source_pairs,m.scene_cuts,m.generated_frames,m.wall_seconds,m.errors)
 generated=sum(m.generated_frames for m in metrics.values()); log.info("completed pairs=%d generated=%d node_wall=%.3fs frame_s=%.2f",pairs,generated,wall,generated/wall if wall>0 else 0)
